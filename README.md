# FlexSLM

FlexSLM extends flexible/slimmable-width nested networks (originally
[FlexViT](#history), for vision transformers) to language models. A single
checkpoint contains N nested-width "flex levels" — from a narrow submodel up
to the full model — that share weights (the narrow levels are literal slices
of the wide ones), so one training run yields a whole family of models at
different compute/quality points. Two architectures are supported:

- **FlexGPT** — GPT-2-style backbone.
- **FlexLLaMA** — LLaMA-style backbone (RoPE, RMSNorm, GQA), with knowledge
  distillation (KD) from real pretrained teachers (LLaMA, TinyLlama, Qwen2.5,
  SmolLM2) and optional warm-starting from those teachers' weights.

## Setup

```
pip3 install -r requirements.txt
```

Configuration lives in `config/`:
- `config/paths.py` — `SCRATCH_PATH` (cluster-wide `/scratch`, for large
  persistent derived data like prepared token corpora), `CHECKPOINT_PATH`,
  `DATA_PATH`, etc.
- `config/hardware.py` — SLURM partition/GPU/time settings per experiment.
  Note: the `gpu_h100` default partition literal here is stale — the cluster
  only has `performance`, `capacity`, and `fp64` (see below).
- `config/wandb.py` — Weights & Biases project name (requires `wandb login`).
- `config/experiments.py` — the `CONFIGS` registry (see below).

## Data pipelines

Three independent FineWeb-Edu loading paths coexist in `utils.py` — pick the
one that matches what your config needs; none of them replace the others.

- **Streaming** (`load_fineweb_edu`) — true streaming, bounded memory, no
  local corpus prep needed. Shards across DDP ranks via
  `split_dataset_by_node`, buffered shuffle. `estimated_steps_per_epoch` is
  attached to the loader since streaming datasets have no exact `len()`.
  Used by the `fineweb.*` configs.
- **Memmap** (`load_fineweb_edu_memmap` / `_MemmapWindowDataset`) — requires
  running `prepare_fineweb.py` first to tokenize the corpus into a `np.memmap`
  `.bin` file under `SCRATCH_PATH/fineweb_bin/<tokenizer>/`. Map-style
  `Dataset`, so `DistributedSampler` shards it exactly and `len()` is exact —
  no memory growth over a long run (verified flat over 2000+ batches, unlike
  streaming). Used by the `fineweb_mm.*` and `fineweb_nc.*` configs.
- `prepare_fineweb.py` is a standalone step, **not** wired into
  `experiment_job.sh` — run it explicitly before using a memmap config:
  ```
  sbatch prepare_fineweb.sh <tokenizer_name> [--sample sample-10BT|sample-100BT|sample-350BT]
  ```

WikiText-103 (`load_wikitext`, used for eval only) takes a `tokenizer_name`
param — pass the model's actual training tokenizer, not the "gpt2" default,
or token ids land outside the embedding table (see [Evaluation](#evaluation)).

## Training

`training.py` is the main Lightning-based trainer (`FlexModelTrainer` /
`FlexLMTrainer`), manual optimization (`automatic_optimization=False`).
`_select_levels()` always trains the min and max flex level every step plus a
random sample of the rest.

- **DDP** (default) and **FSDP** (`use_fsdp=True`) strategies are both wired
  up, but FSDP is currently incompatible with this training loop: it does
  ≥2 forward/backward pairs per step (per the min+max guarantee above), which
  conflicts with FSDP's single-forward-per-step hook state machine
  (`Expects BACKWARD_PRE or BACKWARD_POST ... but got FORWARD`). Stick to DDP;
  reduce `batch_size` if a config needs more memory headroom.
- **`training_memopt.py`** is a separate trainer (`MemOptFlexLMTrainer` /
  `MemOptFlexLMKDTrainer`) for configs that OOM under normal DDP — chiefly KD
  against large-vocabulary teachers (e.g. Qwen2.5's 151936-token vocab blows
  up the softmax/KL-div tensors). It chunks the loss computation (numerically
  exact — verified bit-identical gradients, `max|Δgrad|=0.0`, against the
  normal trainer) and can activation-checkpoint each decoder block. Tag a
  config to use it via the `memopt(...)` helper — it doesn't require
  `training.py` itself to know about memopt configs.
- Checkpointing: `ModelCheckpoint(..., save_last=True)`, mid-epoch cadence at
  `steps_per_epoch // 20`. Training resumes from `last.ckpt` automatically if
  one exists for the config; pass `resume=False` on the `TrainingContext` (or
  the training script's `--no-resume` flag) to force a fresh run instead.
- A rank-0-only experiment banner prints at the start of every run (config
  name, model shape, warm-start/KD teacher info, data pipeline + tokenizer,
  tokens/epoch, resume state, W&B project) — see `test/test_experiment_banner.py`.

### Config registry (`config/experiments.py`)

Experiments are looked up as `"<arch>,<name>"` (e.g.
`flexllama,fineweb.kd_lambda05_1p4B`) via `resolve_from_str`. Config families:

- `fineweb.*` — streaming pipeline.
- `fineweb_mm.*` — memmap pipeline.
- `fineweb_nc.*` — memmap pipeline sized to the
  [nanochat](https://github.com/karpathy/nanochat) token-horizon convention
  (`nanochat_token_horizon(cfg, ratio)`, `tokens = ratio × (transformer_matrices
  + lm_head)`; nanochat's own default ratio is 12, their speedrun uses 8,
  Chinchilla-optimal is 20). `epochs=1` since the horizon already accounts
  for the full token budget.
- `TEACHER_PRESETS` — KD teacher specs (llama-160m, tinyllama-1.1b,
  smollm2-360m/1.7b, qwen2.5-0.5b/1.5b, llama3.2-1b [gated, unverified dims]),
  consumed by `make_flexllama_kd` / `make_flexllama_warmstart_kd`.

## Running experiments

```
sbatch experiment_job.sh <config> [hardware_config]
```
or, to submit every experiment registered under a group at once:
```
./runall.sh <group>
```
`nanoGPT/` and `nanochat/` are reference implementations kept as plain
folders in-repo (not submodules) for comparison while building the memmap
pipeline and token-horizon math — not part of the FlexSLM pipeline itself.

## Evaluation

`eval_job.sh` is the single dispatcher for evaluating a completed checkpoint
(`saved_models/<config>.pt`, written by `utils.save_model()`):

```
sbatch eval_job.sh wiki <config> [dataset] [batch_size]   # eval_wikitext103.py
sbatch eval_job.sh lm   <config> [tasks] [batch_size]     # eval_lm_harness.py
```

- `wiki` — fast per-flex-level WikiText-103 token perplexity, a quick sanity
  cross-check against W&B validation numbers (out-of-distribution for
  FineWeb-Edu-trained models, not a benchmark result). Resolves the
  tokenizer automatically: `--tokenizer` override → KD teacher →
  warm-start source → `"gpt2"` — and hard-fails before running the model if
  the resolved tokenizer's vocab size doesn't match the config, rather than
  crashing deep inside a CUDA kernel.
- `lm` — full [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  suite (`wikitext, lambada_openai, hellaswag, piqa, arc_easy, arc_challenge,
  openbookqa, triviaqa` by default) via the `@register_model("flexllama")` /
  `FlexGPTLMEval`/`FlexLLaMALMEval` classes in `eval_lm_harness.py`, looping
  over every flex level. Needs the `lm_eval` package (pins
  `datasets==3.6.0` — verified the streaming/memmap FineWeb-Edu pipelines are
  unaffected by that downgrade from 4.5.0). Budget `--time` generously: one
  flex level of the full 8-task suite took ~3h from a cold HF-datasets cache;
  a warm cache (datasets already downloaded under `$HOME/.cache/huggingface`
  from a previous run) can finish all levels in well under that. Tokenizer
  resolution (`--tokenizer` override → KD teacher → warm-start source →
  architecture default) and the pre-flight vocab-size check are shared with
  `wiki` mode via `eval_wikitext103.resolve_tokenizer_name`.

## SLURM environment notes

- Partitions: `performance` (5 nodes, 8×RTX 6000 Ada 48GB, needs 32
  CPUs/GPU), `capacity` (8 nodes, 8×L4 23GB, needs 16 CPUs/GPU, max 2
  nodes/job), `fp64` (1 node, 4×MI300A). `MaxTime=7-00:00:00` (168h) is a
  hard per-job ceiling on every partition — check real measured throughput
  (`squeue -o "%L"` + steps/epoch arithmetic) before trusting a long token
  horizon will fit; a nanochat-recipe run at `batch_size=4` measured ~940h
  against the full horizon and had to be reworked.
- `/local_scratch` — node-local NVMe (7TB), **not** shared across nodes and
  **not** auto-cleaned; not visible from the login node either. Use
  `/local_scratch/$USER/$SLURM_JOB_ID` with a cleanup trap.
- `/scratch` (`config.paths.SCRATCH_PATH`) — cluster-wide NFS, visible from
  every node; the right place for large persistent derived data (e.g.
  prepared FineWeb-Edu `.bin` corpora) that's too big for the `$HOME` quota.
- Use `$SLURM_SUBMIT_DIR`, not `$(dirname $0)`/`$HOME`, to resolve the repo
  path inside a job script — `sbatch` copies the script to a spool dir.
- Running a script as `python3 test/foo.py` puts `test/` on `sys.path`, not
  the repo root — set `PYTHONPATH="$SLURM_SUBMIT_DIR"` in the job script.
- TQDM progress bars are `\r`-delimited — pipe a `.out` file through
  `tr '\r' '\n'` before grepping it for anything but the literal last line.
- **Open item**: `submit_flexllama_4gpu.sh` doesn't set `--open-mode=append`,
  so a requeued/restarted job overwrites its own prior log content in place
  (training itself resumes correctly from `last.ckpt` — only log *history*
  is lost). Proposed fix not yet applied.

## Testing

`test/` holds correctness checks that run via `sbatch` (never directly on
the login node): gradient-equivalence for the memopt chunked-loss trainer
(`test_memopt_equivalence.py`), the memmap data pipeline
(`test_memmap_pipeline.py`, `test_memmap_memopt_combo.py`), nanochat
token-horizon math (`test_nanochat_horizon.py`, `test_nanochat_configs.py`),
the experiment banner (`test_experiment_banner.py`), and flex-module/training
sanity checks (`test_modules.py`, `test_training.py`, `test_delta_manager.py`).

## History

FlexSLM began as **FlexViT**, a slimmable Vision Transformer trained on
ImageNet-1k/CIFAR-10 (see git history prior to the `FlexSLM` commit for the
original vision-only workflow: `flex_modules/`'s conv2d/batchnorm/class-token
modules and `plot.py` are holdovers from that era). The project was then
extended to language models (FlexGPT, FlexLLaMA) with all of the above.
