# Implementation log: MedNest (IEA/AIE 2027)

Working log for the FlexSLM medical-benchmark paper. **Every session reads this
first and appends to it before ending** (newest entries at the bottom of §8).
It replaces Claude's per-project memory as the record of decisions and state.

---

## 1. Goal and deadline

- Paper: **"MedNest: Budget-Adaptive Nested Language Models for Medical Question
  Answering at the Edge"**, IEA/AIE 2027 **main track**, **long paper** (12 pages
  *including* references), Springer LNCS/LNAI, double-blind, Microsoft CMT.
- Submission deadline **2026-12-15** (CFP gives no time zone; check CMT).
  Notification 2027-03-10, registration 2027-04-10, camera-ready 2027-04-15,
  conference Naples, 2027-07-20/22 (one author must present).
- Sole author. Nothing is published yet; "FlexSLM" is only the repo name and must
  not appear in the paper (method macro `\method` = MedNest in `paper/main.tex`).
- Draft: `paper/main.tex`, `paper/references.bib`, PDF
  `paper/MedNest_IEA-AIE2027_draft.pdf` (11/12 pages; red `\tbd{}` = missing
  numbers, blue `\todo{}` = writing tasks). `paper/` is gitignored (user's choice).

## 2. Claim and decisions (from the user)

| Decision | Date |
|---|---|
| Claim: nested (elastic) SLM fine-tuning for medical QA on embedded, resource-constrained devices | 2026-11-09 |
| Main backbone **Qwen3-0.6B**; **Qwen3-1.7B** as the scaling backbone once the recipe is final | 2026-11-09 |
| Recipe approved: importance ordering → Stage 1 KD recovery on PMC text → Stage 2 nested SFT (FaithMed schedule) | 2026-11-09 |
| **KD teacher is always a bigger model** than the nested model's full width (1.7B for 0.6B; bigger for 1.7B). No in-place/self distillation | 2026-11-09 |
| **No fine-tuning of Qwen3-4B.** 1.7B-backbone nested runs are ON HOLD until an existing bigger fine-tuned teacher is approved (candidate below) | 2026-11-09 |
| Re-fine-tune Qwen3-1.7B (plain SFT) with the corrected code: 1.7B baseline + Stage-2 teacher for 0.6B | 2026-11-09 |
| **Exactly one seed per experiment** (seed 0), no repeats | 2026-11-09 |
| Device: **Jetson Orin Nano 8 GB dev kit (Super)**, JetPack 6.2.1 (+b38), L4T 36.4.7, 7,619 MB, Python 3.10.12, **15 W** mode (source: `C:\Users\skalra\Workspace\Matryoshka Learning\jetson_logs\exp01_env.jsonl`) | 2026-11-09 |
| Report numbers in `paper/student_results/` (PDF + docx) are trusted; do not re-verify them | 2026-11-09 |
| All experiments run on **DAS-6** | 2026-11-09 |

Bigger-teacher candidate for the 1.7B backbone (not approved yet):
`Intelligent-Internet/II-Medical-8B` (Qwen3-8B, SFT + DAPO; same 151,936-token
vocab; Apache-2.0; answers in `\boxed{}`). Caveats: it is a reasoning model and our
prompts disable thinking; trained on MedQA/MedMCQA-derived synthetic data
(contamination unknown); 8B in bf16 (16 GB) next to a 1.7B student on an A40 is tight.

## 3. Results we already have (trusted, from the student reports)

Qwen3-1.7B, naive nested recipe (no ordering, no Stage 1, broken GQA mapping),
FaithMed's 7 test sets, answer-only, greedy, 32 tokens:

| Model | MedQA | MCQA | HeadQA | Calc | Bullets | Pro-H | Xpert | Avg |
|---|---|---|---|---|---|---|---|---|
| Untrained 1.7B | 48.0 | 43.3 | 62.4 | 9.8 | 31.5 | 36.6 | 9.8 | 34.5 |
| Conventional FT 1.7B | 47.9 | 46.6 | 61.1 | 31.9 | 37.6 | 28.5 | 11.4 | 37.9 |
| Nested L2 1.7B | 49.9 | 48.6 | 63.3 | 33.4 | 43.0 | 30.0 | 12.0 | 40.0 |
| Nested L1 0.82B | 26.6 | 30.3 | 26.4 | 4.7 | 24.2 | 12.0 | 9.2 | 19.1 |
| Nested L0 0.25B | 25.5 | 30.0 | 25.5 | 4.7 | 22.1 | 13.7 | 9.6 | 18.7 |

Lesson: letter-scored MCF for ~0.16B models reported the share of "A" answers
(e.g. MedQA 27.7 = 27.7 % A); cloze scoring removed the artefact.

## 4. Code state (branch `medical_benchmark`)

Base commit `07daef4` (Remove segm folder); the 2026-11-09 work is committed on top
and pushed to `origin/medical_benchmark`.

**Layout (2026-11-09):**
- `training/` package: `trainer.py` (was `training.py`; re-exported by
  `training/__init__.py` so `import training` still works), `memopt.py`, `sft.py`
  (SFT configs `SFT_RUNS`), `run_experiment.py`, `export_flex_hf.py`,
  `recover_ckpt.py`, `prepare_*.py`, `fetch_pmc_commercial_ids.py`.
- `eval/` package: `eval_gen.py` (answer-only FaithMed protocol),
  `eval_lm_harness.py`, `eval_wikitext103.py`, `eval_domain_ppl.py`,
  `compare_paired.py` (McNemar), `medcloze/` task YAMLs.
- `scripts/`: all shell scripts (old ones target the previous `gpu_h100`
  cluster) plus `check_qwen3_warmstart.py`, `bench_memopt.py`; DAS-6 launchers in
  `scripts/das6/`.
- Run from the repo root: `python -m training.run_experiment run flexllama,<cfg>`,
  `python -m eval.eval_gen <model> --out ...`, `python -m training.sft check <cfg>`.

**Fixes and changes made 2026-11-09:**
1. `FlexLLaMAConfig.head_dim` (decoupled head size): Qwen3-0.6B has head_dim 128
   but hidden/heads = 64; the old code assumed equality everywhere (attention,
   loader, exporter). Old path unchanged when `head_dim=None`.
2. GQA-preserving level builder `_flex_llama_dims_fixed_head`
   (`config/experiments.py`): the old builder broke the query-head → KV-group
   mapping in small levels (old 1.7B run: L0 5q/5kv, L1 11q/1kv vs 2 q per KV
   group in Qwen3). Presets `qwen3-0.6b` and `qwen3-1.7b-gqa` use it; old
   `qwen3-1.7b` kept for the existing results. KD-only presets `qwen3-4b`, `qwen3-8b`.
3. Gradient clipping is real now (`FlexModelTrainer._optimizer_step`): manual
   optimization had silently ignored `gradient_clip_val`. Paper configs clip at 1.0.
4. One fixed seed: `TrainingContext.seed` (default 0) → `pl.seed_everything` in `finetune`.
5. KD trainer no longer loads the teacher when `kd_lambda == 0`.
6. Readable output names: `utils.make_str_filename_safe` now gives
   `flexllama__sft.paper.qwen3_1p7b_teacher` (was `flexxxxllamaxn...`);
   `utils.resolve_named_path` still finds legacy-named checkpoints/exports.
7. Storage via env `FLEXSLM_STORE` (checkpoints, saved_models, wandb) in
   `config/paths.py`; `config/hardware.py` emits DAS-6 sbatch args (no `--mem`).
8. FlexViT/vision code removed (networks vit/deit/resnet/vgg, flex conv/bn/
   class/distill tokens, `distillation/`, image loaders, Mixup/timm, `SimpleTrainer`,
   vision tests). Base `PosEmbeddingLayer.forward` restored (was missing).
9. Smoke hooks: `FLEXSLM_SMOKE_STEPS=N` (N train steps), `FLEXSLM_SMOKE_LONGEST=1`
   (longest SFT examples first, to test peak memory).
10. `requirements.txt` rewritten for the LLM stack.

**Tests:** local CPU (`prune_llm` conda env, torch 2.6 CPU, transformers 4.51.1):
`test/test_modules.py test/test_head_dim_decoupled.py test/test_memopt_equivalence.py
test/test_nanochat_horizon.py` → 33 passed. Other tests need FineWeb data on a cluster.

**Paper configs** (`training/sft.py`, `_PAPER` = FaithMed schedule: lr 5e-5, 1 %
warm-up, cosine, 1 epoch, effective batch 32 = 16/GPU × 2 A40, clip 1.0, seed 0,
max_seq 3072):
- `sft.paper.qwen3_1p7b_teacher`: plain SFT of full Qwen3-1.7B (levels [1024,1536,2048]
  hidden, heads 8/12/16, kv 4/6/8; only level 2 trained, no KD). Export →
  `hf_export/flexllama__sft.paper.qwen3_1p7b_teacher/level2` = Stage-2 teacher.
- `sft.paper.qwen3_0p6b_conventional`: plain SFT of full Qwen3-0.6B (0.6B baseline;
  levels hidden 512/768/1024, heads 8/12/16, kv 4/6/8, ffn 1536/2304/3072).

## 5. DAS-6 setup

- Login: `SSH_AUTH_SOCK=$HOME/.ssh/agent.sock ssh -o BatchMode=yes skalra@fs2.das6.science.uva.nl`
  (user loads the key into the agent in Git Bash).
- Env: no DAS-6 env has transformers ≥ 4.51 (checked prune_llm, flexllava-refs,
  matryoshka-mm, base, texcompile). Use `/var/scratch/skalra/prune_llm` (torch
  2.7.0+cu126, lightning 2.5.3) + overlay `/var/scratch/skalra/flexslm/pylibs`
  (transformers 4.51.1, tokenizers 0.21.1, pytest 8.3.5 + deps). Never install into prune_llm.
- Code `~/FlexSLM` (sync: `bash scripts/das6/sync.sh` from local; local repo is the
  source of truth). Heavy files `/var/scratch/skalra/flexslm/{store,hf_cache,data,
  hf_export,results,jobs}`; repo `data/ hf_export/ results/ jobs/` are symlinks there.
  HF cache has Qwen3-0.6B and Qwen3-1.7B. Data `data/medmix_sft` (33,406/500/12,797) copied.
- One-off setup: `bash scripts/das6/setup_env.sh` (login node: overlay, symlinks, model download).
- Jobs (all source `scripts/das6/common.sh gpu|cpu`: CUDA module 12.6, env, GPU check):
  `gpu_tests.sbatch`, `train.sbatch <cfg>` (node205 2×A40; smoke with
  `--export=ALL,SMOKE_STEPS=20`), `export.sbatch <cfg>` (fatq node201),
  `evalgen.sbatch <model>` (eval GPUs, excludes node205).
- Rules: sbatch only; GPU on defq 205–208 (training node205, eval 206–208), CPU on
  fatq 201–203; `--time` = 2× pessimistic estimate (max 150 h); never `--mem`;
  never wait > 10 min for a job (submit, record IDs, chain with `afterok`).

## 6. Experiment plan (paper)

| ID | Run | Status |
|---|---|---|
| T1 | Qwen3-1.7B plain SFT (teacher + 1.7B baseline) + export + eval | submitted 2026-11-09, jobs 28057–28061 (§8) |
| B1 | Qwen3-0.6B conventional SFT + eval | config ready, not submitted |
| B0 | Untrained Qwen3-0.6B / 1.7B eval | not submitted |
| M1 | Importance ordering (Taylor) for Qwen3-0.6B | **code not written** |
| M2 | Stage 1: nested KD recovery on PMC, teacher frozen Qwen3-1.7B | config not written |
| M3 | Stage 2: nested SFT 0.6B, KD from T1 export | config not written |
| A1–A5 | Ablations: no ordering / no Stage 1 / no KD / level weights / sandwich vs one-level sampling | not started |
| I1 | Independent pruned+FT 0.36B and 0.19B baselines | not started |
| J1 | Jetson Orin Nano 15 W: latency, memory, energy, level switch time | not started |
| S1 | 1.7B-backbone nested with bigger teacher | ON HOLD (teacher decision) |

## 7. Open questions for the user

- Approve II-Medical-8B (or another existing fine-tuned >1.7B Qwen3) as the 1.7B-backbone teacher?

## 8. Session log (append newest at the bottom)

### 2026-11-09
- Audited repo and both student reports; venue main track, long paper.
- Literature on nested-training recipes (MatFormer, Flextron, LLaMaFlex, Minitron,
  Nemotron Elastic, Matryoshka LM Suites, US-Nets/BigNAS, Sorted LLaMA) and medical
  recipes (FaithMed, BioMistral, Apollo, MedMobile); summary is in the paper's
  related work and method sections.
- Wrote the paper draft; title fixed by the user.
- Code fixes, cleanup and restructure listed in §4; DAS-6 set up (§5).
- Jobs: first chain 28047–28051 failed at 28047 (pytest overlay lacked
  `exceptiongroup`; fixed in `setup_env.sh`), chain cancelled.
- Second chain 28052–28056: 31/33 unit tests passed; the 2 Qwen3 tests failed because
  prune_llm's `flash_attn` wheel needs GLIBC 2.32 (DAS-6 has 2.28) and
  transformers 4.51 imports it. Fix: `setup_env.sh` copies the DAS-6-built
  flash-attn 2.8.3 from `/var/scratch/skalra/elastoformer_moo/pylibs` into the
  FlexSLM overlay (shadows the broken one). Chain cancelled.
- Third chain: **28057** gpu_tests → **28058** smoke (20 steps, teacher config,
  longest batches) → **28059** T1 teacher training (node205, 8 h limit) → **28060**
  export (fatq) → **28061** evalgen of the teacher's level 2. Logs:
  `~/FlexSLM/jobs/<job-name>-<id>.out`. CUDA confirmed working (A10 visible).
- Code committed and pushed to `origin/medical_benchmark` (see git log).
- Next: check 28052–28056; if green, implement M1 (Taylor ordering), write M2/M3
  configs, submit B0/B1.
