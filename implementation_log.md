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
11. (2026-11-10) `from_pretrained(dtype=)` -> `torch_dtype=` (transformers 4.51) in
    `eval/eval_gen.py`, `training/trainer.py`, `scripts/check_qwen3_warmstart.py`.
12. (2026-11-10) Config flag `level_weighting` on `SFTTrainingContext`
    (`training/level_weights.py`, used by `FlexLMSFTTrainer`): `none` (default, unchanged
    behaviour) | `uncertainty` (NSN/Kendall learned log-variances, own Adam lr
    `level_weight_lr`=1e-2) | `loss_proportional` (w_k = EMA L_k / mean, EMA
    `level_weight_ema`=0.9). Weights sit outside the model and DDP (rank-averaged
    losses, one all-reduce per step), only scale the backward pass, are saved in the
    Lightning checkpoint, never in the exported model; logged losses/val_loss stay
    unweighted; logs `train_level{k}_weight` / `_log_var`. SFT trainer only (Stage 1 KD
    on PMC uses another trainer: add it there for M2 if A4 needs it). Dev config
    `sft.dev.qwen3_0p6b_nested_uncertainty` (smoke only). Tests `test/test_level_weights.py`.
    Later the same day: modes `fixed` (`level_weights_fixed`) and `dwa` (Liu et al. 2019 on
    EMA losses, `dwa_window` 50, `dwa_temperature` 2) added; shared `LevelWeightingMixin`
    used by `FlexLMSFTTrainer` and `MemOptFlexLMKDTrainer` (Stage 1); progress lines now
    print tok/s.
13. (2026-11-10) Lightning compat: `trainer.fit(weights_only=)` and
    `load_from_checkpoint(weights_only=)` only when the installed Lightning takes it
    (`_weights_only_kwarg`; DAS-6 has 2.5.3, local 2.6.1). On 2.5.3 the latter would have
    passed `weights_only` into the trainer's `__init__` and crashed after training.
14. (2026-11-10) `level_sampling="single"` (one uniformly random level per step, seeded by
    (seed, step): same on every rank). Shared `warmup_cosine_scheduler` in `trainer.py`.
15. (2026-11-10) Stage 1 (`training/stage1.py`): `Stage1Context` (token budget
    `train_tokens`, per-step warm-up 1 % + cosine over exactly that budget, lr, KD, level
    weighting/sampling), `FlexLMStage1Trainer` (memopt KD), loader `load_pmc_memmap`
    (random 1,024-token windows from `data/pmc_qwen3/{train,val,test}.bin`, same windows
    for every arm). Data built by `training/prepare_pmc.py` (`scripts/das6/prepare_pmc.sbatch`):
    PMC OA commercial subset from S3, ids shuffled Random(42), docs joined by
    `<|endoftext|>`, val/test docs disjoint from train, 1B train tokens. Per GPU batch
    16 x 1,024 -> 32,768 tokens/step on 2 GPUs.
16. (2026-11-10) DAS-6 GPU jobs set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
    (job 28068 OOM was fragmentation: 36.3 GB allocated + 7.9 GB reserved-free of 44.7 GB).
17. (2026-11-10) Exact micro-batching `micro_batch_size` (sequences per forward/backward
    inside one optimizer step; losses normalised over the whole batch, so the gradient is
    unchanged; tests compare against one pass) in `FlexLMSFTTrainer` and
    `MemOptFlexLMKDTrainer`. Paper SFT configs use 8 (T1 28075 OOM at step 731: 43.5 GB, the
    [N, 151,936] logits + their gradient for 16 x 3,072); Stage 1 uses 4 (28080 OOM at step 0).
    T1 smoke with 8: 41.5 GB peak on the longest batches.
18. (2026-11-10) M1 `training/importance_order.py` (`scripts/das6/importance_order.sbatch`):
    Taylor |x * dL/dx| on 256 x 1,024 PMC train tokens, ported from Elastoformer-MOO for
    MLP channels (down_proj input) and KV groups (o_proj input), plus **residual channels**
    (input of every RMSNorm, summed; Elastoformer does not nest the hidden size). Permutes the
    HF checkpoint (embed/lm_head columns, q/k/v/gate/up columns, o/down rows, residual norms;
    q_norm/k_norm untouched), checks full-model logits unchanged, saves
    `hf_export/qwen3-0.6b.taylor` + `importance.pt` + `ordering.json` (per-level PMC val ppl
    before/after ordering, no training). All Stage-1 configs and A2 start from it
    (`make_stage1(ordering="taylor")`, SFT override `hf_start`); A1 = `s1.qwen3_0p6b.full.no_order`
    -> `sft.paper.qwen3_0p6b_nested.no_order`. Tests `test/test_importance_order.py`.
19. (2026-11-10) Budgets: Stage-1 pilots 20M tokens, final `s1.qwen3_0p6b.full` 100M (user).
    Baseline B2 `sft.paper.qwen3_0p6b_matformer` (one random level/step, CE only, no ordering,
    no Stage 1, no KD: MatFormer's schedule on our levels; FFN-only MatFormer cannot reach 0.19B).

**Tests:** local CPU (`prune_llm` conda env, torch 2.6 CPU, transformers 4.51.1):
`test/test_modules.py test/test_head_dim_decoupled.py test/test_memopt_equivalence.py
test/test_nanochat_horizon.py` → 33 passed; + `test/test_level_weights.py` + `test/test_importance_order.py` → 63 passed (2026-11-10). Other tests need FineWeb data on a cluster.

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
- Rules: sbatch only; GPU on defq 205–208 (training node205, and node206 1x A40 for Stage-1 pilots (user, 2026-11-10): `sbatch --nodelist=node206 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=64 scripts/das6/train.sbatch <cfg>` (64 CPUs since 2026-11-13); eval 206–208), CPU on
  fatq 201–203; `--time` = 2× pessimistic estimate (max 150 h); never `--mem`;
  never wait > 10 min for a job (submit, record IDs, chain with `afterok`).

## 6. Experiment plan (paper)

| ID | Run | Status |
|---|---|---|
| T1 | Qwen3-1.7B plain SFT (teacher + 1.7B baseline) + export + eval | 28075 OOM at step 731; rerun with micro-batch 8: 28082 smoke OK → **28083** train → 28084 export → 28085 eval |
| B1 | Qwen3-0.6B conventional SFT + eval | config ready, not submitted |
| B0 | Untrained Qwen3-0.6B / 1.7B eval | not submitted |
| M1 | Importance ordering for Qwen3-0.6B: a Stage-1 pilot dimension (none / hidden_act / taylor); `FINAL_ORDERING` in `training/importance_order.py` set from the pilots | taylor checkpoint done (28100), hidden_act job 28106; job **28100** (28088/28091/28094 OOM: hook reference cycle kept every graph alive, fixed; 28097 ordered OK, max logit diff 1.65e-4, then crashed in the level-ppl check, fixed) |
| M2 | Stage 1: nested KD recovery on PMC, teacher frozen Qwen3-1.7B | pilots `s1.qwen3_0p6b.pilot.{base,order_hidden_act,order_taylor,levels5,levels5_sw3,grid9_sw3,lr5e-5,lr2e-4,kd1,w_fixed,w_uncertainty,w_lossprop,w_dwa,single}` (20M tokens each, all ordered), final `s1.qwen3_0p6b.full` (100M, only after all hyperparameters incl. levels/sampling/lr/weighting are fixed from the pilots + Stage-2 ablations); PMC data done (1B train tokens, 28079); smoke 28101 |
| M3 | Stage 2: nested SFT 0.6B, KD from T1 export | `sft.paper.qwen3_0p6b_nested` (loads `s1.qwen3_0p6b.full`); full SFT set always |
| A1 | No importance ordering (if FINAL_ORDERING stays none: report the order_* pilots instead) | `s1.qwen3_0p6b.full.no_order` → `sft.paper.qwen3_0p6b_nested.no_order` |
| A2 | No Stage 1 (start = FINAL_ORDERING checkpoint) | `sft.paper.qwen3_0p6b_nested.no_stage1` (smoke 28102) |
| A3 | No KD in Stage 2 | `sft.paper.qwen3_0p6b_nested.no_kd` |
| A4 | Level weights, 5-way (equal = M3, fixed 3/1/1, uncertainty, loss_proportional, dwa) | `sft.paper.qwen3_0p6b_nested.w_*`; same arms as Stage-1 pilots |
| A5 | All levels per step vs one random level | `sft.paper.qwen3_0p6b_nested.single` |
| B2 | MatFormer-style nested SFT baseline | `sft.paper.qwen3_0p6b_matformer` (run at the end) |
| I1 | Independent pruned+FT 0.36B and 0.19B baselines | not started |
| J1 | Jetson Orin Nano 15 W: latency, memory, energy, level switch time | 0.6B shape-only sweep DONE (2026-11-19, random weights, 15 W only); Apollo-2B/Apollo2-2B do not fit in 8 GB; 1.7B pending |
| S1 | 1.7B-backbone nested with bigger teacher | ON HOLD (teacher decision) |

## 7. Open questions for the user

- Approve II-Medical-8B (or another existing fine-tuned >1.7B Qwen3) as the 1.7B-backbone teacher?
- Paper is at the 12-page limit after adding the reference table: something must be cut when results land.

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

### 2026-11-10
- DAS-6 chain 28057–28061 did not run. 28057 (node207, A10): all 33 unit tests
  passed, then `scripts/check_qwen3_warmstart.py` crashed:
  `from_pretrained(..., dtype=...)` is a newer-transformers keyword; 4.51.1 needs
  `torch_dtype=`. 28058–28061 sit in `DependencyNeverSatisfied`.
- Fixed locally (not committed, not synced) in 3 places: `scripts/check_qwen3_warmstart.py`,
  `eval/eval_gen.py` (would have broken every eval), `training/trainer.py` (KD teacher load).
- Resubmitting was blocked by Claude Code's auto-mode permission classifier (the
  scancel + sbatch step). Still to do: `bash scripts/das6/sync.sh`, `scancel 28058-28061`,
  resubmit gpu_tests → smoke (SMOKE_STEPS=20, 1 h) → T1 train → export → evalgen
  `hf_export/flexllama__sft.paper.qwen3_1p7b_teacher/level2`, chained with afterok.
- Discussed learned per-level loss weights (NSN / Kendall uncertainty weighting):
  - The optimum is s_k = log L_k, so the effective weight is 1/L_k (≡ minimising Σ log L_k).
    It **down-weights the high-loss small levels**, the opposite of helping them.
  - At our SFT lr 5e-5 over ~1,044 steps (33,406 / 32), Adam moves s_k by ≤ ~0.05,
    so the weights stay ≈ 1 unless s_k gets its own param group (lr ~1e-2) or the
    closed form 1/stopgrad(EMA L_k) is used.
  - Elastoformer-MOO evidence (SmolLM2-360M, 50M tokens, 1 seed): every weighting
    traded one level against another (fixed 3× on full: full drift +4.4% → +2.8%, f025
    131 → 143), and learned HV weighting did not beat fixed 3×.
  - Plan: treat level weighting as ablation A4 (equal vs uncertainty vs a
    small-favouring variant), not as the main method.
- User asked for the flag: implemented (§4 item 12), 44 local tests pass.
- Resubmitted (user allowed the permissions): 28058–28061 cancelled; **28062**
  gpu_tests (now incl. test_level_weights) → **28063** T1 smoke → **28064** T1 train →
  **28065** export → **28066** evalgen level2. **28067** = 20-step 2×A40 smoke of
  `sft.dev.qwen3_0p6b_nested_uncertainty` (afterok 28062 only, does not gate T1).
- 28062 passed: 44 tests, Qwen3-0.6B and 1.7B warm starts match HF (max logit diff
  < 8e-5, argmax 100 %). 28063 (T1 smoke) running on node205; 28067 waiting for node205.
- No training throughput measured yet (no run has finished). All configs train on the
  full medmix train set (33,406, 1 epoch); there is no data-subset option, and
  SMOKE_STEPS sorts longest-first, so it is unusable for recipe pilots. User wants
  pilots on a small budget first, the full corpus only for the final run (to discuss).
- 28062 tests passed; 28063 and 28067 (smokes) failed: `Trainer.fit(weights_only=)` not in
  Lightning 2.5.3 (fixed, §4 item 13). Resubmitted 28068–28072; 28068 (T1 smoke) reached
  7.5 s/step, 38.9 GB peak on the longest batches, then OOM at step 10 (fragmentation,
  §4 item 16); rank 1 hung, cancelled with 28069–28071.
- User decisions: SFT is cheap (T1 <= ~2.2 h at the smoke's worst-case step time), so every
  SFT experiment is a full run and A4 is a 5-way comparison; pilots-then-full applies to
  Stage 1 only.
- Implemented Stage 1 + pilots + Stage-2/ablation configs (§4 items 14–15, §6); 52 local tests pass.
- Submitted: **28074** T1 smoke → **28075** T1 → **28076** export → **28077** eval level2,
  **28078** Stage-2 smoke (`...nested.no_stage1`, afterok 28076: 0.6B x 3 levels + 1.7B
  teacher at 3,072 tokens), **28079** PMC prep (fatq node202, 1B tokens) → **28080**
  Stage-1 smoke (`s1.dev.qwen3_0p6b_smoke`). 28072 (weighting smoke) still queued for node205.
- Next: M1 (Taylor importance ordering; port from Elastoformer-MOO `--importance-method
  taylor`), then the Stage-1 pilots, then `s1.qwen3_0p6b.full`, then Stage 2 + ablations.
- 28072 (weighting smoke) and 28074 (T1 smoke) passed. **28075 (T1) OOM at step 731/1044**
  (rank 0, 43.5 GB); val_loss had reached 0.371. **28080 (Stage-1 smoke) OOM at step 0.** Fix:
  exact micro-batching (§4 item 17). Failed T1 outputs moved to `*__oom28075`. PMC prep
  28079 done: 1B train + 2M val + 2M test tokens, 76,800 ids read, 126 fetch failures, 51 min.
- User decisions: Stage-1 final run 100M tokens (must fit 5 days), only once every
  hyperparameter (levels, sampling, lr, weighting, ...) is fixed from the analysed pilots;
  report the most relevant ablations; implement M1 now; add SOTA/related work for fair
  comparison; MatFormer runs at the end if needed.
- Implemented M1 (§4 item 18), budgets + B2 MatFormer baseline (item 19); 63 local tests pass.
- Paper: related work + Gemma 3n (MatFormer in production), MedGemma, Medmarks; new Table
  `tab:reference` (FaithMed Table 1 numbers on our 7 benchmarks, reasoning protocol, reference
  only); A4/A5 text updated; baseline (v) MatFormer-style. Compiles, 12 pages, no undefined refs.
  New bib entries marked VERIFY where authors are incomplete.
- Jobs: 28082 T1 smoke OK (41.5 GB) → **28083 T1** (running) → 28084 export → 28085 eval;
  28087 gpu_tests ALL PASSED; 28088 ordering OOM (A10, batch 4), 28091 OOM (1 x 1,024) → **28094** ordering (512-token windows)
  → OOM again (hook reference cycle, fixed); 28097 permutation verified (max |logit diff| 1.65e-4 vs tol 3e-3) but level-ppl
  check consumed a generator (fixed) → **28100** ordering → 28101 Stage-1 smoke; 28102 Stage-2 smoke (afterok 28084, 28100).
- Next: check 28083/28100/28101; read Stage-1 tok/s from 28101 to size pilots (9 x 20M) and the
  100M run; launch pilots; B0/B1 evals still not submitted.
- **M1 result (28100, diagnostics 28104/28105; untrained FlexLLaMA, PMC val ppl L0/L1/L2,
  32 x 512 tokens): importance ordering makes the sub-levels worse.**
  none 4,315/389/10.5 · MLP Taylor 14,679/525 · groups Taylor 13,217/800 · hidden Taylor
  5,570/240 · hidden activation (mean |x| at RMSNorm inputs) 4,530/**170** · all Taylor
  26,763/687 · MLP act×weight (Elastoformer default) 15,091/456 · groups act×weight 7,408/783 ·
  act×weight + hidden act 14,034/723 · MLP Taylor *reversed* 62,282/107,992 · groups reversed
  139,951/372,364. Reversed ≫ worse, so scores and slicing direction are right; the original
  Qwen3-0.6B order already beats every importance order for MLP and KV groups. Hint that it is
  already importance-sorted: 453 of the Taylor top-512 hidden channels lie in the original first
  512 (~256 if random). Untrained ppl in the thousands is only a proxy; what counts is ppl after
  Stage-1 recovery.
- Decision needed (user): treat ordering as a Stage-1 pilot dimension (none / hidden-activation
  only / full Taylor) and pick on post-recovery ppl, instead of assuming M1 helps. Until decided,
  do not launch the pilots. `hf_export/qwen3-0.6b.taylor` exists (full Taylor); 28101 Stage-1 smoke
  uses it (throughput only).
- User decisions: fixed weights 3/1/1 approved; ordering becomes a pilot dimension. Implemented:
  `ORDERINGS = (none, hidden_act, taylor)`, `start_weights()`, `FINAL_ORDERING = "none"`
  (provisional) in `training/importance_order.py`; `--scheme hidden_act` (residual channels by
  mean |x| at RMSNorm inputs, MLP/groups unchanged); pilot base now = original order, new arms
  `order_hidden_act`, `order_taylor` (11 pilots x 20M). 64 local tests pass. Job **28106** builds
  `hf_export/qwen3-0.6b.hidden_act`. Pilots not submitted yet: their --time needs the Stage-1
  throughput from smoke 28101 (now uses the original order). Paper §Importance ordering still
  describes Taylor as the method: rewrite once the pilots decide.
- PMC data: `/var/scratch/skalra/flexslm/data/pmc_qwen3/` (repo `data/` symlink): train.bin 4.0 GB
  (1B uint32 tokens, 75,282 articles), val.bin and test.bin 8 MB each (2M tokens, 152/153 articles),
  meta.json; 3.8 GiB total.
- FaithMed checkpoints: not public (HF API search for faithmed / cxcscmu returns nothing; GitHub
  README only has training code and data-prep scripts, base model passed as a local path).
- 28106 done: hidden_act ordering, untrained ppl L0/L1/L2 4,228 / 170 / 10.6 (none: 4,265 / 313).
- T1 28083: ~9 s/step (step 416 at 65 min; the earlier "14 s/step" was a misread), ~2.6 h total.
- Level count / continuum (user question): GQA (ratio 2) allows head counts 8, 10, ..., 16 in
  [0.5, 1], i.e. at most 5 levels with distinct heads; `_flex_llama_dims_fixed_head` now allows
  repeated head counts (hidden/FFN must still grow), so 7/9-level grids build. A true continuum
  is not supported (flex modules slice by level index into fixed width lists); a dense grid +
  sandwich sampling approximates it. RMSNorm is shared and sliced, so grids add no parameters.
  New pilot arms: `levels5` (all 5 per step, ~1.5x step cost), `levels5_sw3` and `grid9_sw3`
  (min + max + 1 random per step, ~1.0x). Compared at widths 0.5/0.75/1.0 (levels 0,2,4 / 0,4,8).
- Bug fixed: sandwich sampling (`num_levels_per_step`) used an unseeded `random.sample`, so
  DDP ranks could train different middle levels in one step; now seeded by (seed, step).
  66 local tests pass.
- Submitted 14 Stage-1 pilots, all afterok 28101 (Stage-1 smoke), --time 6 h (levels5 9 h, to
  be tightened once 28101 reports tok/s): base 28107, order_hidden_act 28108, order_taylor 28109,
  lr5e-5 28110, lr2e-4 28111, kd1 28112, w_fixed 28113, w_uncertainty 28114, w_lossprop 28115,
  w_dwa 28116, single 28117, levels5 28118, levels5_sw3 28119, grid9_sw3 28120. node205 runs
  them one after another (behind T1 28083 and 28101).
- Pilot scale: stay on 0.6B (main backbone; a 1.7B pilot needs a >1.7B teacher, still on hold,
  and ~3x the compute). Re-check only lr (+ the winning recipe vs base) at 1.7B later.
- Stage-1 teacher = original Qwen/Qwen3-1.7B (HF, frozen); Stage-2 teacher = our T1 export
  (`hf_export/flexllama__sft.paper.qwen3_1p7b_teacher/level2`, from 28083/28084). Not the student
  report's 1.7B (old code).
- node206 for pilots (user): Stage-1 batch is now global (`global_batch_size` 32 sequences, per
  rank = 32 / world size from SLURM_NTASKS/WORLD_SIZE; val 640 / test 1,600 windows fixed), so
  1-GPU and 2-GPU pilots take the same steps and validate on the same windows. 66 tests pass.
  **28121** 1-GPU Stage-1 smoke on node206 passed: **1,593 tok/s**, 24.7 GB peak (batch 32,
  micro 4), ~20.5 s/step; after 20 steps (0.66M tokens) val ppl L0/L1/L2 164 / 41.4 / 10.1
  (untrained 4,265 / 313 / 10.6). Estimates: 20M pilot ≈ 3.5 h on node206, ≈ 1.8 h on node205
  (2 GPUs, if it scales), 100M final ≈ 9 h on node205 (well inside 5 days).
- Moved pilots to node206 (afterok 28121, --time 12 h): w_lossprop 28122, w_dwa 28123,
  single 28124, levels5_sw3 28125, grid9_sw3 28126 (cancelled 28115-28117, 28119, 28120).
  node205 keeps base 28107, order_hidden_act 28108, order_taylor 28109, lr5e-5 28110,
  lr2e-4 28111, kd1 28112, w_fixed 28113, w_uncertainty 28114, levels5 28118 (afterok 28101).

### 2026-11-11
- **T1 finished** (28083 trained 1,044 steps, 28084 export, 28085 eval; Stage-2 smoke 28102 passed at
  28.2 GB, Stage-1 2-GPU smoke 28101 passed at ~1,200 tok/s per GPU). Teacher level2 eval (answer-only,
  32 tokens): MedQA 38.6, MCQA 40.9, HeadQA 47.7, Calc 25.3, Bullets 29.9, Pro-H 20.7, Xpert 11.1,
  **Avg 30.6, no_boxed 0**. **Worse than our untrained 1.7B (34.5) and the old conventional FT (37.9,
  old code)**. Export checks out (bf16 Qwen3ForCausalLM, 1.72B params, config correct, answers well
  formed). Cause unknown. Do NOT start Stage 2 until resolved: the teacher feeds all Stage-2 KD.
  B0 evals submitted to compare under the same pipeline: Qwen3-1.7B **28129**, Qwen3-0.6B **28130**.
  Differences vs the old run: real grad clipping 1.0, effective batch 32 (old 16), 1,044 steps
  (old 2,088), preset qwen3-1.7b-gqa, micro-batch 8.
- **Stage-1 pilots (20M tokens, val ppl L0/L1/L2 on PMC val; untrained 4,265 / 313 / 10.6)**:
  base 17.4/11.3/7.54 · order_hidden_act 17.4/11.4/7.54 · order_taylor **16.6/10.6**/7.55 ·
  lr5e-5 21.5/12.8/7.56 · lr2e-4 **14.8/10.5**/7.71 · w_lossprop 17.0/11.3/7.56 · w_dwa 17.3/11.3/7.51 ·
  single 21.1/12.8/8.14 · kd1 (in progress, 286/610: 31.0/17.7/10.4 at mid-run) · levels5_sw3 (79%) ·
  w_fixed, w_uncertainty, levels5, grid9_sw3 still queued. One seed, so differences of a few
  percent are not established. Findings so far: recovery works (L0 4,265 -> ~17); a higher lr helps the
  small levels (-15% L0) at +2% on L2; all-levels per step beats one level per step (single: +21% L0, +8% L2);
  loss weighting (lossprop, DWA) is within noise of equal weights; Taylor ordering gives -5% on
  L0/L1 despite the worse untrained ppl; hidden_act equals base.
- Runtime: 20M pilot ~2.7 h on node205 (2 GPUs), 2.7 h (single) to 4.8 h (weighted) on node206 (1 GPU).
- **ROOT-CAUSE ANALYSIS of the weak T1 teacher (avg 30.6 vs the student's 37.9)**, status 2026-11-11:
  Ruled out by direct comparison with the student's code (git base 07daef4) and data:
  * data: `data/medmix_sft/{train,val,test}.jsonl` are byte-identical (md5) to the student's copy; same
    pre-rendered chat prompts (`<think>\n\n</think>\n\n`), same 33,406 examples;
  * recipe on paper: lr 5e-5, 1% warm-up + cosine, 1 epoch, effective batch 32 (student 8 x 4 H100; ours
    16 x 2 A40), 1,044 steps, bf16-mixed, max_seq 3072, CE on answer tokens only, same loss code
    (`_step` is line-for-line the student's apart from micro-batching and weight hooks), same scheduler;
  * eval script `eval/eval_gen.py`: only the `dtype=` kwarg changed; export `training/export_flex_hf.py`:
    strict load of every tensor (incl. q/k norms), only `head_dim` handling added; model code only gained
    the optional `head_dim`; trainer only gained clipping, seed, smoke hooks.
  What actually differs: (1) **real gradient clipping at 1.0** (student: none; the old code silently
  ignored it), (2) exact micro-batching (2 passes of 8 per step), (3) seed 0 / data order, (4) 2 A40
  instead of 4 H100, (5) software: transformers 4.51.1, Lightning 2.5.3, torch 2.7.0 (student's env unknown).
  Training itself looks healthy: val CE (answers) falls monotonically 0.465 -> 0.357, token acc 0.81 -> 0.86.
  **Leading hypothesis = clipping**: our supervision is ~7 tokens/example (~230 answer tokens per step), so
  raw gradient norms are probably >> 1; clipped, Adam sees ~constant-norm gradients and keeps full-size
  updates, while the unclipped student run had Adam's second-moment memory dominated by large early
  gradients, i.e. a smaller effective LR late in training => less forgetting (student's teacher kept MedQA
  47.9; ours 38.6). Not yet measured: grad norms are now logged (`grad_norm`, pre-clip, in the `[sft]`
  progress line and as a wandb metric).
  Experiments submitted: **E_A 28131** `sft.diag.qwen3_1p7b_teacher.noclip` (T1 recipe, clip off, 2 GPU,
  node205), **E_D 28132** `sft.diag.qwen3_1p7b_teacher.1gpu` (T1 recipe, clip 1.0, ONE GPU, node206,
  batch 32 per GPU, ~5.5 h) as a control for multi-GPU sync. Pending pilots 28113/28114/28118/28126
  cancelled to free the nodes (kd1 28112 and levels5_sw3 28125 left running); resubmit after.
  Evidence that DDP sync works: Stage-1 1-GPU pilots (node206) and 2-GPU pilots (node205) give matching ppl.
- **B0 finding**: untrained Qwen3-1.7B and 0.6B score ~0 under the 32-token answer-only protocol (no
  `\boxed` in 100% of the answers: the model reasons step by step first). So the student's "untrained"
  rows (1.7B avg 34.5, MedQA 48.0) cannot come from this protocol; they were presumably generated with a
  long budget. The paper's untrained baseline must state its own protocol (long generation), and the
  paragraph in `main.tex` blaming "our short answer-only decoding budget" is wrong. Fidelity check
  **28133**: untrained 1.7B, MedQA only, 1,536 new tokens (should reproduce the student's 48.0).
- The user has the student's checkpoints on SURFdrive (https://surfdrive.surf.nl/apps/files/files/19533381278?dir=/FlexSLM,
  a login-only page: I cannot read it). With the old teacher export I can evaluate it with our pipeline:
  the decisive test of eval vs training. Needs a public share link or an upload to DAS-6.
- **Student's checkpoints fetched** (public SURFdrive share, token in `scripts/das6/fetch_old_ckpts.sbatch`, WebDAV
  via `scripts/fetch_surfdrive.py`, ~240 MB/s) to `/var/scratch/skalra/flexslm/old_ckpts/`: `repo/` (code snapshot
  23-27 Oct, `results/` incl. his `gen.*.json`), `saved_models/` {`..._qwen3_1p7b_teacher.pt` (20 Oct, 7.15 GB,
  the report's teacher), `..._flexxxxslm.pt` (21 Oct, report's nested), `..._10L_teacher.pt`, `..._10L_shard.pt`
  (27 Oct, 7.76 GB: later 10-level runs, 3 sampled per step; `_shard` = SHARD progressive transfer, `_teacher` = KD
  from the teacher; L9 avg 40.0 / 39.0, small levels at chance, NOT the report's models)}. Not downloaded: `checkpoints/`
  (83 GB Lightning ckpts incl. hyper_parameters/optimizer state; only for flexslm, 10L_teacher, 10L_shard).
  The share's `repo/` snapshot is identical to our base commit 07daef4 for training.py, utils.py, eval_gen.py,
  training_memopt.py; sft.py differs only by the 10L/SHARD additions.
- **Student's own eval results** (`old_ckpts/repo/results/gen.*.json`): teacher L2 (t64) 47.9/46.6/61.1/31.9/37.6/
  28.5/11.4 = **37.9** (0% unboxed); nested flexslm L2 49.9/48.6/63.3/33.4/43.0/30.0/12.0 = **40.0**, L1 19.1, L0 18.7;
  kdteacher L2 39.1; `sft.medmix_qwen3_1p7b` (KD from the UNTRAINED 1.7B) L2 26.2 (25% unboxed) and L0/L1 5.7/5.9;
  **untrained 1.7B: `.init-level2.t1024` = 34.5** (3.9% unboxed) => the "untrained" rows were generated with 1,024
  new tokens, the fine-tuned ones with 32-64. Paper must label this and the main-text sentence about a "short
  decoding budget" must be fixed. 10L runs: L9 40.0/39.0, L5 16.7/17.9, L0 18.8/18.1.
- **Weight-change analysis (28135, `scripts/compare_teachers.py`)**, ||W_ft - W_base|| / ||W_base|| vs the HF
  Qwen3-1.7B: old teacher vs OUR T1 teacher: gate/up/down 0.0103-0.0108 vs 0.0253-0.0266 (x2.5), q/k/v/o
  0.0089-0.0099 vs 0.0247-0.0288 (x2.7-2.9), norms x2-3, embeddings 0.0217 vs 0.0196 (same); overall 0.00365 vs
  0.00605; mean |delta| 3.3e-4 vs 6.0e-4. Cosine of the two deltas only 0.16-0.24. **Our fine-tune moved the
  transformer weights ~2.5-3x further than the student's, with the same data, lr and schedule**: strongly consistent
  with the clipping hypothesis (clipped Adam keeps full-size updates; unclipped, early large gradients inflate
  Adam's second moment and damp later updates). E_A (28131, no clipping, grad norms logged) is the test.
  Old teacher exported to `hf_export/old_teacher_1p7b/level2` and evaluated in OUR pipeline: **28136** (32 tokens)
  to settle eval-vs-training.
- E_D (28132, 1-GPU control) cancelled by the user: GPU count cannot change accuracy if gradient sync works (Stage-1
  pilots on 1 and 2 GPUs already agree).
- **Eval/software ruled out**: the student's old teacher, exported and scored with OUR pipeline (28136, 32 tokens):
  47.8/46.2/60.9/32.4/36.9/28.2/11.4 = **37.7** (student 37.9, 0% unboxed). Untrained Qwen3-1.7B on MedQA with 1,536
  new tokens (28133): **48.2** (student 48.0). So eval, export, transformers 4.51.1 and torch 2.7 reproduce the old
  numbers: **the gap (our T1 30.6 vs 37.7) is entirely in training.**
- **Clipping mechanism measured** (E_A 28131 raw grad norms, pre-clip): ~390, 401, 287, 315 at steps 1-4, 85, 18, 10
  by step 7, 5-16 at steps 10-20, 2.8 at step 50, 4.4 at step 100 (clip threshold 1.0) => our T1 clipped on EVERY step,
  mostly by 3-400x, so Adam saw constant-norm gradients; unclipped, the early spike dominates Adam's v and damps
  later updates. Our weights moved 2.5-3x further than the old teacher's (28135). Chained for E_A: export 28137,
  eval 28138, weight comparison 28139. If E_A ~ 37-38 with old-sized weight change: set `gradient_clip_val=None` for the
  SFT configs (`_PAPER`), retrain T1, then proceed to Stage 2. Stage 1 also clips at 1.0: its grad norms are logged in
  the resubmitted pilots; decide after reading them whether key Stage-1 pilots must be re-run unclipped.
- Resubmitted pilots on node206 (clip 1.0 like the finished ones, for comparability; they now log grad_norm):
  w_fixed 28140, w_uncertainty 28141, levels5 28142, grid9_sw3 28143. Finished so far: base, order_hidden_act,
  order_taylor, lr5e-5, lr2e-4, w_lossprop, w_dwa, single, kd1 (28112), levels5_sw3 (28125).
- **Paper: do not edit until all final results exist (user, 2026-11-11).** TODO list for the final paper pass:
  untrained baselines use a 1,024-token protocol (state it, or re-run at a common protocol); remove the sentence
  attributing our untrained 1.7B gap to a "short answer-only decoding budget"; ordering is a pilot dimension
  (Taylor is no longer assumed); reference table / MatFormer baseline text already in; 12-page limit.
- **Paper updated once more at the user's request (2026-11-11), then FROZEN until all final results exist.**
  Changes in `paper/main.tex` (12 pages, no undefined refs; draft PDF refreshed):
  * new Sect. 5.3 "Training cost compared with prior work" + Table `tab:cost` (reported budgets): MatFormer 10-80B
    tokens (78M-850M from scratch; 256 TPUv3 for 850M), Flextron 89.9B + 1.05B router (Llama-2-7B), LLaMaFlex 60.4B
    (Llama-3.1-8B), Minitron 94B per model (15B -> 8B, 4B), Nemotron Elastic 110B (12B -> 9B, 6B); Meditron 48.1B
    (588.8 GPU-h A100, model card), PMC-LLaMA 75B (4.8M papers + 30K books), BioMistral 3B tokens, 1.5 epochs
    (<=20 h on 32 A100), Apollo 2.5B, MedMobile 400K questions x3 epochs (83 h, 4 A100 nodes), FaithMed 28,368 SFT
    samples + 300 RL steps (~32 h on 4 H100); ours 0.10B (Stage 1) + 13.3M (Stage 2; 13,073,630 prompt + 244,682
    answer tokens, 7.3 answer tokens/example, counted with the Qwen3 tokenizer). Our GPU-hours left as \tbd.
  * Importance ordering rewritten as a choice among none / hidden-activation / Taylor, selected after Stage 1
    (states the untrained-ppl finding: Taylor 4,265 -> 26,698 on L0; 453/512 overlap); \tbd{ordering}.
  * Stage 1: 100M tokens, pilots of 20M choose the hyperparameters; ppl thousands -> ~17 after 20M; 13.3M / 0.24M.
  * Stage 2: gradient-norm observation (300-400 early, 3-15 later) and the clipping effect; decision \tbd{no clip}.
  * Evaluation: untrained backbones get 1,024 new tokens (0% at 32); baseline (i) and Table qwen17 dagger note;
    wrong "short decoding budget" sentence replaced; FaithMed 41.0 correctly attributed to agentic search + GRPO
    (the old text called it SFT).
  * Lessons and the 1.7B paragraph shortened to stay at 12 pages; three 9-column tables scaled to \textwidth (they
    were 48-58 pt too wide).
- Optimizer: `training.trainer.adamw` = AdamW with `fused=True` when all params are on CUDA (else default), used by
  `SFTTrainingContext` and `Stage1Context`; prints `[optim] AdamW ... fused True/False` at start. Same algorithm and
  state format, rounding-level differences only; expected gain <0.5% of step time (optimizer ~50-100 ms of a 9-16 s
  step). lr / weight decay unchanged (user only wanted fused). Applies to jobs starting after this sync (pending pilots
  28141-28143 and everything later), not to running ones.

### 2026-11-12
- **E_A settles the T1 root cause: gradient clipping.** `sft.diag.qwen3_1p7b_teacher.noclip` (28131 -> export 28137
  -> eval 28138, 32 tokens): MedQA 46.4, MCQA 46.4, HeadQA 61.2, Calc 30.4, Bullets 34.2, Pro-H 28.7, Xpert 10.3 =
  **avg 36.8**, 0% unboxed (clipped T1 30.6; student's teacher in our pipeline 37.7). Weight change vs HF Qwen3-1.7B
  (28139): ratio new/old **1.00** in every group (clipped T1: 2.5-3x), cosine of the deltas 0.69-0.93 (clipped T1:
  0.16-0.24). The remaining 0.9 points (mostly Bullets -3.4, Calc -1.5) are within seed/data-order noise (Bullets n=298,
  SE 2.7). Raw grad norm 392 at step 1, 3.6 at step 750.
- Decisions taken (pre-agreed conditional from 2026-11-11): `_PAPER` now `gradient_clip_val=None` (all paper SFT runs,
  incl. Stage 2 and B1/B2); **T1 is not retrained**: E_A is exactly the T1 recipe without clipping, so
  `_FT_TEACHER_1P7B` = `hf_export/flexllama__sft.diag.qwen3_1p7b_teacher.noclip/level2` (Stage-2 teacher and the 1.7B
  conventional-FT row of the paper, avg 36.8). The clipped T1 export stays under its name, reference only.
- **Stage-1 pilots complete** (20M tokens, PMC val ppl at widths 0.5 / 0.75 / 1.0; untrained 4,265 / 313 / 10.6):

  | arm | L0 | L1 | L2 | note |
  |---|---|---|---|---|
  | base | 17.4 | 11.3 | 7.54 | lr 1e-4, KD 0.5, 3 levels/step, clip 1.0 |
  | order_hidden_act | 17.4 | 11.4 | 7.54 | = base |
  | order_taylor | **16.6** | **10.6** | 7.55 | -5% / -6% |
  | lr5e-5 | 21.5 | 12.8 | 7.56 | worse |
  | lr2e-4 | **14.8** | **10.5** | 7.71 | -15% / -7%, +2% L2 |
  | kd1 (pure KD) | 25.1 | 15.4 | 9.76 | worse (ppl favours CE; still clearly worse) |
  | w_fixed 3/1/1 | 16.8 | 11.6 | 7.56 | -3% L0, +3% L1 |
  | w_uncertainty | 17.6 | 11.3 | 7.49 | ~ base |
  | w_lossprop | 17.0 | 11.3 | 7.56 | ~ base |
  | w_dwa | 17.3 | 11.3 | 7.51 | ~ base |
  | single | 21.1 | 12.8 | 8.14 | worse everywhere |
  | levels5 (0,2,4) | 17.3 | 11.2 | 7.59 | ~ base at ~2.7x the time (7.3 h vs 2.7 h on 1 vs 2 GPUs) |
  | levels5_sw3 (0,2,4) | 17.1 | 12.2 | 7.55 | middle +8% |
  | grid9_sw3 (0,4,8) | 17.2 | 12.6 | 7.56 | middle +12% (gives 9 deployable widths) |

  Conclusions (one seed): lr and ordering are the only knobs that clearly move the small levels; KD 0.5 > pure KD;
  all levels per step > one random level; level weighting is noise (A4 stays an ablation, equal weights in the
  recipe); dense grids cost the middle width a little. Stage 1 also clipped on every step (grad norm 430-650 at
  steps 1-5, ~15 at step 50, ~5-6 at step 550 vs threshold 1.0).
- Code: `_PAPER` unclipped, teacher repointed (training/sft.py); round-2 pilot arms `noclip`, `order_taylor_lr2e-4`,
  `order_taylor_lr2e-4_noclip` (training/stage1.py). 66 local tests pass. Commit "Paper SFT without gradient clipping...".
- Note: `scripts/das6/sync.sh` must be run with `SSH_AUTH_SOCK=$HOME/.ssh/agent.sock`; without it the sync fails
  silently-ish ("Permission denied") and sbatch runs the OLD code. First submission 28144-28149 was made on stale code
  and cancelled after 9 s (no outputs written); resynced and verified.
- Jobs: **B1** `sft.paper.qwen3_0p6b_conventional` (unclipped) **28150** (node205) -> export **28151** -> eval **28152**;
  Stage-1 round 2: `order_taylor_lr2e-4` **28153**, `order_taylor_lr2e-4_noclip` **28154** (node205, after B1),
  `noclip` **28155** (node206, 1 GPU).
- Next: read round 2; then set `FINAL_ORDERING` (taylor is the current favourite), lr and clipping of
  `s1.qwen3_0p6b.full` (100M, ~9 h on node205), and run A2 `...nested.no_stage1` (needs FINAL_ORDERING). User
  sign-off needed for the final Stage-1 recipe before the 100M run.
- **LLaMaFlex / Minitron ordering (user question).** LLaMaFlex (ICLR 2025): no code, no checkpoints, proprietary
  60.4B-token data; its ordering is one sentence ("accumulated magnitude of activations", citing Flextron + Minitron),
  no formula, no ablation. Flextron: no code/checkpoints. `NVlabs/Minitron` is README/model cards only (HF checkpoints
  Minitron-4B/8B). The real implementation is NVIDIA Model Optimizer `modelopt/torch/prune/plugins/mcore_minitron.py`
  (Megatron only). Its recipe: MLP = down_proj input, head = o_proj input (L2 over head_dim), hidden = OUTPUT of
  input_layernorm + post_attention_layernorm (L2 per site, summed; no final norm); aggregation mean|x| over sequence,
  L2 over samples; Minitron paper: 1,024 samples, one-shot, never compared with Taylor/gradient scores. Our
  `hidden_act` differed (norm inputs, sum |x|, MLP/groups unordered).
- Implemented `--scheme minitron` (`collect_minitron_importance`, port of modelopt; KV-group score = L2 over all its
  channels, since our levels drop whole groups while Minitron prunes query heads inside groups). The script now logs
  `score_seconds` / `score_peak_gb` for every scheme (`[score]` line, ordering.json). New test (matches a manual
  reference, batch-invariant, ordered model is sorted); 67 local tests pass. Pilot arms `order_minitron` (vs
  `order_taylor`) and `order_minitron_lr2e-4` (vs `order_taylor_lr2e-4`). Committed, **NOT yet synced/submitted:
  DAS-6 unreachable (ssh port 22 timeout, 2026-11-12 ~17:40; VPN?).** To run once reachable:
  ```
  SSH_AUTH_SOCK=$HOME/.ssh/agent.sock bash scripts/das6/sync.sh
  # on DAS-6, ~/FlexSLM; all three on node207 in sequence, so the timings are comparable
  o=$(sbatch --parsable --nodelist=node207 scripts/das6/importance_order.sbatch --scheme minitron --out hf_export/qwen3-0.6b.minitron)
  t=$(sbatch --parsable --nodelist=node207 --dependency=afterany:$o scripts/das6/importance_order.sbatch --scheme taylor --out /var/scratch/skalra/flexslm/timing/taylor --eval-seqs 0)
  m=$(sbatch --parsable --nodelist=node207 --dependency=afterany:$t scripts/das6/importance_order.sbatch --scheme minitron --batch-size 8 --out /var/scratch/skalra/flexslm/timing/minitron_b8 --eval-seqs 0)
  sbatch --time=07:00:00 --dependency=afterok:$o scripts/das6/train.sbatch s1.qwen3_0p6b.pilot.order_minitron
  sbatch --time=07:00:00 --dependency=afterok:$o scripts/das6/train.sbatch s1.qwen3_0p6b.pilot.order_minitron_lr2e-4
  ```
  (calibration = the sbatch default 512 x 512 PMC train tokens, batch 1, same as the Taylor checkpoint 28100.)
- **Cluster down (user, 2026-11-12).** Brainstorm on KD / continuum from Flextron, LLaMaFlex, Minitron (+ Llama-3.1-Minitron,
  arXiv 2408.11796). Facts checked in the papers: Flextron trains 3 random sub-models + full per step with plain CE
  against labels (no KD), 89.9B tokens, routers pick per-layer widths; LLaMaFlex: 4 anchor budgets, Gumbel router
  interpolates the budget embedding (interpolated models land on the curve, Table 2), policy-aware modulation (sinusoidal
  width embedding -> MLP scale/shift after elastic MLP/MHA, 0.02M params each, val loss -0.08 at 800 iters), KD not
  described; Minitron: forward KL logit-only best (CE mixing slightly worse 2.117 vs 2.107), T=1 best of {0.1,0.5,1,3},
  top-K hurts, reverse KL worse, intermediate-state losses only help when depth is cut (width-only: no gain), biggest
  teacher best (15B > 8B for 4B); Llama-3.1-Minitron: **teacher correction** (fine-tune the teacher on the distillation
  data first, ~100B tokens, -6% LM val loss; correcting in parallel during distillation is on par).
  Our code: Stage 1 = (1-l) CE + l KL, l 0.5, T 2, teacher = UNTRAINED HF Qwen3-1.7B (never saw PMC); Stage 2 KD covers
  every real token (prompt + answer, `rows = real` when kd_lambda > 0), CE only the answer, KL normalised over all tokens
  (answer tokens ~1.8% of the KD mass).
  Main hypothesis: the Stage-1 teacher is worse on PMC than our student's full level after 20M tokens (L2 7.54), which
  would explain why pure KD (kd1) hurt L2 (9.76) although Minitron finds KD-only best. Not yet measured: needs the
  teacher's PMC val ppl (one short eval job). Missing pilots: kd0 (CE only), T1, corrected teacher. Proposal and
  priorities given to the user in chat; nothing implemented yet.
- **User decisions (2026-11-12):** teacher correction approved (queue when the cluster is up); implement and compare
  ALL four continuum methods; widths below 0.5 are not expected to work zero-shot because they are never trained, so a
  level list down to 0.25 is an ablation for the final experiments only.
- **Implemented locally (cluster down), 79 local tests pass:**
  * `FlexLLaMA.set_widths(hidden, intermediate, heads)`: run at ANY prefix width, uniform or per layer (residual width
    shared); `dims_override` on fm.Linear, `dim_override` on Embedding/RMSNorm, `heads_override` on RoPEAttention
    (KV heads follow the GQA ratio); `set_level_use` clears them. Exact match with a real level of the same width (test).
  * Method 3a `FlexLLaMAConfig.level_norms`: per-width zero-init offsets `level_delta` on every residual RMSNorm,
    linear interpolation in the width between trained levels, clamped outside; folded on export.
  * Method 3b `FlexLLaMAConfig.modulation` = "scale" | "scale_shift": LLaMaFlex modulation (`WidthModulation`,
    sinusoidal 16-dim embedding per fraction of hidden and block width -> MLP 128 -> scale (and shift), zero-init, never
    applied at full width) on attention and MLP outputs; ~0.27M params per module, 56 modules (~15M, training only).
    Export folds the scale into o_proj/down_proj rows (test: HF logits == flex logits); shift cannot be exported to Qwen3
    (no MLP bias) -> NotImplementedError. Continuum params (level_delta, *_mod) get their own AdamW group
    `continuum_lr` 1e-3, no weight decay (LLaMaFlex used 4e-2 for its modulation nets; ours is a guess).
  * Method 2 `Stage1Context.anchor_levels`: anchors always trained, remaining `num_levels_per_step` slots random
    (seeded by (seed, step)). Pilot `grid9_anchors_plus1` = 9-level 1/16 grid, anchors 0/4/8 (= 0.5/0.75/1.0) + 1 random.
  * Methods 1 and 4: `eval/eval_continuum.py` (+ `scripts/das6/eval_continuum.sbatch`): PMC val ppl on the Stage-1 val
    windows (so anchors reproduce the pilots' val ppl) for trained levels, uniform widths 0.25...1.0 step 1/16, and 36
    Mix'n'Match configs (between adjacent anchors: residual of the bigger anchor with k = 7/14/21 layers' MLP+heads at
    the smaller, or the reverse; k layers first / last / spread), with total and non-embedding params; `--hf` for plain
    HF models (teacher PMC ppl). CPU smoke on the real Qwen3-0.6B warm start: levels == set_widths exactly; params
    L0/L1/L2 188M / 364M / 596M, 0.25 width 66M.
  * KD: pilots `kd0`, `T1`, `kd_corrected`, `kd1_corrected`; `s1.qwen3_1p7b.teacher_correction` (full Qwen3-1.7B, one
    level, CE only, 20M PMC tokens, lr 2e-5, no clip) -> export level0 = `stage1.CORRECTED_TEACHER_1P7B`.
    `s1.qwen3_0p6b.full.w025` (fractions 0.25/0.5/0.75/1.0, final ablation only, not queued).
  * **Bug fixed**: `MemOptFlexLMKDTrainer._step` always ran the teacher, so any Stage-1 run with kd_lambda 0 would
    have crashed (teacher not loaded). Now skipped; test: kd0 gradient == CE trainer gradient.
- **To run when DAS-6 is back:** sync, check 28150-28155, then `bash scripts/das6/queue_2026-11-12.sh` (ids appended to
  `jobs/queue_2026-11-12.ids`): PMC ppl of Qwen3-1.7B/0.6B; continuum eval of base(.init), levels5, levels5_sw3,
  grid9_sw3, order_taylor, lr2e-4; Minitron ordering + timing (node207); node205: teacher correction -> export ->
  kd_corrected, kd1_corrected, plus kd0, T1, order_minitron(_lr2e-4); node206: grid9_anchors_plus1, level_norms,
  mod_scale, mod_scale_shift; every pilot chained to its continuum eval. ~20 h of node205, ~52 h of node206 (estimate).
- Comparison protocol for the continuum: PMC val ppl vs params at the 6 untrained 1/16 widths in (0.5, 1) and on the
  Mix'n'Match budgets, against the anchors' ppl (must not get worse) and against levels5/grid9 (trained in-between
  widths). Then the best method goes to Stage 2 and the medical benchmarks.
- **DAS-6 back (2026-11-12 ~19:00), queue submitted** (`jobs/queue_2026-11-12.ids`): PMC ppl Qwen3-1.7B 28156, 0.6B
  28157; continuum evals base.init 28158, base 28159, levels5 28160, levels5_sw3 28161, grid9_sw3 28162, order_taylor
  28163, lr2e-4 28164; Minitron ordering 28165 -> timing taylor 28166 -> minitron_b8 28167 (node207); teacher correction
  28168 -> export 28169 -> PMC ppl 28170; pilots (-> continuum eval): kd0 28171 (28172), T1 28173 (28174), kd_corrected
  28175 (28176), kd1_corrected 28177 (28178), order_minitron 28179 (28180), order_minitron_lr2e-4 28181 (28182) on
  node205; grid9_anchors_plus1 28183 (28184), level_norms 28185 (28186), mod_scale 28187 (28188), mod_scale_shift 28189
  (28190) on node206. The outage interrupted B1 28150 (step 323/1044) and pilot noclip 28155; Slurm requeued both
  (resume from last.ckpt); 28151-28154 still queued behind them.

### 2026-11-13
- No results: nothing ran since the outage. sinfo (12:49): node201-203 and node205-208 all `down`, reason "Node
  unexpectedly rebooted" (2026-11-12 18:20-18:26), node204 down since April. Slurm does not return nodes in that state by
  itself; an admin must resume them. All 41 jobs (28150-28190) still PENDING (ReqNodeNotAvail / Dependency); they start
  automatically once the nodes are resumed, nothing to resubmit.
- 12:57: nodes resumed, jobs started. B1 28150 and pilot noclip 28155 resumed from last.ckpt (saved 16:24 / 16:29 on
  11-12). **PMC val ppl (Stage-1 val windows, 640 x 1,024): Qwen3-0.6B 10.11, Qwen3-1.7B (Stage-1 teacher) 7.92.**
  After 20M tokens the base pilot's full width is at 7.54, i.e. **better than its teacher on PMC**: the teacher-correction
  hypothesis holds (pure KD toward a worse teacher pulled L2 to 9.76 in kd1). Corrected teacher: 28168 -> 28170.
- **User decisions (2026-11-13):** (1) try **II-Medical-8B** as the bigger teacher for the 1.7B backbone (S1), run on
  **hipster, performance partition**; (2) implement I1 now; (3) A3-in-both-stages: decide after the paper is final
  (only if time remains); (4) the continuum is NOT a section: show it in the Pareto plot (anchors = dots, untrained /
  interpolated widths = stars or similar) + a few lines in the method (Sect. 3.4). Keep Stage-1 pilots clipped for now.
- **I1 implemented**: `Stage1Context.train_levels` (+ `FlexLMStage1Trainer._select_levels`): train and validate one
  level of the same nested model = the same importance-ordered prefix, alone. Configs `s1.qwen3_0p6b.full.indep_l{0,1}`
  (100M tokens, same recipe) -> `sft.paper.qwen3_0p6b_indep_l{0,1}` (same Stage-2 schedule/teacher, `train_levels`).
  New `stage1.FINAL_S1` dict (empty until the pilots decide; never holds `ordering`) shared by full, no_order, w025 and
  indep_*, so the final-budget runs cannot drift apart. 80 local tests pass.
- **Hipster** (from the FlexLLaVA hipster launchers): performance = 5 nodes x 8 RTX 6000 Ada 48 GB, 32 CPU/GPU enforced,
  DefMemPerCPU 3000 MB, max 7 days, busy shared cluster (no --exclusive); capacity = L4 24 GB; `module load cuda/12.9.1`.
  Env: existing conda `my_env` (py3.9, torch 2.8.0, transformers 4.57.6, lightning 2.5.2, no pytest), used read-only.
  There is an OLD `~/FlexSLM` clone on hipster (commit b39fbcc, own wandb/jobs from Nov 10): left untouched. Our code
  goes to **`~/FlexSLM-medbench`** (`scripts/hipster/sync.sh`), heavy files `/scratch/skalra/flexslm` (home 52% full).
  `scripts/hipster/setup.sh` streams data/pmc_qwen3 + data/medmix_sft and the Qwen3 HF cache from DAS-6 (hipster
  reaches DAS-6 over ssh) and downloads II-Medical-8B. Job scripts: common.sh, check.sbatch (warm-start + PMC ppl
  parity with DAS-6 under the newer transformers), evalgen.sbatch, eval_continuum.sbatch.
- S1 memory plan: 1.7B nested student (fp32 weights + grads + Adam ~27.6 GB) + 8B bf16 teacher (16.4 GB) does not fit
  48 GB under DDP. Plan: ZeRO-1 (torch ZeroRedundancyOptimizer: Adam state sharded over 4 GPUs, ~34 GB/GPU before
  activations); Lightning consolidates the optimizer state for checkpoints. Not implemented yet; teacher checks first.
- Hipster setup done (`scripts/hipster/setup.sh`): data copied byte-exact from DAS-6 (sizes + md5 match; du shows half
  because of fs compression), Qwen3 cache copied, II-Medical-8B downloaded (Qwen3ForCausalLM, hidden 4096, 36 layers, 8
  KV heads, untied, vocab 151,936; vocab.json/merges.txt md5-identical to Qwen3-1.7B -> KD-compatible).
- ZeRO-1 implemented (`zero_optimizer` on Stage1Context / SFTTrainingContext -> torch ZeroRedundancyOptimizer around the
  same AdamW incl. the continuum group; 2-process gloo test: identical params after 3 steps, full state consolidated on
  rank 0). 81 local tests pass.
- Hipster jobs: check 391237 (warm start + PMC ppl parity), II-Medical-8B PMC ppl 391238, II-Medical-8B evalgen 32 tokens
  391239 -- all PENDING (Priority), estimated start 2026-11-15 17:56. **Scheduler estimates (sbatch --test-only,
  2026-11-13 ~14:30): 1 L4 on capacity -> 2026-12-02; 4 x RTX 6000 Ada on performance -> 2026-12-22 (after the
  deadline).** Our fairshare priority is 5; 46 jobs pending on performance, 274 on capacity. => S1 training on hipster
  is not feasible before 2026-12-15 per the scheduler; needs a user decision (DAS-6 node205 with ZeRO-1 over 2 A40s
  is the fallback).
- User (2026-11-13): keep hipster for the 1.7B experiments; small urgent checks on **capacity** (less busy), heavy jobs on
  **performance**. Checks resubmitted on capacity (1 L4, 16 CPU): check 391242, II-Medical-8B PMC ppl 391243, evalgen
  32 tok (batch 4, 8 h) 391244; the performance copies 391237-391239 stay queued (est. 11-15); cancel whichever pair is
  still pending when the other has run.
- S1 configs: `s1.qwen3_1p7b.full` (100M, FINAL_S1 + S1_1P7B: preset qwen3-1.7b-gqa, hidden 1024/1536/2048, heads
  8/12/16, kv 4/6/8, FFN 3072/4608/6144, teacher II-Medical-8B, ZeRO-1, micro-batch 2; original weight order: make_stage1
  now refuses a non-"none" ordering for presets without an ordered checkpoint), `s1.dev.qwen3_1p7b_smoke`,
  `sft.paper.qwen3_1p7b_nested` (8 per GPU x 4 = 32, micro 2, ZeRO-1, KD from II-Medical-8B). `scripts/hipster/train.sbatch`
  (performance, 4 GPUs, 32 CPU/GPU). **4-GPU memory smoke 391246** (20 steps) queued on performance.
- **Continuum results (methods 1 and 4, existing pilots; PMC val ppl; params total)** -- eval anchors reproduce the
  pilots' val ppl within ~1% (base 17.19/11.22/7.49 vs 17.4/11.3/7.54 logged during training):

  | width | params | base (3 trained) | levels5 (5 trained) | grid9_sw3 (9 trained, 3/step) | untrained init |
  |---|---|---|---|---|---|
  | 0.4375 | 157M | 29.7 | 31.1 | 28.6 | 7,724 |
  | **0.5** | 188M | **17.2** | **17.2** | **17.1** | 4,996 |
  | 0.5625 | 221M | 19.3 | 18.1 | **16.8** | 5,466 |
  | 0.625 | 269M | 18.6 | **13.5** | **15.4** | 1,921 |
  | 0.6875 | 323M | 14.6 | 13.2 | **13.7** | 495 |
  | **0.75** | 364M | **11.2** | **11.1** | **12.5** | 279 |
  | 0.8125 | 408M | 11.7 | 11.0 | **11.5** | 158 |
  | 0.875 | 473M | 11.3 | **9.2** | **10.1** | 57 |
  | 0.9375 | 543M | 9.2 | 8.7 | **8.6** | 16 |
  | **1.0** | 596M | **7.5** | **7.6** | **7.5** | 10.1 |
  (bold = trained width of that run; below 0.4375 everything is in the hundreds/thousands, as expected.)
  Findings: (1) **zero-shot interpolation is not monotone**: just above a trained width the model gets WORSE with more
  parameters (base: 0.5625 19.3 > 0.5 17.2; 0.8125 11.7 > 0.75 11.2), recovering only near the next anchor; most method-1
  points are dominated by the anchor below. (2) More trained widths fix it: levels5's untrained widths are monotone
  except right above 0.5 (18.1); grid9_sw3 is smooth everywhere but its 0.75 anchor is 12% worse (12.5 vs 11.2).
  (3) Mix'n'Match (method 4) on base: residual width of the upper anchor with the LAST k layers' MLP/heads shrunk is
  the best placement (0.5-0.75: 344M 13.7-13.9, 323M 16.1; 0.75-1.0: 569M 9.0, 541M 10.1, 513M 11.0); configurations at
  the lower anchor's residual width are all dominated by that anchor. Taylor ordering lowers everything (base -> taylor
  e.g. 344M 13.7 -> 12.0, 569M 9.0 -> 8.0). => methods 2 and 3 (running) are needed for a usable continuum; Pareto plot:
  anchors = dots, untrained = stars, dominated points visible.
- **Hipster storage rules (user, 2026-11-13):** `/scratch` is wiped every few weeks -> only checkpoints and reproducible
  items there; repo root, data, results and caches in `/home/skalra`; nothing irreplaceable stays in scratch (finals
  are copied off: home if small, SURFdrive if large). The user's existing FlexLLaVA convention is the same (checkpoints
  in `/scratch/skalra/flexllava_saves/checkpoints`, HF cache and wandb in `/home/skalra/flexllava_saves`). Written as a
  **global, all-project note** in `~/.claude/CLAUDE.md` on the local machine AND on hipster (identical, md5 checked;
  there was none before): fair-use rules (sbatch only, no login-node work, never --exclusive/--mem, exact resources and
  ~2x time, performance vs capacity guidance, per-user QoS cap 256 CPUs per partition, no duplicate/speculative jobs,
  smoke tests, <=8-9 parallel ssh to DAS-6), storage rules, and bookkeeping. The cluster policy page itself is behind a
  university login (not readable by Claude).
- **Current (temporary) violation, to be fixed after the queued jobs finish:** our data (4.0 GB pmc_qwen3 + 34 MB
  medmix_sft), results/ and the HF cache (21 GB: Qwen3-0.6B/1.7B + II-Medical-8B 16 GB) are in
  `/scratch/skalra/flexslm`, symlinked into `~/FlexSLM-medbench`. Fix prepared: **`scripts/hipster/migrate_to_home.sbatch`**
  (capacity, 4 CPUs, 6 h; refuses with exit 3 while any `flexslm_*` job of ours is queued/running; copy -> `diff -rq`
  -> swap symlink for a real directory -> delete the scratch copy). Targets: `~/FlexSLM-medbench/data/{pmc_qwen3,
  medmix_sft}`, `~/FlexSLM-medbench/results`, HF cache `~/flexslm_saves/cache/huggingface` (`common.sh` switches
  HF_HOME to it automatically once that directory exists). Stays in scratch: `store/` (checkpoints, wandb),
  `hf_export/`, `jobs/` (Slurm logs; numbers are recorded here). Tested on dummy directories on hipster (symlink
  swap, verification, guard). Home after the move: ~131 GB of 200 GB (now 106 GB). **Do: after the hipster jobs 391237-391246
  (and anything queued later) finish, `sbatch scripts/hipster/migrate_to_home.sbatch` from ~/FlexSLM-medbench, then
  re-check that no flexslm path under /scratch is still needed. Do not re-run setup.sh afterwards as is.**
  Open question to the user: the destination was written as `home/skalra/FlexSLM`; the code (and so data/) lives in
  `~/FlexSLM-medbench` because `~/FlexSLM` is an old clone (commit b39fbcc) with its own wandb/jobs.
- Final checkpoints of the 1.7B runs (S1) must not stay in scratch: export, copy the export to home or SURFdrive, delete
  the Lightning checkpoints (7 GB+ each with optimizer state).
- **User decisions (2026-11-13, later):** (a) move the data into `~/FlexSLM-medbench` (confirmed; the old `~/FlexSLM`
  clone stays untouched); (b) **all Slurm output logs go into the repo root** (`~/FlexSLM-medbench/jobs/`, real
  directory in /home); (c) **no checkpoints (or anything) to SURFdrive unless the user explicitly asks**, final
  checkpoints only; (d) a DAS-6 rules note, with the module/conda lines every GPU batch script must contain.
- **Global notes restructured** into user-level rules files (auto-loaded in every project; verified in the Claude Code
  docs that `~/.claude/rules/*.md` apply machine-wide): `~/.claude/rules/hipster-cluster.md` and
  `~/.claude/rules/das6-cluster.md`, installed identically on the local machine, on hipster and on DAS-6 (md5 checked);
  the single `~/.claude/CLAUDE.md` created earlier today was removed everywhere (verified unmodified first).
  hipster-cluster.md now: logs in repo-root jobs/, SURFdrive only on request, GPU scripts load `cuda/12.9.1` + conda env
  and fail fast if torch sees no GPU. das6-cluster.md: node table (defq node205 2xA40, node206 1xA40, node207 1xA10,
  node208 2xA10, 64 CPUs each, partition OverSubscribe=EXCLUSIVE; fatq node201-203 CPU only, node204 down), 1- vs 2-GPU
  distribution, all CPUs when all GPUs of a node are held, no --mem, --time 2x, no waiting >10 min, the required
  script header (`module load cuda12.6/toolkit/12.6` etc.; versions available 11.7/12.1/12.3/12.6/13.0; **there is no
  conda module on DAS-6**, so `eval "$(conda shell.bash hook)"` / `source .../minimonda3/etc/profile.d/conda.sh` +
  `conda activate`), storage (/home 5 TB for repo roots; /var/scratch/skalra 198 TB for big files; not cleaned; nothing
  leaves DAS-6 unless asked).
- `scripts/hipster/migrate_to_home.sbatch` now also moves `jobs/` (Slurm logs) into the repo root and writes its own log
  to the repo root; retested on dummy dirs on hipster (data, results, jobs, HF cache; store/ and hf_export/ stay in
  scratch). **Still to run once jobs 391237-391246 and anything queued after them have finished.**
- Observation, not changed: `scripts/das6/train.sbatch` asks 2 tasks x 16 CPUs = 32 of node205's 64 CPUs; the DAS-6
  rule says to request all CPUs when holding all GPUs (dataloader throttling). Results are unaffected (throughput
  only); consider 32 per task for new submissions.
- **DAS-6 CPU request fixed (user, 2026-11-13):** `scripts/das6/train.sbatch` now `--cpus-per-task=32` (2 tasks x 32 = all
  64 CPUs of node205); 1-GPU runs on node206 pass `--cpus-per-task=64` (usage comment, `queue_2026-11-12.sh` and §5
  updated). The old 16 never errored: Slurm only confines each task to the cores requested (task/affinity), so each rank
  (main process + 4 dataloader workers + torch threads) shared 16 cores and 32 cores sat idle; partition defq is
  OverSubscribe=EXCLUSIVE, so nobody else could use them. Effect: possible dataloader throttling, never different results.
  **sbatch copies the script at submission, so jobs already submitted keep 16 CPUs/task**: the running 28154 (node205) and
  28183 (node206) and the pending 28168, 28171, 28173, 28175, 28177, 28179, 28181, 28185, 28187, 28189. Not touched
  (changing a running job's CPUs is not possible; cancelling the chain to resubmit is not worth it for a throughput-only
  effect unless the user wants it).

### 2026-11-16
- **State:** DAS-6 queue empty, every job of the 2026-11-12 queue finished (28150-28190). Hipster: checks done
  (391242-391244 on capacity, 391237-391239 on performance, same results); only the 4-GPU S1 memory smoke 391246 is
  still pending (est. start 2026-11-18). `migrate_to_home.sbatch` on hipster still NOT run (waits for 391246).
- **B1 (28150 -> 28151 -> 28152), conventional SFT Qwen3-0.6B, unclipped, 32 tokens:** MedQA 27.9, MCQA 33.2, HeadQA 28.0,
  Calc 23.8, Bullets 26.2, Pro-H 12.7, Xpert 9.7 = **avg 23.1**, 0% unboxed. **Only 1-8 points above the uniform-guess
  rate on every multiple-choice set**; only MedCalc shows clear learning. This is the ceiling for every width of the
  0.6B family under the answer-only protocol => an accuracy-vs-width curve on 0.6B will be nearly flat. Main risk for
  the paper (see "Assessment" below). Untrained 0.6B at 32 tokens: 1.2% (unboxed); at 1,024 tokens NOT run yet.
- **Stage-1 round 2 + KD + ordering + continuum pilots (20M tokens; final eval ppl at 0.5 / 0.75 / 1.0; `results/continuum.*.json`;
  values marked (log) are the last val ppl logged during training, ~1% higher):**
  order_minitron 15.7/10.4/7.49 · **order_minitron_lr2e-4 13.5/9.73/7.63 (best small levels)** · order_taylor_lr2e-4
  14.1/9.94/7.71 (log) · order_taylor_lr2e-4_noclip 17.1/10.9/7.61 (log) · noclip 22.0/13.0/7.56 (log; base 17.4/11.3/7.54)
  => **clipping at 1.0 HELPS Stage 1** (opposite of Stage 2) · kd0 (CE only) 15.5/10.4/6.97 · T1 (tau 1) 15.0/10.2/7.11 ·
  kd_corrected 15.3/10.2/**6.92** · kd1_corrected 18.3/11.4/7.29 · level_norms 17.1/11.2/7.48 · mod_scale 17.0/11.2/7.48 ·
  mod_scale_shift 17.0/11.2/7.48 · grid9_anchors_plus1 17.1/11.2/7.56 with all 9 widths monotone
  (17.1, 16.6, 14.7, 12.8, 11.2, 10.8, 9.8, 8.6, 7.6).
  Corrected teacher (28168, Qwen3-1.7B + 20M PMC tokens): PMC ppl 7.92 -> **5.57**. Ordering cost on one A10, 262k tokens:
  Minitron 43.7 s / 2.5 GB, Taylor 99.9 s / 5.2 GB; untrained ppl after Minitron ordering 4,658 / 998 / 10.6.
  Conclusions (one seed): (1) KD from the uncorrected teacher HURTS (base 17.2/11.2/7.49 vs CE-only 15.5/10.4/6.97);
  corrected-teacher KD 0.5 is on par with / marginally better than CE only (1-2%, not resolvable with one seed); pure KD
  is worse even with the corrected teacher. (2) Minitron ordering > Taylor > original after recovery, and cheapest.
  (3) Continuum: anchors + 1 random width (method 2) is the only method that works; per-width norms and LLaMaFlex modulation
  do nothing (within 5%, still non-monotone). (4) Untested combination = the natural final recipe: minitron + lr 2e-4 +
  clip 1.0 + corrected teacher KD 0.5 (or CE only) + tau 1 or 2 + grid9 anchors + 1.
- **Hipster checks:** warm start matches HF under transformers 4.57.6; Qwen3-1.7B PMC ppl 7.918 (= DAS-6).
  **II-Medical-8B: PMC ppl 7.455** (Qwen3-1.7B 7.92; our corrected 1.7B 5.57) and **0% boxed answers at 32 tokens**
  (no_boxed ~100% on all 7 sets, both runs). => as it is, II-Medical-8B is a poor teacher for the 1.7B backbone: on PMC it
  is barely better than the untrained 1.7B student (Stage 1 would repeat the "teacher worse than student" failure), and for
  Stage 2 it does not produce the answer-only format (same reason the untrained 1.7B could not be the Stage-2 teacher).
  Needs a user decision before any S1 training (see proposals).
- **CFP (i-somet.org/iea-aie2027/cfp.html, read 2026-11-16):** deadline 2026-12-15, long paper 12 pages incl. references,
  LNCS/LNAI, double-blind, CMT; criteria originality, significance, technical merit, clarity; scope "applications of
  applied intelligent systems to solve real-life problems" incl. medicine; special session iSeA covers e-health; best
  papers -> Applied Intelligence special issue. An applied venue: the on-device (Jetson) part is central to the fit.
- **Assessment given to the user:** not enough for acceptance yet. Have: Stage-1 study (27 variants), continuum, cost
  comparison, clipping finding, B1, 1.7B conventional + naive nested (student). Missing: every MedNest accuracy number
  (Stage 2 never run), independent + MatFormer baselines, all Jetson measurements. Risk: 0.6B answer-only ceiling 23.1.
- **Proposals (waiting for user decisions, nothing submitted):**
  P1 run Stage 2 NOW on the best 20M pilot checkpoint (order_minitron_lr2e-4) + A2 no_stage1 + B0 untrained 0.6B at
     1,024 tokens: ~1 day of node205, tells whether accuracy separates the widths at 0.6B before 100M tokens are spent;
  P2 make the 1.7B backbone the accuracy backbone on DAS-6 node205 (2 x A40) instead of hipster (queue estimate is after
     the deadline): Stage-1 with CE only or the corrected 1.7B teacher... conflicts with "teacher always bigger": decision;
  P3 Jetson measurements now (depend only on shapes, not on trained weights): 9 widths, switch time vs reload, power
     modes, memory under a co-running workload; a budget-trace experiment (deadline-miss rate, fixed vs elastic);
  P4 novelty at zero training cost from per-question outputs: confidence-gated width escalation (cascade inside one
     model), cross-width answer consistency (nested vs independent), per-width calibration;
  P5 three seeds for the main Stage-2 rows only (2.6 h each) -- conflicts with the one-seed decision: decision.
- **Paper rewritten at the user's request (2026-11-16)** around elastic inference; `paper/main.tex`, 12 pages, compiles,
  no undefined refs; previous version `paper/main_backup_2026-11-11.tex` (+ `references_backup_2026-11-11.bib`);
  draft PDF refreshed. New: abstract, introduction (three kinds of budget variation: across devices, over time, between
  requests), contributions, Sect. 3.5 "Elastic inference on the device", Sect. 5.1 Stage-1 study + Table `tab:stage1`
  (15 rows, real numbers), Sect. 5.2 continuum + pgfplots Fig. `fig:continuum`, B1 row in `tab:main`, clipping result
  (30.6 vs 36.8) and generation-budget lesson, level shapes/params (188M/364M/596M). Removed to fit 12 pages: Table
  `tab:reference`, the medical half of `tab:cost`, the separate ablation table (Stage-2 ablations become rows of
  `tab:main`), 11 citations. New bib entry `sreenivas2024minitronpractice` (VERIFY authors). The paper now states that
  activation (Minitron) ordering is the final choice: **`FINAL_ORDERING` in code is still "none" and the user has not
  signed off**. 15 `\tbd` remain (all Stage-2 accuracy, Jetson, GPU-hours, untrained 0.6B). Fonts: T1 without cm-super
  makes MiKTeX build bitmap EC fonts (was already the case); fix before submission (e.g. install cm-super).

### 2026-11-16 (later): FINAL RECIPE fixed, final runs submitted
- **User decisions:** (1) Minitron activation ordering is the final ordering; (2) no teacher for the 1.7B backbone,
  train with CE only; (3) seeds: asked what they would verify (answer in "Seeds" below; no extra seeds for now);
  (4) run Stage 2 on the best 20M pilot now, plus the other next actions; Jetson experiments at the very end
  (inference only, about one day).
- **FINAL CONFIG AND TRAINING SETTINGS (MedNest, Qwen3-0.6B).** Code: `importance_order.FINAL_ORDERING`,
  `stage1.FINAL_S1`, `sft._PAPER` / `sft._S2`.

  | Item | Final setting | Evidence (20M pilots, PMC val ppl at 0.5 / 0.75 / 1.0 width) |
  |---|---|---|
  | Backbone / widths | Qwen3-0.6B; 9-width grid, hidden 512...1024 in steps of 64 (1/16), FFN 1536...3072, heads 8,8,10,12,12,12,14,16,16 (GQA ratio 2), depth 28 and head dim 128 fixed; anchors = grid levels 0 / 4 / 8 = 188M / 364M / 596M | grid9_anchors_plus1: all nine widths monotone (17.1 ... 7.6), anchors unchanged |
  | Ordering | Minitron activation magnitude (MLP channels, KV groups, hidden channels), 512 x 512 PMC tokens, `hf_export/qwen3-0.6b.minitron` | 13.5 / 9.73 / 7.63 vs Taylor 14.1 / 9.94 / 7.71 vs original 14.6 / 10.4 / 7.66 (all lr 2e-4); 44 s vs 100 s |
  | Stage 1 data / budget | PMC OA commercial subset, random 1,024-token windows, **100M tokens**, 32 sequences per step (32,768 tokens), 3,051 steps | pilots: 20M |
  | Stage 1 loss | **CE only** (kd_lambda 0, no teacher) | base KD 0.5 from Qwen3-1.7B 17.2 / 11.2 / 7.49 vs CE only 15.5 / 10.4 / 6.97; corrected teacher 15.3 / 10.2 / 6.92 (within 1-2% of CE only); pure KD worse |
  | Stage 1 optimiser | AdamW (fused), **lr 2e-4**, weight decay 0, 1% linear warm-up + cosine to 0, **gradient clipping 1.0**, fp32 weights / bf16 autocast, micro-batch 4 | lr 2e-4: L0 17.2 -> 14.6; no clipping: 22.0 / 13.0 / 7.56 (worse) |
  | Stage 1 levels per step | the 3 anchors + 1 random grid width (seeded by seed and step), equal level weights | one level per step 21.1 / 12.8 / 8.14; level weighting within 3% (noise); per-width norms and modulation: no effect |
  | Stage 2 data | medmix SFT: 33,406 train / 500 val, answer-only target in `\boxed{}`, thinking disabled, max 3,072 tokens | |
  | Stage 2 optimiser | AdamW, lr 5e-5, 1% warm-up + cosine, 1 epoch = 1,044 steps, effective batch 32 (16 per GPU x 2 A40), micro-batch 8, **no gradient clipping** | clipped 1.7B fine-tune 30.6 vs unclipped 36.8 |
  | Stage 2 loss | CE on answer tokens; **KD: decided by the runs below** (KD 0.5, T 2 from the fine-tuned Qwen3-1.7B over all tokens, vs CE only) | not measured yet |
  | Stage 2 levels per step | 3 anchors + 1 random grid width, equal weights | |
  | Seed | 0 everywhere | |
  | Evaluation | greedy, 32 new tokens, last `\boxed{}`; untrained backbones 1,024 new tokens | |

  Qwen3-1.7B backbone: the same recipe (Minitron ordering -> `hf_export/qwen3-1.7b.minitron`, 9-width grid hidden
  1024...2048, anchors about 0.5B / 1.0B / 1.7B, Stage 1 100M tokens CE only lr 2e-4 clip 1.0 micro-batch 2,
  Stage 2 CE only micro-batch 4), **no teacher in either stage**, on DAS-6 node205 (2 x A40, plain AdamW; ZeRO-1 and
  hipster no longer needed). The combination ordering + lr 2e-4 + CE only + grid was never run together: the 20M
  run `s1.qwen3_0p6b.pilot.final` tests it and gates the 100M run.
  Interpretation to confirm with the user: "CE only" was said about the 1.7B teacher. Applied as: 1.7B = CE only in
  both stages; 0.6B Stage 1 = CE only (the pilots support it); 0.6B Stage 2 = both variants run, the better one
  becomes the main row and the other the A3 ablation.
- **Seeds (answer to the user):** repeats verify one thing only: that a difference between two rows is larger than
  the run-to-run variation of training (data order, sampled widths). Not needed for large effects (recovery
  4,996 -> 13.5, clipping 30.6 vs 36.8, one-level-per-step). Needed only for claims resting on small gaps: KD vs CE
  only (1-2%), level weighting, and, if it turns out close (< ~2 points), MedNest vs the independent models and vs
  conventional fine-tuning. Decision: stay with one seed; revisit only if the main comparison comes out that close
  (3 Stage-2 repeats of the two rows involved, ~2-3 h each).
- **Code:** `FINAL_ORDERING = "minitron"`; `ORDERED_BY_PRESET` + `start_weights(..., preset=)` (1.7B ordered
  checkpoint); `FINAL_S1 = dict(lr=2e-4, kd_lambda=0.0, fractions=FRACTIONS_9, anchor_levels=(0, 4, 8),
  num_levels_per_step=4)`, `FINAL_ANCHORS`; new `s1.qwen3_0p6b.pilot.final`; independent baselines train grid levels
  0 and 4; `w025` no longer takes the grid; `S1_1P7B` without teacher / ZeRO-1; `SFTTrainingContext.anchor_levels`;
  `_S2` trains anchors + 1 random width (3-level models still train every level); new
  `sft.pilot.qwen3_0p6b_nested[.no_kd]`; `sft.paper.qwen3_1p7b_nested` CE only. 81 local tests pass
  (`test_experiment_banner.py` fails at import, as before this change). The Stage-2 `w_*` configs (A4) are NOT valid
  for the 9-width model (fixed weights have 3 entries): A4 is reported from the Stage-1 pilots only.
- **Jobs (`scripts/das6/queue_2026-11-16.sh`, ids in `jobs/queue_2026-11-16.ids`; submitted 2026-11-16):**
  node207: **28200** B0 untrained Qwen3-0.6B, 1,024 new tokens. node208: **28201** Minitron ordering of Qwen3-1.7B.
  node206 (1 GPU): **28202** `s1.qwen3_0p6b.pilot.final` (20M) -> continuum 28203; then 100M Stage-1 runs
  indep_l0 **28204**, indep_l1 **28205**, no_order **28206**.
  node205, in order: **28207** Stage 2 on pilot order_minitron_lr2e-4 with KD (export 28208, eval L0-2 28209-28211);
  **28212** same, CE only (28213, 28214-28216); **28217** A2 no Stage 1 (28218, 28219-28221); **28222** B2
  MatFormer-style (28223, 28224-28226); **28227** 1.7B Stage-1 smoke (20 steps); **28228 FINAL Stage 1, 100M**
  (afterok 28202; continuum 28229); **28230** M3 final Stage 2 with KD (28231, eval levels 0-8 28232-28240);
  **28241** M3 CE only (28242, 28243-28251); **28252** indep 0.5 Stage 2 (28253, eval level0 28254); **28255** indep
  0.75 Stage 2 (28256, eval level4 28257); **28258** A1 no-ordering Stage 2 (28259, eval 0/4/8 28260-28262);
  **28263** 1.7B Stage 1, 100M (afterok 28227, afterany 28241; 140 h limit, estimate ~55 h).
  Exports run on node208 (defq, no GPU): fatq node201-203 are held by another user's 15-day jobs.
  Not queued: 1.7B Stage 2 (after 28263), Jetson, continuum eval of no_order.
  Estimates: Stage-2 runs ~2 h each; pilot final ~7 h; 100M on node205 ~18 h; first 0.6B accuracy numbers
  (28209-28216) within ~5 h of submission, final 0.6B table in ~1.5 days.
- **Hipster:** 391246 (II-Medical-8B smoke) cancelled, nothing of ours queued there any more.
  `scripts/hipster/migrate_to_home.sbatch` can run now; not submitted (hipster is no longer needed for the paper:
  ask the user whether to migrate or simply delete `/scratch/skalra/flexslm` there).
- **Jetson comparison candidates** (checked on the Hugging Face hub 2026-11-16; Orin Nano 8 GB has 7.4 GB usable,
  fp16 fits up to about 3B parameters):
  fp16, direct: Qwen3-0.6B, Qwen3-1.7B (our backbones, conventional FT = B1 / E_A), Apollo-0.5B, Apollo-1.8B,
  Apollo-2B, Apollo2-0.5B (medical, Apache-2.0, open), SmolLM2-360M / 1.7B (general); gated but available:
  Gemma 3n E2B (the only released nested model), Gemma 3 270M / 1B, MobileLLM-600M.
  Only with 4-bit quantisation (3.8-4.5B): MedMobile (KrithikV/MedMobile, MIT), MedGemma-4B (gated),
  Llama-3.1-Minitron-4B-Width (pruned + distilled). Too large for the board: BioMistral-7B, Meditron-7B,
  PMC-LLaMA-13B, Mistral-NeMo-Minitron-8B, Nemotron-Elastic-12B, II-Medical-8B.
  No checkpoints: FaithMed, Flextron, LLaMaFlex, MatFormer language models.
  Note: the hub's parameter counts for Qwen3 count the tied embedding twice (0.75B / 2.03B).
- **Paper TODO (not edited after this entry):** Sect. 3.4 Stage 1 is now CE only without a teacher (text still describes
  distillation from a larger teacher as the default; Eq. 1 stays for Stage 2); final-recipe row for Table `tab:stage1`
  from 28202/28203; contribution 2 wording ("teacher must know the domain" -> recovery needs no teacher).
- **Bug found by 28207 and fixed (same day):** every Stage-2 run that starts from a Stage-1 checkpoint failed at load:
  the RoPE tables `self_attn.cos_<level>` / `sin_<level>` are buffers in the state dict sized by max_seq_length
  (Stage 1: 1,024, SFT: 3,072). `training.sft.load_trained_weights`: strict load of every weight, the model keeps its
  own RoPE tables (they depend on positions only); test in `test/test_continuum.py` (82 local tests pass). 28207
  failed, 28212 (same config family) and 28217 (A2; cancelled by mistake, it does not load a Stage-1 checkpoint)
  cancelled with their export/eval jobs 28208-28221. **Resubmitted on node205:** Stage 2 on pilot with KD **28264**
  (export 28265, eval L0-2 28266-28268) -> CE only **28269** (28270, 28271-28273) -> A2 no Stage 1 **28274** (28275,
  28276-28278).
- **Results in, 2026-11-16 ~20:00:**
  * **B2 MatFormer-style (28222, eval 28224-28226), 0.5 / 0.75 / 1.0 width:** L0 24.9/31.4/25.0/3.1/23.2/13.1/9.6 =
    **18.6**; L1 27.6/31.5/24.7/14.5/20.1/13.1/10.3 = **20.2**; L2 27.7/32.3/24.4/17.5/21.8/12.3/9.8 = **20.9**
    (conventional FT 23.1). Multiple-choice sets at the guess rate at every width; only MedCalc separates the widths
    (3.1 / 14.5 / 17.5; conventional 23.8).
  * **Minitron ordering of Qwen3-1.7B (28201):** 143.8 s, 6.7 GB on an A10, max logit diff 3.9e-4 ->
    `hf_export/qwen3-1.7b.minitron`.
  * **1.7B Stage-1 smoke (28227, 20 steps, 2 x A40, CE only, 4 widths per step):** passed, **36.9 GB peak**, 418 tok/s
    per rank => 100M tokens in ~33 h if both ranks count (28263 has a 140 h limit).
  * **pilot final 28202** at step 240/610: val ppl 16.0 / 11.2 / (full not read) on the anchors, all nine widths
    monotone so far. B0 28200 at 12,000/12,797.

### 2026-11-17
- **State (16:27):** running: **28230** M3 final Stage 2 with KD (node205, step ~728/1044, ~5.3 h total, 28.9 GB),
  **28206** A1 Stage 1 no_order 100M (node206, step ~1340/3051, ~18 h total). Done since the last entry: 28200 (B0),
  28202/28203 (pilot final), 28204/28205 (indep Stage 1, 100M), 28228/28229 (FINAL Stage 1, 100M), 28264-28268
  (Stage 2 on pilot, KD). Queue order on node205 is now 28230 -> 28241 -> ... with 28269 (pilot CE only) and 28274
  (A2) pending (Resources) behind it. `sacct` returns nothing on DAS-6 (accounting off): read times from logs.
- **New results:**
  * **FINAL Stage 1 (28228, 100M tokens, continuum 28229), PMC val ppl over the 9 grid widths:** 9.02, 8.86, 8.41,
    7.85, **7.42**, 7.32, 7.07, 6.79, **6.57** (anchors 0.5/0.75/1.0 = 9.02/7.42/6.57; monotone). The half-width
    model beats the pretrained full Qwen3-0.6B (10.11). Uniform width 0.25 (untrained): 1,579.
  * **pilot final (28202, 20M, same recipe):** 12.60, 12.19, 11.18, 10.02, 9.18, 8.93, 8.35, 7.73, 7.27 -> anchors
    12.6/9.18/7.27, better on L0/L1 than every single-change pilot (best was 13.5/9.73).
  * **B0 untrained Qwen3-0.6B, 1,024 tokens (28200):** 32.5/32.3/42.5/3.5/24.2/19.2/9.8 = **23.4**, 3.6% unboxed
    (= conventional FT 23.1).
  * **Stage 2 on pilot order_minitron_lr2e-4, KD (28264, evals 28266-28268), 3 levels:** L2 (596M) 37.9/39.3/45.0/
    27.5/28.9/18.1/11.6 = **29.8**; L1 (364M) 29.0/33.6/31.6/24.8/26.8/14.2/10.5 = **24.4**; L0 (188M) 25.8/31.9/24.3/
    21.5/20.8/11.7/10.2 = **20.9** (0% unboxed). Exact McNemar, pooled over 12,797 questions (`eval.compare_paired`):
    L2 vs conventional FT +7.7 pooled (2,401 vs 1,422 discordant, p<1e-4); L1 vs conventional FT 596M +1.3 (p=0.0017);
    vs MatFormer-style L2/L1/L0 +9.3/+3.4/+1.7, all p<1e-4. Recovery breaks the 0.6B "ceiling" of 23.1.
- **Apollo comparison (user question; published numbers, arXiv 2403.03640, Table 3/5):** 3-shot, own prompt, up to 128
  tokens, regex option extraction; same MedQA (1,273) / MedMCQA val (4,183) test sets. Apollo-0.5B (Qwen-0.5B) 33.0 /
  37.8, Apollo-1.8B (Qwen-1.8B) 42.2 / 45.0, Apollo-2B (Gemma-2B) 38.3 / 42.0 (MMLU-medical 45.9 / 49.1 / 52.9); 2.5B
  tokens of ApolloCorpora. MedNest pilot 596M (zero-shot, 32 tokens, 33M training tokens): 37.9 / 39.3 -> above
  Apollo-0.5B, below 1.8B/2B; our 364M 29.0/33.6 and 188M 25.8/31.9 are below Apollo-0.5B. Apollo2-0.5B/2B numbers
  exist only as images on the HF cards (not read). Not run in our pipeline (would need ~1,024 tokens: they do not
  answer in `\boxed{}`); proposal to the user.
- **Paper updated at the user's request** (backup `paper/main_backup_2026-11-16.tex` + bib): Stage 1 = CE only,
  final-recipe description; Table `tab:stage1` + rows "final 20M" and "same, 100M"; Fig. `fig:continuum` + 100M curve;
  Table `tab:main` filled: untrained 0.6B, MatFormer-style x3, MedNest pilot x3 (+ McNemar text), final row \tbd;
  Apollo reference paragraph; abstract/contribution 2 numbers; generation-budget lesson + 0.6B. 13 pages: the user
  will handle length/formatting at the end (do not trim; content cut for length was restored).
- **W&B: one project `MedNest` (user).** `config/wandb.py` `wandb_target()`: every `MedNest_*` project name ->
  project `MedNest`, old name = group + tag (trainer.py; configs unchanged; test in test_continuum.py; 83 local tests
  pass). Synced to DAS-6 16:3x: pending jobs (28241, 28252, 28255, 28258, 28263, 28269, 28274, ...) log into MedNest;
  running 28230/28206 untouched (they stay in MedNest_s2_nested / MedNest_s1_full_no_order: copy them with
  `scripts/das6/wandb_to_mednest.sbatch` once finished). Existing runs: `wandb sync -p MedNest` does NOT work for
  online-logged runs (28305: prints "done", uploads nothing; cancelled) -> `scripts/wandb_to_mednest.py` copies via
  the API (config + full history at the same steps + summary, same id/name, group/tag = old project,
  `copied_from(_created_at)` in config; old projects untouched). Test 28307 (1 run), full copy **28308** (node208).
  Old per-config projects can be deleted in the W&B UI after checking (not done; ask).
- **node208 is off limits (user, 2026-11-17): keep it free for another user unless the user says otherwise.**
  Pending jobs updated with scontrol: 7 exports (28231, 28242, 28253, 28256, 28259, 28270, 28275) ReqNodeList node208 ->
  node207; 29 evals ExcNodeList node205,node206 -> +node208 (all evals now run on node207 only). Scripts: every
  `#SBATCH --exclude=node205` -> `node205,node208`; queue_2026-11-16.sh exports/ordering on node207; memory note.
- **28263 (1.7B Stage 1, ~33-40 h) re-chained** to afterany 28241, 28255, 28258, 28274 (it only waited for 28241 and
  would have taken node205 before A1 / indep 0.75 / A2). node205 order now: 28230 -> {28269, 28241} -> 28274 ->
  28252 -> 28255 -> 28258 (after 28206) -> 28263 (Slurm picks among eligible jobs by priority).
- Apollo evals in our pipeline (`--prompt_format apollo_answer`: Apollo's "User:..\nAssistant:" with the assistant
  turn pre-filled "The correct answer is", first option letter / value; 16 tokens; 0-shot): 28311 Apollo-0.5B,
  28312 Apollo2-0.5B, 28314 Apollo-2B, 28315 Apollo2-2B; Apollo-1.8B needs the Qwen-1 custom code (reviewed: identical
  to Qwen/Qwen-1_8B's except a gradient-checkpointing method; pinned commit 73072ac), loaded from a dot-free local
  snapshot `hf_export/apollo_1p8b` (`scripts/das6/apollo18_eval.sbatch`): 28313 failed (dot in repo name), 28321 OOM at
  batch 32 on an A10, **28325** batch 8. With our boxed protocol and 1,024 tokens Apollo scores 0% (never boxes; 28309/28310).
- **Apollo in our pipeline (apollo_answer, 0-shot), MedQA/MCQA/HeadQA/Calc/Bullets/Pro-H/Xpert = avg (avg w/o MedCalc):**
  Apollo-0.5B 30.2/33.2/31.9/2.0/22.8/15.9/10.2 = **20.9** (24.0); Apollo2-0.5B 30.6/31.5/32.4/1.5/30.9/13.8/10.9 = **21.7**
  (25.0); Apollo-1.8B 37.5/39.0/43.6/2.6/31.2/15.8/10.2 = **25.7** (29.6); Apollo-2B 34.1/36.4/38.2/2.5/20.8/16.3/10.9 =
  **22.7** (26.1); Apollo2-2B 49.1/43.5/43.3/1.0/35.2/22.5/11.2 = **29.4** (34.1). (Reported 3-shot: 0.5B MedQA 33.0, 2B 38.3:
  our 0-shot numbers are close, so the protocol is not unfair to them.) MedNest pilot 596M 29.8 (30.1 w/o Calc), 364M 24.4
  (24.3), 188M 20.9 (20.8). McNemar (pooled, same questions) 596M vs: Apollo-0.5B/Apollo2-0.5B/Apollo-2B p<1e-11 (also w/o
  Calc); Apollo-1.8B all p=8e-10, w/o Calc 32.9 vs 32.1 p=0.08; Apollo2-2B all p=0.74, w/o Calc 32.9 vs 35.6 p=3e-7 (Apollo
  better). 364M vs 0.5B Apollos w/o Calc: level (p 0.65-0.97). Our MedCalc advantage is in-distribution (its train split is
  in medmix). Paper: "Comparison with Apollo" paragraph now uses these numbers (14 pages; length left to the user).
  Untrained Qwen3-0.6B (1,024 tokens, reasoning) w/o Calc = 26.8: above our 364M width (but ~30x the generated tokens).
- **I1 full-width control queued (user, 2026-11-17), all on node206:** `s1.qwen3_0p6b.full.indep_full` (final Stage-1 recipe,
  100M tokens, only the full width / grid level 8 trained) **28330** (afterany 28206, 20 h limit) -> continuum (PMC ppl)
  **28331** -> Stage 2 `sft.paper.qwen3_0p6b_indep_full` (_S2 = KD 0.5 from the fine-tuned 1.7B, train_levels [8], 1 GPU x
  batch 32, micro-batch 8 -> same effective batch 32 and 1,044 steps as the 2-GPU runs) **28332** (10 h limit) -> export
  **28333** (node206, no GPU) -> eval level8 **28334**. Purpose: separates the gain of nesting from the gain of PMC
  recovery + distillation (MedNest 596M 29.8 vs conventional FT 23.1). Estimate: Stage 1 ~6-8 h (indep_l0/l1 took 4.2 / 5.0 h
  on node206), Stage 2 ~3-4 h, eval 3 min -> result ~20 h from now (node206 frees at ~03:30 on 11-18). Test added (84 pass).
  Paper: Table `tab:main` row "Independent (596M)" (\tbd), baseline (iii) text covers the full width.
- **Jetson scripts prepared (user, 2026-11-17)** in `jetson/` (README there):
  * `mednest_jetson/elastic.py`: every width = standard HF Qwen3 whose parameters are prefix **views** of the
    full-width export (zero copy; 9 widths cost one model's memory; switch = pick a view); same attention kernel
    as the loaded model; `materialize` mode (contiguous copy on switch) as fallback if strided matmuls are slow;
    `check_views` vs the exported dense levels. `measure.py`: MemPoller (MemTotal-MemAvailable, unified
    memory), page-cache drop, `load_width_from_full` (read one width's prefix slices from the single full
    checkpoint = memory-elastic deployment). `workload.py`: 210-question stratified subset
    (`jetson/questions/subset.jsonl`, 30 per benchmark, seed 0, with test_index; `data/` is gitignored, hence
    `questions/`), prompts/scoring from eval/eval_gen.py, timed greedy decoder (TTFT / decode ms per token).
  * Scripts: `check_env.py` (stack, rails, sudo, views == exported levels ON the board), `bench_widths.py`
    (per width + dense baselines: latency, memory, energy net of idle, subset accuracy), `bench_switch.py`
    (view / materialize / reload warm+cold / slice-load / all-resident memory), `bench_trace.py` (urgent/routine
    stream while nvpmodel goes 15W->7W->15W; policies fixed:max, fixed:min, elastic, reload; deadlines `auto`
    from the measured profiles), `aggregate.py` (CSV/LaTeX, on-device vs cluster per-question agreement),
    `run_all.sh` (QUICK=1 smoke ~10 min; full ~2 h estimate), `stage_models.sh` (DAS-6 rsync + Hub models pinned:
    Qwen3-0.6B c1899de, Apollo-0.5B 1a57c38, Apollo2-0.5B 044ffc6, Apollo-2B 4049fc0, Apollo2-2B ee3b5f0,
    Apollo-1.8B 73072ac local snapshot), `requirements-jetson.txt` (transformers 4.51.1 as on the cluster; board
    torch 2.8.0 NVIDIA wheel via venv --system-site-packages).
  * `jetson/nestbench/{device,power,timing,results}.py` vendored UNCHANGED (md5 in its __init__) from
    Matryoshka Learning/jetson_eval_harness; its telemetry test copied as test/test_jetson_device_telemetry.py.
    New CPU tests test/test_jetson_elastic.py (views == dense prefix slices, bit-exact at full width; share
    storage; materialize; slice-load; timed greedy == generate; subset). 29 pass.
  * DAS-6 dry run of the whole chain on the pilot export: `scripts/das6/jetson_validate.sbatch` -> **28335** (node207).
- **Jetson scripts validated end to end on a cluster GPU (A10, node207; 28340, pilot export, 3 widths, no rails):**
  check_env all required checks pass (view == exported level: max|dlogit| 0, argmax 100%; graph decode == static
  cache 100%; static vs eager differs only at bf16 near-ties, max logit gap 0.000 <= 0.25). Earlier runs 28335-28339
  found and fixed: MemPoller `_stop` name clash with Thread internals; eager decoding is launch-bound at batch 1
  (every width ~32 ms/token on the A10) so the benchmarks decode with a **static KV cache + CUDA-graph steps**
  (`mednest_jetson/decode.py`; 4.8 / 6.0 / 7.2 ms/token for the 3 widths; widths share one full-width KV buffer per
  length bucket 512/1024/2048/4096, because one cache sized for the longest prompt slowed short requests by 25%).
  On-device vs cluster per-question agreement 95-100% (differences = bf16 ties). Switch: view 0.012 ms, materialize
  165 ms, separate model load 0.25-0.40 s (warm), slice load 0.32-0.43 s; resident weights 1.19 GB elastic vs 2.30 GB
  for 3 separate models. Trace (deadlines auto) behaves as designed: elastic 20.8% missed vs fixed:max 37.5%,
  reload 66.7%.
  **Known artifact, to fix before the board runs:** in bench_trace the first use of a (width, cache bucket) pays view
  construction and CUDA-graph capture (~100-230 ms) inside that request's latency (visible as "switch_ms" of
  fixed policies); warm every (width, bucket) before the measured stream. Not fixed yet (user asked to commit once
  validation passed). Energy/power/thermal paths are untested until the board (rails absent here: recorded as None).
- **Warm-up artifact fixed and revalidated (28341, node207, all required checks pass):** `bench_widths` and `bench_trace`
  now build views, KV caches and CUDA graphs for every (width, cache-length bucket) the questions need BEFORE the measured
  block (`GraphDecoder.warmup`, `ElasticDecoders.prepare_all`, `prepare()` on the proxies; `prepare_ms` / a `warmup`
  record are logged). Trace: fixed policies now show 0.3 ms switch and 0 ms capture (was 117-233 ms), elastic 8.3% missed
  vs fixed:max 8.3% / fixed:min 0% on this A10 (deadlines are derived from the profile, so the toy numbers mean little),
  reload 45.8% missed; the reload path reports graph capture of the freshly loaded model separately (`capture_ms`,
  4.1 s over 7 reloads). Test added (33 pass).
  **Open risk for the board:** prefill is eager and launch-bound. TTFT on the A10 is flat (~34 ms) from 100 to 900 prompt
  tokens and varies ~34 <-> ~56 ms between blocks/runs (28340 vs 28341, levels 1-2 only) without any change in decode
  speed: host-side jitter, cause not found. On the Orin's slower CPU, TTFT may be dominated by launch overhead and
  hide the width effect. Mitigations if the board shows it: interleave and repeat width blocks (compare), report prefill
  also at long prompts, or CUDA-graph the prefill at bucketed prompt lengths. Decode (graph) is not affected.

### 2026-11-18
- **Completed 0.6B runs (single seed, 32-token answer-only, avg of 7 benchmarks; pooled McNemar over 12,797 questions):**
  * **M3 final, KD (28230), 9 widths L0..L8 = 21.4 21.4 21.4 22.3 22.5 22.7 24.4 26.9 27.2**: flat at the guess-rate ceiling up
    to ~0.8 width (408M), knee at 473M+. L8 vs conventional FT 23.1: +4.1 (pooled 30.2 vs 24.8, p 2e-32).
  * **M3 CE-only (28241) L0..L8 = 21.6 21.1 21.7 21.7 21.7 21.9 21.5 22.2 22.6**: no separation at all (L8 vs M3-KD: 24.5 vs 30.2
    pooled, p 1e-36). **Stage-2 KD from the fine-tuned 1.7B is what makes the full width work** (pilot: 29.8 KD vs 27.5 CE-only).
  * **A1 no importance ordering (28258) L0/L4/L8 = 22.1 / 22.5 / 28.5** vs ordered 21.4 / 22.5 / 27.2: ordering does NOT help
    downstream (L8 pooled 30.2 vs 30.7 for no-order, p 0.14; L4 p 0.56). It only helped Stage-1 perplexity.
  * **Independent baselines (28252, 28255): 0.5 width 21.3 (nested L0 21.4, p 0.70), 0.75 width 22.3 (nested L4 22.5, p 0.12)**: nested == independent
    at the narrow widths (accuracy parity; advantage = one model, memory, switching).
  * **Final (100M, CE-only Stage 1, 9 widths) is WORSE than the 20M pilot (3 widths, Stage-1 KD 0.5 from untrained 1.7B):
    L8 27.2 vs pilot L2 29.8 (pooled p 5e-9), 364M 22.5 vs 24.4.** Perplexity-selected recipe lost on accuracy. Confounded
    (KD vs CE Stage 1, 3 vs 9 widths, 20M vs 100M tokens). Discriminating run not queued: Stage 2 on `s1.qwen3_0p6b.pilot.final`
    (20M, CE-only, 9 widths). A2 (no Stage 1, 28274) and the full-width control (28332) are running.
  * **Full-width Stage 1 control (28330):** PMC ppl 6.48 at full width vs nested 6.57 (nesting costs ~1.4% there); the other widths
    of that model are untrained-by-design (1,124 ... 7.31).
- **Controller / trace discussion:** the only switching policy in the repo is `elastic` in `jetson/bench_trace.py:131` (rule: widest width
  whose PREDICTED latency in the current power mode meets the request's deadline, else narrowest; predictor = per-mode linear
  fit TTFT(prompt tokens) + decode ms/token x p90 answer length from bench_widths records; the urgent/routine label is given).
  Not covered: memory budget, energy/battery, temperature, accuracy/confidence. The old harness's exp09 is scaffolding only.
  Trace results so far are validation on an A10 (toy): see the answer given to the user; not informative about the benefit.
- **Queued (user, 2026-11-18): Stage 2 from the 20M final-recipe checkpoint** `sft.pilot.qwen3_0p6b_final9` (= M3's Stage 2: KD from the
  fine-tuned 1.7B, 9-width grid, anchors + 1 random width; start = `s1.qwen3_0p6b.pilot.final`: 20M tokens, CE-only, 9 widths):
  train **28347** (node205, afterany 28274, ~5.5 h) -> export **28348** (node207) -> eval levels 0-8 **28349-28357** (node207).
  Reading: ~29.8 at the full width => the 100M budget is what hurts; ~27 => Stage-1 CE-only / 9 widths is. Config + test in
  training/sft.py, test/test_continuum.py (17 pass), uncommitted.
- **28263 (1.7B Stage 1, 100M, ~35 h) is on HOLD** (`scontrol hold 28263`; release with `scontrol release 28263`). It was queued on
  2026-11-16, before any 0.6B Stage-2 result, as the 1.7B scaling run of the same recipe (CE-only, Minitron ordering, 100M, 9 widths);
  that recipe gave a worse 0.6B model than the 20M pilot and the ordering showed no downstream benefit, so the 1.7B run would
  inherit an unvalidated recipe. Decision pending with the user (cancel / release / run a ~7 h 20M 1.7B pilot of the fixed recipe first).
- **HydraLLM comparison (user question, 2026-11-18; arXiv 2610.23455 + github tuhh-ncps/hydrallm @64c15ff, read in full):**
  * Slicing operator = ours (prefix of embedding/residual, query heads with KV following, FFN, LM-head columns; one knob k).
    Gemma 3 270M is MQA (1 KV head, never sliced), 4 widths 25/50/75/100%. "Head stacking" = one k sampled per step
    (weights 1/1/1/3) = our `single` pilot, which was worse than all/anchor levels per step. Not a better method.
  * Recipe (configs/train_llm/pt_hydra_default.yaml): FineWeb-Edu (general), seq 3,072, eff. batch 16, 20k steps =
    **0.98B tokens** (long run 1M steps = 49B), lr 3e-4 cosine, wd 0.1, clip 1.0, z-loss 1e-5, **KD weight 0.1, T 2 from
    the frozen ORIGINAL Gemma 3 270M** (same model, not bigger), KD on label tokens. Selected by downstream benchmarks.
  * They do NOT heal the small widths on knowledge tasks: H1/H2 HellaSwag 26.2/26.9 (chance 25), TriviaQA 0.15/0.23 (full
    pretrained ~11; H4 after training 5.5). Only ARC-Easy (log-likelihood scoring) separates (H2 46.3). 49x more tokens:
    ARC-E +4-5, HellaSwag +0-2, TriviaQA worse (H4 5.5 -> 3.6). No-teacher ablation halves TriviaQA (H4 5.5 -> 2.2).
  * Their LM-level recovery is weaker than ours: ppl 50%/100% = 71.6/44.3 = 1.62x; ours 0.5/1.0 = 9.02/6.57 = 1.37x.
  * Our budget: Stage 1 100M = 10x fewer tokens; per non-embedding parameter 0.23 vs ~10 tok/param.
- **Full-width control done (28330 -> 28332 -> 28334), Stage 1 final recipe at full width only + Stage 2:** 36.1/37.2/43.5/
  25.6/28.9/16.1/10.4 = **avg 28.3** (M3 nested L8 27.2, pilot L2 29.8). Nesting costs ~1 point at full width (1 seed);
  the 20M KD pilot beats the full-width-only model of the final recipe => the final Stage-1 recipe (CE only on PMC), not
  nesting, is the regression. Together with indep 0.5/0.75 == nested: narrow widths are at the guess rate because of
  capacity x task (medical MCQ is knowledge-heavy like TriviaQA), not because of the nested recipe.
- Proposed pilots (waiting for user): P0 log-likelihood (cloze) eval of existing exports per width (medqa/medmcqa cloze +
  ARC-E + HellaSwag; needs lm-eval in our overlay); P1 Stage 1 20M CE + KD 0.1 T2 (teacher choice = user decision:
  original Qwen3-1.7B complies with the bigger-teacher rule, original Qwen3-0.6B = HydraLLM-exact); P2 = P1 + 50%
  FineWeb-Edu replay. More Stage-1 tokens is not recommended (28347 already tests 20M vs 100M).
- **User decisions (2026-11-18, later):** (1) medical data ONLY in every stage, no general data; other medical corpora may
  be mixed in; (2) teacher ablation with several teachers, incl. medical SOTA models; (3) KD-weight pilots 0.1 / 0.3 / 0.5
  with the same (final) recipe; (4) question: provenance of eval protocols / cloze / OLMES; (5) END GOAL: be better than or
  on par with FaithMed and Apollo at matched FLOPs and parameter count.
- **Teacher compatibility (HF hub, 2026-11-18; logit KD needs identical token ids):** vocab.json/merges.txt blob ids identical
  to Qwen3 (4783fe10ac3a / 31349551d90c) for Apollo-0.5B (qwen2), Apollo2-0.5B (qwen2), **Apollo2-7B** (qwen2, vocab padded
  to 152,064 -> slice logits to 151,936), **II-Medical-8B** (qwen3), HuatuoGPT-o1-7B (same merges; Qwen2.5). NOT compatible
  (Gemma, 256k vocab): Apollo-7B, Apollo2-2B. FaithMed: still no checkpoints on the hub. Apollo-0.5B / Apollo2-0.5B are
  weaker than our student (20.9 / 21.7 vs 29.8) -> not teachers. Known PMC ppl: Qwen3-1.7B 7.92, II-Medical-8B 7.46,
  corrected 1.7B 5.57; our final student full width 6.57.
- Medical corpora on the hub (ungated): FreedomIntelligence/ApolloCorpus (apache-2.0), MedRAG/textbooks (MedQA's 18
  textbooks), MedRAG/statpearls, MedRAG/pubmed, epfl-llm/guidelines (license "other"); licences to check per source.
- Protocol provenance: generation protocol = FaithMed's scoring (`rewards/medmix_reward.py` re-implemented in
  eval/eval_gen.py), thinking off, 32 tokens; `eval/medcloze/*.yaml` are our own lm-eval-harness tasks ("Question: ..
  \nAnswer:" + answer text, acc/acc_norm; MMLU medical via lm-eval's `*_continuation` tasks; medmcf = letter format);
  OLMES is only cited (gu2025olmes), not implemented; lm-eval is not installed in the current DAS-6 overlay.
- **User decisions (2026-11-18, evening):** (1) go ahead with the eval-only step (teacher screen + inference-compute
  curve); (2) a 7-8B teacher only after the pilots show that Stage-1 KD helps, and only if that teacher was trained on
  our PMC data with our logit scheme (i.e. PMC-corrected like the 1.7B); (3) run the inference-compute comparison for
  FaithMed; (4) **28263 (1.7B Stage 1) CANCELLED** (`scancel`, done) until the recipe is final; (5) Stage-1 objective =
  HydraLLM's **CE + lambda KL**, sweep lambda 0.1 / 0.3 / 0.5; (6) pilot HydraLLM's weighted sampling (one width per
  step, full width 3x as likely); keep it as default if faster at similar accuracy.
- **A2 no Stage 1 (28274 -> evals) done: L0/L1/L2 (0.5/0.75/1.0) = 20.8 / 22.7 / 30.5 avg (pooled full width 33.4).**
  Full width beats the 20M KD pilot (pooled 32.5, p 0.015) and M3 (30.2, p < 1e-4): **so far Stage 1 on PMC does not
  help downstream at any width** (narrow widths also equal M3's). The KD sweep tests whether KD from the PMC-corrected
  teacher changes that; if no arm beats A2, the honest recipe is "no Stage 1" (or Stage 1 must be justified otherwise).
- **Code (uncommitted, 51 tests pass, job 28360):** `training/trainer.py` `kd_mix` / `KD_FORMS` (convex = old default,
  additive = CE + lambda KL) used by FlexLMKDTrainer and MemOptFlexLMKDTrainer; `level_sampling_weights` for
  `level_sampling="single"` (`random.choices`, seeded by (seed, step); weights None keeps the old `choice` draws).
  `training/stage1.py`: `Stage1Context.kd_form`, `.level_sampling_weights`; `S1_KD_TEACHERS` (corr = PMC-corrected 1.7B,
  orig = Qwen3-1.7B), `S1_KD_LAMBDAS`, `HYDRA_WEIGHTS_9` = (1 x8, 3) (P(full) 3/11), `kd_pilot_name`, 12 configs
  `s1.qwen3_0p6b.pilot.final.kdadd{0p1,0p3,0p5}.{corr,orig}[.hydra]` (= pilot.final + KD). `training/sft.py`: Stage 2
  `sft.pilot.qwen3_0p6b_final9.<same suffix>[.1gpu]` = exactly final9's (M3's) Stage 2. `eval/eval_gen.py`:
  `--prompt_format think` (Qwen3 thinking ON), `--budgets` (accuracy at every smaller decoding budget from one
  generation: a truncated greedy/sampled trajectory IS the smaller-budget run), `--per_source N`, `--temperature/
  --top_p/--top_k/--seed`, per-sample `gen_tokens` / `prompt_tokens`. evalgen.sbatch env BUDGETS / PER_SOURCE /
  SAMPLE=qwen3; eval_continuum.sbatch --hf passes extra args; new scripts/das6/pytest.sbatch,
  scripts/das6/queue_2026-11-18.sh, test/test_kd_sweep.py. test/test_experiment_banner.py is stale (imports a removed
  `_print_experiment_banner`; pre-existing, not touched).
- **Queued 2026-11-18 16:51 (jobs/queue_2026-11-18.ids):** tests 28361 -> smoke 28362 (kdadd0p1.corr) -> 28363 (.hydra),
  node206. Stage 1 (20M, lambda additive, corrected teacher): **lambda 0.1 28364 (node205, after 28347)**, **Hydra lambda 0.1
  28365**, **lambda 0.3 28366**, **lambda 0.5 28367** (node206); PMC continuum 28368-28371. Stage 2 (node205) -> export
  -> eval levels 0/4/8: 0.1 28372 -> 28373 -> 28374-76; hydra 28377 -> 28378 -> 28379-81; 0.3 28382 -> 28383 -> 28384-86;
  0.5 28387 -> 28388 -> 28389-91. lambda 0 = 28347 chain. Teacher screen (PMC ppl, node207): Apollo2-7B 28392,
  HuatuoGPT-o1-7B 28393, Qwen3-8B 28394. Inference-compute curve (node207; thinking on, Qwen3 sampling T 0.6 / p 0.95 /
  k 20 seed 0, 300 questions per set, budgets 64..4096): smoke 28395 -> Qwen3-1.7B 28396, Qwen3-0.6B 28397.
  Compare MedNest at 32 tokens on the same 2,100 questions (first 300 per set of the existing samples files).
- Next: teacher ablation (orig 1.7B) at the best lambda; Hydra sampling becomes default only if its L8/L4 are within
  noise of the lambda-0.1 sandwich arm (pooled McNemar) — it takes ~1/2 of the Stage-1 time. Paper TODOs (no edits now):
  Stage-1 objective form, A2 result, inference-compute figure.
- **Paper rewritten (user, 2026-11-18): ONE re-training step.** `paper/main.tex` = ordering + nested SFT with KD (the old
  Stage 2) only; every Stage-1 (PMC pre-training) section, table, the continuum figure and the cost table were removed;
  PMC pre-training appears only as one ablation row (+20M tokens: 20.9/24.4/29.8 vs 20.8/22.7/30.5; full width p 0.015,
  364M p 0.008, 188M p 0.19). Backups of the two-stage version: `paper/main_backup_2026-11-18.tex`,
  `references_backup_2026-11-18.bib`, old PDF `MedNest_IEA-AIE2027_draft_backup_2026-11-17.pdf`. New PDF
  `paper/MedNest_IEA-AIE2027_draft.pdf` (12 pages incl. refs, builds clean).
  Main result = A2 (`sft.paper.qwen3_0p6b_nested.no_stage1`, 3 widths): 30.5/22.7/20.8. Pooled McNemar (computed locally
  from copied samples): L2 vs conventional +8.5 pooled (2,567 vs 1,473, p<1e-4); L1 vs conventional 25.2 vs 24.8 p 0.35
  (w/o Calc 0.42); L0 vs conventional -1.3 p 0.001; vs MatFormer all widths p<1e-4; L2 vs untrained t1024 p<1e-4; L2 vs
  FT 1.7B -6.5. Apollo (L2): 0.5B/Apollo2-0.5B/1.8B/2B all p<1e-3 also w/o Calc; Apollo2-2B overall p 0.15, w/o Calc
  33.9 vs 35.6 pooled p 0.0013 (Apollo better). L1 vs 0.5B Apollos: better overall (p 0.026/0.008) only via MedCalc;
  w/o Calc below (p 0.026/0.12). Cost: A2 4:15:47 on 2xA40 (28274), teacher E_A 3:01:01 (28131) -> ~15 GPU-h.
  **Placeholders (red) needing runs NOT yet queued:** Stage-2-only ablations (no KD, no ordering, one random width per
  step), independent 188M/364M (SFT only), 1.7B MedNest (ordering + nested SFT; teacher decision: bigger teacher or CE
  only), Jetson table. FaithMed table rows wait for 28396/28397 (+ prompt-token counts -> GFLOPs). Bib still has the old
  "VERIFY" notes in several entries.
- **Hipster queue 2026-11-18 17:27 (scripts/hipster/queue_2026-11-18.sh; ids in ~/FlexSLM-medbench/queue_2026-11-18.hipster.ids),
  all capacity (performance --test-only said 2026-12-04; user: ignore estimates, queue and wait):** migrate_to_home
  397255 (data/results/jobs/HF cache -> /home; SKIP_JOB_CHECK) -> fetch exports from DAS-6 397256 (E_A teacher,
  qwen3-0.6b.minitron, qwen3-1.7b.minitron -> /scratch hf_export) + tests 397257 -> smoke 397258 (0.6B single, 2xL4)
  397259 (1.7B, 4xL4 ZeRO-1 + act. ckpt) -> trainings (A2 + one change; 0.6B on 2xL4, micro-batch 4, same 32/step):
  no_kd 397260 (export 397261, evals 397262-64), no_order 397265 (397266, 397267-69), single 397270 (397271,
  397272-74), indep_l0 397275 (397276, 397277), indep_l1 397278 (397279, 397280), indep_l2 (full width alone WITH KD)
  397281 (397282, 397283), 1.7B MedNest CE-only `sft.paper.qwen3_1p7b_nested.no_stage1` 397284 (397285, 397286-88).
  Results land in ~/FlexSLM-medbench/results (home). New: scripts/hipster/{pytest,fetch_exports,export}.sbatch,
  hipster evalgen -> capacity L4; configs in training/sft.py, test in test/test_kd_sweep.py.
- Paper: ablation table separates "independent (with KD)" (tbd) from "conventional FT (no KD)" 23.1.
  Note (user question): the 1.7B teacher = full-width conventional FT only (train_levels [2], CE, no clip), not nested.
- **W&B consolidated into the single project `MedNest` (user, 2026-11-18); old per-config projects deleted.** node208 was used for
  these jobs on the user's explicit OK (standing rule unchanged: node208 only when told). Procedure (`scripts/wandb_to_mednest.py`):
  copy the two remaining finished runs (28230 ouvny4cs, 28206 qp73a2io) -> verify every old run against its copy (run id, name, group,
  state, history rows, config, final summary: **42 of 42 OK**; two fresh copies needed a second pass because W&B summaries lag a few
  seconds) -> back up every old run (config, summary, full history) as JSON in
  `/var/scratch/skalra/flexslm/wandb_backup_2026-11-18/<project>/<run id>.json` (42 files, 3.6 MB) -> delete the old projects with
  the GraphQL `deleteModel` mutation (wandb 0.19.11 has no project-delete call; one project first as a test, then the other 38).
  **39 projects (`MedNest_*`, 42 runs) deleted, no empty shells left.** Untouched on purpose: the entity's unrelated projects
  (FlexLLaVA, FlexViT, ViT-ImageNet-Adaptivity, FlexLLaMA_*, Scala, dimenet++, huggingface, segmenter, ...). `MedNest` now has 52
  runs (47 finished, 3 failed = the crashed copies, 2 running: A2 28274 and the full-width Stage 2 28332), grouped by the old
  per-config name (group + tag). New runs go to `MedNest` directly (config/wandb.py). The jobs' local wandb run directories under
  `/var/scratch/skalra/flexslm/store/wandb/job_*` also still exist. `scripts/wandb_inventory.py` = read-only inventory.

### 2026-11-18 (late): status check, hipster fix, maintenance, commit
- **DAS-6 (~21:00):** 28347 (Stage 2 of the 20M final-recipe pilot) at step 903/1044, still running on node205 (export 28348 and
  evals 28349-57 follow); 28361 unit tests 51 passed, smokes 28362 / 28363 (kdadd0p1.corr, hydra) passed; **28365 (Hydra
  lambda 0.1 Stage 1) running on node206 (3.7 h)**; 28364 (node205), 28366 / 28367 (node206) wait for those nodes (no job has
  failed). node207: 28396 (1.7B reasoning curve) running, 3.8 h, 768/2098 questions. Nothing new has finished that gives a result.
- **Hipster (397255-397288):** migrate_to_home 397255 COMPLETED (3 min), fetch of the 3 exports from DAS-6 397256 COMPLETED
  (hf_export has the E_A teacher, qwen3-0.6b.minitron, qwen3-1.7b.minitron; home now 129 GB / 200 GB). **397257 (unit tests)
  FAILED: `my_env` has no pytest** (never install into my_env); the two smoke jobs then sat on DependencyNeverSatisfied.
  Fix: `scontrol update JobId=397258/397259 Dependency=` (smokes now pending on Priority, whole chain intact) and
  tests rerun as **397369** with pytest pip-installed into `~/flexslm_saves/pytest_overlay` (scripts/hipster/pytest.sbatch).
  The new Stage-2-only ablation test (test_stage2_only_ablations_change_one_thing_each) has NOT run anywhere yet (DAS-6 GPU
  nodes are all busy, pytest.sbatch pins node207; the hipster smokes also resolve the single / 1.7B configs).
- **Bug found:** `sbatch --export=ALL,...,BUDGETS=64,128,...` splits at the commas, so 28395/28396 got BUDGETS=64 only (the smoke
  showed only "budget 64"). Samples files still hold every full generation + gen_tokens, so **28396's curve is rescored
  offline: `python -m eval.rescore_budgets results/gen.Qwen-Qwen3-1.7B.t4096.pthink.n300.sq3 --budgets 64,...,4096
  --tokenizer Qwen/Qwen3-1.7B --per_source 300`** (new eval/rescore_budgets.py; needs transformers, run as a job).
  28397 cancelled and resubmitted as **28405** with `env BUDGETS=... sbatch` (queue_2026-11-18.sh fixed the same way).
- **Maintenance 2026-11-19 ~13:00 (no reservation shown by scontrol on either cluster):** jobs running during a shutdown are
  killed; Slurm requeues batch jobs (JobRequeue=1 on both; same as 2026-11-12, when B1 28150 / noclip 28155 were requeued) and
  training resumes from `last.ckpt` (saved every 5% of an epoch, i.e. ~13 min of a 4.3 h Stage 2; resume=True by default).
  Pending jobs and their dependency chains survive. DAS-6 (ReturnToService=1): after an unexpected reboot the nodes stayed DOWN
  until an admin resumed them (2026-11-12 -> 11-13); hipster has ReturnToService=2 (nodes return by themselves).
  Evals / exports / continuum jobs restart from scratch (short). Stage-1 runs resume the same way.

### 2026-11-19
- **Overnight failure (my bug):** training/sft.py (ablation configs of 2026-11-18) passed `hf_start` twice for `no_order` ->
  TypeError on import of config.experiments -> every job that started after the 2026-11-18 evening sync died: DAS-6 28364,
  28366, 28367 (Stage 1), 28348 (export of the finished 28347), 28369 (continuum), 28377 (Stage 2 Hydra); hipster 397258/59
  (smokes; they caught it) and 397369 (tests); all their dependents sat on DependencyNeverSatisfied. Fixed in sft.py (dict
  merge); **52 tests pass (DAS-6 28406, incl. test_stage2_only_ablations_change_one_thing_each)**. Lesson: run the unit tests
  AFTER the last code edit and before queueing (the 28361 run predated the ablation configs).
- **Finished:** 28347 (Stage 2 of the 20M final-recipe pilot, 5 h 29 min on 2 x A40, val answer-token acc 0.826 ... 0.853 over the
  9 widths); 28365 (Stage 1, Hydra sampling, additive lambda 0.1, corrected 1.7B teacher, 20M tokens): **PMC ppl L0..L8 = 15.30 / 13.70
  / 12.10 / 10.90 / 10.20 / 9.61 / 8.89 / 8.26 / 7.87** vs CE-only sandwich pilot.final 12.60 ... 9.18 ... 7.27 (worse at every
  width; accuracy needs its Stage 2); 28392-94 teacher screen (PMC val ppl, same 640 x 1,024 windows): **HuatuoGPT-o1-7B 5.013,
  Qwen3-8B 5.574, Apollo2-7B 5.652** (Qwen3-1.7B 7.92, II-Medical-8B 7.46, corrected 1.7B 5.57, our student 6.57); 28395/28396
  reasoning-curve smoke and Qwen3-1.7B thinking run (2,098 questions; JSON has budget 64 only -> rescore 28444).
- **Resubmitted (maintenance 2026-11-19 ~13:00 was imminent; queued at 12:56):** DAS-6 `scripts/das6/queue_2026-11-19.sh`
  (ids in jobs/queue_2026-11-19.ids): export 28407 + evals L0-8 28408-16 (of 28347); Hydra continuum 28417, Stage 2 28418, export 28419, evals
  28420-22; Stage 1 lambda 0.1 28423 (continuum 28424, Stage 2 28425, export 28426, evals 28427-29), lambda 0.3 28430 (28431, 28432,
  28433, 28434-36), lambda 0.5 28437 (28438, 28439, 28440, 28441-43); rescore of the 1.7B curve 28444; 0.6B curve 28405 running.
  Hipster `scripts/hipster/queue_2026-11-19.sh`: tests 398150 -> smokes 398151 / 398152 -> 7 trainings 398153 ... 398177 (+ exports,
  evals) as in the 2026-11-18 chain (ids in ~/FlexSLM-medbench/queue_2026-11-19.hipster.ids).
- **Hipster (2026-11-19 15:50 check):** not rebooted (up 76 days). Tests 398150 passed. **Smoke 398151 (0.6B, 2 x L4 24 GB, micro-batch 4)
  ran OUT OF MEMORY** in sft.py's KL (22 GB card; the DAS-6 recipe needs ~21 GB at micro-batch 8 on a 48 GB A40) and hung; my
  `scancel -n a,b,c` had also not cancelled the 2026-11-18 chain (7 trainings on DependencyNeverSatisfied). Per the hipster rules
  (multi-GPU / > 24 GB -> performance) everything moved to **performance, 2 x RTX 6000 Ada 48 GB, 16 per GPU x 2 = 32 per step,
  micro-batch 8 (0.6B) / 4 (1.7B) = A2's and sft.paper.qwen3_1p7b_nested's settings; the L4-specific micro-batch / ZeRO /
  checkpointing overrides were dropped.** Cancelled every flexslm job there and resubmitted (queue_2026-11-19.sh, ids in
  ~/FlexSLM-medbench/queue_2026-11-19b.hipster.ids): tests 398291 -> smokes 398292 (0.6B single) / 398293 (1.7B) -> trainings
  398294 (no_kd), 398299 (no_order), 398304 (single), 398309 (indep_l0), 398312 (indep_l1), 398315 (indep_l2), 398318 (1.7B),
  each with export (398295 ...) and 32-token evals on capacity L4. Per-user QoS cap 256 CPUs in performance = 4 concurrent
  2-GPU jobs at most.
- **DAS-6 back after the maintenance (15:48):** rebooted ~14:05, nodes 205-207 up (204 down since April), jobs survived: running
  28418 (Hydra Stage 2, node205), 28430 (lambda 0.3 Stage 1, node206), 28405 (0.6B reasoning curve, node207; it RESTARTED from
  scratch, was at 1600/2098 before).
- **Jetson (2026-11-19): shape-only 15 W sweep of the 0.6B MedNest done; 2B Apollo baselines do not fit; 1.7B pending.**
  * Board: IP changed to 10.113.206.56 (`~/.ssh/config` host `jetson` updated). Disk was 97% full: backed up four HF caches (clip-vit-large 4.8,
    SmolLM2-1.7B 3.2, phi-2 2.6, TinyLlama 2.1 GB) as one tar + MobileLLaMA-1.4B as a second tar to
    `C:\Users\skalra\Workspace\Matryoshka Learning\jetson_eval_harness\jetson_hf_cache_backup_2026-11-19*.tar` (size and listing verified), then deleted them
    on the board (15 GB free; `apt clean` run by the user). Repo cloned to `~/FlexSLM` (commit a6a3ad2), venv `~/mednest-venv` made with `virtualenv`
    (python3.10-venv needs sudo). Staged from the Hub: Qwen3-0.6B, Apollo-0.5B, Apollo2-0.5B (Apollo-2B / Apollo2-2B: config + tokenizer only).
    DAS-6 was in maintenance, so no trained export was staged.
  * **Decision (user): experiments without accuracy/performance measurement use random weights of the exact MedNest shapes, not checkpoints; 15 W only.**
    New `jetson/make_random_export.py` (9-width grid of Qwen3-0.6B: hidden 512..1024, heads 8..16, kv 4..8, mlp 1536..3072; level 8 + anchors 0/4 with
    weights, all levels with config + tokenizer), `jetson/run_shape_only.sh` (preflight, widths + dense baselines, switch cost; `BACKBONE=1p7b` for the 1.7B
    shapes from `jetson/widths_qwen3_1p7b.json` = s1.qwen3_1p7b.full; `SKIP_2B=1`), `jetson/drop_file_cache.py`. Edits: `MEDNEST_FORCE_LEN=1`
    (decode exactly max_new tokens, ignore EOS: `workload.eos_ids`), `bench_widths.py` (config-only dense models built on the GPU from the config,
    OOM recorded as `kind: oom`, `import torch._dynamo` first because the Gemma modules fail on a circular import otherwise),
    `bench_switch.py` (OOM per step recorded), `elastic.check_views` (skips levels without weights: the staged export only has weights for 0/4/8).
    Not run on purpose: 7 W profile, 15 W <-> 7 W trace, subset accuracy (all need trained weights or the 7 W mode). Code changes are NOT committed.
  * **Results** (15 W, 210 questions, forced 32 new tokens, bf16, graph decode; `jetson_logs_mednest/mednest-results/shape_only_full/`, backup in
    `C:\Users\skalra\Workspace\jetson_logs_mednest`): decode ms/token and net J/answer by level 0..8 = 21.0/4.0, 22.3/4.4, 25.2/5.3, 28.0/6.3, 29.3/6.8,
    30.8/7.3, 34.3/8.4, 37.5/9.5, 38.7/10.0 (params 188M..596M); TTFT 180-200 ms at every width. Same-shape dense baselines = nested widths within 1%
    (conventional 38.7 ms/tok, indep-l0 20.7, indep-l1 29.4). Apollo-0.5B 24.7 ms/tok 3.5 J, Apollo2-0.5B 28.1 / 3.8 J. Switching: view 0.037 ms,
    materialize ~749 ms (and 2x the memory), slice-load from the full file 0.75-1.1 s (warm cache only: no passwordless sudo for drop_caches).
    Memory: all 9 widths as views 1.19 GB vs 6.77 GB for 9 separate models (3 separate: 2.30 GB).
    **Caveat: `peak_sys_mem_mb` grows 5.0 -> 7.3 GB across the targets of one process (nothing is released between models); use weight bytes for the
    paper's memory numbers, or one process per model.** The Qwen3-0.6B reasoning baseline has only 35 answers (5 per benchmark x 1,024 tokens, 456 J each);
    its accuracy fields and all accuracy fields of this sweep are meaningless (random weights / forced length).
  * **Apollo-2B / Apollo2-2B (config-only, random weights): DO NOT FIT in bf16 on the 8 GB Orin Nano (decision 2026-11-19: report them as "does not fit").**
    After the user's reboot (first CUDA process, 6.7 GB free, 15 W, `jetson_logs_mednest/mednest-results/shape_only_2b_clean/`) both failed at weight
    allocation with the CUDA allocator assert (`NVML_SUCCESS == r INTERNAL ASSERT`, which this Jetson torch build raises on OOM; recorded as `kind: oom`).
    Apollo-2B = 2.51B params = 5.01 GB bf16, Apollo2-2B ~ 5.2 GB; the largest single allocation the board accepted was 5-5.5 GB. One earlier 6-question run of
    Apollo-2B alone did complete (118 ms/tok, 16.4 J/answer, 321 ms TTFT) but could not be reproduced (4 later attempts, also after the reboot): treat as
    borderline, do not report it as a measurement. About 0.9 GB shows as "used" with no process running after a CUDA process exits. Not tried: 8-bit/4-bit
    (different protocol, needs a quantisation library on the board).
    **Paper TODO (edit only at the end): the comparison table (paper/main.tex ~l.197-198) lists Apollo-2B and Apollo2-2B as accuracy baselines; in the Jetson
    table give them "does not fit (5.0 / 5.2 GB bf16 > 8 GB shared memory)" in place of latency/energy.**
  * **1.7B backbone (same protocol): prepared, started, cancelled by the user at 16:05 (to run later).** Estimate ~3 h (2.5-4 h). Command on the board:
    `BACKBONE=1p7b MEDNEST_OUT=~/mednest-results/shape_only_1p7b bash jetson/run_shape_only.sh` (Qwen3-1.7B config + tokenizer staged; the random export
    is rebuilt by the script). Expect OOM for the materialize step and the 3-resident-models step (full width is ~4 GB).

- **Jetson (2026-11-19, evening): 1.7B shape-only sweep at 15 W finished (19:16 to 20:41 board time, ~85 min; faster than the 3 h estimate).**
  * Results in `jetson_logs_mednest/mednest-results/shape_only_1p7b/` (copy verified: 4 files, same sizes) plus `shape_1p7b.out`.
  * Decode ms/token by level 0..8 = 32.8, 36.4, 42.8, 49.3, 53.3, 58.2, 66.0, 73.5, 78.9 (params 508M..1.72B); median TTFT ~180-260 ms, mean 215-371 ms (long tail, max 3.8 s at level 6). Energy fields not yet tabulated here.
  * Elastic view switch 0.049 ms mean (60 switches over levels 0/4/8). Full-width resident weights 3.44 GB (all 9 widths as views).
  * **Does not fit (OOM, recorded as `kind: oom`):** the three same-shape dense baselines (conventional, indep-l0, indep-l1) next to the resident 3.4 GB model; switch steps B (materialize), C, D, E (several models resident). Report as "does not fit in 8 GB" for the 1.7B case, or rerun the dense baselines one process each (not done).
  * Preflight: PASS with warnings for `check_views` at levels 4 and 8 (dense copy does not fit next to the resident model). Code fix (`check_env.py` reports skipped as warning, `elastic.check_views` catches OOM) is still uncommitted.
  * Same caveat as 0.6B: `peak_sys_mem_mb` (6.7 to 7.6 GB) is unreliable across targets; use weight bytes.
  * **Decision (user, 2026-11-19): the paper uses only the 0.6B switching analysis.** The 1.7B switch-cost steps B-E that ran out of memory (materialize, separate-model and slice-load at level 8, all-9-resident ~18 GB) are not rerun and not reported. 1.7B dense baselines rerun one process per model into `shape_only_1p7b/widths_15W_dense_*.jsonl` (log `~/shape_1p7b_dense.out` on the board).
  * **1.7B dense baselines, one process each (15 W, 210 questions, forced 32 tokens; `shape_only_1p7b/widths_15W_dense_*.jsonl`, backed up locally):** indep-l0-shape (508M) 32.3 ms/token, median TTFT 207 ms; indep-l1-shape (1.03B) 53.0 ms/token, TTFT 215 ms. These match the nested widths 0 and 4 (32.8, 53.3 ms/token), as on 0.6B. conventional-shape (level-8 shape, 1.72B, 3.44 GB) still does NOT FIT even alone (allocator assert); the elastic model of the same size did load, so the dense path (from_pretrained to CPU, then .to(cuda): two copies in shared memory) needs more peak memory than the elastic load. Not investigated further.
  * **1.7B conventional-shape (dense, 1.72B, 3.44 GB) investigation (2026-11-19, evening): borderline, not reproducible either way.** Memory traces (free -m every 2 s) show: after from_pretrained (CPU copy) + page cache of the 3.4 GB safetensors file, free memory is ~130-230 MB at the move to the GPU; the CUDA caching allocator on this Jetson torch raises the NVML assert instead of reclaiming page cache. Outcomes with identical settings varied: a foreground probe and a 6-question run loaded fine; the 210-question run failed at load (x3) and once at the static KV-cache allocation of the longest prompt bucket (load passed after adding `_evict_page_cache` in `bench_widths.py` before and after `.to(device)`). Result: report conventional-shape (level-8 shape) as "does not fit / borderline in 8 GB" unless a direct-to-GPU loader (no CPU copy; accelerate is not installed) is written. Latency of that width is already given by the nested level 8 (78.9 ms/token); the l0/l1 dense baselines match nested widths within ~1.5%.
  * **1.7B conventional-shape, final (user decision 2026-11-19): "does not fit" for the 210-question run; the 6-question probe is reported as a reduced-sample reading.** The public HF Qwen3-1.7B checkpoint (`~/mednest/models/Qwen3-1.7B`, 4.06 GB with a separate lm_head) also failed at load (allocator assert), so it brings no improvement. Two independent 6-question probes of the level-8-shape dense model (random weights, 15 W, forced 32 tokens, `jetson_logs_mednest/mednest-results/probe_conv/widths_probe.jsonl`, `widths_probe2_20261119_221828.jsonl`): 77.0 and 76.9 ms/token, median TTFT 256 and 251 ms, net 20.9 J/answer (gross 31.1 J), 3.44 GB weights. Compare nested level 8: 78.9 ms/token (n=210). Mark as n=6 wherever used.
- **Controller ideation (user question, 2026-11-19; discussion only, no code):** proposed a two-tier controller. (1) Deploy-time selector:
  device profile (memory, power mode, backend) -> feasible widths -> pick 1-2 "anchor" widths; a compiled backend (delta file / per-level
  HF export -> dense -> GGUF/TensorRT) has fixed shapes, so it is chosen here, not per request. (2) Runtime per-request controller over
  in-process views (0.037 ms switch): widest width whose predicted latency (existing `elastic` predictor x a feedback correction from
  observed latency, for thermal/contention) meets the deadline, plus an energy virtual queue (drift-plus-penalty) for a battery budget.
  Observation from M3 (28230) + Jetson 15 W: on the (ms/token, accuracy) plane the upper convex hull is only L0 / L7 / L8, so an
  average-budget controller time-shares those; intermediate widths only matter for hard per-request deadlines. Confidence/router
  controllers need the oracle bound from per-question correctness at all 9 widths first (not computed). Risk to check: int4 llama.cpp
  of the full 0.6B may beat bf16 L0 in PyTorch; compare widths on the same backend.

### 2026-11-20
- **Results check (DAS-6 + hipster, 14:30).** ssh works again via `~/.ssh/agent.sock` in Git Bash (`export SSH_AUTH_SOCK=~/.ssh/agent.sock`);
  the Windows ssh-agent service is disabled. DAS-6 has no Slurm accounting (`sacct` disabled): read `jobs/*.out` instead.
- **Finished (all queued 2026-11-19 DAS-6 jobs except the lambda 0.5 tail and two failures).** PMC ppl L0 / L4 / L8 (Stage 1 continuum):
  Hydra lambda 0.1 (28417) 15.24 / 10.13 / 7.85; plain lambda 0.1 (28424) 12.34 / 9.02 / 7.18; lambda 0.3 (28431) 12.20 / 8.93 / 7.12;
  lambda 0.5 (28438) 12.17 / 8.90 / 7.10. Plain beats Hydra at every width; larger lambda lowers ppl slightly (0.05-0.2).
  Stage 2 val answer-token acc L0 -> L8: final-recipe pilot 0.826 -> 0.853; lambda 0.1 0.826 -> 0.854; lambda 0.3 0.827 -> 0.852 (flat across lambda).
  32-token evals L0 -> L8 (acc %): pilot medqa 25.8 -> 37.4, medmcqa 32.2 -> 38.7, head_qa 24.7 -> 44.9, medbulltes 20.1 -> 28.9;
  Hydra medqa 28.4 -> 37.3, medmcqa 31.8 -> 38.0; lambda 0.1 medqa 26.3 -> 37.9; lambda 0.3 medqa 26.7 -> 37.3, medmcqa 31.9 -> 39.6, medbulltes 18.8 -> 31.2.
  Lambda differences are mostly within the +-1-2.6 CI. Not all 9 levels of the pilot (28409-15) were read.
- **Running:** 28439 (lambda 0.5 Stage 2, node205, 67% at 14:33); 28440 export, 28441-43 evals depend on it.
- **Failures and fixes:** 28405 (0.6B think curve, node207 A10 22 GB) CUDA OOM at question 2016/2098 (bs 32, 4096 tokens) -> resubmitted as
  **28459** on node206 (A40), bs 32, 12 h. 28444 (rescore of the 1.7B curve) `No module named 'eval'`: `python -I` ignores the cwd;
  resubmitted as **28458** (fatq node202, `python -m`, no `-I`); `queue_2026-11-19.sh` fixed locally (not synced). ids in jobs/queue_2026-11-20.ids.
- **Hipster:** nothing started. 398292 / 398293 (smokes) PENDING (Priority); 7 trainings + exports + evals on Dependency.
  Only pytest 398150 / 398291 completed.

### 2026-11-20
- **Width controller built in `controller/` (separate from the pipeline; imports nothing from it; numpy only, runs locally;
  not committed).** Inputs are copies (gitignored `controller/data/`): M3 samples levels 0-8 + conventional from DAS-6 results,
  Jetson 0.6B 15 W shape-only `widths_15W.jsonl` / `switch.jsonl`. Tier 1 `deploy.py` (views vs compiled anchors), tier 2
  `policies.py` (Fixed, DeadlinePolicy +/- latency feedback, EnergyQueuePolicy = drift-plus-penalty, task-aware with shrunk
  per-set tables), `offline.py` (Pareto / hull / oracle), `simulate.py` (budget, thermal), `test_controller.py` (6 pass).
  Calibration/evaluation split of the test questions (seed 0): controller statistics never see the scored half.
  Energy model reproduces the measured J/answer (L0 4.02 / L8 9.97 vs 4.0 / 10.0).
- **Results (simulation, not a device run; see controller/README.md):** hull {L0, L7, L8}; energy queue beats the best fixed width
  within the same budget by +2.3 to +3.6 macro points (paired bootstrap CIs exclude 0, 6,423 questions); task-aware over agnostic
  +0.5..1.2 (significant only at the lowest budget). Latency feedback under a x1.3 slowdown: urgent misses 100% -> 49% (eos
  answers; floor 46%) / -> 22.5% (32-token answers), accuracy -0.3 n.s. Oracle router 50.3 is mostly not deployable (correlated
  errors + lucky guesses; needs a confidence signal not stored in samples files).
- Next (not done): Jetson runner for policies.py with trained weights + real 15 W <-> 7 W switches; GGUF/TensorRT profiles of 2-3
  widths and of int4 full; rerun on the `final9` Stage-2 evals when they land. `scripts/das6/queue_2026-11-19.sh` shows as
  modified in git; not touched by this session.
- **Jetson runner for the controller written (2026-11-20, not run yet; not committed):** `controller/jetson_runner.py` (board;
  imports jetson/mednest_jetson + nestbench read-only: ElasticModel views, ElasticDecoders, PowerSampler, set_power_mode),
  `controller/run_jetson.sh` (QUICK=1 smoke / full 630 x 6 policies), `controller/make_bundle.py` -> tracked
  `controller/bundles/{paper.qwen3_0p6b_nested.n630.s0.json, smoke.json, profile.0p6b.15W.jsonl}` (stratified 90 per test set from
  the evaluation half; calibration-half tables), `controller/analyze_board.py`, `controller/das6_dryrun.sbatch` (node207, trained M3,
  no rails). Per-request energy = VDD_IN integrated over the request window minus idle floor; the policy observes measured
  latency/energy. Default `--score cluster`: random-weight export on the board decodes the cluster's per-width answer length, correctness
  from the cluster (no checkpoint on the board); `--score device` with the trained export. Default disturbance = co-running CUDA load
  (`15W:210,15W+gpu:210,15W:210`); 7 W phases possible via PHASES (needs passwordless nvpmodel; 7 W was not used so far by decision).
  Tests: 7 pass locally (incl. window energy + phase parsing); analyze_board checked on a synthetic log. **DAS-6 dry run NOT submitted:
  fs2.das6 ssh timed out (2026-11-20 afternoon).** Next: sync + `sbatch controller/das6_dryrun.sbatch`, then the board.
- **Controller DAS-6 dry run 28462 (node207 A10, trained M3, smoke bundle 42 req, keep/keep+gpu/keep, no rails) PASSED (~3.5 min).**
  7 unit tests ok; all three runs (score cluster 6 policies, score device 2 policies, no-deadlines 2 policies) + analyze_board
  completed; outputs `/var/scratch/skalra/flexslm/controller_dryrun/28462/`. Mechanics only (A10 is ~4x faster than the Jetson
  profile: measured/predicted 0.23, with GPU load 0.29 = +24%, so the load phase works). Open-loop `deadline` dropped widths it did
  not need (35/42 at L8, acc 23.8); `deadline+fb` learned rho ~0.25 and stayed at L8 (26.2), the intended feedback effect.
  Energy = profile model (no rails). Device scoring: accuracy at L8 = cluster (26.2) but answer agreement 36/42 (85.7%): A10 vs A40,
  graph/static-cache decoding vs HF generate, near-chance model with small margins; check agreement on the board.
  Note: the queue's warm-start backlog assumes the budget binds; where the device is far below budget (this A10) it under-spends
  for ~40 requests (queue:0.5 stayed at L7). On the board the budget is from the same 15 W profile, so it binds.
- **Full 7-benchmark averages, Stage-1 lambda pilots (32-token evals, mean of the 7 sets / mean w/o MedCalc / pooled over 12,797 q; computed 2026-11-20 from results/gen.*final9*-level{0,8}.t32.json):**
  L8: CE-only Stage 1 (pilot final9) 29.8 / 30.0 / 32.2; KD 0.1 29.2 / 29.6 / 31.7; KD 0.1 Hydra 28.9 / 29.5 / 31.5; KD 0.3 29.7 / 30.2 / 32.2; KD 0.5 30.0 / 30.4 / 32.5.
  L0: 21.1 / 21.3 / 21.3 (hydra) / 21.2 / 21.4 (avg), all at the guess rate. Differences <= 1.1 avg (pooled SE ~0.4): no resolvable winner; KD 0.5 nominally
  best, not better than CE-only Stage 1 beyond noise. Hydra sampling is nominally worst.
- **Correction (2026-11-20):** the Stage-2 teacher / 1.7B baseline is the UNCLIPPED E_A export (avg 36.8); 30.6 was the clipped T1 (reference only).
- **8B teacher cost estimate (extrapolation, nothing run):** 1.7B full FT = 3.0 h for 1 epoch on 2 x A40 (28131, 41.7 GB peak). 8B full FT needs ZeRO-3 / 8 GPUs
  (hipster performance); LoRA fits 2 x A40. See chat for numbers.
- **HuatuoGPT-o1-7B LoRA teacher: prepared, NOT run (2026-11-20, user request).** Files (uncommitted, synced to hipster ~/FlexSLM-medbench):
  `training/finetune_teacher_lora.py`, `scripts/hipster/teacher_lora.sbatch`, `test/test_teacher_lora.py` (4 tests, hipster pytest 398907 PASSED, tiny Qwen2 on CPU).
  * Design: standalone peft + torchrun DDP (no LoRA existed in the repo; the Stage-2 teacher only needs an HF dir). Data = data/medmix_sft through
    `training.sft.QADataset/collate_qa` with the Qwen3 tokenizer (KD needs identical ids), CE on answer tokens only, LM head applied to the answer positions
    only (no [B,S,152k] logits). LoRA r 64 / alpha 128 / dropout 0.05 on q,k,v,o,gate,up,down; lr 1e-4, 3% warmup + cosine, global batch 32, 1 epoch
    (~1,043 steps), max_seq 3072, bf16 + activation checkpointing, no grad clipping (repo's Stage-2 finding; grad_norm logged). Not chosen by experiment.
  * Export: merge adapter, trim embedding + lm_head from 152,064 to 151,936 rows (trainer.py softmaxes teacher logits as [.., student_vocab]), save with the
    Qwen3 tokenizer to `hf_export/medmix_huatuogpt_o1_7b_lora`; safetensors header check of the shapes. Final adapter also copied to ~/flexslm_saves/teacher_lora/<job>.
  * Known risk, logged at startup: Huatuo has no `<think>`/`</think>` embeddings (ids 151667/8 are untrained padding rows) but the Stage-2 prompt ends with
    `<think>\n\n</think>\n\n`; LoRA does not train embeddings. Check the printed row norms and the teacher's val answer-token acc / 7-benchmark eval.
  * Smoke (not submitted): `sbatch --gres=gpu:rtx_6000_ada:2 --cpus-per-task=64 --time=01:30:00 --export=ALL,SMOKE_STEPS=20 scripts/hipster/teacher_lora.sbatch`
    (longest sequences first, memory + throughput + projected full-run time, merge/save/verify into scratch then deleted; also downloads ~15 GB into the home HF cache,
    home had 72 GB free). Full run: `sbatch scripts/hipster/teacher_lora.sbatch` (4 x RTX 6000 Ada, --time 10 h placeholder; set from the smoke projection).
  * Still needed before it can be a Stage-2 teacher: eval of the merged teacher (32-token, 7 sets), and a Stage-2 config with `teacher_hf_model` pointing to it
    (7B bf16 teacher + student on one A40 will not fit at the current micro-batch; hipster 48 GB or a smaller micro-batch).
- **Student final report read (2026-11-20, `paper/student_results/ReportErasmus_1.docx`; job logs `Nextcloud/FlexSLM/repo/jobs`, checkpoints
  `Nextcloud/FlexSLM/saved_models`, student code github berraylvc/flexSlm_uva, evals ran on Snellius, not DAS-6).** Spot-checked logs match the report
  (evalgen60k L9/L5, evalgenKVT L9/L5). New 1.7B results, same 32-token protocol, 12,797 q, KD 0.5 T2 all tokens from the SAME-SIZE fine-tuned 1.7B
  ("SFT teacher", student's eval 37.9; our E_A 36.8): 10 levels 244M..1,721M, SHARD sampling (largest + smallest + 1 random middle), Minitron
  reordering + kv_group (= our GQA-preserving slicing). avg7 L0 / L5 (~900M) / L9: run 4 (SFT teacher, reorder + kv_group, 10k cap) 19.8 / 23.1 / 39.3
  (`..._10L_kv_teacher.pt`); run 6 (+60k MedMCQA) 19.9 / 23.9 / 40.2 (gain only on MedMCQA, in-distribution; breaks FaithMed's 10k cap);
  progressive teacher (run 3) worse at L9 (37.6); head reordering (run 5) n.s. 3-level old-mapping run: full 40.0 vs teacher 37.9 (+1.8 pooled
  p<1e-4) = born-again effect. Only L0/L5/L9 were evaluated (L1-L4, L6-L8 not).
- **Also landed:** 28458 rescore of Qwen3-1.7B thinking curve (300/set): budget 512 0.2, 1024 8.8, 2048 26.4, **4096 38.5 (mean 2,308 generated
  tokens)** = FaithMed's reported 38.3 for the untrained 1.7B (reproduced). 28439-43 KD 0.5 lambda arm L0/L4/L8 evals done (see entry above). 28459
  (0.6B curve) running on node206. Hipster: nothing started; smokes 398292/398293 still PENDING (Priority) since 2026-11-19, all 7 trainings on
  Dependency. DAS-6 node205 and node207 idle at ~19:00.
- **Assessment given to the user (no runs submitted):** no 7-8B teacher needed (Stage-1 KD sweep gave no resolvable gain = the user's own gate for it
  is not met; narrow widths are capacity-limited, which a bigger teacher does not fix; same-size fine-tuned teacher already lifts the 1.7B full width
  above conventional FT). Proposed minimal remaining set: (1) 1.7B MedNest in our code with KD from the same-size E_A teacher (needs the user to
  relax the "teacher bigger than the full width" rule for 1.7B), or adopt student run 4 as the 1.7B row; (2) eval all 10 levels of student run 4 (knee
  of the 1.7B curve); (3) independent full width WITH KD (indep_l2) + no_kd: move from hipster to idle DAS-6; (4) 0.6B curve 28459 + MedNest on the
  same 300/set subset. Key weakness: 1.7B ~900M width (23.1-23.9) < 0.6B MedNest full width (30.5 at 596M).

### 2026-11-20 controller committed; Jetson not reachable
- Commit 24fc7fe adds `controller/` (16 files; `data/`, `out/` gitignored; README updated with dry run 28462).
- Board run not started: `ssh jetson` (192.168.0.101) gives no ping and no port 22 while the laptop is on
  192.168.0.106 (board off/asleep or a new DHCP address). When it is up: update ~/FlexSLM on the board, then
  `QUICK=1 bash controller/run_jetson.sh && bash controller/run_jetson.sh` in tmux.

### 2026-11-20 evening: controller on the Jetson (board back up, 192.168.0.101)
- Board disk was 98% (1.3 GB free). User asked to delete saved models no longer needed: deleted `~/mednest/models/mednest-random-1p7b`
  (6.2 GB; rebuilt by `BACKBONE=1p7b jetson/run_shape_only.sh`) and the weight files of Qwen3-1.7B, Apollo-0.5B, Apollo2-0.5B (Hub, re-downloadable;
  their measurements are done). Config + tokenizer kept (config-only runs), `WEIGHTS_DELETED.txt` in each. Kept Qwen3-0.6B (source for
  `--random-from`), Apollo-2B/Apollo2-2B (config only). Now 15 GB free.
- Board repo `~/FlexSLM` stays at a6a3ad2 with uncommitted jetson edits (same content as the laptop's up to CRLF, checked by md5);
  `controller/` copied over with tar (no git pull).
- New `jetson_runner.py --random-from <base dir>`: random seeded weights built in GPU memory, no export on disk; `run_jetson.sh` falls back to it.
- Smoke (`~/mednest-results/controller_smoke`, copy in `jetson_logs_mednest/mednest-results/controller_smoke/`): 42 requests, 15 W,
  rails measured (idle 3.86 W). measured/predicted latency 1.11 at 15 W, 1.49 with the GPU load; deadline+fb rho 1.25 -> 1.48 -> 1.16.
  fixed:L8 26.2 acc 3.48 J 23.8% miss; deadline+fb 26.2 3.05 J 9.5%; queue-task:0.5 23.8 2.08 J 4.8%.
  Caveat: per-request net energy in the +gpu phase includes the load generator's power (window integral of VDD_IN minus idle).
- Full run started 18:58 board time in tmux `ctrl`, out `~/mednest-results/controller_full`.
- **Energy attribution fix (load phases), 2026-11-20:** `jetson_runner.py` now measures a load baseline (`--load-baseline-s 5`: board power with
  the phase's load running and no request) at the start and end of each loaded phase (`kind: load_baseline` records) and subtracts it, not idle,
  from that phase's requests; the policies observe this attributed energy. Records also keep `energy_gross_j`, `energy_net_idle_j`
  (old definition), `baseline_w`, `baseline_src`. Bounds: the load slows while sharing the GPU, so attributed = lower bound, idle-net = upper
  bound of the inference's marginal energy. `analyze_board.py` prints J/answer per phase in both definitions and the baselines.
  The full run started 18:58 uses the OLD code: its +gpu-phase energy and the queue policies' choices there include the load's power; latency,
  misses and rho are unaffected. Board went off the network again (~19:1x) before the fix could be copied. To do when it is back: copy
  `controller/` (tar), QUICK=1 smoke to check the baselines, then rerun the full run.
- **User decisions (2026-11-20, evening):** (1) 1.7B teacher = the SAME-SIZE conventionally fine-tuned Qwen3-1.7B (E_A); the
  "teacher bigger than the full width" rule is relaxed for the 1.7B backbone (born-again distillation). (2) Items 1-2 on DAS-6,
  item 3 (Stage-2-only ablations, indep, 1.7B CE-only) stays on hipster only, not urgent. (3) Paper updated now with all results.
- **Item 1, 1.7B MedNest with KD (`sft.paper.qwen3_1p7b_nested.no_stage1.kd`, new in training/sft.py; = A2 on Qwen3-1.7B, 3 widths
  0.5/0.75/1.0, Minitron order, KD 0.5 T2 from E_A).** Smoke 28463 OOM at micro-batch 4 (KL of the longest batch, 44.1 / 44.7 GB) ->
  micro-batch 2; chain 28464-28468 cancelled. **Resubmitted: smoke 28469 -> train 28470 (node205, 40 h limit) -> export 28471
  (fatq node202; node201 is another user's) -> evals L0-L2 28472-28474** (jobs/queue_2026-11-20.ids). 27 local tests pass.
- **Item 2, all 10 widths of the intern's run 4 (`sft.medmix_qwen3_1p7b_10L_kv_teacher`):** her code cloned to
  `/var/scratch/skalra/flexslm_student/code` @1e5fdfa (her exporter repeats kv heads per query head for HF, e.g. L0 5q/3kv -> 253M
  exported); warm-start dir `code/hf_export/qwen3_1p7b_reordered` = Qwen3-1.7B config + tokenizer only (export needs nothing else);
  checkpoint uploading to `flexslm_student/home/FlexSLM/saved_models/` (her code reads $HOME/FlexSLM/saved_models; HOME is
  overridden for her python only). New `scripts/das6/export_student.sbatch` (fatq node202) -> `hf_export/student__<config>/level0-9`,
  then OUR evalgen.sbatch. Not submitted yet (upload in progress).
- **Head reordering (user question):** our ordering always includes KV groups (heads move with their query heads) together with MLP
  and hidden channels; its downstream effect was nil (A1 no-order 28258: L8 28.5 vs 27.2 ordered, p 0.14; only Stage-1 ppl improved).
  The intern's run 4 ordered hidden + MLP only; run 5 added head ordering (in GQA groups): +0.5 / +0.2 / -0.5 (n.s. / n.s. / p 0.038).
  What matters is the GQA pairing (kv_group), not head order. Stage-2-only no_order ablation still pending on hipster.
- **New numbers used in the paper:** MedNest 596M (A2 L2) on the 300-per-set subset (2,098 q) = **31.5** (full set 30.5); Qwen3-1.7B
  reasoning on the subset: mean prompt 296 tokens, mean generated 2,308, avg 38.46. GFLOPs/question = 2 x params x (296 + gen):
  1.7B budgets 1,024 / 2,048 / 4,096 = 4,444 / 6,894 / 8,958; MedNest 596M <= 391, 1.72B <= 1,128. Jetson 0.6B table from
  `jetson/aggregate.py` on shape_only_full (medians): L0/L4/L8 TTFT 188/195/197 ms, 21.0/29.3/38.7 ms/token, answer 0.84/1.10/1.40 s,
  4.0/6.7/10.0 J; dense same-shape within 1%; Apollo-0.5B / Apollo2-0.5B 24.7 / 28.1 ms/token (16-token answers, 3.5 / 3.8 J);
  Qwen3-0.6B reasoning 68.1 s, 456 J (n=35); switch view 0.039 ms median, TTFT after switch 135 vs 134 ms; separate-model load
  0.47-0.82 s warm (first loads 5.1 / 9.3 s), slice-load 0.75-1.08 s, materialize 0.53 s; memory 1.19 / 2.30 / 6.77 GB.
- **Paper updated (backup `paper/main_backup_2026-11-20.tex`, `references_backup_2026-11-20.bib`; PDF rebuilt, 15 pages, no errors):**
  abstract (1.7B sentence, Jetson numbers), teacher rule (same-size teacher for 1.7B, + furlanello2018born in bib), 1.7B params
  0.51/1.03/1.72B, Table qwen17 + ten-width rows (run 4) and the intern's conventional FT 37.9 (marked "independent
  implementation"), FaithMed table/text (1.7B curve, MedNest subset 31.5, FLOPs, Jetson 3-minute answer), ablations (GQA pairing,
  head ordering, progressive teacher), new Sect. 5.4 + Table jetson, limitations (useful widths near full size), conclusion.
  Fixed a pre-existing `\` -> `\` in the ablation table. Remaining red: 1.7B 3-width rows, Qwen3-0.6B reasoning row, independent
  and ablation rows, Separate-shape accuracies, subset number of MedNest 1.7B.
- **Rerun with the fix, 2026-11-20 19:20-19:30 (board back):** the old full run (18:58, old energy code) was stopped at 2,520 / 3,780 requests (4 of 6
  policies; queue policies not yet run) -> `~/mednest-results/controller_full_oldenergy`. First fix (subtract the load's standalone power, measured
  at phase start) gave far-apart bounds on the smoke run (L8 in +gpu: 1.81 vs 4.27 J; clean 3.10) and the load's power drifted 8.1 -> 10.4 W over
  a 14-request phase (GPU clocks rising) -> smoke kept as `controller_smoke_loadbase`, the just-started full run aborted (`controller_full_aborted_loadbase`).
- **Final attribution (`load_ops`):** the GPU load counts finished matmuls in a /dev/shm counter; baseline windows (load alone, 5 s at phase start
  and end) give J per op = (P_load - P_idle) / ops rate; each request is charged gross - idle*dt - J_per_op * (load ops during the request).
  Bounds kept per request: `energy_net_loadpower_j` (lower), `energy_net_idle_j` (upper). Smoke (`controller_smoke`): J/op 30.3-32.9 mJ at
  start/end while the power drifted 8.1 -> 10.4 W (ops 137 -> 198/s), i.e. the per-op cost is stable; attributed J/answer in +gpu vs clean 15W:
  fixed:L8 2.86 vs 3.09 [bounds 1.81, 4.30], deadline+fb 2.54 vs 2.71, queue-task:0.5 1.80 vs 1.78. measured/predicted latency 1.09 / 1.44.
- Full run with this code started right after the smoke in tmux `ctrl` (~19:31), out `~/mednest-results/controller_full`. Copies of all smoke
  dirs and the old full run in `jetson_logs_mednest/mednest-results/`.
- **User (2026-11-20, late): item 1 dropped. The 1.7B result is the intern's run 4 checkpoint (`..._10L_kv_teacher.pt`); no rerun in
  our code.** Smoke 28469 (micro-batch 2) had also run out of memory (37 GB static after step 1: fp32 weights + grads + AdamW + bf16
  teacher; she trained on 80 GB H100s); the ZeRO-1 resubmission (smoke 28475 -> 28476 -> 28477 -> 28478-28480) was cancelled
  before it ran. Config `sft.paper.qwen3_1p7b_nested.no_stage1.kd` (micro-batch 2 + zero_optimizer) stays in training/sft.py, unused.
- **Item 2 upload:** scp from the laptop failed (Nextcloud online-only placeholder: "Permission denied" on read; DAS-6 connection
  reset). The user downloads the checkpoint on DAS-6 directly from SURFdrive WebDAV (curl, WebDAV app password, in tmux) to
  `/var/scratch/skalra/flexslm_student/home/FlexSLM/saved_models/`; expected size 7,763,415,647 bytes. Then submit
  `sbatch scripts/das6/export_student.sbatch sft.medmix_qwen3_1p7b_10L_kv_teacher` + evalgen of
  `hf_export/student__sft.medmix_qwen3_1p7b_10L_kv_teacher/level0..9` (afterok).
- **Paper:** the empty 3-width 1.7B rows are removed; Table qwen17 now presents run 4 as MedNest (1.72B / 0.90B / 0.24B) with the
  recipe differences in the text (ten widths, sandwich sampling, hidden + FFN ordering, same-size teacher 37.9 = "second
  conventional FT run" row) + a red row for the other seven widths; method gives the ten widths 0.24B-1.72B. PDF rebuilt (15 pages).
- **Item 2 submitted (2026-11-20 ~19:47).** Public SURFdrive link of the user's `saved_models` folder (download permission; two earlier links were
  file drops) fetched over public WebDAV on a fatq job: download **28481** (node202, curl resumable, size check 7,763,415,647) -> export
  **28482** (`scripts/das6/export_student.sbatch sft.medmix_qwen3_1p7b_10L_kv_teacher`, intern's code @1e5fdfa) -> evals L0..L9
  **28483-28492** (our evalgen.sbatch, 32 tokens; node207 only, ~15 min each). Exports in
  `/var/scratch/skalra/flexslm/hf_export/student__sft.medmix_qwen3_1p7b_10L_kv_teacher/level0-9`.
- **Full run 19:31-20:05 (load_ops code) INVALID for 4 of 6 policies** -> kept as `controller_full_noload` (board and local copy). The GPU load
  process died at start-up with CUDA out of memory (NvMap error 12, ~1 GB for its CUDA context next to the resident model + 9 decoders sized for
  the longest of 630 prompts; the 42-request smoke has shorter buffers and was fine). `Load` only checked liveness after 3 s, so fixed:L0,
  deadline, deadline+fb ran their "+gpu" phase without load (0 ops/s, 4.3-4.5 W baselines); queue:0.5 got the load only late in the phase.
  fixed:L8 and queue-task:0.5 had it. Fix: `Load` empties the caching allocator, waits (<= 60 s) until the load has finished >= 10 matmuls,
  retries 3x, otherwise aborts; a load that exits mid-phase aborts the run; `analyze_board.py` warns about loaded-phase requests without load work.
  Full rerun started 20:07 in tmux `ctrl` -> `~/mednest-results/controller_full`.
- **A2 (paper main model, 3 widths) in the controller (user, 2026-11-20 ~20:30):** samples copied from DAS-6
  (`gen.hf_export-flexllama__sft.paper.qwen3_0p6b_nested.no_stage1-level{0,1,2}`; DAS-6 ssh needs `SSH_AUTH_SOCK=~/.ssh/agent.sock`) and
  stored a second time as `..._grid048-level{0,4,8}` (A2 levels 0/1/2 = grid shapes 0/4/8, so the 9-level Jetson profile applies).
  `data_io.A2G`, `simulate.py --model`. Offline (eos): A2 L0/L4/L8 = 20.8/22.7/30.5 at 1.23/1.81/2.49 J; hull {L0, L8}, L4 2.57 points
  below the L0/L8 mix; oracle 45.8 (55.7 if independent). Budget sim: task-aware queue - best fixed within budget = +3.3 [2.7, 4.0] (1.43 J),
  +4.3, +4.9, +6.9, +7.0 [5.3, 8.9] (2.26 J); task-aware - agnostic +1.9 [1.3, 2.5] at 1.43 J, +1.6 [0.8, 2.3] at 2.05 J, n.s. elsewhere;
  the queue time-shares L0/L8 and almost never uses L4. Thermal x1.3: urgent-slow misses 98.5% open loop -> 48.6% with feedback
  (floor: L0 45.4%), overall 14.8% -> 9.1%, accuracy -0.47 [-0.97, +0.02]. Outputs `controller/out/*A2*`.
  Board: bundle `paper.qwen3_0p6b_nested.no_stage1.A2.n630.s0.json` (levels 0/4/8); A2 full run queued behind the M3 rerun
  (tmux `ctrlA2`, `~/run_A2_after.sh` waits for runner PID 7044) -> `~/mednest-results/controller_full_A2`.
- **Board results 2026-11-20 (valid; all loads ran, J/op 30.6-32.8 mJ, no warnings)** — M3 rerun `controller_full` (20:07-20:40) and A2
  `controller_full_A2` (20:41-21:13), 630 requests x 6 policies, copies in `jetson_logs_mednest/mednest-results/`. A2 / M3 acc, J, miss:
  fixed full 30.8/3.04/28.1 | 25.7/3.05/26.3; fixed narrowest 20.3/1.25/7.1 | 19.8/1.27/4.4; deadline 30.0/2.68/20.8 | 24.9/2.75/21.9;
  deadline+fb 28.6/2.48/11.0 | 24.0/2.49/14.9; queue:0.5 24.1/1.78/6.7 | 23.3/1.78/6.8; queue-task:0.5 25.4/1.78/5.4 | 21.9/1.78/5.2.
  Feedback - open loop: misses -9.8 [-12.5, -7.6] (A2), -7.0 [-9.5, -4.4] (M3); urgent misses under load 95 -> 30 % (A2), 98 -> 28 % (M3);
  acc -1.4 [-3.0, +0.04], -0.95 [-2.5, +0.6]. Queues vs fixed narrowest (only width within 1.78 J): A2 +3.8 [0.9, 6.7] / task +5.1 [1.9, 8.2],
  M3 +3.5 [0.6, 6.4] / task +2.1 n.s.; vs the L0/L8 mix at equal energy (`controller/board_compare.py`) +0.4 to +2.0, n.s.
  measured/predicted latency 1.06 (15 W) / 1.41 (load). README "Board results" section written.
- **Paper (user asked, 2026-11-20 evening; overrides the no-paper-edits rule for this change):** controller added: method Sect. 3.4 rewritten (three
  rules, drift-plus-penalty with \cite{neely2010stochastic}, compiled engines), new Sect. 5.5 "Choosing the width at run time" + Table 7
  (A2 budget simulation; board A2 + nine-width M3), contribution 5, one abstract sentence, limitation replaced. 17 pages (user: ignore the limit,
  will edit). Backups `paper/main_backup_2026-11-20b.tex`, `references_backup_2026-11-20b.bib`. paper/ is gitignored, so not committed.

### 2026-11-23: 1.7B ten-width evals and 0.6B reasoning curve analysed, paper filled
- **DAS-6 queue empty, all done; nodes 205-208 and fatq idle.** Student export 28482 + evals 28483-28492 (all 12,797 q, 0% unboxed) and 28459 (0.6B thinking) finished.
- **Ten-width 1.7B (intern run 4, our eval), avg / w/o MedCalc:** 0.24B 19.4/20.6, 0.40B 19.6/21.0, 0.50B 20.0/21.1, 0.62B 20.9/21.7, 0.75B 21.5/21.9,
  0.90B 23.3/23.6, 1.05B 24.5/24.7, 1.33B 30.7/31.2, 1.53B 34.6/35.5, 1.72B 39.0/40.4. Report had 19.8/23.1/39.3 at L0/L5/L9 (within 0.4). Params = shared
  weights with K = ceil(H/2) (the HF exports repeat kv heads, e.g. L0 253M, L5 951M, L8 1,623M on disk). Teacher (28136, our eval) 37.7, report 37.9.
  McNemar pooled: full vs teacher p=7e-7 (966 vs 759), vs E_A 36.8 p=3e-9; each step above 0.75B p<1e-4; 1.33B vs A2 596M (30.5) p=0.31;
  1.05B < A2 596M; 0.24B < A2 188M p=9e-4 (w/o Calc p=0.04); 1.53B < E_A p=5e-12.
- **Subset of 300/set (2,098 q):** Qwen3-0.6B thinking 15.3/23.5/26.3 at 1,024/2,048/4,096 (845/1,179/1,358 mean tokens); MedNest 596M 31.5 is above (p=2e-5; w/o Calc
  32.3 vs 29.8, p=0.06). MedNest 1.72B 39.6 vs Qwen3-1.7B thinking 38.5 (2,308 tokens): p=0.35 (w/o Calc 41.2 vs 42.2, p=0.43), 6.5 tokens per answer.
- **Attribution (old 100M-Stage-1 recipe, nine widths, finished earlier):** full width alone with KD (indep_full) 28.3 vs nested M3 full 27.2 (p=0.14, so no resolvable
  cost or gain from nesting); without KD nested full 22.6 (p=0.34 vs conventional 23.1; p=1e-36 vs M3): the gain over conventional FT comes from KD.
  indep_l1 / indep_l0 (Stage 1 + KD) 22.3 / 21.3 vs M3 nested same shape 22.5 / 21.4 (n.s.). Needs the A2-recipe control (indep_l2) to transfer.
- **Floor:** best constant letter per set, mean of the six multiple-choice sets = 22.4. A2 w/o MedCalc: 596M 31.0, 364M 22.4, 188M 21.4; conventional FT 596M 22.9.
  So "364M matches conventional FT (p=0.35)" compares two models at the letter-prior floor.
- **Hipster:** 398309/398310 (indep_l0) and 398312/398313 (indep_l1) trained and exported (hf_export/flexllama__sft.paper.qwen3_0p6b_nested.no_stage1.indep_l{0,1});
  their evals 398311/398314 PENDING (capacity, 6 of 8 nodes down, est. 11-24 16:54). Trainings 398294 no_kd (est. 11-23 20:35), 398299 no_order (11-24 06:36),
  398304 single / 398315 indep_l2 / 398318 1.7B CE-only (11-24 23:31). Nothing was submitted, moved or cancelled in this session.
- **Paper:** backup `paper/main_backup_2026-11-23.tex`. Filled Table qwen17 (all ten widths, ours; bold removed), Table faithmed (three 0.6B thinking rows, MedNest 1.72B subset), text, abstract,
  conclusion, limitation (39.3 -> 39.0, teacher 37.9 -> 37.7). PDF 18 pages; draft PDF refreshed. Remaining red: independent rows, ablation rows, Separate-shape accuracies, method/related-work TODOs.
  Not changed (user to decide): the "364M matches" claim and the attribution of the +7.4 to nesting + KD.

### 2026-11-23 (later): two runs moved to DAS-6, 1.7B controller status
- **User OK'd moving the full-width-alone (indep_l2) and no-KD runs to DAS-6 and cancelling 398318 (1.7B CE-only).** Submitted on DAS-6 (ids in jobs/queue_2026-11-23.ids; same code as A2,
  node205, 2 x A40, 16/GPU; hipster sft.py only lacks the unused 1.7B `.kd` config):
  indep_l2 train **28495** -> export 28496 -> eval L2 28497; no_kd train **28498** -> export 28499 -> evals L0-2 28500-28502 (evals pinned to node207, node208 excluded).
  After 28495 was running, cancelled on hipster: 398294-398298 (no_kd chain), 398315-398317 (indep_l2 chain), 398318-398322 (1.7B CE-only). Still on hipster: no_order 398299-398303, single
  398304-398308, indep_l0/l1 evals 398311/398314.
- **1.7B controller, preliminary simulation (laptop, scratch only, nothing added to controller/data):** run 4 per-question results + the 2026-11-19 1.7B shape-only profile
  (`shape_only_1p7b/widths_15W.jsonl`, 9 widths 0.51-1.72B). Eight of the ten student widths (L2-L9) have a profiled shape within 3% of their parameters; L0/L1 (0.24B/0.40B) have none. Eos answers, accuracy all 12,797 q / budgets on
  the 6,423-q half: J/answer 1.97 (0.50B) ... 3.99 (1.33B), 4.55 (1.53B), 4.84 (1.72B); energy hull = {0.50B, 1.72B} only: 1.33B / 1.53B lie 2.7 / 2.5 points below the line,
  1.05B 4.2. Task-aware queue vs best fixed within budget +4.9 ... +12.6 (the last at 3.85 J, where 1.33B misses the budget by 0.05 J). Slowdown x1.3: deadline misses 15.1% -> 5.9% with feedback, acc -1.25 [-1.95, -0.59].
  Cross-backbone: 0.6B full width 30.5% at 2.49 J is ~7 points above the 1.7B energy hull at that cost and cheaper than the 1.33B width (30.7% at 3.99 J).
- **Pending for 1.7B on the board (nothing run):** (a) exact ten-shape profile: ten-width json; (b) odd query-head widths (5/7/9/11/15 heads, L0 L1 L3 L5 L8) are not prefix views in `elastic.py`: HF Qwen3
  needs kv heads that divide query heads, the exporter repeats kv heads (15/15), so the view build raises; needs a custom attention forward or the dense proxy shapes; (c) `jetson_runner.py --random-from` hardcodes the
  0.6B grid widths, `run_jetson.sh` defaults to Qwen3-0.6B; (d) bundle + profile for run 4; (e) board run (memory next to a 3.44 GB model and the GPU load; ~2.2x the 0.6B run's latency, 630 x 6 policies). Board not reachable now (192.168.0.101 timed out).

### 2026-11-23 (evening): two-model simulation (controller/two_model.py, simulation only)
- **Setup:** arms = A2 188/364/596M, run 4 at 8 widths (0.50-1.72B, closest profiled shape within 3%), conventional 596M (23.1) and the 1.72B teacher of run 4 (37.7), same shapes.
  Inputs copied to controller/data (gitignored): `gen.student1p7b_grid-level*`, `gen.conv1p7b_teacher-level9`, `jetson/widths_15W_1p7b.jsonl` (2026-11-19). Output `controller/out/two_model.txt`. 6,423-q evaluation half.
- **Energy, J/answer (15 W profile):** 188M 1.23, 596M 2.49, 0.50B 1.97, 1.33B 3.99, 1.72B 4.84. Hulls: PAIR = {596M, 1.72B}; nested 1.7B = {0.50B, 1.72B}; BOTH = {188M, 596M, 1.72B}.
- **Task-aware queue, macro acc at 2.8 / 3.2 / 3.6 / 4.0 J:** N17 nested 1.7B 27.2 / 30.0 / 36.4 / 37.9; PAIR (MedNest full widths, 4.63 GB) 32.1 / 33.4 / 36.8 / 38.6;
  PAIRc (conventional FT) 28.2 / 30.2 / 34.5 / 36.9; BOTH nested models 32.4 / 35.9 / 37.8 / 38.6. PAIR - N17 +4.8 [3.0, 6.5], +3.3 [1.6, 5.0], +0.3 n.s., +0.7 [0.2, 1.2]; PAIR - PAIRc +3.8, +3.2, +2.2, +1.6 (all CIs > 0);
  BOTH - PAIR +0.3 n.s., +2.5 [1.6, 3.6], +1.1 [0.1, 2.0], +0.1 n.s. Below 2.4 J PAIR and N17 cannot run; nested 0.6B gives 25.1 / 29.4 / 30.0 at 1.6 / 2.0 / 2.4 J. Differences of about a point at single budgets are policy noise (BOTH < N06 at 2.0 J).
- **Deadlines (30% urgent, x1.3 slowdown, feedback; urgent 400 ms), acc / miss%:** N06 29.1 / 4.1; N17 33.4 / 8.3; PAIR 36.5 / 10.9; BOTH 35.5 / 4.9. PAIR - N17 acc +3.1 [1.9, 4.2] with misses +2.6; BOTH - PAIR misses -6.0 with acc -1.0.
- **Checks:** the queue's gains are not an artifact of the unweighted mean over a stream dominated by MedMCQA: on pooled accuracy and on an equal-per-set stream the A2 (paper table) gains are +3.4 to +7.7 over
  the best fixed width and +1.0 to +3.2 over the hull mix at the same energy (agnostic > hull: it spends energy on short prompts).
- **Feasibility not shown:** 1.0 GB (indep-l0 shape) OOMed next to the resident 3.44 GB model (shape_only_1p7b, one process with decoders already prepared, nothing released; confounded). *-load is not a tuned policy:
  assumed 1.7B load 1.9 s (scaled from the 0.6B 0.65 s, not measured), never returns to 1.72B; its numbers are not used.
- DAS-6 at ~20 min: 28495 at 169/1044 steps (~1.7 h left), no errors.

### 2026-11-23: co-residency test on the Jetson (10.113.206.56, freshly booted, 15 W, PyTorch, bf16, random weights in GPU memory; latency/memory only)
- Scripts `controller/coresident_test.py`, `run_coresident.sh` (benchmark decoder unchanged), `run_coresident_lean.sh` (`--lean`: last-position logits only, file page cache evicted after build). One fresh process per case,
  28 questions incl. the 4 longest (2,009 tokens), 32 forced tokens. Raw files in `jetson_logs_mednest/mednest-results/{coresident,coresident_lean}`. One run per case, so fit/no-fit near the limit is not a probability.
- Unmodified harness: 596M alone and 1.72B alone fit (39.0 / 77.6 ms/token); 596M + 1.72B, 1.72B + nested 0.6B, nested 1.7B + nested 0.6B all FAIL (allocator assert / CUDA OOM) while preparing the second model's decoder; with GPU load also fail. 1.72B alone + load fits (avail 0.6 GB at the end).
- Lean: 1.72B alone ends with 1.9 GB available (peak alloc 4.07 GB vs 4.39), 596M alone fine. PAIR: models resident (4.6 GB weights, 5.3-5.4 GB with KV/graphs), ran 96 generations (the 24 ordinary questions) then failed on the first of the four longest prompts (free 138 MB).
  PAIR + GPU load: all 112 generations finished, but with 0.5 GB swap used at the end, 51,662 pages (~200 MB) swapped out during the run and 0.2 GB available. pair3 (1.72B + nested 0.6B with 3 widths) and both (nested 1.7B 3 widths + nested 0.6B 3 widths) failed when preparing the second model's graphs (CUDA graphs for 4 cache buckets x 3 widths), with and without load.
  Speed when co-resident: 77.5 / 39.1 ms/token vs 77.6 / 39.0 alone (no cost); TTFT after a model switch 234 / 195 ms vs 238 / 195 repeat; under load 115.2 (1.72B, same as alone under load 115.3), TTFT after a switch 389 vs 350, and one answer took 10.1 s.
- **Reading:** without quantisation the two-model pair (4.63 GB bf16 weights + KV + runtime) does not fit reliably on this 8 GB board (first-prompt failures or swapping); nested 1.7B alone (3.44 GB) has 1.9 GB of margin. Not tested: loading two trained checkpoints from files
  (worse: CPU copy + page cache), leaner KV caches / fewer CUDA graphs, int8/int4 (would change the argument), 1.7B load time. Every process after the first starts with ~1 GB less MemAvailable than after boot (presumably driver-held, not confirmed; no reboot done).
- DAS-6 28495 running (1 h 07 m elapsed at this check), 28498 waiting for node205.

### 2026-11-23 (late afternoon): int8 test, PARTIAL (board run and cluster evals still going)
- **Setup.** Weight-only int8, symmetric, one scale per output row (max-abs/127), embedding / output head / norms in bf16; a narrower nested width = prefix slice of the shared int8 weights + row scales. Same scheme on the cluster as fake int8 (`eval/fake_quant.py`, `test/test_fake_quant.py`,
  `scripts/das6/fakequant.sbatch`; exports `hf_export/int8__student_1p7b/level{9,7,6,4,2}`, `int8__a2_0p6b/level{2,1,0}`; narrower levels checked to be exact prefixes of the full one). Board: `controller/int8_test.py`, `run_int8.sh` (tmux `int8`, out `~/mednest-results/int8`), one fresh process per case.
- **No native int8 CUDA matmul in the board's torch 2.8** (`_weight_int8pack_mm` not implemented for CUDA); my layer dequantises on the fly (3.5-9x slower per matrix than bf16; self-test exact). So int8 speed and energy are NOT results; memory is. int8 596M: 167.8 ms/token vs 39.0 bf16.
- **Finished on the board:** (1) nested 1.7B **bf16**, five even-head widths with exact shapes (lean; ordinary prompts, 14 q): ms/token 31.6 / 41.9 / 53.0 / 64.1 / 77.5, TTFT 195 / 199 / 218 / 257 / 264 ms (no penalty after a width switch), net J per forced-32-token answer 7.3 / 10.5 / 14.3 / 18.0 / 22.1 (0.50 / 0.75 / 1.05 / 1.33 / 1.72B).
  This replaces the closest-shape profile for these five widths. (2) int8 596M alone: 0.76 GB allocated after build, 4.0 GB available, 596M. int8 1.72B alone: 2.04 GB allocated, 2.9 GB available after build, still running. Not finished: int8 nested, pair, pairn, both, and the three +GPU-load cases.
- **CORRECTION of the co-residency entry above:** "nested 1.7B leaves about 1.9 GB of margin" was measured for the 1.72B full width ALONE (one width, one set of decoders). With the five widths prepared (decoders + CUDA graphs for all of them) nested 1.7B bf16 ended at 0.67 GB available / 0.17 GB free.
  The memory comparison with the bf16 pair (which failed or swapped) still favours nested; the size of the margin does not.
- **Finished on the cluster:** int8 1.72B (28504) 39.3 vs bf16 39.0 (macro; w/o Calc 40.7 vs 40.4), p = 0.52 McNemar, 93.6% of answers identical. Not done: 28505-28511 (1.7B at 1.33 / 1.05 / 0.75 / 0.50B; 0.6B at all three widths).
- **DAS-6: root put all nodes into drain at 16:27 ("software update")**: node205 drains after 28495 (running, 1 h 34 m at 16:34; `resume=True`, checkpointed), all others (201-204, 206-208) drained. Queued and waiting for nodes to come back:
  28498 (no_kd), 28496-28497, 28499-28502, 28505-28511. Nothing of mine is affected except waiting; no resubmission needed unless jobs are cancelled by the maintenance. Hipster: only no_order / single chains and 2 independent evals pending, all with Priority.
- Next when the board run ends: copy `~/mednest-results/int8`, memory table (int8 nested vs pair vs both), then decide whether the int8 pair fits.

### 2026-11-23 16:43: int8 board run STOPPED at the user's request ("no new exps for now on board"); partial result
- The queue (`run_int8.sh`, tmux `int8`) was stopped after the case in progress: SIGSTOP on the driver did not hold (the process stayed running), so `controller/int8_test.py` was moved aside on the board; the 7 remaining cases exited at once
  ("can't open file", in run.log; no result files written). File restored afterwards; the board is idle, no tmux session. **Not run: int8 nested, pair, pairn, both, and nested / pair / both + GPU load.** So there is NO measured nested-vs-pair result at int8.
- **Finished int8 cases (one process each, 8 ordinary + 8 long-prompt generations, memory is the result):** 596M alone: 0.76 GB allocated after build, 1.09 GB with KV/graphs, 4.0 GB available at the end. 1.72B alone: 2.04 GB after build, 2.36 GB with KV/graphs, 2.7 GB available at the end, no swapping.
  int8 arithmetic (on-the-fly dequantise) 167.8 ms/token (596M, bf16 39.0) and 485.9 ms/token (1.72B, bf16 77.5): kernel cost, NOT int8 speed. `cuda_peak_alloc_gb` of the int8 cases (1.67 / 4.95) includes building in bf16 before quantising, so it is not serving memory.
- **Expectation, not measured:** by these sizes the int8 pair (1.72B + 596M) would allocate about 2.8 GB of weights + ~0.6 GB KV/graphs, next to ~5.4 GB available at process start, so it should fit; nested int8 1.7B about the same as 1.72B alone (+ CUDA graphs of the extra widths). Memory would then stop separating nested from the pair.
- **Cluster:** int8 1.72B 39.3 vs bf16 39.0 (p = 0.52). int8 evals 28505-28511 and the no_kd chain still pending: DAS-6 nodes drained by root for a software update since 16:27 (node201 back to idle at 16:42, 205 draining under 28495, the rest drained). Resume: nothing to resubmit unless cancelled.
- To resume later: copy nothing; on the board `cd ~/FlexSLM && bash controller/run_int8.sh` reruns everything (it overwrites `~/mednest-results/int8`; pass OUT=... to keep the current files).

### 2026-11-25: all 11-23 jobs finished (DAS-6 + hipster); A2 ablations, independent controls and int8 analysed
- **Queues empty on both clusters, nothing failed** (DAS-6 maintenance did not cancel anything; node204 still drain*). Nothing submitted, cancelled or deleted this session.
  Per-question files copied to the laptop scratchpad only; analysis = per-set tables, pooled exact McNemar, macro difference with a set-stratified paired bootstrap (2,000).
- **Macro avg / w/o MedCalc, all 12,797 q, 0 unboxed everywhere.** A2 (method) 188M 20.8/21.4, 364M 22.7/22.4, 596M 30.5/31.0; conventional 596M 23.1/22.9.
  no_order (398299-303, hipster) 20.1/20.9, 22.2/22.5, 30.7/31.2. single (398304-308) 20.2/21.7, 20.9/21.4, 27.5/28.7. no_kd (28498-502) 20.0/21.1, 20.8/20.8, 25.7/25.8.
  indep (398311 / 398314 / 28497) L0 20.7/21.5, L1 22.1/21.9, L2 31.1/31.6. int8 A2 20.5/21.0, 22.7/22.5, 30.5/31.0.
  int8 1.7B (28504-508) vs bf16: 0.50B 19.9 vs 20.0, 0.75B 21.6 vs 21.5, 1.05B 24.4 vs 24.5, 1.33B 30.7 vs 30.7, 1.72B 39.3 vs 39.0.
- **Differences to A2 at the same width (macro, 95% CI; pooled McNemar p):** full width (596M): no_order +0.2 [-0.6, +1.0] p=0.38; single -2.9 [-3.9, -1.9] p=6e-11; no_kd -4.7 [-5.9, -3.6] p=1e-32;
  indep_l2 +0.6 [-0.2, +1.4] p=0.41; int8 +0.1 [-0.3, +0.5] (92% identical answers). 364M: single -1.8 [-2.7, -0.8], no_kd -1.9 [-2.9, -0.9], no_order -0.6 n.s., indep -0.6 n.s., int8 0.0.
  188M: all within [-1.7, +0.6], nothing resolvable by macro CI. vs conventional: A2 596M +7.4 [6.1, 8.7], indep_l2 +8.0 [6.6, 9.3], no_kd +2.7 [1.4, 4.0] (p=2e-14). int8 1.7B: every width within ±0.3, all p > 0.5, 87-96% identical.
- **Reading.** (1) Nesting costs nothing at the full width (indep_l2 = A2, n.s.) and the smaller widths match their independently trained slices: one model serves three widths for the cost of one.
  (2) The +7.4 over conventional FT comes mostly from KD (no_kd -4.7); nested training without KD still adds +2.7 over conventional, which the old Stage-1 recipe did NOT show (22.6 vs 23.1, p=0.34). Configs differ only in
  training all three widths (Taylor start is function-preserving at full width), so in this recipe nesting acts like a regulariser that KD makes redundant (indep_l2 + KD = A2). Single-run, one seed: report it, don't build on it.
  (3) Importance ordering has no measurable effect in A2 (no_order = A2 at every width). (4) All widths every step beats one random width per step (-2.9 / -1.8), but single used about half the training time
  (1 h 05 vs 2 h 19 on hipster): not compute-matched. (5) int8 weight-only costs no accuracy at any width of either backbone (only memory was measured on the board; speed/energy not).
  (6) Caveat for every 188M/364M contrast: w/o MedCalc they all sit at 20.8-22.5, i.e. at the letter-prior floor (22.4); their differences are mostly MedCalc and say little.
- Hipster: eval results are safe in ~/FlexSLM-medbench/results (home), but the no_order / single / indep_l0 / indep_l1 exports are only on /scratch (`hf_export` -> /scratch/skalra/flexslm/hf_export, wiped every few weeks);
  user to decide whether they are needed again (e.g. for board runs) before scratch is wiped. Nothing deleted. Paper not edited (TODO at the end: ablation table, independent rows, int8 row).
- **15:37 hipster -> DAS-6: the four A2-ablation exports are being MOVED to DAS-6 (user: "instead can we move these exports to das-6").** Home-copy job 402267 cancelled before it started.
  New `scripts/hipster/move_exports_das6.sbatch` (capacity, 2 CPUs): per export tar stream to DAS-6 `/var/scratch/skalra/flexslm/hf_export/<same name>` (via `<name>.partial`), SHA-256 of the stream on both ends +
  per-file sizes compared, then `.complete` on DAS-6 and ONLY THEN `rm -rf` of the hipster copy; refuses to overwrite an existing DAS-6 export. Smoke **402275** (300 kB dummy `hf_export/_movetest_20261125`)
  -> real move **402276** (afterok): no_order / single / indep_l0 / indep_l1, 2.34 GB each. Both pending (capacity busy). Ids in `~/FlexSLM-medbench/queue_2026-11-25.hipster.ids`,
  logs `~/FlexSLM-medbench/jobs/flexslm_to_das6-<id>.out`. After it: remove the dummy `_movetest_20261125` on DAS-6; the eval results stay in hipster home (`~/FlexSLM-medbench/results`).
- **15:50 hipster eval results MOVED to DAS-6 (user).** All 20 files of `~/FlexSLM-medbench/results` (none were on DAS-6; 17 MB): gen.* of no_order L0-2, single L0-2, indep_l0, indep_l1
  (json + samples), `gen.Intelligent-Internet-II-Medical-8B.t32.*`, `pmcppl.Intelligent-Internet-II-Medical-8B.json`, `pmcppl.hipster.Qwen-Qwen3-1.7B.json`
  -> DAS-6 `/var/scratch/skalra/flexslm/results` (= `~/FlexSLM/results`). Streamed via the laptop (login nodes; small), no overwrite (`tar -k`, pre-check), SHA-256 of all 20 identical on both sides,
  then deleted on hipster: hipster `results/` is now empty. DAS-6 `results/` is the single place for all eval results.

### 2026-11-25: int8 board run resumed
- Jetson reachable at 10.113.206.56 (freshly rebooted, 15 W, idle; old 192.168.0.101 entry times out). `controller/int8_test.py` on the board = repo version (checked).
- New `controller/run_int8_resume.sh` runs only the 7 cases that did not run on 11-23: int8 nested / pair / pairn / both, then nested / pair / both + GPU load. tmux `int8r`, out `~/mednest-results/int8_resume` (first run's files in `~/mednest-results/int8` untouched).
  Started 16:45 board time. First case (int8 nested 1.7B, five widths) built: 2.04 GB allocated, 2.43 GB with KV + graphs of all five widths, 2.7 GB available after prepare (bf16 nested: 4.0 GB / 0.67 GB).

### 2026-11-25 17:07: int8 resume, first results (2 of 7 cases done; run still going in tmux `int8r`)
- Copied to `jetson_logs_mednest/mednest-results/int8_resume/`. One process per case, 4 ordinary + 4 longest prompts (2,009 tokens), lean decoder, no swapping in either.
- **int8 nested 1.7B, five widths (0.50-1.72B):** 2.04 GB allocated after build, 2.43 GB with KV + CUDA graphs of all five widths; at the end 2.3 GB available, peak used 5.3 GB. All 80 generations OK.
- **int8 pair (1.72B + 596M):** 2.80 GB after build, 3.43 GB with both models' decoders; at the end 1.9 GB available, peak used 5.8 GB. All 32 generations OK, including the long prompts.
- **Reading:** at int8 the pair FITS with margin (1.9 GB vs 2.3 GB for nested 1.7B). The bf16 memory argument for nesting (pair fails / swaps) does not carry over to int8; nested still uses 1.0 GB less than the pair and switches widths for free, but both fit.
- Still running / queued: int8 pairn (1.72B + nested 0.6B), both (nested 1.7B + nested 0.6B), and nested / pair / both + GPU load. Accuracy: int8 1.72B = bf16 (39.3 vs 39.0, p 0.52); 28505-28511 still pending on DAS-6 (nodes drained for maintenance on 11-23, not checked today).

### 2026-11-25 17:10: int8 accuracy, all widths (DAS-6 28504-28511 complete, no errors; queue empty, nodes back except node204)
- Weight-only int8 (shared row scales, narrower widths = slices of the full model's int8 weights) vs bf16, macro avg, McNemar over 12,797 q:
  1.7B: 1.72B 39.3 vs 39.0 (p 0.52) | 1.33B 30.7 vs 30.8 (p 0.88) | 1.05B 24.4 vs 24.5 (p 0.51) | 0.75B 21.6 vs 21.5 (p 0.82) | 0.50B 19.9 vs 20.0 (p 0.73).
  0.6B: 596M 30.5 vs 30.5 (p 0.79) | 364M 22.8 vs 22.7 (p 0.52) | 188M 20.5 vs 20.8 (p 0.041; w/o Calc 21.0 vs 21.4). Identical answers 86-96%.
- **Reading:** int8 costs no accuracy at any width except a 0.3-point loss at 188M (borderline, p 0.04; that width is near the guess rate anyway). Shared int8 weights keep the nested property.
  With the board result (int8 pair fits with 1.9 GB free), an int8 deployment can hold two separate full-width models; the paper's memory argument for nesting must be stated for bf16 only.
- (Correction: the A2 ablations and independent controls were already analysed by the parallel session, entry "2026-11-25: all 11-23 jobs finished" above.)

### 2026-11-25: paper red cells filled (user: "finalise the missing red results")
- Source: the parallel session's analysis (log entry "2026-11-25: all 11-23 jobs finished"), per-set numbers recomputed from the DAS-6 samples files (same values). Backup `paper/main_backup_2026-11-25.tex`; PDF rebuilt (18 pages, no errors), draft PDF refreshed.
- Table main: new row Independent (596M) 39.5/40.7/47.1/28.0/30.9/20.3/11.0 = 31.1 (now bold where best; MedNest 596M keeps bold only on HeadQA), Independent 364M 22.1, 188M 20.7; caption defines Independent.
- Accuracy text: independent narrow models no better (p 0.63; nested 364M ahead p 0.03, pooled); full width alone + KD 31.1 = MedNest 30.5 (p 0.41): nesting costs nothing at full width, the gain over conventional FT is mostly distillation.
- Table ablation: no KD 20.0/20.8/25.7 (p<1e-4), no ordering 20.1/22.2/30.7 (0.38), single width per step 20.2/20.9/27.5 (<1e-4, about half the training time), independent 20.7/22.1/31.1 (0.41); three new text paragraphs (distillation, all widths per step, ordering).
- Jetson table: Separate 188M / 364M accuracy = independent models 20.7 / 22.1.
- No 	bd left. Remaining blue TODOs: gap search before submission, Fig. 1. Not added (user to decide): int8 results (accuracy = bf16 at every width; int8 pair fits on the board), two-model simulation, 18 pages vs 12-page limit.

### 2026-11-25: paper, three open items done (user: all except the page limit)
- Backups `paper/main_backup_2026-11-25b.tex`, `references_backup_2026-11-25b.bib`. PDF rebuilt, 19 pages, no errors / undefined refs; draft PDF refreshed. No \tbd or \todo left in the text.
- **Gap check (web search 2026-11-25):** no work found that combines nested / elastic widths with a small medical language model or medical edge use; nested LLM work stays general-domain and >= billions of parameters (MatFormer, Nemotron Elastic, Star Elastic, Accordion). The intro's "to our knowledge" sentence stays; its TODO removed.
  Closest new paper cited in related work: MedPsy (Vitabile et al., NeurIPS 2026, arXiv 2611.02011): Qwen3-1.7B/4B medical models for laptops/phones, distilled from a 235B teacher, quantised, FIXED sizes. bib key `vitabile2026medpsy` (venue from its arXiv comment; full text not read).
- **Fig. 1 (TikZ, `fig:overview`, Sect. 3.4):** nested prefix slices of one weight matrix (188M/364M/596M) + controller loop (request, device state, controller, answer at width l, measured latency/energy feedback).
- **int8 + co-residency** paragraph in Sect. 5.4: bf16 pair does not fit reliably (OOM on longest prompts / swapping under load), speed unchanged when co-resident; int8 accuracy within +-0.3 at all widths (188M -0.3, p 0.04); int8 memory nested 1.7B 2.43 GB, pair 3.43 GB (1.9 GB to spare), 1.72B + nested 0.6B 3.46 GB; no int8 kernel, so no int8 speed/energy. "Both nested models" int8 case and the +load int8 cases were still running on the board (tmux int8r) and are NOT in the paper.
- **Two-model simulation** paragraph in Sect. 5.5 (numbers from `controller/two_model.py`, approximate 1.7B profile, co-residency assumed and flagged): pair vs nested 1.7B +4.8 / +3.3 at 2.8 / 3.2 J, level at 3.6-4.4 J; adding the narrow 0.6B widths +2.5 at 3.2 J, only option below 2.4 J, misses 4.9% vs 10.9% at -1.0 acc.
- Limitations: int8 speed/energy and the two-model comparison (simulation) named as open.

### 2026-11-25 17:32: int8 resume, status (still running, tmux `int8r`)
- Done, all fit, no swapping: nested 1.7B (2.43 GB with KV+graphs; 2.3 GB available at end), pair (3.43 GB; 1.9 GB), pairn = 1.72B + nested 0.6B (3.46 GB; 1.5 GB available, peak used 6.1 GB). All generations OK incl. 2,009-token prompts.
- Running: `both` (nested 1.7B five widths + nested 0.6B three widths): built and prepared at 3.53 GB allocated, 96 of 128 generations done without error so far. Not started: nested / pair / both + GPU load.
- The slow on-the-fly int8 layer (3.5-9x bf16 per matrix) makes each case take 5-25 min; remaining run ~1 h. Copies in `jetson_logs_mednest/mednest-results/int8_resume/`.

### 2026-11-25 17:38: int8 resume, all four no-load cases done (load cases running, tmux `int8r`)
- `both` (nested 1.7B five widths + nested 0.6B three widths, int8): FITS, all 128 generations OK; 3.53 GB allocated with KV+graphs, 0.92 GB available at the end, peak used 6.7 GB, 37 pages swapped out (~0.15 MB, negligible).
  So at int8 every combination fits: nested 1.7B 2.43 GB (2.3 GB left), pair 3.43 GB (1.9), 1.72B + nested 0.6B 3.46 GB (1.5), both nested 3.53 GB (0.9). In bf16 only single models / nested 1.7B fit (pair, pairn, both failed).
- Running: nested + GPU load (load started with 2.1 GB available, 10 of 80 generations so far). Then pair + load, both + load. Paper (Sect. 5.4) already states the first three int8 fits; `both` int8 fitting could be added.

### 2026-11-25 17:45: int8 board run cancelled by the user (will resume later); paper sentence added
- tmux `int8r` killed during "nested + GPU load" (10 of 80 generations, no result line; load process gone too). Board idle, no tmux session. Not run: nested / pair / both + GPU load.
  To resume: copy nothing; on the board edit `controller/run_int8_resume.sh` to keep only the three load cases (or run them by hand) with OUT=~/mednest-results/int8_load so the finished files stay.
- Paper Sect. 5.4: one sentence added: both nested models together (eight widths) fit in int8 (3.53 GB, 0.9 GB to spare). PDF rebuilt (19 pages, no errors).
