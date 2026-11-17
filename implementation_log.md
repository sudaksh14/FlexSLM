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
| J1 | Jetson Orin Nano 15 W: latency, memory, energy, level switch time | not started |
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
