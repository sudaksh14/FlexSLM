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
- Rules: sbatch only; GPU on defq 205–208 (training node205, and node206 1x A40 for Stage-1 pilots (user, 2026-11-10): `sbatch --nodelist=node206 --gres=gpu:1 --ntasks-per-node=1 scripts/das6/train.sbatch <cfg>`; eval 206–208), CPU on
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
