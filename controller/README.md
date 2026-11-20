# Width controller (MedNest)

Chooses the MedNest width per deployment and per request. Kept apart from the training / eval / Jetson
pipeline: it reads copies of their outputs and changes none of their files. The analysis and simulation are pure
numpy (laptop); only `jetson_runner.py` imports the Jetson harness (`jetson/mednest_jetson`, `jetson/nestbench`),
read-only, to load views, decode and sample power.

## Design: two tiers

| Tier | When | Decides | Code |
|---|---|---|---|
| 1. Deploy time | once per device | backend (elastic **views** vs **compiled** fixed-shape engines) and which widths to put on the device | `policies.deploy_select`, `deploy.py` |
| 2. Runtime | per request | width, from deadline, observed latency, energy budget, task type | `policies.py` |

- **views** (PyTorch + CUDA graphs): one weight set (1.19 GB for 0.6B), every width resident, 0.037 ms switch,
  so the runtime tier can switch on every request.
- **compiled** (TensorRT / llama.cpp GGUF): shapes fixed at build time, no weight sharing between engines,
  switching = reload (~1 s). Each width is a dense Qwen3 (`hf_export/<cfg>/level<k>`, or the legacy delta file
  `networks/level_delta_utils.py`). Tier 1 keeps 1-2 anchors; two resident anchors are switched for free by
  the same runtime queue.

Runtime policies (same `choose(Request)` / `observe(level, ms, J)` interface):
- `Fixed(k)`: a dense deployment.
- `DeadlinePolicy`: widest width whose predicted latency (Jetson profile, prompt length) meets the deadline
  (= `elastic` in `jetson/bench_trace.py`); `feedback=True` scales predictions by an EWMA of observed/predicted
  latency (throttling, co-running load).
- `EnergyQueuePolicy`: Lyapunov drift-plus-penalty for a long-run budget B (J/answer): argmin −V·acc + Q·E over
  the deadline-feasible widths, Q ← max(0, Q + E − B), warm-started at the equilibrium backlog. `task_aware`
  uses per-test-set accuracy (shrunk toward the global table, κ = 500).

## Inputs (`data/`, gitignored; copies)

- `data/samples/gen.hf_export-flexllama__sft.paper.qwen3_0p6b_nested-level{0..8}.t32.samples.jsonl` (+ `.json`,
  conventional): M3 eval outputs from DAS-6 `/var/scratch/skalra/flexslm/results/` (copied 2026-11-20), in
  `data/medmix_sft/test.jsonl` order (checked line by line).
- `data/jetson/{widths_15W,switch}.jsonl`: 0.6B shape-only run, Jetson Orin Nano 15 W, from
  `C:\Users\skalra\Workspace\jetson_logs_mednest\mednest-results\shape_only_full\`.

Re-copy for another model: `ssh das6 'cd /var/scratch/skalra/flexslm/results && tar cf - gen.<model>-level?.t32.samples.jsonl' | tar xf - -C controller/data/samples`, then `--model <model>`.

## Run

```
cd controller
python test_controller.py                 # 7 unit tests, synthetic inputs
python data_io.py                         # profile fit
python offline.py [--answer eos|forced32] # Pareto / hull / oracle
python simulate.py budget  [--answer ...] [--V 10] [--task-table shrunk|scaled|raw]
python simulate.py thermal [--answer ...] [--slowdown 1.3] [--urgent-level 8 --urgent-scale 1.0]
python deploy.py --backend views|compiled --weight-mb N [--p90-ms D] [--energy-j B]
```
Outputs: `out/*.json` and `out/*.txt` (gitignored).

## What is real and what is synthetic

Real: per-question correctness at each width (scored only on the evaluation half; every controller statistic
comes from the seeded calibration half), Jetson latency/power vs prompt length, latency noise (empirical
observed/predicted ratios). The energy model reproduces the measured J/answer (forced 32 tokens: L0 4.02 J,
L8 9.97 J model vs 4.0 / 10.0 measured). Synthetic: request order, deadlines, the slowdown phase.
`--answer eos` (default) assumes generation stops after `\boxed{X}` (~4 tokens, from the generated text);
`forced32` is the Jetson bench setting.

## Results (M3 final, 0.6B, 15 W, 2026-11-20)

Offline (`out/offline.eos.txt`): upper convex hull in (J, acc) and (ms, acc) = **{L0, L7, L8}**; L1-L6 lie
0.5-2.4 points under the L0/L7 time-share at equal energy. Oracle per-question router 50.3 macro (vs L8 27.2),
but answers are strongly correlated across widths (any-width-correct 50.3 vs 86.5 if independent) and part
of the oracle is lucky guesses; a deployable router needs a confidence signal the samples files do not store.

Budget scenario (`out/budget.eos.txt`, 6,423 evaluation questions, paired 95% bootstrap CI):

| Budget J/ans | best fixed width within budget | queue (task-agnostic) | queue (task-aware) | task-aware − fixed |
|---|---|---|---|---|
| 1.43 | L1 21.4 | 22.5 | 23.7 | +2.3 [+1.1, +3.5] |
| 1.64 | L2 21.4 | 23.7 | 24.2 | +2.7 [+1.2, +4.2] |
| 1.85 | L4 22.7 | 24.7 | 25.5 | +2.8 [+1.2, +4.3] |
| 2.05 | L5 22.5 | 25.5 | 26.1 | +3.6 [+2.0, +5.1] |
| 2.26 | L6 24.1 | 25.7 | 26.3 | +2.3 [+1.0, +3.5] |

(L8 = 26.9 at 2.42 J.) Task-aware over task-agnostic: +0.5 to +1.2, CI excludes 0 only at 1.43 J. With
`forced32` the same pattern (task-aware − fixed +2.0 to +3.1, all CIs exclude 0). The κ / table variant was chosen after looking
at the evaluation half (raw, shrunk, scaled compared), a mild selection effect.

Thermal scenario (`out/thermal.eos.txt`; 30 % urgent with deadline = median predicted L8 latency, latency
×1.3 in the middle third): urgent misses during the slowdown 100 % open-loop → 49 % with feedback (floor
46 % = even L0 misses there); overall misses 15.9 % → 10.4 %, accuracy −0.3 [−0.95, +0.33]. With `forced32`
(wider latency range): 100 % → 22.5 %. Feedback + task-aware queue at 1.85 J: 25.0 acc, 7.7 % misses, 1.84 J.

## On the Jetson (`jetson_runner.py`)

Serves a request stream for real with the same `policies.py` objects: one export loaded once, every width a view,
CUDA-graph decoders (`jetson/mednest_jetson`), per-request latency measured and per-request energy integrated
from the VDD_IN trace over the request's own window, net of the idle floor; the policy observes both. The
controller only knows ONE profile (15 W, no load: `bundles/profile.0p6b.15W.jsonl`); phases disturb the device
mid-stream (`15W:210,15W+gpu:210,15W:210`: a co-running CUDA matmul loop; or `7W` via nvpmodel; `+cpu`).

| File | Where | What |
|---|---|---|
| `make_bundle.py` | laptop | `bundles/<name>.json`: calibration-half tables + a stratified evaluation-half stream (90 per test set by default) with the cluster's answer and correctness at every width; `bundles/smoke.json` (42) |
| `jetson_runner.py` | board | the run; writes `$MEDNEST_OUT/controller_<tag>.jsonl` (setup, idle, one record per request incl. rho/Q, per-policy summary) |
| `run_jetson.sh` | board | `QUICK=1` smoke (~5 min), full 630 requests x 6 policies (~35 min estimate); env `PHASES`, `NO_DEADLINES=1`, `ELASTIC`, `SCORE` |
| `analyze_board.py` | laptop | per-policy accuracy, J, latency, misses per phase, paired CI vs a reference policy, measured/predicted latency per phase, rho per phase |
| `das6_dryrun.sbatch` | DAS-6 node207 | mechanics check on an A10 with the trained export (no rails, no nvpmodel); both scorings |

`--score cluster` (default): no checkpoint on the board. The random-weight shape-only export decodes, per
request and width, exactly as many tokens as the trained model produced on the cluster (EOS ignored); correctness
is the cluster's answer at that width. Latency/energy depend only on shapes and token counts, so this measures
the controller's real cost; accuracy is the trained model's. `--score device` (trained export staged):
real greedy answers with EOS, scored on the board, agreement with the cluster recorded.

```
# laptop
python controller/make_bundle.py --n 630            # (done: bundles/ is tracked and synced with the code)
# board (tmux, venv active, ~/FlexSLM updated)
QUICK=1 bash controller/run_jetson.sh && bash controller/run_jetson.sh
# laptop, after copying $MEDNEST_OUT back
python controller/analyze_board.py <dir>/controller_full.jsonl --ref fixed:L8
```

## Limits / next steps

- DAS-6 dry run 28462 (A10, trained export, smoke bundle) passed: feedback learned rho ~0.25 and kept L8
  (26.2) where open-loop `deadline` dropped to 23.8; the GPU-load phase raised latency +24 %; device answers
  agree with the cluster's on 85.7 % (same accuracy). Energy there is the profile model (no rails), and the
  queue's warm start under-spends on a device far faster than the profile. Board runs pending.
- Compiled path not measured: convert 2-3 widths to GGUF / TensorRT and profile them on the board (and the
  int4 full model, the obvious reviewer comparison).
- Router / cascade: needs per-question confidence (answer-letter margin) from eval; not in the samples files.
- Re-run on the next Stage-2 model (e.g. the 20M-recipe `final9` evals): if L1-L6 improve, the hull changes
  and so does every table above.
