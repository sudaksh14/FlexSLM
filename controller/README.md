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

- A2 (`data_io.A2G`, the paper's main 0.6B model: fine-tuning only, 3 widths 188M / 364M / 596M):
  `gen.hf_export-flexllama__sft.paper.qwen3_0p6b_nested.no_stage1-level{0,1,2}.t32.samples.jsonl`, stored a second time as
  `..._no_stage1_grid048-level{0,4,8}`: its widths have the shapes of grid levels 0 / 4 / 8, so the 9-level Jetson
  profile applies unchanged.

Re-copy for another model (Git Bash: `export SSH_AUTH_SOCK=~/.ssh/agent.sock` first): `ssh das6 'cd /var/scratch/skalra/flexslm/results && tar cf - gen.<model>-level?.t32.samples.jsonl' | tar xf - -C controller/data/samples`, then `--model <model>`.

## Run

```
cd controller
python test_controller.py                 # 7 unit tests, synthetic inputs
python data_io.py                         # profile fit
python offline.py [--answer eos|forced32] [--model M] # Pareto / hull / oracle (M: default M3; A2 = data_io.A2G)
python simulate.py budget  [--answer ...] [--V 10] [--task-table shrunk|scaled|raw] [--model M]
python simulate.py thermal [--answer ...] [--slowdown 1.3] [--urgent-level 8 --urgent-scale 1.0] [--model M]
python board_compare.py <dir>/controller_full.jsonl   # board run: policy vs the fixed min/max mix at equal energy
python two_model.py                        # two separate full-width models vs the nested models (simulation)
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

### A2, the paper's 3-width model (`out/*A2*`, same protocol)

A2 L0 / L4 / L8 (188M / 364M / 596M) = 20.8 / 22.7 / 30.5 at 1.23 / 1.81 / 2.49 J (eos answers). Hull = **{L0, L8}**:
the 364M width lies 2.6 points under the L0/L8 time-share at its energy, so the queue time-shares the two ends and
almost never picks it (14-102 of 6,423 requests). Oracle 45.8 (55.7 if independent errors).

| Budget J/ans | best fixed width within budget | queue (task-agnostic) | queue (task-aware) | task-aware − fixed | task-aware − agnostic |
|---|---|---|---|---|---|
| 1.43 | L0 21.1 | 22.5 | 24.4 | +3.3 [+2.7, +4.0] | +1.9 [+1.3, +2.5] |
| 1.64 | L0 21.1 | 24.8 | 25.4 | +4.3 [+3.5, +5.1] | +0.6 [−0.2, +1.2] |
| 1.85 | L4 22.8 | 27.2 | 27.7 | +4.9 [+3.3, +6.6] | +0.5 [−0.3, +1.3] |
| 2.05 | L4 22.8 | 28.1 | 29.7 | +6.9 [+5.2, +8.7] | +1.6 [+0.8, +2.3] |
| 2.26 | L4 22.8 | 29.6 | 29.8 | +7.0 [+5.3, +8.9] | +0.2 [−0.1, +0.5] |

The gains over a fixed width are larger than on M3 (+2.3 to +3.6) because A2's accuracy jumps between 364M and 596M:
any budget that admits some full-width answers beats every fixed width it can afford. Thermal (×1.3): urgent misses
during the slowdown 98.5 % open-loop → 48.6 % with feedback (floor: L0 misses 45.4 %), overall 14.8 % → 9.1 %,
accuracy −0.47 [−0.97, +0.02].

## Board results (Jetson Orin Nano, 15 W, 2026-11-20)

630 requests (90 per test set, evaluation half) × 6 policies, one request at a time, phases `15W:210,15W+gpu:210,15W:210`
(the middle third runs a co-running CUDA matmul load, 60 % duty), 30 % urgent requests (deadline = median predicted
full-width latency, 359 ms), the rest 869 ms. `--score cluster` with random weights built in memory (`--random-from`):
latency and energy are measured, correctness is the cluster's answer at the chosen width. Energy: VDD_IN over each
request, net of idle (3.9 W); in the load phase also net of the load's own work (below). Outputs (copies):
`jetson_logs_mednest/mednest-results/controller_full/` (M3) and `controller_full_A2/` (A2).

| Policy | M3 acc | M3 J | M3 miss | A2 acc | A2 J | A2 miss | A2 urgent misses, load phase |
|---|---|---|---|---|---|---|---|
| fixed full width | 25.7 | 3.05 | 26.3 % | 30.8 | 3.04 | 28.1 % | 100 % |
| fixed narrowest | 19.8 | 1.27 | 4.4 % | 20.3 | 1.25 | 7.1 % | 40.6 % |
| deadline (open loop) | 24.9 | 2.75 | 21.9 % | 30.0 | 2.68 | 20.8 % | 95.3 % |
| deadline + feedback | 24.0 | 2.49 | 14.9 % | 28.6 | 2.48 | 11.0 % | 29.7 % |
| queue, B = 1.78 J | 23.3 | 1.78 | 6.8 % | 24.1 | 1.78 | 6.7 % | 21.9 % |
| queue task-aware, B = 1.78 J | 21.9 | 1.78 | 5.2 % | 25.4 | 1.78 | 5.4 % | 20.3 % |

- The device was 6 % slower than its profile at 15 W and 41 % slower under the load (measured / predicted latency
  1.06 / 1.41); the feedback factor tracked it (A2 deadline+fb: rho 0.99 → 1.54 → 1.13 at the end of the three phases).
- Feedback vs open loop (paired, same stream): misses −7.0 points [−9.5, −4.4] (M3) and −9.8 [−12.5, −7.6] (A2), urgent
  misses under load 98 → 28 % (M3) and 95 → 30 % (A2), 0.21-0.26 J less per answer, accuracy −0.95 [−2.5, +0.6] and
  −1.4 [−3.0, +0.04]: feedback drops width when the device slows, which costs a little accuracy.
- Energy queues at 1.78 J stay on budget (1.78 J measured) and beat the best fixed width that fits it (the narrowest):
  M3 +3.5 [+0.6, +6.4] (agnostic), A2 +3.8 [+0.9, +6.7] (agnostic) and +5.1 [+1.9, +8.2] (task-aware). Against
  the L0/L8 mix at the same measured energy (`board_compare.py`): +0.4 to +2.0, CIs include 0. Task-aware vs agnostic:
  A2 +1.3 [−0.3, +3.0], M3 −1.4 [−3.7, +1.0]; 630 requests cannot resolve 1-2 points (the 6,423-request
  simulation above does).
- Widths used (A2): deadline+fb 113 / 56 / 461 requests at 188M / 364M / 596M; the queues 357-362 / 0-6 / 267-268.
  The middle width matters only to the deadline rule (as the widest width that meets a deadline), never to the
  energy budget.
- Urgent deadlines miss even at full width in the nominal phase (50-60 %), by construction: the deadline is the median
  predicted latency and the device ran 6 % slower than predicted.

**Energy attribution under a co-running load.** Subtracting idle alone charges the request for the load's power
(upper bound); subtracting the load's standalone power over-corrects, because the load slows while it shares the GPU
(lower bound), and its power drifted 8.1 → 10.4 W over a phase as the GPU clocks rose. The runner counts the load's
finished matmuls (a `/dev/shm` counter), measures J per matmul in load-only windows at the start and end of the phase
(30.6-32.8 mJ in every window, stable while the power drifted), and charges each request
`gross − idle·dt − J_per_op·(load ops during the request)`. Full-width energy per answer: 3.00 J without load, 3.13 J
under load (bounds 1.88 / 4.68). Every record keeps both bounds (`energy_net_loadpower_j`, `energy_net_idle_j`).

**Load start-up.** The load's CUDA context needs ~1 GB of the shared 8 GB next to the resident model and the nine
decoders; in a first full run it died of out-of-memory at start-up for 4 of 6 policies, unnoticed (kept as
`controller_full_noload`, invalid). `Load` now empties the caching allocator, waits until the load has finished
matmuls (retries 3×), and aborts the run otherwise; a load that exits mid-phase also aborts; `analyze_board.py` warns
about loaded-phase requests without load work.

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

`--score cluster` (default): no checkpoint on the board. `run_jetson.sh` uses the random export if present,
else `--random-from ~/mednest/models/Qwen3-0.6B` (seeded random weights built in GPU memory from the config,
the 9 widths of `jetson/make_random_export.py`; nothing written to disk). The random weights decode, per
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
  (26.2) where open-loop `deadline` dropped to 23.8; device answers agree with the cluster's on 85.7 % (same
  accuracy). The queue's warm start under-spends on a device far faster than the profile.
- Board runs use cluster correctness with random weights (`--score cluster`); a `--score device` run with the
  trained A2 export staged on the board would check that the device's own answers agree.
- Only one disturbance (GPU load) and one budget (1.78 J) were run on the board; 7 W and CPU load are implemented
  (`PHASES=15W:210,7W:210,15W:210`, `+cpu`) but not run.
- Compiled path not measured: convert 2-3 widths to GGUF / TensorRT and profile them on the board (and the
  int4 full model, the obvious reviewer comparison).
- Router / cascade: needs per-question confidence (answer-letter margin) from eval; not in the samples files.
- Re-run on the next Stage-2 model (e.g. the 20M-recipe `final9` evals): if L1-L6 improve, the hull changes
  and so does every table above.
