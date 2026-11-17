# MedNest on the Jetson Orin Nano

On-device measurements for the paper (Sect. "On-device elastic inference"): what each MedNest width
costs on the board, what switching width costs against the alternatives, and whether choosing the
width per request pays off when the budget changes.

Board: Jetson Orin Nano 8 GB developer kit (Super), JetPack 6.2.1, L4T 36.4.7, Python 3.10, torch
2.8.0 (NVIDIA wheel), CUDA 12.6, 16 GB swap (`jetson_logs/exp01_env.jsonl` of the earlier project).
The board setup itself (flashing, swap, passwordless sudo for `nvpmodel` / `jetson_clocks` /
`tegrastats`) is in `Matryoshka Learning/jetson_eval_harness/jetson_setup.md`; it is already done.

## How elastic inference works here

A MedNest width is a prefix slice of the full-width weights in every dimension (the exporter writes
exactly that). The board loads the full-width export once; each width is a standard Qwen3 model
whose parameters are **views** into those tensors (`mednest_jetson/elastic.py`). Switching width =
using another view: nothing is loaded or copied, and nine widths cost the memory of one model.
`check_env.py` proves on the board that every view gives the exported checkpoint's answers.

Two alternatives are measured against it (`bench_switch.py`): separate models loaded from disk on
every change, and *slice loading*, i.e. reading only one width's prefix from the single full
checkpoint (one artifact for every device; the memory-elastic deployment).

## Steps

```bash
# 0. code + environment (once)
rsync -a --exclude hf_export --exclude data --exclude results <machine>:FlexSLM/ ~/FlexSLM/   # or git clone
python3 -m venv --system-site-packages ~/mednest-venv && . ~/mednest-venv/bin/activate
pip install -r ~/FlexSLM/jetson/requirements-jetson.txt
python3 -c "import torch; assert torch.cuda.is_available(), 'CPU-only torch wheel'"

# 1. models (~12 GB; set FINAL=flexllama__sft.paper.qwen3_0p6b_nested.no_kd if the CE-only run wins)
bash ~/FlexSLM/jetson/stage_models.sh

# 2. smoke test (~10 min), then the sweep (~2 h, estimate), inside tmux
cd ~/FlexSLM && QUICK=1 bash jetson/run_all.sh
bash jetson/run_all.sh

# 3. copy $MEDNEST_OUT back and build the tables (anywhere)
python3 jetson/aggregate.py --results <dir> --cluster mednest:8=results/gen.hf_export-flexllama__sft.paper.qwen3_0p6b_nested-level8.t32.samples.jsonl ...
```

## Scripts

| Script | What it measures | Output |
|---|---|---|
| `check_env.py` | stack, GPU/bf16, rails, power modes, sudo, models; views == exported levels | `env.jsonl` |
| `bench_widths.py` | per width and per dense baseline: TTFT, ms/token, ms/answer (median, p95), weight bytes, peak RAM, energy/answer (VDD_IN, gross and net of idle), accuracy on the subset | `widths_<tag>.jsonl` |
| `bench_switch.py` | switch time (view / materialize), TTFT right after a switch, reload from disk (warm/cold cache), slice load from the full file, memory of all separate models resident | `switch.jsonl` |
| `bench_trace.py` | urgent/routine request stream while the power mode changes (15W -> 7W -> 15W); policies fixed-max, fixed-min, elastic, reload: deadline misses, accuracy, latency, energy | `trace.jsonl` |
| `aggregate.py` | CSV + LaTeX tables; agreement of on-device answers with the cluster's per-question outputs | `derived/` |
| `make_subset.py` | the committed 210-question subset (30 per benchmark, seed 0, with test indices) | `questions/subset.jsonl` |
| `stage_models.sh` | models from DAS-6 (rsync) and the Hub (pinned commits) | `~/mednest/models` |

Shared flags: `--power-mode 15W|7W|25W|MAXN_SUPER` (set via `sudo -n nvpmodel`), `--power-source
auto|sysfs|tegrastats|none`, `--dtype bf16` (as on the cluster), `--cooldown-c 50`, `--warmup 3`,
`--out-dir` / `$MEDNEST_OUT`, `--tag`. Every output file starts with a provenance record (board,
JetPack, power mode, clocks, git commit, argv); nothing is overwritten.

## Protocol notes

- Batch 1, greedy, KV cache, 32 new tokens, bf16: the deployment case and the cluster's decoding.
  Apollo: its own format with the answer turn opened ("The correct answer is"), 16 tokens. The
  untrained Qwen3-0.6B reasons first: 1,024 tokens, 5 questions per benchmark (each is ~1,000 steps).
- Accuracy on the full test sets comes from the cluster; the board answers a 210-question subset
  to show the deployed model answers the same way (`aggregate.py --cluster ...`).
- Memory: on Jetson the GPU shares RAM with the CPU, so peak system memory in use
  (MemTotal - MemAvailable) is reported next to the CUDA allocator's peak, which misses the CUDA
  context.
- Energy: VDD_IN (module input) from the INA3221 via sysfs at 5 ms; net = gross - idle floor x time.
  The module rail excludes the carrier board (say so in the paper).
- Trace deadlines default to `auto`: urgent = predicted latency of the middle width in the first
  mode (so the full width cannot meet it), routine = 1.2 x the full width's latency in the slowest
  mode. The predictor is fitted on `bench_widths` answers of the same board and mode.
- `jetson/nestbench/` (device, power, timing, results) is vendored unchanged from the earlier
  harness; see its `__init__.py` for provenance.
- Dry run on a cluster GPU (no rails): `sbatch scripts/das6/jetson_validate.sbatch`.
