"""nestbench (vendored) -- device, power, timing and result plumbing for the MedNest Jetson runs.

device.py, power.py, timing.py and results.py are copied UNCHANGED from the author's earlier harness
(Matryoshka Learning/jetson_eval_harness/nestbench, NEST-VLM DATE 2027 work, Sept 2026), where they
were tested on the same Jetson Orin Nano 8 GB (Super) board. Copy date 2026-11-17; md5 at copy time:
  device.py  85ef78381b0c9b9743799427994788eb   power.py   18f30f8c6593ecfa18f18c5e4d6e59b7
  timing.py  0cb82284f4462e94a5fbcfa5b78ef91f   results.py 55359b1ee145e301877d5f75a8ccbace
Their docstrings still mention NEST-VLM (vision prefill); the code is model-agnostic. Fix bugs in
both copies, or not at all. The model-specific parts of the old harness (modelsource, modelutil,
cli, exp0*.py) are VLM-specific and not used: MedNest's own code is in jetson/mednest_jetson/.

device     Jetson identification, nvpmodel power modes, clocks, thermals, cool-down.
power      VDD rail sampling (INA3221 sysfs or tegrastats) -> energy in joules, idle floor.
timing     synchronised timers, warm-up loops, CUDA allocator + /proc memory snapshot.
results    run provenance + append-only JSON-lines writer, stat_block (median / p95 ...).
"""

__version__ = "1.0.0+mednest"
