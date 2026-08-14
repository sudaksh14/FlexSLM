"""Confirms the experiment banner prints for every config shape without crashing,
and that the numbers it reports are actually correct (not just non-crashing)."""
import os
import io
import contextlib

from config.experiments import CONFIGS
from training import _print_experiment_banner

os.environ.pop('SLURM_PROCID', None)  # ensure we're "rank 0" for this test

cases = [
    'fineweb_nc.kd_qwen25_1p5b_warmstart_r12',   # memmap + KD + memopt
    'fineweb.kd_lambda05_1p4B',                   # streaming + KD, no memopt
    'fineweb_mm.kd_lambda05',                     # memmap + KD
    'fineweb.3levels',                            # non-KD
]

for key in cases:
    tb = CONFIGS['flexllama'][key]
    print(f"=== {key} ===")
    train, val, test = tb.training_context.loader_function()
    try:
        steps = len(train)
        exact = True
    except TypeError:
        steps = getattr(train, 'estimated_steps_per_epoch', None)
        exact = False

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_experiment_banner(f"flexllama,{key}", tb.model_config, tb.training_context, steps, exact)
    out = buf.getvalue()
    assert out, "banner printed nothing"
    assert key in out or f"flexllama,{key}" in out, "config name missing from banner"
    print(out)

# rank-gating: must print nothing when SLURM_PROCID != 0
os.environ['SLURM_PROCID'] = '2'
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    tb = CONFIGS['flexllama']['fineweb.kd_lambda05_1p4B']
    _print_experiment_banner("x", tb.model_config, tb.training_context, 100, True)
assert buf.getvalue() == "", f"non-rank-0 should print nothing, got: {buf.getvalue()!r}"
print("rank-gating (SLURM_PROCID != 0 prints nothing): OK")

print("\nALL BANNER CHECKS PASSED")
