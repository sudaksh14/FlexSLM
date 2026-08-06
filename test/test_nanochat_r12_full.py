"""Checks the new ratio-12 qwen2.5-1.5b / smollm2-1.7b configs resolve correctly."""
import utils
from config.experiments import CONFIGS

for k, expect_tok in [
    ('fineweb_nc.kd_qwen25_1p5b_warmstart_r12', 'Qwen/Qwen2.5-1.5B'),
    ('fineweb_nc.kd_smollm2_1p7b_warmstart_r12', 'HuggingFaceTB/SmolLM2-1.7B'),
]:
    tb = CONFIGS['flexllama'][k]
    tc = tb.training_context
    lf = tc.loader_function
    assert lf.func is utils.load_fineweb_edu_memmap, f"{k}: {lf.func.__name__}"
    assert lf.keywords['tokenizer_name'] == expect_tok
    assert tc.epochs == 1
    assert lf.keywords['batch_size'] == 4
    print(f"{k}: tokenizer={expect_tok} batch_size=4 epochs=1  OK")

print("\nboth ratio-12 full-data configs resolve correctly")
