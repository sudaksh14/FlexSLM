"""Checks the fineweb_nc.* (nanochat-recipe) configs resolve as intended."""
import utils
from config.experiments import CONFIGS

nc = {k: v for k, v in CONFIGS['flexllama'].items() if k.startswith('fineweb_nc.')}
assert nc, "no fineweb_nc.* configs found"

for k, tb in sorted(nc.items()):
    tc = tb.training_context
    lf = tc.loader_function
    assert lf.func is utils.load_fineweb_edu_memmap, f"{k}: wrong loader {lf.func.__name__}"
    assert tc.epochs == 1, f"{k}: epochs={tc.epochs}, the nanochat recipe wants a single pass"
    print(f"{k}\n   trainer={tb.training_method.__name__} epochs={tc.epochs} "
          f"patience={tc.patience} tokenizer={lf.keywords['tokenizer_name']} "
          f"batch_size={lf.keywords['batch_size']}")

print(f"\n{len(nc)} nanochat-recipe configs, all epochs=1 on the memmap loader: OK")

# the other families must be unaffected
stream = CONFIGS['flexllama']['fineweb.kd_lambda05_1p4B'].training_context
assert stream.loader_function.func is utils.load_fineweb_edu, "streaming config changed"
assert stream.epochs == 3, stream.epochs
mm = CONFIGS['flexllama']['fineweb_mm.kd_lambda05'].training_context
assert mm.loader_function.func is utils.load_fineweb_edu_memmap
assert mm.epochs == 3, mm.epochs
print("streaming and fineweb_mm.* configs unchanged: OK")
