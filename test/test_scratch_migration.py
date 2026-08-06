"""Confirms fineweb_bin_dir now resolves under SCRATCH_PATH and the migrated
llama-160m data is readable there through the real loader (not a synthetic .bin)."""
import config.paths as paths
import utils

resolved = utils.fineweb_bin_dir("JackFram/llama-160m")
print(f"SCRATCH_PATH = {paths.SCRATCH_PATH}")
print(f"fineweb_bin_dir(...) = {resolved}")
assert str(paths.SCRATCH_PATH) in resolved, "fineweb_bin_dir did not move to SCRATCH_PATH"
assert "/home/" not in resolved, f"still resolving under home: {resolved}"

train, val, test = utils.load_fineweb_edu_memmap(
    max_seq_length=1024, batch_size=8, num_workers=0,
    tokenizer_name="JackFram/llama-160m", val_batches=4, test_batches=4)
x, y = next(iter(train))
assert x.shape == (8, 1024) and x.max() < 32000
print(f"loaded real migrated data: train batch {tuple(x.shape)}, "
      f"epoch length {len(train)} batches, max token id {int(x.max())}")
print("\nSCRATCH MIGRATION VERIFIED")
