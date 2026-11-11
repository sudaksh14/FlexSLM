"""
Stage 1 of the MedNest recipe: nested KD recovery of Qwen3-0.6B on PMC full text,
frozen Qwen3-1.7B teacher (always larger than the student), before the Stage-2
nested SFT (training/sft.py).

Data: pre-tokenized PMC Open Access (commercial-use subset) from
training/prepare_pmc.py, read as random fixed-length windows from a memmap
(utils._MemmapWindowDataset). Every run with the same seed reads the same windows,
so pilot arms differ only in the recipe, never in the data, and no run depends
on the network.

Budget: `train_tokens` sets the number of training windows (train_tokens /
max_seq_length, shared across DDP ranks), and the learning-rate schedule (linear
warm-up, cosine to 0, stepped per optimizer step) is laid out over exactly that
many steps, so a pilot is a complete short run, not a truncated long one.

Configs (registered into CONFIGS['flexllama'] by config/experiments.py):
  s1.qwen3_0p6b.pilot.<arm>   20M-token recipe pilots (see S1_PILOT_ARMS)
  s1.qwen3_0p6b.full          the final Stage-1 run (budget S1_FULL_TOKENS);
                              Stage 2 (sft.paper.qwen3_0p6b_nested*) loads it.
  s1.dev.qwen3_0p6b_smoke     tiny budget, for smoke tests of the pipeline.

Ordering (M1) is a pilot dimension: base = original weight order, order_hidden_act and
order_taylor start from the checkpoints training/importance_order.py writes (run
scripts/das6/importance_order.sbatch [--scheme hidden_act] first). The final run uses
importance_order.FINAL_ORDERING, set from the pilots.
"""
import dataclasses
import functools
import json
import logging
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from training import FlexTrainingContext, TrainerBuilder
from training.level_weights import CONTEXT_DEFAULTS as _LW_DEFAULTS, check_args as check_level_weighting
from training.importance_order import FINAL_ORDERING
from training.memopt import MemOptFlexLMKDTrainer
from training.trainer import adamw, check_level_sampling, warmup_cosine_scheduler

PMC_DATA_DIR = os.path.join("data", "pmc_qwen3")
S1_PILOT_TOKENS = 20_000_000  # 1/5 of the final budget
S1_FULL_TOKENS = 100_000_000  # user, 2026-11-10: 100M, must fit in 5 days; run once the recipe is final


def _world_size() -> int:
    # Set before trainer.fit() by srun (SLURM_NTASKS: 2 on node205, 1 on node206) or torchrun.
    return int(os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or 1)


def load_pmc_memmap(data_dir: str = PMC_DATA_DIR, max_seq_length: int = 1024, global_batch_size: int = 32,
                    train_tokens: int = S1_PILOT_TOKENS, val_windows: int = 640, test_windows: int = 1600,
                    seed: int = 1337, num_workers: int = 4, **_ignored):
    """Train windows from train.bin (train_tokens // max_seq_length of them), val
    and test windows from the separate val.bin / test.bin, whose documents are
    disjoint from the training documents (prepare_pmc.py).

    global_batch_size is the number of sequences per optimizer step over all GPUs;
    each rank's loader gets global_batch_size / world size, so a run on node205
    (2 x A40, 16 per GPU) and on node206 (1 x A40, 32) take the same steps over the
    same windows and validate on the same 640 windows."""
    world = _world_size()
    if global_batch_size % world:
        raise ValueError(f"global_batch_size {global_batch_size} not divisible by {world} GPUs")
    batch_size = global_batch_size // world
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No pre-tokenized PMC at {data_dir}. Build it first (CPU job):\n"
            f"  sbatch scripts/das6/prepare_pmc.sbatch")
    with open(meta_path) as f:
        meta = json.load(f)
    if train_tokens > meta["train_tokens"]:
        logging.warning(f"train_tokens {train_tokens:,} > {meta['train_tokens']:,} prepared: "
                        f"windows will repeat text (more than one pass).")

    def collate(batch):
        ids = torch.stack(batch)
        return ids, ids

    def make_loader(split, num_windows, shuffle, split_seed):
        ds = utils._MemmapWindowDataset(os.path.join(data_dir, f"{split}.bin"), meta["dtype"],
                                        max_seq_length, num_windows, seed=split_seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=collate, pin_memory=True, drop_last=True)

    return (make_loader("train", max(1, train_tokens // max_seq_length), True, seed),
            make_loader("val", val_windows, False, seed + 1),
            make_loader("test", test_windows, False, seed + 2))


@dataclasses.dataclass
class Stage1Context(FlexTrainingContext):
    kd_lambda: float = 0.5
    kd_temperature: float = 2.0
    teacher_hf_model: str = "Qwen/Qwen3-1.7B"
    num_levels_per_step: int = None
    level_sampling: str = "sandwich"
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.01
    train_tokens: int = S1_PILOT_TOKENS
    activation_checkpointing: bool = True
    loss_chunk_size: int = 1024
    # Sequences per forward/backward inside a step (exact, see memopt.py): 16 x 1,024
    # in one pass ran out of memory at step 0 (job 28080).
    micro_batch_size: int = 4
    level_weighting: str = "none"
    level_weight_lr: float = 1e-2
    level_weight_ema: float = 0.9
    level_weights_fixed: tuple = None
    dwa_window: int = 50
    dwa_temperature: float = 2.0

    def __init__(self, train_tokens=S1_PILOT_TOKENS, data_dir=PMC_DATA_DIR, max_seq_length=1024,
                 global_batch_size=32, kd_lambda=0.5, kd_temperature=2.0, teacher_hf_model="Qwen/Qwen3-1.7B",
                 num_levels_per_step=None, level_sampling="sandwich", lr=1e-4, weight_decay=0.0,
                 warmup_ratio=0.01, activation_checkpointing=True, loss_chunk_size=1024,
                 micro_batch_size=4, level_weighting="none", level_weight_lr=1e-2, level_weight_ema=0.9,
                 level_weights_fixed=None, dwa_window=50, dwa_temperature=2.0,
                 epochs=1, patience=1000, **kwargs):
        loader = functools.partial(load_pmc_memmap, data_dir=data_dir, max_seq_length=max_seq_length,
                                   global_batch_size=global_batch_size, train_tokens=train_tokens)
        FlexTrainingContext.__init__(self, loader, patience=patience, epochs=epochs, **kwargs)
        check_level_weighting(level_weighting, level_weight_lr, level_weight_ema,
                              level_weights_fixed, dwa_window, dwa_temperature)
        check_level_sampling(level_sampling)
        self.train_tokens = train_tokens
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature
        self.teacher_hf_model = teacher_hf_model
        self.num_levels_per_step = num_levels_per_step
        self.level_sampling = level_sampling
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.activation_checkpointing = activation_checkpointing
        self.loss_chunk_size = loss_chunk_size
        self.micro_batch_size = micro_batch_size
        self.level_weighting = level_weighting
        self.level_weight_lr = level_weight_lr
        self.level_weight_ema = level_weight_ema
        self.level_weights_fixed = tuple(level_weights_fixed) if level_weights_fixed else None
        self.dwa_window = dwa_window
        self.dwa_temperature = dwa_temperature

    def make_optimizer(self, model):
        return adamw(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def make_step_scheduler(self, optimizer, total_steps: int):
        return warmup_cosine_scheduler(optimizer, total_steps, self.warmup_ratio)


class FlexLMStage1Trainer(MemOptFlexLMKDTrainer):
    """MemOptFlexLMKDTrainer (chunked CE + KL over every token, activation
    checkpointing, level weighting) with the per-step warm-up + cosine schedule
    laid out over the run's exact number of optimizer steps."""

    def configure_optimizers(self):
        ctx = self.training_context
        optimizer = ctx.make_optimizer(self.submodel)
        scheduler = ctx.make_step_scheduler(optimizer, int(self.trainer.estimated_stepping_batches))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_train_epoch_end(self):
        # The schedule is stepped per optimizer step in _step, never per epoch.
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], prog_bar=True, sync_dist=True)


torch.serialization.add_safe_globals([Stage1Context, FlexLMStage1Trainer, load_pmc_memmap])

# 5 levels: one per head count in [8, 16] (GQA ratio 2); 9 levels: hidden/FFN every 1/16 of
# the width, head counts repeat (8, 8, 10, 10, ..., 16). Sampling 3 per step over a dense grid
# approximates training on a continuum of widths at the cost of the 3-level recipe.
FRACTIONS_5 = (0.5, 0.625, 0.75, 0.875, 1.0)
FRACTIONS_9 = tuple(0.5 + k / 16 for k in range(9))

# (arm name, overrides of the base pilot recipe). Base: original weight order, lr 1e-4,
# KD lambda 0.5, T 2, every level every step, equal weights, 20M tokens.
S1_PILOT_ARMS = [
    ("base", {}),
    ("order_hidden_act", dict(ordering="hidden_act")),  # M1: residual channels by activation
    ("order_taylor", dict(ordering="taylor")),          # M1: everything by Taylor score
    ("lr5e-5", dict(lr=5e-5)),
    ("lr2e-4", dict(lr=2e-4)),
    ("kd1", dict(kd_lambda=1.0)),                       # pure KD, no CE
    ("w_fixed", dict(level_weighting="fixed", level_weights_fixed=(3.0, 1.0, 1.0))),
    ("w_uncertainty", dict(level_weighting="uncertainty")),
    ("w_lossprop", dict(level_weighting="loss_proportional")),
    ("w_dwa", dict(level_weighting="dwa")),
    ("single", dict(level_sampling="single")),          # one random level per step
    # Level count / dense grid. Every grid contains the paper's widths 0.5 / 0.75 / 1.0, so
    # the arms are compared on those levels (5 levels: 0, 2, 4; 9 levels: 0, 4, 8).
    ("levels5", dict(fractions=FRACTIONS_5)),                                   # all 5 per step
    ("levels5_sw3", dict(fractions=FRACTIONS_5, num_levels_per_step=3)),        # min, max + 1 random
    ("grid9_sw3", dict(fractions=FRACTIONS_9, num_levels_per_step=3)),          # dense grid, 3 per step
]


def make_stage1(train_tokens: int, wandb_project_name: str, ordering: str = "none",
                fractions=(0.5, 0.75, 1.0), **overrides) -> TrainerBuilder:
    """Qwen3-0.6B with GQA-preserving levels at hidden fractions 0.5 / 0.75 / 1.0
    (the paper's three levels), KD from the frozen Qwen3-1.7B. ordering (M1, see
    training/importance_order.ORDERINGS) picks the warm-start checkpoint: "none" is
    Hugging Face Qwen3-0.6B, the others the checkpoints importance_order.py writes."""
    from config.experiments import make_flexllama_warmstart_kd
    from training.importance_order import start_weights
    model_config = make_flexllama_warmstart_kd("qwen3-0.6b", fractions=tuple(fractions)).model_config
    model_config = dataclasses.replace(model_config, pretrained_hf_model=start_weights(ordering))
    kwargs = dict(train_tokens=train_tokens, gradient_clip_val=1.0, seed=0,
                  wandb_project_name=wandb_project_name)
    kwargs.update(overrides)
    return TrainerBuilder(FlexLMStage1Trainer, model_config, Stage1Context(**kwargs))


def register_stage1_configs(configs: dict) -> None:
    """Called from config/experiments.py before register_sft_configs (the Stage-2
    configs load the Stage-1 result)."""
    flex = configs["flexllama"]
    for arm, overrides in S1_PILOT_ARMS:
        flex[f"s1.qwen3_0p6b.pilot.{arm}"] = make_stage1(
            S1_PILOT_TOKENS, f"MedNest_s1_pilot_{arm}", **overrides)
    flex["s1.qwen3_0p6b.full"] = make_stage1(S1_FULL_TOKENS, "MedNest_s1_full", ordering=FINAL_ORDERING)
    # A1: the final recipe without importance ordering (Stage 2: ...nested.no_order). Only
    # meaningful if the pilots choose an ordering; if FINAL_ORDERING stays "none", A1 is
    # reported from the order_* pilots instead and this config is not run.
    flex["s1.qwen3_0p6b.full.no_order"] = make_stage1(S1_FULL_TOKENS, "MedNest_s1_full_no_order",
                                                      ordering="none")
    flex["s1.dev.qwen3_0p6b_smoke"] = make_stage1(2_000_000, "MedNest_s1_dev")
