from typing import Callable, Optional
import dataclasses
import os
import datetime
import logging
import random

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch

import flex_modules as fm
from networks.config import ModelConfig, FlexModelConfig
import config.hardware as hardware
import config.paths as paths
import config.wandb
import utils
from timm.loss import SoftTargetCrossEntropy


@dataclasses.dataclass
class TrainingContext(utils.SelfDescripting):
    loader_function: Callable[[], tuple[DataLoader, DataLoader, DataLoader]]
    patience: int = 5
    epochs: int = 10
    label_smoothing: float = 0.0
    gradient_clip_val: Optional[float] = None

    wandb_project_name: str = config.wandb.WANDB_PROJECT_NAME

    unittest_mode: bool = False
    resume: bool = True  # if a checkpoint exists at this config's path, resume from it; set False to force a fresh start
    use_fsdp: bool = False  # shard params/grads/optimizer state across GPUs (FULL_SHARD) instead of full DDP replication

    def make_optimizer(self, model) -> torch.optim.Optimizer:
        raise NotImplementedError()

    def make_scheduler(self, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        raise NotImplementedError()


class BaseTrainer:
    def get_model(self) -> nn.Module:
        raise NotImplementedError()

    def run_training(self, conf_description: str) -> None:
        raise NotImplementedError()


@dataclasses.dataclass
class TrainerBuilder:
    training_method: type[BaseTrainer]
    model_config: ModelConfig
    training_context: TrainingContext

    def __init__(self, training_method: type[BaseTrainer], model_config: ModelConfig, training_context: TrainingContext):
        self.training_method = training_method
        self.model_config = model_config
        self.training_context = training_context

    def build(self):
        return self.training_method(
            self.model_config, self.training_context)

    def run_training(self, conf: str):
        return self.build().run_training(conf)

    def __call__(self, conf: str):
        return self.run_training(conf)


@dataclasses.dataclass
class FlexTrainingContext(TrainingContext):
    load_from: Optional[str] = None
    load_flex_from: Optional[tuple[int, ...]] = None
    load_flex_to: Optional[tuple[int, ...]] = None

    distill: bool = False


class FlexModelTrainer(pl.LightningModule, BaseTrainer):
    def __init__(self, model_config: ModelConfig, training_context: FlexTrainingContext) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_context = training_context
        self.submodel = self.model_config.make_model()
        self.distill_net = None
        self.Mixup = utils.mixup_fn_cifar100
        self.automatic_optimization = False

    def get_model(self) -> nn.Module:
        return self.submodel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.submodel(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch

        if stage == "train":
            x,y = self.Mixup(x, y)
            opt = self.optimizers()
            opt.zero_grad()

        if self.distill_net is not None:
            self.distill_net.eval()
            for p in self.distill_net.parameters():
                p.requires_grad_(False)
            y_loss = self.distill_net(x)
        else:
            y_loss = y

        total_loss = 0.0

        for i in range(self.submodel.max_level() + 1):
            self.submodel.set_level_use(i)
            logits = self(x)
            
            # Handle soft labels for Mixup/CutMix
            if y.ndim == 2:
                loss = SoftTargetCrossEntropy()(logits, y_loss)
                acc = (logits.argmax(1) == y.argmax(1)).float().mean()
            else:
                loss = F.cross_entropy(logits, y_loss, label_smoothing=self.training_context.label_smoothing)
                acc = (logits.argmax(1) == y).float().mean()

            self.log(f"{stage}_level{i}_loss", loss,
                     prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_acc",  acc,
                     prog_bar=(stage != 'train'), sync_dist=True)
            if stage == "train":
                self.manual_backward(loss)
            total_loss += loss.clone().detach()

        self.log(f"{stage}_loss", total_loss, prog_bar=(
            stage != 'train'), sync_dist=True)
        if stage == "train":
            opt.step()

    def on_train_epoch_end(self):
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        self.lr_schedulers().step()

    def training_step(self, b, _) -> torch.Tensor:
        return self._step(b, "train")

    def validation_step(self, b, _) -> torch.Tensor:
        return self._step(b, "val")

    def test_step(self, b, _) -> torch.Tensor:
        return self._step(b, "test")

    def handle_load_from(self):
        if self.training_context.load_from is None:
            return

        from run_experiment import resolve_from_str
        load_from_tconfig: TrainerBuilder = resolve_from_str(
            self.training_context.load_from)
        lmodel = utils.load_model(
            self.training_context.load_from, load_from_tconfig.model_config)

        if self.training_context.load_flex_from is None and self.training_context.load_flex_to is None:
            utils.flexible_model_copy(lmodel, self.submodel)
        elif self.training_context.load_flex_from is None:
            assert (isinstance(self.submodel, fm.Module))
            for l in self.training_context.load_flex_to:
                self.submodel.set_level_use(l)
                utils.flexible_model_copy(lmodel, self.submodel)
        elif self.training_context.load_flex_to is None:
            assert (isinstance(lmodel, fm.Module))
            for l in self.training_context.load_flex_from:
                lmodel.set_level_use(l)
                utils.flexible_model_copy(lmodel, self.submodel)
        else:
            assert (isinstance(self.submodel, fm.Module))
            assert (isinstance(lmodel, fm.Module))
            for lf, lt in zip(self.training_context.load_flex_from, self.training_context.load_flex_to):
                lmodel.set_level_use(lf)
                self.submodel.set_level_use(lt)
                utils.flexible_model_copy(lmodel, self.submodel)

    def handle_pretrained_hf(self):
        name = getattr(self.model_config, 'pretrained_hf_model', None)
        if name is None:
            return
        from networks.flexgpt import FlexGPT
        from networks.flexllama import FlexLLaMA
        if isinstance(self.submodel, FlexGPT):
            utils.load_gpt2_weights_into_flexgpt(self.submodel, name)
        elif isinstance(self.submodel, FlexLLaMA):
            utils.load_llama_weights_into_flexllama(self.submodel, name)
        else:
            raise ValueError(f"No pretrained loader registered for {type(self.submodel).__name__}")

    def handle_distill(self):
        if self.training_context.distill:
            distill_config = self.model_config.no_prebuilt()
            if isinstance(distill_config, FlexModelConfig):
                distill_config = distill_config.create_base_config(
                    self.submodel.max_level())
            self.distill_net = distill_config.make_model()

    def train_loop(self, trainer, conf_description):
        trainer = finetune(
            trainer, self.training_context,
            conf_description, self.model_config)

    def run_training(self, conf_description: str) -> None:
        torch.set_float32_matmul_precision('high')

        self.handle_load_from()
        self.handle_pretrained_hf()
        self.handle_distill()
        self.train_loop(self, conf_description)

        utils.save_model(conf_description, self.submodel)

    def configure_optimizers(self):
        optimizer = self.training_context.make_optimizer(self.submodel)
        scheduler = self.training_context.make_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


class FlexLMTrainer(FlexModelTrainer):
    """
    Trainer function for FlexSLM models without KD
    """

    def _select_levels(self, stage: str) -> list[int]:
        """
        Levels to run this step. Validation/test always cover every level (for
        honest per-level metrics). Training samples num_levels_per_step of them
        when set and smaller than the total - the min and max level are always
        included (never looped over/dropped), with the remaining slots drawn at
        random from the levels in between. num_levels_per_step >= total levels
        (or unset) trains every level, same as looping over all of them.
        """
        all_levels = list(range(self.submodel.max_level() + 1))
        num_sample = getattr(self.training_context, "num_levels_per_step", None)
        if stage != "train" or num_sample is None or num_sample >= len(all_levels):
            return all_levels

        required = {all_levels[0], all_levels[-1]}
        pool = [l for l in all_levels if l not in required]
        num_extra = max(0, num_sample - len(required))
        extra = random.sample(pool, k=min(num_extra, len(pool)))
        return sorted(required | set(extra))

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> None:
        input_ids, _ = batch  # both tensors are identical; labels are derived by shifting

        if stage == "train":
            opt = self.optimizers()
            opt.zero_grad()

        levels = self._select_levels(stage)

        total_loss = 0.0
        vocab_size = self.submodel.config.vocab_size

        for i in levels:
            self.submodel.set_level_use(i)
            logits = self(input_ids)  # [B, S, vocab_size]

            # CLM loss: predict token t+1 from token t at every position
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),  # [B*(S-1), vocab_size]
                input_ids[:, 1:].reshape(-1),            # [B*(S-1)]
            )
            ppl = torch.exp(loss.detach())

            self.log(f"{stage}_level{i}_loss", loss,
                     prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_ppl",  ppl,
                     prog_bar=(stage != "train"), sync_dist=True)

            if stage == "train":
                self.manual_backward(loss)
            total_loss += loss.clone().detach()

        self.log(f"{stage}_loss", total_loss,
                 prog_bar=(stage != "train"), sync_dist=True)

        if stage == "train":
            opt.step()


class FlexLMKDTrainer(FlexLMTrainer):
    """
    FlexLMTrainer extended with knowledge distillation from a frozen teacher model
    """

    def __init__(self, model_config, training_context: FlexTrainingContext):
        super().__init__(model_config, training_context)
        # Store as plain Python attribute, not a PyTorch submodule, so Lightning
        # doesn't include teacher weights in checkpoints.
        object.__setattr__(self, '_teacher', None)

    def _load_teacher(self):
        from transformers import AutoModelForCausalLM
        teacher_name = getattr(self.training_context, 'teacher_hf_model', None) or 'gpt2'
        # bfloat16: teacher is frozen/eval-only, no need for the fp32 default's
        # memory cost (halves weight memory, e.g. ~6.2GB->~3.1GB for a 1.5B teacher).
        teacher = AutoModelForCausalLM.from_pretrained(teacher_name, dtype=torch.bfloat16)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    def _step(self, batch, stage: str) -> None:
        input_ids, _ = batch

        kd_lambda    = self.training_context.kd_lambda
        T            = self.training_context.kd_temperature
        vocab_size   = self.submodel.config.vocab_size

        if stage == "train":
            opt = self.optimizers()
            opt.zero_grad()

        # Teacher forward no grad, done once per batch
        with torch.no_grad():
            teacher_logits = self._teacher(input_ids).logits  # [B, S, V]

        levels = self._select_levels(stage)

        total_loss = 0.0

        for i in levels:
            self.submodel.set_level_use(i)
            logits = self(input_ids)  # [B, S, V]

            # Shift by one position for CLM
            student = logits[:, :-1]              # [B, S-1, V]
            teacher = teacher_logits[:, :-1]      # [B, S-1, V]
            targets = input_ids[:, 1:].reshape(-1)  # [B*(S-1)]

            # CE loss against true tokens
            ce_loss = F.cross_entropy(
                student.reshape(-1, vocab_size),
                targets,
            )

            # KL loss against teacher (soft targets, temperature T)
            kl_loss = F.kl_div(
                F.log_softmax(student / T, dim=-1).reshape(-1, vocab_size),
                F.softmax(teacher / T, dim=-1).reshape(-1, vocab_size),
                reduction="batchmean",
            ) * (T ** 2)

            loss = (1 - kd_lambda) * ce_loss + kd_lambda * kl_loss
            ppl  = torch.exp(ce_loss.detach())

            self.log(f"{stage}_level{i}_loss",    loss,    prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_ce_loss", ce_loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_kl_loss", kl_loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_ppl",     ppl,     prog_bar=(stage != "train"), sync_dist=True)

            if stage == "train":
                self.manual_backward(loss)
            total_loss += loss.clone().detach()

        self.log(f"{stage}_loss", total_loss, prog_bar=(stage != "train"), sync_dist=True)

        if stage == "train":
            opt.step()

    def run_training(self, conf_description: str) -> None:
        object.__setattr__(self, '_teacher', self._load_teacher())
        super().run_training(conf_description)

    def _ensure_teacher(self):
        if self._teacher is None:
            object.__setattr__(self, '_teacher', self._load_teacher())
        object.__setattr__(self, '_teacher', self._teacher.to(self.device))

    def on_fit_start(self):
        self._ensure_teacher()

    def on_test_start(self):
        self._ensure_teacher()


import functools
torch.serialization.add_safe_globals([
    TrainingContext, FlexTrainingContext, FlexModelTrainer, FlexLMTrainer, FlexLMKDTrainer,
    functools.partial, utils.load_wikitext, utils.load_openwebtext, utils.load_fineweb_edu,
    utils.load_data, utils.load_imagenet, utils.load_dummy_data])


class SimpleTrainer(pl.LightningModule, BaseTrainer):
    def __init__(self, model_config: ModelConfig, training_context: TrainingContext) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_context = training_context
        self.submodel = self.model_config.make_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.submodel(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_acc",  acc,
                 prog_bar=(stage != 'train'), sync_dist=True)
        return loss

    def training_step(self, b, _) -> torch.Tensor:
        return self._step(b, "train")

    def validation_step(self, b, _) -> torch.Tensor:
        return self._step(b, "val")

    def test_step(self, b, _) -> torch.Tensor:
        return self._step(b, "test")

    def run_training(self, conf_description: str) -> None:
        torch.set_float32_matmul_precision('high')
        model = self.submodel
        trainer = self

        trainer = finetune(
            trainer, self.training_context,
            conf_description, self.model_config)

        utils.save_model(conf_description, trainer.submodel)

    def configure_optimizers(self):
        optimizer = self.training_context.make_optimizer(self.submodel)
        scheduler = self.training_context.make_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


logger = None


def _print_experiment_banner(conf_description, model_config, config, steps_per_epoch, steps_are_exact) -> None:
    """
    Prints a one-glance summary (config name, model/teacher, dataset size) at the
    top of the job's output log. Rank-0 only, so a 4-GPU DDP job doesn't print it
    4 times - identifying which SLURM_PROCID is rank 0 has to happen the same way
    utils.load_fineweb_edu does it (torch.distributed isn't initialized yet here,
    this runs before trainer.fit()), via the env var SLURM sets per task.
    """
    if int(os.environ.get('SLURM_PROCID', 0)) != 0:
        return

    loader = getattr(config, 'loader_function', None)
    loader_kwargs = getattr(loader, 'keywords', {}) or {}
    pipeline = getattr(getattr(loader, 'func', None), '__name__', 'unknown')
    batch_size = loader_kwargs.get('batch_size')
    max_seq_length = loader_kwargs.get('max_seq_length')

    hidden_dims = getattr(model_config, 'hidden_dims', None)
    num_levels = (len(hidden_dims) if hidden_dims else getattr(model_config, 'max_level', lambda: None)() or 0)
    max_width = hidden_dims[-1] if hidden_dims else None
    pretrained = getattr(model_config, 'pretrained_hf_model', None)

    teacher = getattr(config, 'teacher_hf_model', None)
    kd_lambda = getattr(config, 'kd_lambda', None)

    tokens_per_epoch = (steps_per_epoch * batch_size * max_seq_length
                        if steps_per_epoch and batch_size and max_seq_length else None)
    tokens_label = "tokens/epoch (exact)" if steps_are_exact else "tokens/epoch (estimate)"

    lines = [
        "=" * 88,
        f"EXPERIMENT: {conf_description}",
        f"  Model:    {type(model_config).__name__}  vocab={getattr(model_config, 'vocab_size', '?')}"
        f"  layers={getattr(model_config, 'num_layers', '?')}  levels={num_levels}  max_width={max_width}",
    ]
    if pretrained:
        lines.append(f"  Warm start: {pretrained}")
    if teacher:
        lines.append(f"  KD teacher: {teacher}  (kd_lambda={kd_lambda}, kd_temperature={getattr(config, 'kd_temperature', '?')})")
    lines.append(
        f"  Data:     pipeline={pipeline}  tokenizer={loader_kwargs.get('tokenizer_name', '?')}"
        f"  batch_size={batch_size}  max_seq_length={max_seq_length}"
    )
    if tokens_per_epoch:
        lines.append(f"            {tokens_per_epoch:,} {tokens_label}  ({steps_per_epoch:,} steps/epoch)")
    lines.append(
        f"  Training: epochs={getattr(config, 'epochs', '?')}  patience={getattr(config, 'patience', '?')}"
        f"  resume={getattr(config, 'resume', '?')}  W&B project={getattr(config, 'wandb_project_name', None)}"
    )
    lines.append("=" * 88)
    print("\n".join(lines), flush=True)


def finetune(model: pl.LightningModule, config: TrainingContext, conf_description, model_config) -> pl.LightningModule:
    global logger

    if config.unittest_mode:
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
        logging.getLogger(
            'lightning_fabric.utilities.distributed').setLevel(logging.ERROR)

    # Persistent directory, survives crashes; temp dirs lose weights if anything
    # between trainer.fit() and utils.save_model() raises.
    ckpt_dir = paths.CHECKPOINT_PATH / utils.make_str_filename_safe(conf_description)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resume from a prior run's checkpoint if one exists at this path, checked
    # before trainer.fit() so an interrupted/preempted job picks back up (full
    # training state - weights, optimizer, epoch/step counters) instead of
    # silently restarting from the warm-start/random init every time. Resumes
    # from last.ckpt (always saved, see save_last=True below), not best-model.ckpt -
    # resuming should continue from the most recent point, not roll back to
    # whichever earlier step happened to have the best val_loss so far.
    resume_ckpt_path = ckpt_dir / 'last.ckpt'
    resume_from = str(resume_ckpt_path) if (config.resume and resume_ckpt_path.exists()) else None
    if resume_from:
        logging.info(f"Found existing checkpoint at {resume_from}, resuming training from it.")
    elif config.resume is False and resume_ckpt_path.exists():
        logging.info(f"resume=False - ignoring existing checkpoint at {resume_ckpt_path}, starting fresh.")

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=config.patience, mode='min', verbose=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best-model',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,  # always overwrite last.ckpt too, even when val_loss doesn't improve - this is what resume uses
    )

    callbacks = [early_stopping, checkpoint_callback]

    kwargs = dict()
    if config.wandb_project_name is not None:
        if logger is None:
            num_levels = model_config.max_level() + 1
            wandb_config = dict(model_config.get_flat_dict())
            wandb_config['num_levels'] = num_levels
            wandb_config['num_levels_per_step'] = getattr(config, 'num_levels_per_step', None) or num_levels
            logger = WandbLogger(
                project=config.wandb_project_name,
                name=f"{conf_description}_das6",
                config=wandb_config,
                save_dir=paths.LOG_PATH,
                dir=paths.LOG_PATH,
                log_model=False)
        kwargs['logger'] = logger
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    else:
        kwargs['logger'] = False

    if config.unittest_mode:
        kwargs['enable_progress_bar'] = False
        kwargs['enable_model_summary'] = False

    train_loader, val_loader, test_loader = config.loader_function()

    # Space checkpoints through the epoch (every ~1/20th of it) rather than only
    # at epoch end - for a streaming IterableDataset (unknown length) an "epoch"
    # only ends once the whole requested corpus is exhausted, which for a large
    # corpus could be hours; without this, a preempted job could have nothing to
    # resume from at all. len() works for ordinary (materialized) loaders;
    # streaming loaders instead carry an estimate (see utils.load_fineweb_edu).
    try:
        steps_per_epoch = len(train_loader)
        steps_are_exact = True
    except TypeError:
        steps_per_epoch = getattr(train_loader, 'estimated_steps_per_epoch', None)
        steps_are_exact = False
    val_check_interval = max(1, steps_per_epoch // 20) if steps_per_epoch else 1.0

    _print_experiment_banner(conf_description, model_config, config, steps_per_epoch, steps_are_exact)

    if config.use_fsdp:
        # FULL_SHARD (ZeRO-3 equivalent): params/grads/optimizer state sharded
        # across GPUs instead of fully replicated per-rank like DDP. state_dict_type
        # stays 'full' (the default) so checkpoints remain a single unsharded
        # state_dict - compatible with the existing resume/save_model/warm-start
        # loading code, which all expect that format.
        strategy = FSDPStrategy(process_group_backend='nccl')
    else:
        strategy = DDPStrategy(process_group_backend='nccl', find_unused_parameters=True)
    trainer = pl.Trainer(
        **kwargs,
        max_epochs=config.epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=val_check_interval,
        enable_checkpointing=True,
        accelerator="gpu",
        devices="auto",
        num_nodes=utils.get_num_nodes(),
        strategy=strategy,
        precision='bf16-mixed'
    )

    try:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)
    except Exception as exc:
        # Training may have fully completed before a post-epoch callback (e.g.
        # scheduler step) raised.  Try to recover and permanently save the best
        # checkpoint so the run is not completely lost.
        best_path = getattr(checkpoint_callback, 'best_model_path', '')
        if best_path:
            logging.warning(
                f"trainer.fit() raised {exc!r}. Attempting recovery from {best_path}")
            try:
                recovered = type(model).load_from_checkpoint(best_path)
                utils.save_model(conf_description, recovered.submodel)
                logging.warning("Recovery succeeded. Weights saved to permanent storage.")
            except Exception as rec_exc:
                logging.error(
                    f"Recovery failed: {rec_exc!r}. Checkpoint preserved at: {best_path}")
        raise

    # Load best checkpoint and save immediately, before test() can crash.
    if utils.get_num_nodes() > 1:
        if trainer.is_global_zero:
            model = type(model).load_from_checkpoint(
                checkpoint_callback.best_model_path)
            utils.save_model(conf_description, model.submodel)
    else:
        model = type(model).load_from_checkpoint(
            checkpoint_callback.best_model_path)
        utils.save_model(conf_description, model.submodel)

    # Testing is informational, a failure here must not erase already-saved weights.
    try:
        trainer.test(model, dataloaders=test_loader, verbose=False)
    except Exception as exc:
        logging.warning(f"trainer.test() failed (weights already saved): {exc!r}")

    return model
