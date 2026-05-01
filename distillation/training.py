# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
from timm import create_model
import torch
from torch.nn import functional as F
import timm
import math
import sys
from typing import Iterable, Optional
import torch.nn as nn
import torch
from torch.nn import functional as F
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import random
import dataclasses
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import config.paths as paths
from typing import Callable
import logging
import tempfile

import utils
from training import TrainerBuilder, FlexModelConfig, ModelConfig, BaseTrainer, TrainingContext


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                # We provide the teacher's targets in log probability because we use log_target=True
                # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            # We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # But we also experiments output_kd.size(0)
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


def load_teacher_model(model_name, model_path, num_classes=1000):
    teacher_model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )
    if model_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            model_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    teacher_model.load_state_dict(checkpoint['model'])
    teacher_model.to(utils.get_device())
    teacher_model.eval()

def load_teacher():
    """
    Simple loader for a teacher model with a given architecture.

    Returns
    -------
    model : torch.nn.Module
    Teacher model loaded with the state dict and moved to device in eval mode.
    """
    # Create a timm ViT without distillation

    ### REMINDER: add own model!!!!!

    model = timm.create_model('deit3_base_patch16_224.fb_in22k_ft_in1k', pretrained=True, num_classes=1000)
    model.eval().to(utils.get_device())
    return model

@dataclasses.dataclass
class ScalaDistillContext(TrainingContext):
    teacher_loader: Callable[[], nn.Module] = lambda: load_teacher()
    ce_coefficient: float = 1.0
    transfer_type: str = 'progressive'
    distillation_type: str = 'none'
    distill_type: str = 'soft'
    ce_type: str = 'one_hot'
    token_type: str = 'cls_token'
    mixup_fn: Optional[Callable[[torch.Tensor, torch.Tensor],
                                tuple[torch.Tensor, torch.Tensor]]] = None
    cosub: bool = False
    base_criterion: Callable[[], nn.Module] = nn.CrossEntropyLoss
    distillation_alpha: float = 0.5
    distillation_tau: float = 1.0

    make_optimizer: Callable[[nn.Module], torch.optim.Optimizer] = None
    make_scheduler: Callable[[torch.optim.Optimizer],
                             torch.optim.lr_scheduler.LRScheduler] = None


class ScalaDistillTrainer(pl.LightningModule, BaseTrainer):
    def __init__(self, model_config: FlexModelConfig, training_context: ScalaDistillContext) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model_config', 'training_context'])
        self.model_config = model_config
        self.training_context = training_context
        self.submodel = self.model_config.make_model()
        self.automatic_optimization = False

        self.ce_coefficient = training_context.ce_coefficient
        self.transfer_type = training_context.transfer_type
        self.distillation_type = training_context.distillation_type
        self.distill_type = training_context.distill_type
        self.ce_type = training_context.ce_type
        self.token_type = training_context.token_type
        self.mixup_fn = training_context.mixup_fn
        self.cosub = training_context.cosub

        teacher = training_context.teacher_loader()
        teacher.to(utils.get_device())

        self.criterion = DistillationLoss(
            training_context.base_criterion(),
            teacher,
            self.distillation_type,
            training_context.distillation_alpha,
            training_context.distillation_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.submodel(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _):
        samples, targets = batch
        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)

        # Did not include cosub option

        opt = self.optimizers()
        opt.zero_grad()

        # Did not include warmup

        # full model
        self.submodel.set_level_use(self.submodel.max_level())
        outputs = self.submodel(samples)
        loss_full = self.criterion(samples, outputs, targets)
        self.manual_backward(loss_full)
        loss_full = loss_full.clone().detach()

        if self.token_type == 'dist_token':
            token = int(1)
        elif self.token_type == 'cls_token':
            token = int(0)

        if self.ce_type == 'one_hot':
            ce_targets = targets
        elif self.ce_type == 'teacher':
            ce_targets = outputs[0].detach().argmax(dim=1)

        ce_loss = torch.nn.CrossEntropyLoss()

        # 3q model
        level_3q = random.randint(1, self.submodel.max_level() // 2)
        self.submodel.set_level_use(level_3q)
        output_3q = self.submodel(samples)
        if self.distillation_type == 'none':
            if self.distill_type == 'hard':
                loss_3q = F.cross_entropy(
                    output_3q, outputs.detach().argmax(dim=1))
            else:
                loss_3q = nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(
                    dim=1)(output_3q), nn.Softmax(dim=1)(outputs.detach()))
            loss_3q_ce = self.ce_coefficient * \
                ce_loss(output_3q, ce_targets)
            loss_3q = loss_3q + loss_3q_ce
        else:
            if self.distill_type == 'hard':
                loss_3q = F.cross_entropy(
                    output_3q[token], outputs[token].detach().argmax(dim=1))
            else:
                loss_3q = nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(
                    output_3q[token]), nn.Softmax(dim=1)(outputs[token].detach()))
            loss_3q_ce = self.ce_coefficient * \
                ce_loss(output_3q[0], ce_targets)
            loss_3q = loss_3q + loss_3q_ce

        self.manual_backward(loss_3q)
        loss_3q = loss_3q.clone().detach()
        loss_3q_ce = None

        # 2q model
        if self.transfer_type == 'US':
            teacher_2q = outputs
        else:
            teacher_2q = output_3q

        level_2q = random.randint(
            self.submodel.max_level() // 2 + 1, self.submodel.max_level() - 1)
        self.submodel.set_level_use(level_2q)
        output_2q = self.submodel(samples)
        if self.distillation_type == 'none':
            if self.distill_type == 'hard':
                loss_2q = F.cross_entropy(
                    output_2q, teacher_2q.detach().argmax(dim=1))
            else:
                loss_2q = nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(
                    dim=1)(output_2q), nn.Softmax(dim=1)(teacher_2q.detach()))
            loss_2q_ce = self.ce_coefficient * \
                ce_loss(output_2q, ce_targets)
            loss_2q = loss_2q + loss_2q_ce
        else:
            if self.distill_type == 'hard':
                loss_2q = F.cross_entropy(
                    output_2q[token], teacher_2q[token].detach().argmax(dim=1))
            else:
                loss_2q = nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(
                    output_2q[token]), nn.Softmax(dim=1)(teacher_2q[token].detach()))
            loss_2q_ce = self.ce_coefficient * \
                ce_loss(output_2q[0], ce_targets)
            loss_2q = loss_2q + loss_2q_ce

        self.manual_backward(loss_2q)
        loss_2q = loss_2q.clone().detach()
        loss_2q_ce = None

        # 1q model
        if self.transfer_type == 'US':
            teacher_1q = outputs
        else:
            teacher_1q = output_2q
        self.submodel.set_level_use(0)
        output_1q = self.submodel(samples)
        if self.distillation_type == 'none':
            if self.distill_type == 'hard':
                loss_1q = F.cross_entropy(
                    output_1q, teacher_1q.detach().argmax(dim=1))
            else:
                loss_1q = nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(
                    dim=1)(output_1q), nn.Softmax(dim=1)(teacher_1q.detach()))
            loss_1q_ce = self.ce_coefficient * \
                ce_loss(output_1q, ce_targets)
            loss_1q = loss_1q + loss_1q_ce
        else:
            if self.distill_type == 'hard':
                loss_1q = F.cross_entropy(
                    output_1q[token], teacher_1q[token].detach().argmax(dim=1))
            else:
                loss_1q = nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(
                    output_1q[token]), nn.Softmax(dim=1)(teacher_1q[token].detach()))
            loss_1q_ce = self.ce_coefficient * \
                ce_loss(output_1q[0], ce_targets)
            loss_1q = loss_1q + loss_1q_ce

        self.manual_backward(loss_1q)
        loss_1q = loss_1q.clone().detach()
        loss_1q_ce = None

        opt.step()

        loss = loss_full + loss_3q + loss_2q + loss_1q

        self.log(
            f"train_level{self.submodel.max_level()}_loss", loss_full, sync_dist=True)
        self.log(f"train_level{level_3q}_loss", loss_3q, sync_dist=True)
        self.log(f"train_level{level_2q}_loss", loss_2q, sync_dist=True)
        self.log(f"train_level0_loss", loss_1q, sync_dist=True)
        self.log("train_loss", loss, sync_dist=True)

        
    def on_train_epoch_end(self):
        # Log the learning rate at the end of each epoch
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        self.lr_schedulers().step(self.current_epoch)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        x, y = batch
        total_loss = 0.0

        for i in range(self.submodel.max_level() + 1):
            self.submodel.set_level_use(i)
            logits = self(x)
            loss = F.cross_entropy(
                logits, y, label_smoothing=self.training_context.label_smoothing)
            acc = (logits.argmax(1) == y).float().mean()
            self.log(f"val_level{i}_loss", loss,
                     prog_bar=False, sync_dist=True)
            self.log(f"val_level{i}_acc",  acc, prog_bar=False, sync_dist=True)
            total_loss += loss.clone().detach()

        self.log(f"val_loss", total_loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        x, y = batch
        total_loss = 0.0

        for i in range(self.submodel.max_level() + 1):
            self.submodel.set_level_use(i)
            logits = self(x)
            loss = F.cross_entropy(
                logits, y, label_smoothing=self.training_context.label_smoothing)
            acc = (logits.argmax(1) == y).float().mean()
            self.log(f"test_level{i}_loss", loss,
                     prog_bar=False, sync_dist=True)
            self.log(f"test_level{i}_acc",  acc, prog_bar=False, sync_dist=True)
            total_loss += loss.clone().detach()

        self.log(f"test_loss", total_loss, prog_bar=True, sync_dist=True)

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


def finetune(model: pl.LightningModule, config: TrainingContext, conf_description, model_config) -> pl.LightningModule:

    if config.unittest_mode:
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
        logging.getLogger(
            'lightning_fabric.utilities.distributed').setLevel(logging.ERROR)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=config.patience, mode='min', verbose=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=paths.CHECKPOINT_PATH,
        filename=f"{conf_description}_best_model",
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    callbacks = [early_stopping, checkpoint_callback]

    kwargs = dict()
    if config.wandb_project_name is not None:
        logger = WandbLogger(
            project=config.wandb_project_name,
            name=f"{conf_description}_ivi",
            config=model_config.get_flat_dict(),
            save_dir=paths.LOG_PATH,
            dir=paths.LOG_PATH,
            log_model=False,
            offline=True)
        kwargs['logger'] = logger
    else:
        kwargs['logger'] = False

    if config.unittest_mode:
        kwargs['enable_progress_bar'] = False
        kwargs['enable_model_summary'] = False

    ddp = DDPStrategy(process_group_backend='nccl', find_unused_parameters=True)
    trainer = pl.Trainer(
        **kwargs,
        max_epochs=config.epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=True,
        accelerator="gpu",
        devices="auto",
        num_nodes=utils.get_num_nodes(),
        strategy=ddp,
        precision='bf16-mixed'
    )

    train_loader, val_loader, test_loader = config.loader_function()
    trainer.fit(model, train_loader, val_loader)

    if utils.get_num_nodes() > 1:
        if trainer.is_global_zero:
            model = type(model).load_from_checkpoint(
                    checkpoint_callback.best_model_path, training_context=config, model_config=model_config)
        
    else:
        model = type(model).load_from_checkpoint(
                checkpoint_callback.best_model_path, training_context=config, model_config=model_config)
    trainer.test(model, dataloaders=test_loader, verbose=False)

    return model
