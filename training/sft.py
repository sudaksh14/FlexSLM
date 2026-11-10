"""
Supervised fine-tuning of a FlexLLaMA on question/answer pairs. Same FlexSLM
training (all levels, CE + KD) as FlexLMKDTrainer, with CE on the answer tokens
only. Two kinds of runs:
  - from a trained FlexLLaMA checkpoint (sft.medqa_full5ep: data/medqa_sft from
    prepare_medqa_sft.py, starts from saved_models/<pmc.full5ep>.pt);
  - from Hugging Face weights (sft.medmix_qwen3_1p7b: Qwen3-1.7B backbone, FaithMed's
    medmix data from prepare_medmix_sft.py and FaithMed's SFT schedule).

    python -m training.sft check      # login node, no GPU: validates config, weights and data
"""
import dataclasses
import functools
import json
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset

import config.paths as paths
import utils
from training import FlexLMKDTrainer, FlexTrainingContext, TrainerBuilder
from training.importance_order import FINAL_ORDERING, start_weights
from training.trainer import check_level_sampling, warmup_cosine_scheduler
from training.level_weights import LevelWeightingMixin, check_args as check_level_weighting

IGNORE_INDEX = -100
SFT_DATA_DIR = os.path.join("data", "medqa_sft")

# FaithMed's SFT recipe (train_sft.yaml, paper Sec. 5): LR 5e-5, cosine with a 1%
# linear warmup, effective batch 32 (8 per GPU x 4 GPUs), one epoch as in the paper,
# on the Qwen3-1.7B backbone with FaithMed's medmix data.
_QWEN3_FAITHMED = dict(
    hf_preset="qwen3-1.7b", data_dir=os.path.join("data", "medmix_sft"),
    lr=5e-5, warmup_ratio=0.01, epochs=1, patience=1000, batch_size=8,
    max_seq_length=3072, activation_checkpointing=True)

# Corrected setup for the paper runs (2026-11-09): the GQA-preserving level
# builder (presets with a head_dim), gradient-norm clipping at 1.0 (the HF
# Trainer / LLaMA-Factory default that FaithMed's recipe runs with; the runs
# above did not clip) and one fixed seed. The effective batch is 32 sequences,
# i.e. batch_size x number of GPUs: 16 x 2 on node205 (2x A40).
# micro_batch_size 8: job 28075 (1.7B, 16 x 3,072 per GPU in one pass) ran out of
# memory at step 731 (43.5 GB of 44.7 GB on an A40); two passes of 8 give the same step.
_PAPER = dict(_QWEN3_FAITHMED, batch_size=16, micro_batch_size=8, gradient_clip_val=1.0, seed=0)
_FT_TEACHER_1P7B = os.path.join(
    "hf_export", utils.make_str_filename_safe("flexllama,sft.paper.qwen3_1p7b_teacher"), "level2")
# Stage 2 (nested SFT of Qwen3-0.6B): the paper schedule, KD (lambda 0.5, T 2)
# from the fine-tuned 1.7B, all levels every step. hf_preset None: the model and
# its start weights come from the Stage-1 config named in SFT_RUNS.
_S2 = dict(_PAPER, hf_preset=None, teacher_hf_model=_FT_TEACHER_1P7B,
           tokenizer_name="Qwen/Qwen3-0.6B", kd_lambda=0.5, kd_temperature=2.0,
           num_levels_per_step=None)

# (new config name, config it starts from, overrides). A start config of None
# warm-starts from Hugging Face weights instead (see _hf_base).
SFT_RUNS = [
    ("sft.medqa_full5ep", "pmc.full5ep", {}),
    # KD from the frozen, untrained Qwen3-1.7B, which answers in prose rather than
    # in the boxed format of the targets.
    ("sft.medmix_qwen3_1p7b", None, dict(_QWEN3_FAITHMED,
        wandb_project_name="FlexLLaMA_sft_medmix_qwen3")),
    # Stage 1, the teacher: plain SFT of the full Qwen3-1.7B (max level only, no KD),
    # i.e. FaithMed's SFT itself.
    ("sft.medmix_qwen3_1p7b_teacher", None, dict(_QWEN3_FAITHMED,
        kd_lambda=0.0, train_levels=[2],
        wandb_project_name="FlexLLaMA_sft_medmix_qwen3_teacher")),
    # Stage 2: the FlexSLM run above, unchanged except that KD now comes from the
    # stage-1 teacher (exported with export_flex_hf.py), which answers in \boxed{}.
    ("sft.medmix_qwen3_1p7b_kdteacher", None, dict(_QWEN3_FAITHMED,
        teacher_hf_model=os.path.join("hf_export", "flexxxxllamaxnsftxmmedmixxxx_qwen3_1p7b_teacher", "level2"),
        tokenizer_name="Qwen/Qwen3-1.7B",
        wandb_project_name="FlexLLaMA_sft_medmix_qwen3_kdteacher")),
    # The runs above used KD on the answer tokens only. The two below use KD on
    # every token, as in FlexLMKDTrainer (new names, so earlier results are kept).
    ("sft.medqa_full5ep_flexslm", "pmc.full5ep", dict(
        wandb_project_name="FlexLLaMA_sft_medqa_flexslm")),
    ("sft.medmix_qwen3_1p7b_flexslm", None, dict(_QWEN3_FAITHMED,
        teacher_hf_model=os.path.join("hf_export", "flexxxxllamaxnsftxmmedmixxxx_qwen3_1p7b_teacher", "level2"),
        tokenizer_name="Qwen/Qwen3-1.7B",
        wandb_project_name="FlexLLaMA_sft_medmix_qwen3_flexslm")),

    # ---- Paper runs (IEA/AIE 2027), corrected code ----
    # Conventional fine-tuning of the full Qwen3-1.7B (max level only, no KD).
    # It is both the 1.7B "conventional FT" baseline and, exported with
    # export_flex_hf.py to _FT_TEACHER_1P7B, the fine-tuned teacher for the
    # Qwen3-0.6B Stage-2 runs (the KD teacher is always larger than the student).
    ("sft.paper.qwen3_1p7b_teacher", None, dict(_PAPER,
        hf_preset="qwen3-1.7b-gqa", kd_lambda=0.0, train_levels=[2],
        level_fractions=(0.5, 0.75, 1.0),
        wandb_project_name="MedNest_qwen3_1p7b_teacher")),
    # Conventional fine-tuning of the full Qwen3-0.6B: the 0.6B baseline.
    ("sft.paper.qwen3_0p6b_conventional", None, dict(_PAPER,
        hf_preset="qwen3-0.6b", kd_lambda=0.0, train_levels=[2],
        level_fractions=(0.5, 0.75, 1.0),
        wandb_project_name="MedNest_qwen3_0p6b_conventional")),
    # M3, Stage 2: nested SFT of Qwen3-0.6B (all levels every step) starting from
    # the Stage-1 result, KD from the fine-tuned 1.7B (T1 export). The variants
    # below change one thing each and are always full runs (the whole SFT set).
    ("sft.paper.qwen3_0p6b_nested", "s1.qwen3_0p6b.full", dict(_S2,
        wandb_project_name="MedNest_s2_nested")),
    # A4: level weighting, 5-way with the equal-weight M3 run above.
    ("sft.paper.qwen3_0p6b_nested.w_fixed", "s1.qwen3_0p6b.full", dict(_S2,
        level_weighting="fixed", level_weights_fixed=(3.0, 1.0, 1.0),
        wandb_project_name="MedNest_s2_w_fixed")),
    ("sft.paper.qwen3_0p6b_nested.w_uncertainty", "s1.qwen3_0p6b.full", dict(_S2,
        level_weighting="uncertainty", wandb_project_name="MedNest_s2_w_uncertainty")),
    ("sft.paper.qwen3_0p6b_nested.w_lossprop", "s1.qwen3_0p6b.full", dict(_S2,
        level_weighting="loss_proportional", wandb_project_name="MedNest_s2_w_lossprop")),
    ("sft.paper.qwen3_0p6b_nested.w_dwa", "s1.qwen3_0p6b.full", dict(_S2,
        level_weighting="dwa", wandb_project_name="MedNest_s2_w_dwa")),
    # A5: one uniformly random level per step (MatFormer) instead of all levels.
    ("sft.paper.qwen3_0p6b_nested.single", "s1.qwen3_0p6b.full", dict(_S2,
        level_sampling="single", wandb_project_name="MedNest_s2_single")),
    # A2: no Stage 1 (warm start straight from Qwen3-0.6B with the final recipe's ordering).
    ("sft.paper.qwen3_0p6b_nested.no_stage1", None, dict(_S2,
        hf_preset="qwen3-0.6b", level_fractions=(0.5, 0.75, 1.0), hf_start=start_weights(FINAL_ORDERING),
        wandb_project_name="MedNest_s2_no_stage1")),
    # A1: no importance ordering (Stage 1 and Stage 2 both without it).
    ("sft.paper.qwen3_0p6b_nested.no_order", "s1.qwen3_0p6b.full.no_order", dict(_S2,
        wandb_project_name="MedNest_s2_no_order")),
    # A3: no KD in Stage 2 (CE on the answers only).
    ("sft.paper.qwen3_0p6b_nested.no_kd", "s1.qwen3_0p6b.full", dict(_S2,
        kd_lambda=0.0, wandb_project_name="MedNest_s2_no_kd")),
    # Baseline B2: MatFormer-style nested fine-tuning of the same three levels: one
    # uniformly random level per step, CE only, no ordering, no Stage 1, no KD.
    # (MatFormer nests only the FFN, which cannot reach a 0.19B level of Qwen3-0.6B.)
    ("sft.paper.qwen3_0p6b_matformer", None, dict(_PAPER,
        hf_preset="qwen3-0.6b", kd_lambda=0.0, level_fractions=(0.5, 0.75, 1.0),
        level_sampling="single", wandb_project_name="MedNest_b2_matformer")),
    # Development only (smoke tests of the level_weighting flag, not a paper run):
    # nested SFT of Qwen3-0.6B, every level, no KD, no importance ordering.
    ("sft.dev.qwen3_0p6b_nested_uncertainty", None, dict(_PAPER,
        hf_preset="qwen3-0.6b", kd_lambda=0.0, level_fractions=(0.5, 0.75, 1.0),
        level_weighting="uncertainty",
        wandb_project_name="MedNest_dev_level_weighting")),
]


class QADataset(Dataset):
    """Pre-tokenized (prompt_ids, answer_ids) pairs from a {prompt, answer} jsonl."""

    def __init__(self, path: str, tokenizer_name: str, max_seq_length: int):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        prompts = tokenizer([r["prompt"] for r in rows], add_special_tokens=False)["input_ids"]
        answers = tokenizer([r["answer"] for r in rows], add_special_tokens=False)["input_ids"]

        self.items = []
        self.num_truncated = 0
        for p, a in zip(prompts, answers):
            a = a[:max_seq_length // 2]
            if len(p) + len(a) > max_seq_length:
                # Same as eval: keep the end of the prompt (the "Answer:" cue).
                p = p[-(max_seq_length - len(a)):]
                self.num_truncated += 1
            self.items.append((p, a))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_qa(batch, pad_id: int = 0):
    """Right-pads to the longest sequence in the batch. Returns input ids, labels
    (the input ids on answer positions, IGNORE_INDEX on prompt and padding) and a
    mask of the real (prompt + answer, non-padding) positions. Right padding needs
    no attention mask: attention is causal, so real tokens never attend to the
    padding that follows them."""
    length = max(len(p) + len(a) for p, a in batch)
    input_ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), length), IGNORE_INDEX, dtype=torch.long)
    real = torch.zeros((len(batch), length), dtype=torch.bool)
    for row, (p, a) in enumerate(batch):
        ids = torch.tensor(p + a, dtype=torch.long)
        input_ids[row, :len(ids)] = ids
        labels[row, len(p):len(ids)] = ids[len(p):]
        real[row, :len(ids)] = True
    return input_ids, labels, real


def load_qa_sft(data_dir: str = SFT_DATA_DIR, tokenizer_name: str = "JackFram/llama-160m",
                batch_size: int = 16, max_seq_length: int = 1024, **_ignored):
    train = QADataset(os.path.join(data_dir, "train.jsonl"), tokenizer_name, max_seq_length)
    val = QADataset(os.path.join(data_dir, "val.jsonl"), tokenizer_name, max_seq_length)
    shuffle = True
    if os.environ.get("FLEXSLM_SMOKE_LONGEST"):
        # Smoke test of peak memory: the longest examples first, unshuffled.
        train.items.sort(key=lambda pa: -(len(pa[0]) + len(pa[1])))
        shuffle = False
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle,
                              num_workers=0, collate_fn=collate_qa)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_qa)
    # The benchmark test sets are evaluated separately with eval_job.sh; the
    # held-out SFT split doubles as the test loader here.
    return train_loader, val_loader, val_loader


@dataclasses.dataclass
class SFTTrainingContext(FlexTrainingContext):
    kd_lambda: float = 0.5
    kd_temperature: float = 2.0
    teacher_hf_model: str = "JackFram/llama-160m"
    num_levels_per_step: int = None
    lr: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = None
    activation_checkpointing: bool = False
    train_levels: list = None
    # Per-level loss weights (training/level_weights.py, which documents them):
    # "none" | "fixed" | "uncertainty" (NSN / Kendall) | "loss_proportional" | "dwa".
    level_weighting: str = "none"
    level_weight_lr: float = 1e-2     # uncertainty: Adam lr of the log-variances
    level_weight_ema: float = 0.9     # loss_proportional, dwa: loss EMA decay
    level_weights_fixed: tuple = None # fixed: one weight per level, smallest first
    dwa_window: int = 50
    dwa_temperature: float = 2.0
    # "sandwich" (default: every level, or num_levels_per_step with min and max
    # always in) | "single" (one uniformly random level per step, MatFormer-style).
    level_sampling: str = "sandwich"
    # Sequences per forward/backward pass inside one optimizer step (None = whole
    # batch). Exact: losses are normalised over the whole batch.
    micro_batch_size: int = None

    def __init__(self, load_from: str, teacher_hf_model: str, tokenizer_name: str,
                 kd_lambda=0.5, kd_temperature=2.0, num_levels_per_step=None,
                 lr=2e-5, weight_decay=0.0, epochs=3, patience=5,
                 batch_size=16, max_seq_length=1024, data_dir=SFT_DATA_DIR,
                 warmup_ratio=None, activation_checkpointing=False, train_levels=None,
                 level_weighting="none", level_weight_lr=1e-2, level_weight_ema=0.9,
                 level_weights_fixed=None, dwa_window=50, dwa_temperature=2.0,
                 level_sampling="sandwich", micro_batch_size=None, **kwargs):
        loader = functools.partial(load_qa_sft, data_dir=data_dir, tokenizer_name=tokenizer_name,
                                   batch_size=batch_size, max_seq_length=max_seq_length)
        FlexTrainingContext.__init__(self, loader, patience=patience, epochs=epochs,
                                     load_from=load_from, **kwargs)
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature
        self.teacher_hf_model = teacher_hf_model
        self.num_levels_per_step = num_levels_per_step
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.activation_checkpointing = activation_checkpointing
        self.train_levels = train_levels
        check_level_weighting(level_weighting, level_weight_lr, level_weight_ema,
                              level_weights_fixed, dwa_window, dwa_temperature)
        check_level_sampling(level_sampling)
        self.level_weighting = level_weighting
        self.level_weight_lr = level_weight_lr
        self.level_weight_ema = level_weight_ema
        self.level_weights_fixed = tuple(level_weights_fixed) if level_weights_fixed else None
        self.dwa_window = dwa_window
        self.dwa_temperature = dwa_temperature
        self.level_sampling = level_sampling
        if micro_batch_size is not None and (int(micro_batch_size) != micro_batch_size or micro_batch_size < 1):
            raise ValueError(f"micro_batch_size must be a positive integer, got {micro_batch_size!r}")
        self.micro_batch_size = micro_batch_size

    def make_optimizer(self, model):
        return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def make_scheduler(self, optimizer):
        # Stepped once per epoch (FlexModelTrainer.on_train_epoch_end).
        return CosineAnnealingLR(optimizer, T_max=max(1, self.epochs), eta_min=self.lr * 0.1)

    def make_step_scheduler(self, optimizer, total_steps: int):
        """Linear warmup over the first warmup_ratio of the steps, then cosine to 0,
        stepped after every optimizer step: the schedule HF Trainer (and so
        LLaMA-Factory) uses for lr_scheduler_type=cosine with warmup_ratio."""
        return warmup_cosine_scheduler(optimizer, total_steps, self.warmup_ratio)


def trained_model_path(config_name: str):
    return utils.resolve_named_path(paths.TRAINED_MODELS, config_name, ".pt")


class FlexLMSFTTrainer(LevelWeightingMixin, FlexLMKDTrainer):
    """FlexLMKDTrainer for question/answer data. The KD term is FlexSLM's own: over
    every real token, as in FlexLMKDTrainer, with the same lambda and T. Only CE is
    restricted to the answer tokens, the SFT convention (FaithMed's
    train_on_prompt: false). The loss is evaluated in row chunks into a detached
    proxy of the logits, as in training_memopt, so a 152k vocab fits in memory.
    With level_weighting set, each level's training loss is scaled by its weight
    (training/level_weights.py); logged losses and val_loss stay unweighted."""

    def handle_load_from(self):
        if self.training_context.load_from is None:
            return
        path = trained_model_path(self.training_context.load_from)
        self.submodel.load_state_dict(torch.load(path, map_location="cpu"))
        logging.info(f"SFT: loaded starting weights from {path}")

    def handle_pretrained_hf(self):
        # Starting from a trained checkpoint, the model config may still name the
        # original warm-start checkpoint; loading it here would overwrite the
        # weights from handle_load_from.
        if self.training_context.load_from is None:
            super().handle_pretrained_hf()

    def on_fit_start(self):
        super().on_fit_start()
        if self.training_context.activation_checkpointing:
            from training.memopt import enable_activation_checkpointing
            enable_activation_checkpointing(self.submodel)

    def _select_levels(self, stage: str) -> list[int]:
        # train_levels trains (and validates, so val_loss and the best checkpoint
        # follow the same model) a fixed subset of levels, e.g. [2] for the plain
        # SFT of the full model that serves as a KD teacher.
        levels = getattr(self.training_context, "train_levels", None)
        return list(levels) if levels else super()._select_levels(stage)

    def configure_optimizers(self):
        ctx = self.training_context
        if ctx.warmup_ratio is None:
            return super().configure_optimizers()
        optimizer = ctx.make_optimizer(self.submodel)
        scheduler = ctx.make_step_scheduler(optimizer, int(self.trainer.estimated_stepping_batches))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_train_epoch_end(self):
        if self.training_context.warmup_ratio is None:
            return super().on_train_epoch_end()
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=True, sync_dist=True)

    def _hidden_rows(self, input_ids, rows):
        # Final hidden states at the selected prediction positions, so the LM head
        # (and the [N, V] logits) only covers real tokens, never padding.
        m = self.submodel
        hidden = m.norm(m.layers(m.embed_tokens(input_ids)))
        return hidden[:, :-1][rows]

    def _step(self, batch, stage: str) -> None:
        input_ids, labels, real = batch

        kd_lambda = self.training_context.kd_lambda
        T = self.training_context.kd_temperature
        chunk = getattr(self.training_context, "loss_chunk_size", None) or 1024
        training = stage == "train"

        targets = labels[:, 1:]          # IGNORE_INDEX except on answer tokens
        answer = targets != IGNORE_INDEX
        # Prediction positions: every real next token for KD (FlexLMKDTrainer's
        # positions, minus padding); with no KD, only the answer tokens CE uses.
        rows = real[:, 1:] if kd_lambda > 0 else answer
        # Normalisers over the whole batch, so micro-batches sum to the same loss
        # (and the same gradient) as one pass over the batch.
        n_ce = int(answer.sum())
        n_kd = int(rows.sum())

        if training:
            opt = self.optimizers()
            opt.zero_grad()

        levels = self._select_levels(stage)
        # Weight of each level's loss in the backward pass only (1.0 when off).
        weights = {i: self._level_weight(i, training) for i in levels}
        zero = torch.zeros((), device=input_ids.device)
        ce_loss = {i: zero for i in levels}
        kl_loss = {i: zero for i in levels}
        correct = {i: 0 for i in levels}

        # micro_batch_size splits the batch by sequence: the [N, V] logits and their
        # gradient (about 30 GB for 16 x 3,072 tokens and Qwen3's 151,936-token vocab)
        # only exist for one micro-batch at a time.
        bsz = input_ids.shape[0]
        mb = getattr(self.training_context, "micro_batch_size", None) or bsz
        for lo in range(0, bsz, mb):
            mb_ids, mb_rows = input_ids[lo:lo + mb], rows[lo:lo + mb]
            row_targets = targets[lo:lo + mb][mb_rows]   # IGNORE_INDEX on prompt rows: CE skips them
            n_rows = int(mb_rows.sum())
            if n_rows == 0:
                continue
            teacher_rows = None
            if kd_lambda > 0:
                with torch.no_grad():
                    teacher_rows = self._teacher.model(mb_ids).last_hidden_state[:, :-1][mb_rows]

            for i in levels:
                self.submodel.set_level_use(i)
                logits = self.submodel.lm_head(self._hidden_rows(mb_ids, mb_rows))  # [N, V]
                proxy = logits.detach().requires_grad_(True) if training else logits
                w = weights[i]
                for start in range(0, n_rows, chunk):
                    end = min(start + chunk, n_rows)
                    s = proxy[start:end].float()
                    t = row_targets[start:end]
                    # sum / N reproduces the mean over the answer tokens (CE) and
                    # kl_div(reduction='batchmean') over the real tokens (KD).
                    ce_chunk = F.cross_entropy(s, t, ignore_index=IGNORE_INDEX, reduction="sum") / n_ce
                    loss_chunk = (1 - kd_lambda) * ce_chunk
                    if teacher_rows is not None:
                        with torch.no_grad():
                            teacher = self._teacher.lm_head(teacher_rows[start:end]).float()
                        kl_chunk = F.kl_div(F.log_softmax(s / T, dim=-1), F.softmax(teacher / T, dim=-1),
                                            reduction="sum") / n_kd * (T ** 2)
                        loss_chunk = loss_chunk + kd_lambda * kl_chunk
                        kl_loss[i] = kl_loss[i] + kl_chunk.detach()
                    if training:
                        (loss_chunk * w if w != 1.0 else loss_chunk).backward()
                    ce_loss[i] = ce_loss[i] + ce_chunk.detach()
                    on_answer = t != IGNORE_INDEX
                    correct[i] += int((s.detach().argmax(-1)[on_answer] == t[on_answer]).sum())

                if training and proxy.grad is not None:
                    self.manual_backward(logits, gradient=proxy.grad)
                del logits, proxy

        total_loss = 0.0
        for i in levels:
            loss = (1 - kd_lambda) * ce_loss[i] + kd_lambda * kl_loss[i]
            self.log(f"{stage}_level{i}_loss", loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_ce_loss", ce_loss[i], prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_kl_loss", kl_loss[i], prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_answer_token_acc", correct[i] / max(1, n_ce),
                     prog_bar=(stage != "train"), sync_dist=True)
            self._record_level(i, loss, weights[i], training)
            total_loss += loss

        self.log(f"{stage}_loss", total_loss, prog_bar=(stage != "train"), sync_dist=True)

        if training:
            self._finish_level_weights(training, input_ids.device)
            self._optimizer_step(opt)
            if self.training_context.warmup_ratio is not None:
                self.lr_schedulers().step()
            # tok/s counts every real (prompt + answer) token of this rank's batch.
            self._report_progress("sft", int(real.sum()), kd_rows=n_kd, answer_rows=n_ce)


torch.serialization.add_safe_globals([SFTTrainingContext, FlexLMSFTTrainer, load_qa_sft])


def _tokenizer_for(builder) -> str:
    # Same preference order as eval_wikitext103.resolve_tokenizer_name, which
    # eval_lm_harness.py uses, so SFT data and evaluation are tokenized
    # identically. Not imported from there: that module imports run_experiment,
    # which imports config.experiments, which is still loading when this runs.
    return (getattr(builder.training_context, "teacher_hf_model", None)
            or getattr(builder.model_config, "pretrained_hf_model", None)
            or "JackFram/llama-160m")


def make_sft(base_name: str, base: TrainerBuilder, **overrides) -> TrainerBuilder:
    """SFT config keeping `base`'s model architecture, KD teacher, KD
    lambda/temperature and level sampling. Starts from the trained weights of
    flexllama,<base_name>, or from the model config's Hugging Face warm start
    when base_name is None."""
    ctx = base.training_context
    kwargs = dict(
        load_from=f"flexllama,{base_name}" if base_name is not None else None,
        teacher_hf_model=getattr(ctx, "teacher_hf_model", None) or "JackFram/llama-160m",
        tokenizer_name=_tokenizer_for(base),
        kd_lambda=getattr(ctx, "kd_lambda", 0.0),
        kd_temperature=getattr(ctx, "kd_temperature", 2.0),
        num_levels_per_step=getattr(ctx, "num_levels_per_step", None),
        wandb_project_name="FlexLLaMA_sft_medqa",
    )
    kwargs.update(overrides)
    model_config = base.model_config
    if kwargs.get("max_seq_length", model_config.max_seq_length) > model_config.max_seq_length:
        model_config = dataclasses.replace(model_config, max_seq_length=kwargs["max_seq_length"])
    return TrainerBuilder(FlexLMSFTTrainer, model_config, SFTTrainingContext(**kwargs))


def _hf_base(preset: str, fractions=None, hf_start=None) -> TrainerBuilder:
    """A FlexLLaMA whose max level is the Hugging Face model `preset`, warm-started
    from its weights and distilled from it (the TEACHER_PRESETS warm-start setup).
    config.experiments is still importing when this runs, but the function used
    here is defined before register_sft_configs is called."""
    from config.experiments import make_flexllama_warmstart_kd
    builder = make_flexllama_warmstart_kd(preset, kd_lambda=0.5, kd_temperature=2.0, fractions=fractions)
    if hf_start is not None:  # same architecture, other start weights (e.g. importance-ordered)
        builder.model_config = dataclasses.replace(builder.model_config, pretrained_hf_model=hf_start)
    return builder


def register_sft_configs(configs: dict) -> None:
    """Called at the end of config/experiments.py. A missing base config only
    skips its SFT entry, so it can never break the other configs."""
    for name, base_name, overrides in SFT_RUNS:
        overrides = dict(overrides)
        preset = overrides.pop("hf_preset", None)
        fractions = overrides.pop("level_fractions", None)
        hf_start = overrides.pop("hf_start", None)
        base = _hf_base(preset, fractions, hf_start) if preset else configs.get("flexllama", {}).get(base_name)
        if base is None:
            logging.warning(f"SFT: base config flexllama,{base_name} not found, skipping {name}")
            continue
        configs["flexllama"][name] = make_sft(base_name, base, **overrides)


def check(only: str = None) -> None:
    """Runs on a login node without a GPU: everything the training job needs
    except the training itself (Hugging Face start weights are checked by the
    loader at training start, too large to load twice here)."""
    from training.run_experiment import resolve_from_str

    for name, base_name, _ in SFT_RUNS:
        if only is not None and name != only:
            continue
        conf = f"flexllama,{name}"
        print(f"== {conf}")
        builder = resolve_from_str(conf)
        ctx = builder.training_context
        cfg = builder.model_config
        print(f"teacher            : {ctx.teacher_hf_model}")
        print(f"kd_lambda / T      : {ctx.kd_lambda} / {ctx.kd_temperature}")
        print(f"levels per step    : {ctx.num_levels_per_step or 'all'}")
        print(f"level weighting    : {getattr(ctx, 'level_weighting', 'none')}  "
              f"fixed {getattr(ctx, 'level_weights_fixed', None)}  sampling {getattr(ctx, 'level_sampling', 'sandwich')}")
        print(f"lr / epochs        : {ctx.lr} / {ctx.epochs}  warmup_ratio {ctx.warmup_ratio}  "
              f"train_levels {getattr(ctx, 'train_levels', None) or 'all'}")
        print(f"hidden dims        : {list(cfg.hidden_dims)}  heads {list(cfg.num_heads)}  "
              f"kv {list(cfg.num_kv_heads)}  mlp {list(cfg.intermediate_dims)}")
        print(f"layers / vocab     : {cfg.num_layers} / {cfg.vocab_size}  max_seq {cfg.max_seq_length}  "
              f"qk_norm {getattr(cfg, 'qk_norm', False)}  tie {cfg.tie_embeddings}")

        if ctx.load_from is not None:
            base = resolve_from_str(f"flexllama,{base_name}")
            print(f"base trainer       : {base.training_method.__name__}")
            path = trained_model_path(ctx.load_from)
            print(f"start weights      : {path} (exists: {path.exists()})")
            model = cfg.make_model()
            model.load_state_dict(torch.load(path, map_location="cpu"))
            print("start weights load : ok (all keys match)")
        else:
            print(f"start weights      : {cfg.pretrained_hf_model} (Hugging Face warm start)")

        train_loader, val_loader, _ = ctx.loader_function()
        tokenizer_name = ctx.loader_function.keywords["tokenizer_name"]
        print(f"tokenizer          : {tokenizer_name}")
        for split, loader in (("train", train_loader), ("val", val_loader)):
            ds = loader.dataset
            lengths = [len(p) + len(a) for p, a in ds.items]
            print(f"{split:5s} examples     : {len(ds)} (max len {max(lengths)}, "
                  f"mean {sum(lengths) / len(lengths):.0f}, truncated {ds.num_truncated})")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        input_ids, labels, _ = collate_qa([val_loader.dataset[0]])
        answer = input_ids[0][labels[0] != IGNORE_INDEX]
        print(f"example prompt end : {tokenizer.decode(input_ids[0][labels[0] == IGNORE_INDEX][-12:])!r}")
        print(f"example answer     : {tokenizer.decode(answer)!r}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        check(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print("usage: python -m training.sft check [config name, e.g. sft.medmix_qwen3_1p7b]")
