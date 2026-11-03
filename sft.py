"""
Supervised fine-tuning of a FlexLLaMA on question/answer pairs. Same FlexSLM
training (all levels, CE + KD) as FlexLMKDTrainer, with CE on the answer tokens
only. Two kinds of runs:
  - from a trained FlexLLaMA checkpoint (sft.medqa_full5ep: data/medqa_sft from
    prepare_medqa_sft.py, starts from saved_models/<pmc.full5ep>.pt);
  - from Hugging Face weights (sft.medmix_qwen3_1p7b: Qwen3-1.7B backbone, FaithMed's
    medmix data from prepare_medmix_sft.py and FaithMed's SFT schedule).

    python sft.py check      # login node, no GPU: validates config, weights and data
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

IGNORE_INDEX = -100
SFT_DATA_DIR = os.path.join("data", "medqa_sft")

# FaithMed's SFT recipe (train_sft.yaml, paper Sec. 5): LR 5e-5, cosine with a 1%
# linear warmup, effective batch 32 (8 per GPU x 4 GPUs), one epoch as in the paper,
# on the Qwen3-1.7B backbone with FaithMed's medmix data.
_QWEN3_FAITHMED = dict(
    hf_preset="qwen3-1.7b", data_dir=os.path.join("data", "medmix_sft"),
    lr=5e-5, warmup_ratio=0.01, epochs=1, patience=1000, batch_size=8,
    max_seq_length=3072, activation_checkpointing=True)

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
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
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

    def __init__(self, load_from: str, teacher_hf_model: str, tokenizer_name: str,
                 kd_lambda=0.5, kd_temperature=2.0, num_levels_per_step=None,
                 lr=2e-5, weight_decay=0.0, epochs=3, patience=5,
                 batch_size=16, max_seq_length=1024, data_dir=SFT_DATA_DIR,
                 warmup_ratio=None, activation_checkpointing=False, train_levels=None, **kwargs):
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

    def make_optimizer(self, model):
        return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def make_scheduler(self, optimizer):
        # Stepped once per epoch (FlexModelTrainer.on_train_epoch_end).
        return CosineAnnealingLR(optimizer, T_max=max(1, self.epochs), eta_min=self.lr * 0.1)

    def make_step_scheduler(self, optimizer, total_steps: int):
        """Linear warmup over the first warmup_ratio of the steps, then cosine to 0,
        stepped after every optimizer step: the schedule HF Trainer (and so
        LLaMA-Factory) uses for lr_scheduler_type=cosine with warmup_ratio."""
        warmup = math.ceil(self.warmup_ratio * total_steps)

        def factor(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, factor)


def trained_model_path(config_name: str):
    return paths.TRAINED_MODELS / f"{utils.make_str_filename_safe(config_name)}.pt"


class FlexLMSFTTrainer(FlexLMKDTrainer):
    """FlexLMKDTrainer for question/answer data. The KD term is FlexSLM's own: over
    every real token, as in FlexLMKDTrainer, with the same lambda and T. Only CE is
    restricted to the answer tokens, the SFT convention (FaithMed's
    train_on_prompt: false). The loss is evaluated in row chunks into a detached
    proxy of the logits, as in training_memopt, so a 152k vocab fits in memory."""

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
            from training_memopt import enable_activation_checkpointing
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
        row_targets = targets[rows]      # IGNORE_INDEX on prompt rows: CE skips them
        n_ce = int(answer.sum())
        n_kd = int(rows.sum())

        if training:
            opt = self.optimizers()
            opt.zero_grad()

        teacher_rows = None
        if kd_lambda > 0:
            with torch.no_grad():
                teacher_rows = self._teacher.model(input_ids).last_hidden_state[:, :-1][rows]

        total_loss = 0.0
        for i in self._select_levels(stage):
            self.submodel.set_level_use(i)
            logits = self.submodel.lm_head(self._hidden_rows(input_ids, rows))  # [N, V]
            proxy = logits.detach().requires_grad_(True) if training else logits

            ce_loss = kl_loss = 0.0
            correct = 0
            for start in range(0, n_kd, chunk):
                end = min(start + chunk, n_kd)
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
                    kl_loss = kl_loss + kl_chunk.detach()
                if training:
                    loss_chunk.backward()
                ce_loss = ce_loss + ce_chunk.detach()
                on_answer = t != IGNORE_INDEX
                correct += int((s.detach().argmax(-1)[on_answer] == t[on_answer]).sum())

            if training and proxy.grad is not None:
                self.manual_backward(logits, gradient=proxy.grad)

            kl_loss = kl_loss if torch.is_tensor(kl_loss) else torch.zeros((), device=input_ids.device)
            loss = (1 - kd_lambda) * ce_loss + kd_lambda * kl_loss
            self.log(f"{stage}_level{i}_loss", loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_ce_loss", ce_loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_kl_loss", kl_loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_answer_token_acc", correct / max(1, n_ce),
                     prog_bar=(stage != "train"), sync_dist=True)
            total_loss += loss

        self.log(f"{stage}_loss", total_loss, prog_bar=(stage != "train"), sync_dist=True)

        if training:
            opt.step()
            if self.training_context.warmup_ratio is not None:
                self.lr_schedulers().step()
            # Per-rank progress in the .out log (the progress bar only appears at the
            # end): a rank that stops printing is the one stuck in a collective.
            step = self.global_step
            if step < 20 or step % 50 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                print(f"[sft] rank {self.global_rank} step {step} kd_rows {n_kd} answer_rows {n_ce} "
                      f"max_mem {mem:.1f}GB", flush=True)


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


def _hf_base(preset: str) -> TrainerBuilder:
    """A FlexLLaMA whose max level is the Hugging Face model `preset`, warm-started
    from its weights and distilled from it (the TEACHER_PRESETS warm-start setup).
    config.experiments is still importing when this runs, but the function used
    here is defined before register_sft_configs is called."""
    from config.experiments import make_flexllama_warmstart_kd
    return make_flexllama_warmstart_kd(preset, kd_lambda=0.5, kd_temperature=2.0)


def register_sft_configs(configs: dict) -> None:
    """Called at the end of config/experiments.py. A missing base config only
    skips its SFT entry, so it can never break the other configs."""
    for name, base_name, overrides in SFT_RUNS:
        overrides = dict(overrides)
        preset = overrides.pop("hf_preset", None)
        base = _hf_base(preset) if preset else configs.get("flexllama", {}).get(base_name)
        if base is None:
            logging.warning(f"SFT: base config flexllama,{base_name} not found, skipping {name}")
            continue
        configs["flexllama"][name] = make_sft(base_name, base, **overrides)


def check(only: str = None) -> None:
    """Runs on a login node without a GPU: everything the training job needs
    except the training itself (Hugging Face start weights are checked by the
    loader at training start, too large to load twice here)."""
    from run_experiment import resolve_from_str

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
        print("usage: python sft.py check [config name, e.g. sft.medmix_qwen3_1p7b]")
