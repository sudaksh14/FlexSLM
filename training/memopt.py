"""
Memory-optimized variants of the FlexSLM language-model trainers.

These subclass the trainers in training.py and keep the nested-width training
algorithm byte-for-byte identical: same level selection (_select_levels, min and
max always included), same per-level gradient accumulation, one optimizer step
per batch, same DDP strategy, same logged metric names. Only *how* the loss is
evaluated and how activations are stored changes - never *what* is computed.

Both optimizations are mathematically exact (not approximations):

1. Chunked loss. The KD loss materializes several [N, V] tensors, where
   N = batch * (seq_len - 1) and V = vocab_size, and autocast promotes softmax /
   log_softmax / cross_entropy to fp32. For Qwen2.5 (V=151936) at batch=4,
   seq=1024 that is ~2.5GB *per tensor*, and the loss needs three of them live at
   once - which is exactly where the 1.5B runs were OOMing. Here the loss is
   summed over row-chunks of size loss_chunk_size instead. Because
   cross_entropy(reduction='mean') == sum/N and kl_div(reduction='batchmean')
   == sum/N over the same N, summing per-chunk sums and dividing by N once gives
   an identical result. During training each chunk is backwarded immediately into
   a detached proxy of the logits (freeing that chunk's graph), then the
   accumulated proxy gradient is pushed through the model in a single backward -
   standard chunked-loss, exact same gradients as one big backward.

2. Activation checkpointing (opt-in via TrainingContext.activation_checkpointing).
   Decoder blocks recompute their activations during backward instead of storing
   them. Applied by wrapping each block's forward in place, so module structure
   and state_dict keys are untouched and existing checkpoints / warm-start loading
   stay compatible.

Neither changes the model, the optimizer, or the level schedule, so runs are
directly comparable against the training.py trainers.
"""
import torch
import torch.nn.functional as F

from training import FlexLMTrainer, FlexLMKDTrainer
from training.level_weights import LevelWeightingMixin

DEFAULT_LOSS_CHUNK_SIZE = 1024


def enable_activation_checkpointing(model) -> int:
    """
    Wraps each decoder block's forward in torch.utils.checkpoint, in place.

    Patches the bound forward rather than swapping the module for a wrapper so
    that parameter names (and therefore state_dict keys, checkpoints and
    warm-start loading) are unchanged. Returns the number of blocks wrapped.
    """
    import torch.utils.checkpoint as cp

    layers = getattr(model, "layers", None)
    if layers is None:
        return 0

    wrapped = 0
    for block in layers.children():
        if getattr(block, "_flexslm_ckpt_wrapped", False):
            continue
        inner = block.forward

        def forward(x, _inner=inner):
            # Only checkpoint when a graph is actually being built; under
            # torch.no_grad (validation/test) plain forward is both correct and
            # cheaper, and cp.checkpoint would warn about having nothing to save.
            if torch.is_grad_enabled() and x.requires_grad:
                return cp.checkpoint(_inner, x, use_reentrant=False)
            return _inner(x)

        block.forward = forward
        block._flexslm_ckpt_wrapped = True
        wrapped += 1
    return wrapped


def memopt(training_context, loss_chunk_size: int = DEFAULT_LOSS_CHUNK_SIZE,
           activation_checkpointing: bool = True):
    """
    Tags an existing TrainingContext with the memory-optimization settings and
    returns it, so configs can opt in without training.py needing to know about
    them (the trainers read both via getattr with defaults).
    """
    training_context.loss_chunk_size = loss_chunk_size
    training_context.activation_checkpointing = activation_checkpointing
    return training_context


class _MemOptMixin:
    """Shared chunked-loss plumbing for the memory-optimized trainers."""

    def _chunk_size(self) -> int:
        return getattr(self.training_context, "loss_chunk_size", None) or DEFAULT_LOSS_CHUNK_SIZE

    def _maybe_checkpoint_activations(self) -> None:
        if getattr(self.training_context, "activation_checkpointing", False):
            enable_activation_checkpointing(self.submodel)

    def on_fit_start(self):
        super().on_fit_start()
        self._maybe_checkpoint_activations()

    def _flat_views(self, logits, input_ids, vocab_size):
        """(student rows, target rows, N) with the standard CLM shift applied."""
        student = logits[:, :-1].reshape(-1, vocab_size)
        targets = input_ids[:, 1:].reshape(-1)
        return student, targets, targets.numel()

    def _push_grad_through_model(self, student_rows, proxy) -> None:
        """Single model backward using the gradient accumulated on the proxy."""
        if proxy.grad is not None:
            self.manual_backward(student_rows, gradient=proxy.grad)


class MemOptFlexLMTrainer(_MemOptMixin, FlexLMTrainer):
    """FlexLMTrainer with chunked CE loss and optional activation checkpointing."""

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> None:
        input_ids, _ = batch

        if stage == "train":
            opt = self.optimizers()
            opt.zero_grad()

        levels = self._select_levels(stage)
        vocab_size = self.submodel.config.vocab_size
        chunk = self._chunk_size()
        training = stage == "train"

        total_loss = 0.0

        for i in levels:
            self.submodel.set_level_use(i)
            logits = self(input_ids)

            student_rows, targets, n_rows = self._flat_views(logits, input_ids, vocab_size)
            proxy = student_rows.detach().requires_grad_(True) if training else student_rows

            loss = 0.0
            for start in range(0, n_rows, chunk):
                end = min(start + chunk, n_rows)
                # reduction='sum' / n_rows == reduction='mean' over all rows
                chunk_loss = F.cross_entropy(
                    proxy[start:end], targets[start:end], reduction="sum") / n_rows
                if training:
                    chunk_loss.backward()
                loss = loss + chunk_loss.detach()

            ppl = torch.exp(loss)

            self.log(f"{stage}_level{i}_loss", loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_ppl", ppl, prog_bar=(stage != "train"), sync_dist=True)

            if training:
                self._push_grad_through_model(student_rows, proxy)
            total_loss += loss

        self.log(f"{stage}_loss", total_loss, prog_bar=(stage != "train"), sync_dist=True)

        if training:
            self._optimizer_step(opt)


class MemOptFlexLMKDTrainer(_MemOptMixin, LevelWeightingMixin, FlexLMKDTrainer):
    """FlexLMKDTrainer with chunked CE+KL loss and optional activation checkpointing.
    Also supports the level_weighting flag (training/level_weights.py); with it
    unset (the default) the gradients are those of FlexLMKDTrainer."""

    def _step(self, batch, stage: str) -> None:
        input_ids, _ = batch

        kd_lambda = self.training_context.kd_lambda
        T = self.training_context.kd_temperature
        vocab_size = self.submodel.config.vocab_size
        chunk = self._chunk_size()
        training = stage == "train"

        if training:
            opt = self.optimizers()
            opt.zero_grad()

        levels = self._select_levels(stage)
        weights = {i: self._level_weight(i, training) for i in levels}  # 1.0 unless level_weighting is set
        zero = torch.zeros((), device=input_ids.device)
        ce_loss = {i: zero for i in levels}
        kl_loss = {i: zero for i in levels}
        # Whole-batch normaliser, so micro-batches give the same loss and gradient.
        n_total = input_ids.shape[0] * (input_ids.shape[1] - 1)

        # micro_batch_size (training context, None = whole batch) splits the batch by
        # sequence, so teacher and student [N, V] logits exist for one micro-batch at a
        # time (job 28080: 16 x 1,024 tokens x 151,936 vocab ran out of memory at step 0).
        bsz = input_ids.shape[0]
        mb = getattr(self.training_context, "micro_batch_size", None) or bsz
        for lo in range(0, bsz, mb):
            mb_ids = input_ids[lo:lo + mb]
            # Teacher forward, no grad, once per micro-batch (shared by all levels)
            with torch.no_grad():
                teacher_rows = self._teacher(mb_ids).logits[:, :-1].reshape(-1, vocab_size)

            for i in levels:
                self.submodel.set_level_use(i)
                logits = self(mb_ids)
                student_rows, targets, n_rows = self._flat_views(logits, mb_ids, vocab_size)
                proxy = student_rows.detach().requires_grad_(True) if training else student_rows
                w = weights[i]
                for start in range(0, n_rows, chunk):
                    end = min(start + chunk, n_rows)
                    s = proxy[start:end]
                    t = teacher_rows[start:end]
                    # sum/n_total reproduces cross_entropy(reduction='mean') and
                    # kl_div(reduction='batchmean') over the whole batch, chunk by chunk.
                    ce_chunk = F.cross_entropy(s, targets[start:end], reduction="sum") / n_total
                    kl_chunk = F.kl_div(
                        F.log_softmax(s / T, dim=-1),
                        F.softmax(t / T, dim=-1),
                        reduction="sum",
                    ) / n_total * (T ** 2)
                    if training:
                        chunk_loss = (1 - kd_lambda) * ce_chunk + kd_lambda * kl_chunk
                        (chunk_loss * w if w != 1.0 else chunk_loss).backward()
                    ce_loss[i] = ce_loss[i] + ce_chunk.detach()
                    kl_loss[i] = kl_loss[i] + kl_chunk.detach()
                if training:
                    self._push_grad_through_model(student_rows, proxy)
                del logits, student_rows, proxy
            del teacher_rows

        total_loss = 0.0
        for i in levels:
            loss = (1 - kd_lambda) * ce_loss[i] + kd_lambda * kl_loss[i]
            self.log(f"{stage}_level{i}_loss", loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_ce_loss", ce_loss[i], prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_kl_loss", kl_loss[i], prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_ppl", torch.exp(ce_loss[i]), prog_bar=(stage != "train"), sync_dist=True)
            self._record_level(i, loss, weights[i], training)
            total_loss += loss

        self.log(f"{stage}_loss", total_loss, prog_bar=(stage != "train"), sync_dist=True)

        if training:
            self._finish_level_weights(training, input_ids.device)
            self._optimizer_step(opt)
            if getattr(self.training_context, "warmup_ratio", None) is not None:
                self.lr_schedulers().step()  # per-step schedule (Stage-1 contexts)
            self._report_progress("kd", input_ids.numel(), levels=",".join(map(str, levels)))


torch.serialization.add_safe_globals([MemOptFlexLMTrainer, MemOptFlexLMKDTrainer])
