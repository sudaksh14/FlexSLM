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
            opt.step()


class MemOptFlexLMKDTrainer(_MemOptMixin, FlexLMKDTrainer):
    """FlexLMKDTrainer with chunked CE+KL loss and optional activation checkpointing."""

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

        # Teacher forward, no grad, once per batch (unchanged from FlexLMKDTrainer)
        with torch.no_grad():
            teacher_logits = self._teacher(input_ids).logits

        levels = self._select_levels(stage)
        teacher_rows = teacher_logits[:, :-1].reshape(-1, vocab_size)

        total_loss = 0.0

        for i in levels:
            self.submodel.set_level_use(i)
            logits = self(input_ids)

            student_rows, targets, n_rows = self._flat_views(logits, input_ids, vocab_size)
            proxy = student_rows.detach().requires_grad_(True) if training else student_rows

            ce_loss = 0.0
            kl_loss = 0.0
            for start in range(0, n_rows, chunk):
                end = min(start + chunk, n_rows)
                s = proxy[start:end]
                t = teacher_rows[start:end]

                # sum/n_rows reproduces cross_entropy(reduction='mean') and
                # kl_div(reduction='batchmean') exactly, chunk by chunk.
                ce_chunk = F.cross_entropy(s, targets[start:end], reduction="sum") / n_rows
                kl_chunk = F.kl_div(
                    F.log_softmax(s / T, dim=-1),
                    F.softmax(t / T, dim=-1),
                    reduction="sum",
                ) / n_rows * (T ** 2)

                if training:
                    ((1 - kd_lambda) * ce_chunk + kd_lambda * kl_chunk).backward()
                ce_loss = ce_loss + ce_chunk.detach()
                kl_loss = kl_loss + kl_chunk.detach()

            loss = (1 - kd_lambda) * ce_loss + kd_lambda * kl_loss
            ppl = torch.exp(ce_loss)

            self.log(f"{stage}_level{i}_loss", loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_ce_loss", ce_loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_kl_loss", kl_loss, prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_ppl", ppl, prog_bar=(stage != "train"), sync_dist=True)

            if training:
                self._push_grad_through_model(student_rows, proxy)
            total_loss += loss

        self.log(f"{stage}_loss", total_loss, prog_bar=(stage != "train"), sync_dist=True)

        if training:
            opt.step()


torch.serialization.add_safe_globals([MemOptFlexLMTrainer, MemOptFlexLMKDTrainer])
