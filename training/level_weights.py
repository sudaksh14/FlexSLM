"""
Per-level loss weights for nested training: the `level_weighting` config flag.

Every training step sums one loss per level, L = sum_k w_k * L_k. Modes:

  none               w_k = 1 (the default, and the behaviour of all runs so far).
  fixed              w_k = level_weights_fixed[k], constant.
  uncertainty        NSN / Kendall et al. (2018) homoscedastic-uncertainty weighting:
                     each level has a learned log-variance s_k and contributes
                     exp(-s_k) * L_k + s_k. The optimum is s_k = log L_k, i.e.
                     w_k -> 1 / L_k: high-loss (small) levels get LESS weight.
  loss_proportional  w_k = EMA(L_k) / mean_j EMA(L_j): high-loss (small) levels get
                     MORE weight; mean weight 1 over the levels seen so far.
  dwa                Dynamic Weight Averaging (Liu et al., CVPR 2019):
                     w_k = K * softmax_k(r_k / T), r_k = EMA(L_k) now / EMA(L_k)
                     dwa_window steps ago, so levels whose loss falls slowest get
                     more weight. EMA-smoothed because one step's loss is noisy
                     (DWA was defined on epoch means; our SFT is one epoch).
                     w_k = 1 until a level has dwa_window steps of history.

The weights live outside the model and outside DDP: they are plain Python floats,
updated from the step's per-level losses averaged over all ranks (one all-reduce
per step), so every rank holds identical weights, the exported model is unchanged
and the weights are not part of the global gradient-norm clip. They only scale
the backward pass; logged losses and val_loss stay unweighted. The s_k use their
own Adam (torch defaults, lr `level_weight_lr`) on the analytic gradient
d/ds_k [exp(-s_k) L_k + s_k] = 1 - exp(-s_k) L_k, the same gradient autograd would
give. Their lr is separate on purpose: at the SFT lr (5e-5) over ~1k steps Adam
moves s_k by at most ~0.05, so the weights would stay ~1.

Note (Adam): Adam normalises each parameter's update, so a common scale on all
weights cancels; only the relative weights on parameters that several levels
share matter.
"""
import collections
import math
import time
from typing import Optional, Sequence

import torch
import torch.distributed as dist

MODES = ("none", "fixed", "uncertainty", "loss_proportional", "dwa")

# Context fields (and their defaults) read by LevelWeighter.from_context. Contexts
# pickled before a field existed fall back to these.
CONTEXT_DEFAULTS = dict(
    level_weighting="none",
    level_weight_lr=1e-2,        # uncertainty: Adam lr of the log-variances
    level_weight_ema=0.9,        # loss_proportional, dwa: loss EMA decay
    level_weights_fixed=None,    # fixed: one weight per level, smallest level first
    dwa_window=50,               # dwa: steps between the two compared EMAs
    dwa_temperature=2.0,         # dwa: softmax temperature (Liu et al. use 2)
)

_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8  # torch.optim.Adam defaults


def check_args(level_weighting="none", level_weight_lr=1e-2, level_weight_ema=0.9,
               level_weights_fixed=None, dwa_window=50, dwa_temperature=2.0) -> None:
    if level_weighting not in MODES:
        raise ValueError(f"level_weighting must be one of {MODES}, got {level_weighting!r}")
    if not level_weight_lr > 0:
        raise ValueError(f"level_weight_lr must be positive, got {level_weight_lr}")
    if not 0.0 <= level_weight_ema < 1.0:
        raise ValueError(f"level_weight_ema must be in [0, 1), got {level_weight_ema}")
    if level_weighting == "fixed":
        if not level_weights_fixed or any(not w > 0 for w in level_weights_fixed):
            raise ValueError(f"level_weighting='fixed' needs positive level_weights_fixed, got {level_weights_fixed}")
    if int(dwa_window) < 1 or not dwa_temperature > 0:
        raise ValueError(f"dwa_window must be >= 1 and dwa_temperature > 0, got {dwa_window}, {dwa_temperature}")


def uncertainty_grad(log_var: float, loss: float) -> float:
    """d/ds [exp(-s) * L + s]."""
    return 1.0 - math.exp(-log_var) * loss


class LevelWeighter:
    def __init__(self, num_levels: int, mode: str = "none", lr: float = 1e-2, ema_decay: float = 0.9,
                 fixed: Optional[Sequence[float]] = None, dwa_window: int = 50, dwa_temperature: float = 2.0):
        check_args(mode, lr, ema_decay, fixed, dwa_window, dwa_temperature)
        if mode == "fixed" and len(fixed) != num_levels:
            raise ValueError(f"level_weights_fixed has {len(fixed)} weights for {num_levels} levels")
        self.num_levels = num_levels
        self.mode = mode
        self.lr = lr
        self.ema_decay = ema_decay
        self.fixed = [float(w) for w in fixed] if fixed is not None else None
        self.dwa_window = int(dwa_window)
        self.dwa_temperature = dwa_temperature
        self.log_vars = [0.0] * num_levels
        self._m = [0.0] * num_levels
        self._v = [0.0] * num_levels
        self._t = [0] * num_levels  # per level: a level is only updated on steps that train it
        self.ema: list[Optional[float]] = [None] * num_levels
        self._history = [collections.deque(maxlen=self.dwa_window + 1) for _ in range(num_levels)]

    @classmethod
    def from_context(cls, ctx, num_levels: int) -> "LevelWeighter":
        a = {k: getattr(ctx, k, v) for k, v in CONTEXT_DEFAULTS.items()}
        return cls(num_levels, mode=a["level_weighting"], lr=a["level_weight_lr"],
                   ema_decay=a["level_weight_ema"], fixed=a["level_weights_fixed"],
                   dwa_window=a["dwa_window"], dwa_temperature=a["dwa_temperature"])

    @property
    def active(self) -> bool:
        return self.mode != "none"

    def _dwa_weights(self) -> list[float]:
        ready = [len(h) > self.dwa_window for h in self._history]
        if not all(ready):
            return [1.0] * self.num_levels
        r = [h[-1] / h[0] for h in self._history]
        z = [math.exp(x / self.dwa_temperature) for x in r]
        return [self.num_levels * x / sum(z) for x in z]

    def weight(self, level: int) -> float:
        if self.mode == "fixed":
            return self.fixed[level]
        if self.mode == "uncertainty":
            return math.exp(-self.log_vars[level])
        if self.mode == "loss_proportional":
            seen = [e for e in self.ema if e is not None]
            if self.ema[level] is None or not seen:
                return 1.0
            return self.ema[level] / (sum(seen) / len(seen))
        if self.mode == "dwa":
            return self._dwa_weights()[level]
        return 1.0

    def observe(self, level: int, loss: float) -> None:
        """Update with this step's (rank-averaged, unweighted) loss of `level`."""
        if self.mode == "uncertainty":
            g = uncertainty_grad(self.log_vars[level], loss)
            self._t[level] += 1
            t = self._t[level]
            self._m[level] = _BETA1 * self._m[level] + (1 - _BETA1) * g
            self._v[level] = _BETA2 * self._v[level] + (1 - _BETA2) * g * g
            m_hat = self._m[level] / (1 - _BETA1 ** t)
            v_hat = self._v[level] / (1 - _BETA2 ** t)
            self.log_vars[level] -= self.lr * m_hat / (math.sqrt(v_hat) + _EPS)
        elif self.mode in ("loss_proportional", "dwa"):
            prev = self.ema[level]
            self.ema[level] = loss if prev is None else self.ema_decay * prev + (1 - self.ema_decay) * loss
            self._history[level].append(self.ema[level])

    def observe_step(self, losses: dict, device) -> None:
        """Average this step's per-level losses over all ranks (one all-reduce;
        every rank calls it once per training step) and update every level that
        any rank trained."""
        if self.mode in ("none", "fixed"):
            return
        buf = torch.zeros(2, self.num_levels, dtype=torch.float64, device=device)
        for level, loss in losses.items():
            buf[0, level] = float(loss)
            buf[1, level] = 1.0
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(buf)
        for level in range(self.num_levels):
            n = float(buf[1, level])
            if n > 0:
                self.observe(level, float(buf[0, level]) / n)

    def state_dict(self) -> dict:
        return dict(mode=self.mode, log_vars=list(self.log_vars), m=list(self._m), v=list(self._v),
                    t=list(self._t), ema=list(self.ema), history=[list(h) for h in self._history])

    def load_state_dict(self, state: dict) -> None:
        if state.get("mode") != self.mode:
            raise ValueError(f"checkpoint level_weighting {state.get('mode')!r} != config {self.mode!r}")
        self.log_vars = list(state["log_vars"])
        self._m, self._v, self._t = list(state["m"]), list(state["v"]), list(state["t"])
        self.ema = list(state["ema"])
        for h, saved in zip(self._history, state.get("history", [[]] * self.num_levels)):
            h.clear()
            h.extend(saved)


class LevelWeightingMixin:
    """For the Flex LM trainers: the level weighter (built lazily from the training
    context), its checkpoint state, per-level weight logging, and a progress line
    with throughput. Usage inside _step:
        w = self._level_weight(i, training)          # scale level i's backward by w
        self._record_level(i, loss, w, training)     # after level i
        self._finish_level_weights(training, device) # once, after all levels
    """

    @property
    def level_weighter(self) -> LevelWeighter:
        lw = self.__dict__.get("_level_weighter_obj")
        if lw is None:
            lw = LevelWeighter.from_context(self.training_context, self.submodel.max_level() + 1)
            self.__dict__["_level_weighter_obj"] = lw
        return lw

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["level_weighter"] = self.level_weighter.state_dict()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if "level_weighter" in checkpoint:
            self.level_weighter.load_state_dict(checkpoint["level_weighter"])

    def _level_weight(self, level: int, training: bool) -> float:
        lw = self.level_weighter
        return lw.weight(level) if (training and lw.active) else 1.0

    def _record_level(self, level: int, loss, w: float, training: bool) -> None:
        lw = self.level_weighter
        if not (training and lw.active):
            return
        self.log(f"train_level{level}_weight", w, prog_bar=False, sync_dist=False)
        if lw.mode == "uncertainty":
            self.log(f"train_level{level}_log_var", lw.log_vars[level], prog_bar=False, sync_dist=False)
        self.__dict__.setdefault("_step_losses", {})[level] = loss

    def _finish_level_weights(self, training: bool, device) -> None:
        """Weights for the next step, from this step's rank-averaged losses."""
        if training:
            self.level_weighter.observe_step(self.__dict__.pop("_step_losses", {}), device)

    def _report_progress(self, tag: str, n_tokens: int, every: int = 50, **extra) -> None:
        """Per-rank progress line in the .out log: step, tokens/s since the last
        line, peak memory. A rank that stops printing is the one stuck in a collective."""
        d = self.__dict__
        now = time.time()
        d["_tok_since"] = d.get("_tok_since", 0) + n_tokens
        step = self.global_step
        if not (step < 20 or step % every == 0):
            return
        last = d.get("_last_report")
        rate = d["_tok_since"] / (now - last) if last else float("nan")
        d["_last_report"], d["_tok_since"] = now, 0
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        fields = " ".join(f"{k} {v}" for k, v in extra.items())
        print(f"[{tag}] rank {self.global_rank} step {step} {fields} tok/s {rate:.0f} "
              f"max_mem {mem:.1f}GB", flush=True)
