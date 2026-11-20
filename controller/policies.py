"""Runtime width-selection policies (tier 2 of the controller) and the deploy-time selector (tier 1).

Every runtime policy has the same two calls, so the simulator and a later on-device runner can share them:
    level = policy.choose(Request)                  # before serving
    policy.observe(level, latency_ms, energy_j)     # after serving

Policies (all analytic, no learning; statistics come from a calibration split, never from the scored half):
- Fixed(level)                   one width for every request (the dense-model deployment).
- DeadlinePolicy(feedback=False) widest width whose PREDICTED latency meets the request deadline, else the
                                 narrowest (= jetson/bench_trace.py `elastic`, open loop).
- DeadlinePolicy(feedback=True)  same, prediction scaled by rho = EWMA(observed / predicted latency): follows
                                 thermal throttling, other processes on the shared 8 GB, clock changes.
- EnergyQueuePolicy              Lyapunov drift-plus-penalty for a long-run energy budget B (J/answer):
                                 pick argmin_k  -V*acc_k + Q*E_k  over the deadline-feasible widths, then
                                 Q <- max(0, Q + E_used - B). Linear in (acc, E), so it only ever picks
                                 vertices of the upper convex hull and time-shares two of them to sit at B.
                                 task_aware=True uses acc per test set (the request's task type is known
                                 at deployment, e.g. which clinical form is open), else one accuracy per width.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from data_io import Profile


@dataclass
class Request:
    prompt_tokens: int
    deadline_ms: Optional[float] = None     # None = no latency requirement
    task: Optional[int] = None              # index into data_io.SOURCES, if the deployment knows it


class Policy:
    name = "policy"

    def choose(self, req: Request) -> int:
        raise NotImplementedError

    def observe(self, level: int, latency_ms: float, energy_j: float) -> None:
        pass


class Fixed(Policy):
    def __init__(self, level: int):
        self.level = level
        self.name = f"fixed:L{level}"

    def choose(self, req):
        return self.level


class _Predictor:
    """Latency/energy prediction from the device profile, optionally corrected by feedback."""

    def __init__(self, prof: Profile, levels: Sequence[int], answer_tokens: float, feedback: bool, alpha: float):
        self.prof, self.levels, self.n = prof, list(levels), answer_tokens
        self.feedback, self.alpha, self.rho = feedback, alpha, 1.0
        self._last_pred: Dict[int, float] = {}

    def latency(self, level: int, prompt_tokens: int) -> float:
        raw = float(self.prof.latency_ms(level, prompt_tokens, self.n))
        self._last_pred[level] = raw
        return raw * self.rho

    def energy(self, level: int, prompt_tokens: int) -> float:
        return float(self.prof.energy_j(level, self.latency(level, prompt_tokens)))

    def update(self, level: int, latency_ms: float) -> None:
        if self.feedback and level in self._last_pred:
            r = latency_ms / self._last_pred[level]
            self.rho = (1 - self.alpha) * self.rho + self.alpha * r

    def feasible(self, req: Request, margin: float) -> List[int]:
        if req.deadline_ms is None:
            return list(self.levels)
        ok = [l for l in self.levels if self.latency(l, req.prompt_tokens) * margin <= req.deadline_ms]
        return ok or [min(self.levels)]


class DeadlinePolicy(Policy):
    def __init__(self, prof: Profile, levels: Sequence[int], answer_tokens: float = 6, feedback: bool = False,
                 alpha: float = 0.2, margin: float = 1.0):
        self.pred = _Predictor(prof, levels, answer_tokens, feedback, alpha)
        self.margin = margin
        self.name = "deadline+fb" if feedback else "deadline"

    def choose(self, req):
        return max(self.pred.feasible(req, self.margin))

    def observe(self, level, latency_ms, energy_j):
        self.pred.update(level, latency_ms)


class EnergyQueuePolicy(Policy):
    def __init__(self, prof: Profile, levels: Sequence[int], acc: np.ndarray, budget_j: float, V: float = 1.0,
                 answer_tokens: float = 6, feedback: bool = True, alpha: float = 0.2, margin: float = 1.0,
                 task_aware: bool = False, typical_prompt_tokens: int = 300):
        """acc: [L] accuracy per width (task-agnostic) or [S, L] per task (task_aware=True), in %."""
        self.pred = _Predictor(prof, levels, answer_tokens, feedback, alpha)
        self.acc, self.B, self.V, self.Q = np.asarray(acc, float), budget_j, V, 0.0
        self.margin, self.task_aware = margin, task_aware
        self.idx = {l: i for i, l in enumerate(levels)}
        self.name = f"queue{'-task' if task_aware else ''}@{budget_j:.2f}J"
        self.Q = self._equilibrium_backlog(levels, typical_prompt_tokens)

    def _equilibrium_backlog(self, levels, prompt_tokens) -> float:
        """Warm start: the backlog at which the two hull widths bracketing B tie, Q* = V * d(acc)/d(E).
        Starting at 0 the queue first overspends until Q reaches ~Q* (an O(V) transient, which over T
        requests inflates the mean energy by Q*/T)."""
        from offline import upper_hull
        E = [float(self.prof_energy(l, prompt_tokens)) for l in levels]
        a = list(self.acc.mean(0) if self.acc.ndim == 2 else self.acc)
        h = upper_hull(E, a)
        for i, j in zip(h, h[1:]):
            if E[i] <= self.B <= E[j]:
                return self.V * (a[j] - a[i]) / (E[j] - E[i])
        return 0.0

    def prof_energy(self, level, prompt_tokens):
        p = self.pred.prof
        return p.energy_j(level, p.latency_ms(level, prompt_tokens, self.pred.n))

    def _acc(self, level, task):
        a = self.acc[task] if (self.task_aware and task is not None) else (
            self.acc.mean(0) if self.acc.ndim == 2 else self.acc)
        return a[self.idx[level]]

    def choose(self, req):
        cand = self.pred.feasible(req, self.margin)
        score = {l: -self.V * self._acc(l, req.task) + self.Q * self.pred.energy(l, req.prompt_tokens) for l in cand}
        return min(score, key=lambda l: (score[l], -l))

    def observe(self, level, latency_ms, energy_j):
        self.pred.update(level, latency_ms)
        self.Q = max(0.0, self.Q + energy_j - self.B)


# ----------------------------------------------------------------------------- tier 1: deploy time
def deploy_select(prof: Profile, acc: Sequence[float], levels: Sequence[int], *, backend: str,
                  weight_budget_bytes: float, p90_deadline_ms: Optional[float] = None,
                  energy_budget_j: Optional[float] = None, mean_energy: Optional[Sequence[float]] = None,
                  p90_latency: Optional[Sequence[float]] = None, max_anchors: int = 2) -> dict:
    """Pick what to put on the device, once.

    backend="views":    one weight set (all widths are views); every hull width stays selectable at runtime.
    backend="compiled": fixed-shape engines (TensorRT / GGUF from the delta file or per-level HF export) share
                        no weights, so keep at most `max_anchors` widths; with an energy budget, the two hull
                        widths that bracket it (the runtime queue time-shares them), else the widest width
                        meeting the p90 deadline.
    Returns a manifest dict (levels to materialise, memory, reason).
    """
    from offline import upper_hull
    levels = list(levels)
    E = list(mean_energy) if mean_energy is not None else [prof.cost[l].net_power_w for l in levels]
    T = list(p90_latency) if p90_latency is not None else [0.0] * len(levels)
    ok = [i for i in range(len(levels)) if p90_deadline_ms is None or T[i] <= p90_deadline_ms]
    if not ok:
        ok = [0]
    hull = [i for i in upper_hull(E, acc) if i in ok]
    if backend == "views":
        need = prof.resident_bytes
        if need > weight_budget_bytes:
            return {"backend": "views", "fits": False, "need_bytes": need,
                    "reason": "full weight set does not fit; use backend=compiled with one narrow width"}
        return {"backend": "views", "fits": True, "need_bytes": need, "runtime_levels": [levels[i] for i in hull],
                "reason": "all widths resident as views; runtime chooses among the hull widths"}
    fits = [i for i in hull if prof.cost[levels[i]].weight_bytes <= weight_budget_bytes]
    if not fits:
        return {"backend": "compiled", "fits": False, "reason": "no width fits the weight budget"}
    if energy_budget_j is not None and max_anchors >= 2:
        lo = max([i for i in fits if E[i] <= energy_budget_j], default=fits[0])
        hi = min([i for i in fits if E[i] >= energy_budget_j and i != lo], default=None)
        pick = [lo] + ([hi] if hi is not None else [])
        need = sum(prof.cost[levels[i]].weight_bytes for i in pick)
        while need > weight_budget_bytes and len(pick) > 1:
            pick = pick[:1]
            need = prof.cost[levels[pick[0]]].weight_bytes
        reason = "hull widths bracketing the energy budget (runtime time-shares them)"
    else:
        # one width and no time-sharing: the hull does not matter, any width within budget may be best
        single = [i for i in ok if prof.cost[levels[i]].weight_bytes <= weight_budget_bytes]
        pick = [max(single, key=lambda i: (acc[i], -E[i]))]
        need = prof.cost[levels[pick[0]]].weight_bytes
        reason = "most accurate single width within the deadline and weight budget"
    return {"backend": "compiled", "fits": True, "need_bytes": need, "anchors": [levels[i] for i in pick],
            "reason": reason}
