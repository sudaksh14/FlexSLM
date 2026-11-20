"""Unit tests on synthetic inputs (no data files needed). Run: python controller/test_controller.py
(or pytest controller/test_controller.py)."""
import numpy as np

from data_io import LevelCost, Profile
from offline import hull_value, pareto, upper_hull
from policies import DeadlinePolicy, EnergyQueuePolicy, Fixed, Request, deploy_select


def toy_profile():
    # 3 widths: latency 100/150/200 ms + 0 per prompt token, power 2/3/4 W, 100/200/300 MB
    cost = {k: LevelCost(level=k, params=0, weight_bytes=(k + 1) * 100_000_000, ttft_a=100 + 50 * k, ttft_b=0.0,
                         tpot_ms=0.0, net_power_w=2.0 + k, kv_cache_bytes=0) for k in range(3)}
    return Profile(mode="toy", levels=[0, 1, 2], cost=cost, resident_bytes=300_000_000,
                   ratio_samples=np.ones(10), chars_per_token=4.0)


def test_pareto_and_hull():
    cost, acc = [1, 2, 3, 4, 5], [20, 20, 21, 25, 26]
    assert pareto(cost, acc) == [0, 2, 3, 4]       # index 1: same accuracy, more cost
    assert upper_hull(cost, acc) == [0, 3, 4]      # index 2 lies under the 0 -> 3 segment
    assert abs(hull_value(cost, acc, [0, 3, 4], 2.5) - 22.5) < 1e-9


def test_deadline_picks_widest_feasible():
    p = DeadlinePolicy(toy_profile(), [0, 1, 2], answer_tokens=0)
    assert p.choose(Request(prompt_tokens=10, deadline_ms=160)) == 1
    assert p.choose(Request(prompt_tokens=10, deadline_ms=50)) == 0     # nothing fits: narrowest
    assert p.choose(Request(prompt_tokens=10)) == 2                     # no deadline: widest


def test_feedback_tracks_slowdown():
    p = DeadlinePolicy(toy_profile(), [0, 1, 2], answer_tokens=0, feedback=True, alpha=0.5)
    for _ in range(20):
        lvl = p.choose(Request(prompt_tokens=10, deadline_ms=210))
        p.observe(lvl, latency_ms=1.5 * (100 + 50 * lvl), energy_j=0)
    assert abs(p.pred.rho - 1.5) < 1e-3
    assert p.choose(Request(prompt_tokens=10, deadline_ms=210)) == 0      # 1.5 x 150 > 210


def test_queue_meets_budget_and_time_shares():
    prof = toy_profile()
    acc = np.array([20.0, 21.0, 30.0])                 # width 1 under the hull -> never used
    pol = EnergyQueuePolicy(prof, [0, 1, 2], acc, budget_j=0.5, V=10, answer_tokens=0, feedback=False)
    used, en = [], []
    for _ in range(4000):
        l = pol.choose(Request(prompt_tokens=10))
        e = float(prof.energy_j(l, prof.latency_ms(l, 10, 0)))   # 0.2 / 0.45 / 0.8 J
        pol.observe(l, 0, e)
        used.append(l)
        en.append(e)
    assert abs(np.mean(en) - 0.5) < 0.01
    assert set(used) == {0, 2}


def test_task_aware_prefers_sensitive_task():
    prof = toy_profile()
    acc = np.array([[20, 20, 40], [20, 20, 21]], float)    # task 0 gains from width, task 1 does not
    pol = EnergyQueuePolicy(prof, [0, 1, 2], acc, budget_j=0.5, V=10, answer_tokens=0, feedback=False,
                            task_aware=True)
    wide = {0: 0, 1: 0}
    for i in range(4000):
        t = i % 2
        l = pol.choose(Request(prompt_tokens=10, task=t))
        pol.observe(l, 0, float(prof.energy_j(l, prof.latency_ms(l, 10, 0))))
        wide[t] += l == 2
    assert wide[0] > 5 * max(1, wide[1])


def test_deploy_select():
    prof = toy_profile()
    acc, E, T = [20, 21, 30], [0.2, 0.45, 0.8], [100, 150, 200]
    v = deploy_select(prof, acc, [0, 1, 2], backend="views", weight_budget_bytes=1e9, mean_energy=E, p90_latency=T)
    assert v["fits"] and v["runtime_levels"] == [0, 2]
    c = deploy_select(prof, acc, [0, 1, 2], backend="compiled", weight_budget_bytes=250e6, mean_energy=E,
                      p90_latency=T)
    assert c["anchors"] == [1]                         # single width: best that fits, hull or not
    c2 = deploy_select(prof, acc, [0, 1, 2], backend="compiled", weight_budget_bytes=1e9, energy_budget_j=0.5,
                       mean_energy=E, p90_latency=T)
    assert c2["anchors"] == [0, 2]


def test_runner_helpers():
    from jetson_runner import parse_phases, window_energy_j

    class T:   # fake PowerTrace: 1 W (1000 mW) constant, samples every 10 ms from t=0 to t=1 s
        t = [i / 100 for i in range(101)]
        rails = {"VDD_IN": [1000.0] * 101}

    assert abs(window_energy_j(T, "VDD_IN", 0.205, 0.705, wait_s=0) - 0.5) < 1e-9   # clipped to the window
    assert window_energy_j(T, None, 0, 1) is None
    assert parse_phases("15W:10,15W+gpu:20,keep+cpu:5") == [("15W", None, 10), ("15W", "gpu", 20), ("keep", "cpu", 5)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
