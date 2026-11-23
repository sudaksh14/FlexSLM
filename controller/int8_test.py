"""Nested 1.7B vs the two-model pair at int8 on the Jetson (board only; memory / latency / energy, NO accuracy: random weights).

    python3 controller/int8_test.py --case nested --prec int8 --out DIR/nested.int8.jsonl [--load]
Same decoder and lean settings as coresident_test.py (last-position logits, page cache evicted, one fresh process per case).
int8 = weight-only, symmetric, one scale per output row (max-abs / 127), embedding / output head / norms stay bf16; the layer
multiplies int8 by its row scale in bf16 on the fly, so it has no optimised kernel (torch 2.8 has no CUDA int8 matmul): 3.5-9x slower per matrix than bf16.
Absolute int8 speed is therefore NOT a result; memory is, and int8 energy is not measured.
A narrower width of a nested model is a prefix view of the shared int8 weights and row scales (eval/fake_quant.py is the same scheme).

Cases   small  596M alone            big    1.72B alone
        nested 1.7B, the five even-head widths of run 4 (0.50 / 0.75 / 1.05 / 1.33 / 1.72B), one weight set
        pair   1.72B + 596M          pairn  1.72B + nested 0.6B (188M / 364M / 596M)
        both   nested 1.7B (5 widths) + nested 0.6B (3 widths)
Energy: VDD_IN integrated over a block of requests per width minus idle power (32 forced tokens).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "jetson"))
os.environ["MEDNEST_FORCE_LEN"] = "1"
import torch  # noqa: E402
from torch import nn  # noqa: E402

from coresident_test import meminfo, vmstat  # noqa: E402

# (hidden, heads, kv, mlp) per width; last = the full width
W06 = {"188M": (512, 8, 4, 1536), "364M": (768, 12, 6, 2304), "596M": (1024, 16, 8, 3072)}
W17 = {"0.50B": (1024, 8, 4, 2944), "0.75B": (1280, 10, 5, 3840), "1.05B": (1536, 12, 6, 4800),
       "1.33B": (1792, 14, 7, 5248), "1.72B": (2048, 16, 8, 6144)}
KEYS = ("hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size")


class Int8Linear(nn.Module):
    def __init__(self, w, s):
        super().__init__()
        self.w, self.s = nn.Parameter(w, requires_grad=False), nn.Parameter(s, requires_grad=False)

    def forward(self, x):
        return nn.functional.linear(x, self.w.to(x.dtype) * self.s)


def quantise(w):
    s = (w.float().abs().amax(1, keepdim=True) / 127).clamp_min(1e-30).to(torch.bfloat16)
    return torch.round(w.float() / s.float()).clamp_(-127, 127).to(torch.int8), s


def _set(model, name, mod):
    parent, _, attr = name.rpartition(".")
    setattr(model.get_submodule(parent), attr, mod)


class Int8Elastic:
    """Interface the decoders use: .levels, .width(level). One shared int8 full model, narrower widths are prefix views."""

    def __init__(self, full, configs):
        self.full, self.configs = full, configs
        self.levels = list(range(len(configs)))
        self.q = {}
        for name, m in list(full.model.named_modules()):
            if isinstance(m, nn.Linear):
                w, s = quantise(m.weight.data)
                self.q["model." + name] = (w, s)
                _set(full.model, name, Int8Linear(w, s))
                del m
        torch.cuda.empty_cache()
        self.views = {}

    def width(self, l):
        if l not in self.views:
            self.views[l] = self._build(self.configs[l])
        return self.views[l]

    def _build(self, cfg):
        from transformers import AutoModelForCausalLM

        from mednest_jetson.elastic import _prefix, _rebuild_meta_buffers
        if all(getattr(cfg, k) == getattr(self.full.config, k) for k in KEYS):
            return self.full
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation=self.full.config._attn_implementation)
        fp = dict(self.full.named_parameters(remove_duplicate=False))
        for name, mod in list(m.model.named_modules()):
            if isinstance(mod, nn.Linear):
                w, s = self.q["model." + name]
                _set(m.model, name, Int8Linear(w[:mod.out_features, :mod.in_features], s[:mod.out_features]))
        for name, p in list(m.named_parameters(remove_duplicate=False)):
            if p.is_meta and not name.startswith("lm_head"):
                parent, _, attr = name.rpartition(".")
                src = fp[name]
                setattr(m.get_submodule(parent), attr, nn.Parameter(_prefix(src.detach(), p.shape), requires_grad=False))
        m.lm_head.weight = m.model.embed_tokens.weight
        _rebuild_meta_buffers(m, self.full.lm_head.weight.device)
        m.eval()
        m.generation_config = self.full.generation_config
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["small", "big", "nested", "pair", "pairn", "both"])
    ap.add_argument("--prec", default="int8", choices=["bf16", "int8"])
    ap.add_argument("--models", default=os.path.expanduser("~/mednest/models"))
    ap.add_argument("--subset", default=os.path.join(REPO, "jetson", "questions", "subset.jsonl"))
    ap.add_argument("--n", type=int, default=14, help="ordinary questions (plus the 4 longest)")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--no-energy", action="store_true", help="int8 here has no native kernel (dequantise on the fly), so its energy is not an int8 number")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import subprocess
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from jetson_runner import Load, window_energy_j
    from mednest_jetson import elastic as el
    from mednest_jetson import workload as wl
    from mednest_jetson.decode import ElasticDecoders, max_len_for
    from mednest_jetson.measure import MemPoller
    from nestbench.power import PowerSampler, measure_idle

    out = open(args.out, "w")

    def rec(**k):
        out.write(json.dumps(k) + "\n")
        out.flush()
        print({a: (round(b, 2) if isinstance(b, float) else b) for a, b in k.items()}, flush=True)

    # kernel self-test: the int8 layer equals a bf16 layer with the dequantised weights
    torch.manual_seed(0)
    w = (torch.randn(256, 384, device="cuda") * 0.05).to(torch.bfloat16)
    q, s = quantise(w)
    x = torch.randn(4, 384, device="cuda").to(torch.bfloat16)
    a, b = Int8Linear(q, s)(x), nn.functional.linear(x, (q.to(torch.bfloat16) * s))
    rec(kind="selftest", equal=bool(torch.equal(a, b)), int8_bytes=q.numel() * q.element_size())

    def build(name, widths):
        d = os.path.join(args.models, name)
        top = list(widths.values())[-1]
        cfg = AutoConfig.from_pretrained(d)
        assert all(getattr(cfg, k) == v for k, v in zip(KEYS, top)), (name, top)
        cfg.tie_word_embeddings = True
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(0)
        with torch.device("cuda"):
            full = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16).eval()
        full.generation_config = GenerationConfig.from_pretrained(d)
        cfgs = []
        for h, a_, k_, m_ in widths.values():
            c = cfg.__class__.from_dict(cfg.to_dict())
            for k, v in zip(KEYS, (h, a_, k_, m_)):
                setattr(c, k, v)
            cfgs.append(c)
        if args.prec == "int8":
            e = Int8Elastic(full, cfgs)
            for l in e.levels:
                e.width(l)
        else:
            e = el.ElasticModel.from_widths(full, [dict(zip(KEYS, v)) for v in widths.values()])
            e.build_all()
        for l in e.levels:
            m = e.width(l)
            f = m.forward
            m.forward = lambda *a, _f=f, **k: _f(*a, **{**k, "logits_to_keep": 1})
        return e, list(widths), AutoTokenizer.from_pretrained(d)

    cases = {"small": [("Qwen3-0.6B", {"596M": W06["596M"]})], "big": [("Qwen3-1.7B", {"1.72B": W17["1.72B"]})],
             "nested": [("Qwen3-1.7B", W17)],
             "pair": [("Qwen3-1.7B", {"1.72B": W17["1.72B"]}), ("Qwen3-0.6B", {"596M": W06["596M"]})],
             "pairn": [("Qwen3-1.7B", {"1.72B": W17["1.72B"]}), ("Qwen3-0.6B", W06)],
             "both": [("Qwen3-1.7B", W17), ("Qwen3-0.6B", W06)]}
    rec(kind="start", case=args.case, prec=args.prec, load=args.load, **meminfo(), **vmstat())
    load = None
    try:
        mods = []
        for name, widths in cases[args.case]:
            e, labels, tok = build(name, widths)
            torch.cuda.empty_cache()
            mods.append((e, labels, tok))
            rec(kind="built", model=name, widths=labels, cuda_alloc_gb=torch.cuda.memory_allocated() / 1e9,
                cuda_reserved_gb=torch.cuda.memory_reserved() / 1e9, **meminfo())
        subprocess.run([sys.executable, os.path.join(REPO, "jetson", "drop_file_cache.py"), args.models,
                        os.path.expanduser("~/mednest-results"), os.path.expanduser("~/.cache"), os.environ.get("VIRTUAL_ENV", "/tmp")],
                       check=True, stdout=subprocess.DEVNULL)
        torch.cuda.empty_cache()
        rows = [json.loads(l) for l in open(args.subset, encoding="utf8")]
        tok = mods[0][2]
        enc = [wl.encode(tok, wl.prompt_for(r, "as_is"), "as_is", "cpu") for r in rows]
        plen = [int(v.shape[1]) for v in enc]
        long4 = sorted(range(len(rows)), key=lambda i: -plen[i])[:4]
        pick = list(range(0, len(rows), max(1, len(rows) // args.n)))[:args.n]
        needs = sorted({plen[i] + 32 for i in pick + long4})
        ml = max_len_for([plen[i] for i in pick + long4], 32)
        decs = []
        for e, labels, _ in mods:
            d = ElasticDecoders(e, ml, use_graph=True)
            ms = d.prepare_all(needs)
            decs.append(d)
            rec(kind="prepared", ms=ms, cuda_alloc_gb=torch.cuda.memory_allocated() / 1e9,
                cuda_reserved_gb=torch.cuda.memory_reserved() / 1e9, kv_gb=d.kv_bytes() / 1e9, **meminfo())
        arms = [(mi, li) for mi, (e, labels, _) in enumerate(mods) for li in range(len(labels))]
        for mi, li in arms:
            decs[mi].get(li).generate(enc[pick[0]].cuda(), 4, [])
        if args.load:
            load = Load("gpu", 0.6, 2048, 4)
            rec(kind="load_running", attempts=load.start_attempts, **meminfo())
        v0, t0 = vmstat(), time.time()
        with MemPoller() as mp:
            for i in pick + long4:
                for mi, li in arms:
                    for rep in (0, 1):
                        r = decs[mi].get(li).generate(enc[i].cuda(), 32, [])
                        out.write(json.dumps(dict(kind="gen", model=mods[mi][1][li], rep=rep, prompt=plen[i], long=i in long4,
                                                  ttft_ms=r["ttft_ms"], ms_tok=r["ms_per_decode_token"], total_ms=r["total_ms"])) + "\n")
                out.flush()
        v1 = vmstat()
        rec(kind="alternation", seconds=time.time() - t0, mem_peak_used_mb=mp.peak_mb,
            cuda_peak_alloc_gb=torch.cuda.max_memory_allocated() / 1e9, swap_out_pages=v1["pswpout"] - v0["pswpout"], **meminfo())
        if not args.load and not args.no_energy:
            time.sleep(5)
            idle = measure_idle(seconds=10.0)
            idle_w = idle.get("idle_power_w")
            with PowerSampler(source="auto", period_ms=5) as ps:
                rail = ps.trace.total_rail() if ps.trace.n() else None
                for mi, li in arms:
                    e0 = time.monotonic()
                    tot, n = 0.0, 0
                    for i in pick:
                        r = decs[mi].get(li).generate(enc[i].cuda(), 32, [])
                        tot += r["total_ms"]
                        n += 1
                    e1 = time.monotonic()
                    g = window_energy_j(ps.trace, rail, e0, e1)
                    rec(kind="energy", model=mods[mi][1][li], n=n, mean_ms=tot / n, gross_j=(g / n if g else None),
                        net_j=((g - idle_w * (e1 - e0)) / n if g and idle_w else None), idle_w=idle_w)
        if load is not None:
            rec(kind="load_ops", ops=load.ops(), alive=load.procs[0].poll() is None)
        rec(kind="done", ok=True)
    except Exception as ex:                                   # out of memory is an outcome, not a crash
        rec(kind="failed", error=type(ex).__name__, message=str(ex)[:300], **meminfo(), **vmstat())
        sys.exit(3)
    finally:
        if load is not None:
            load.stop()


if __name__ == "__main__":
    main()
