"""Greedy decoding with a static KV cache and CUDA-graph decode steps.

Why: at batch 1 a 0.2-0.6B model in eager PyTorch is bound by Python and kernel-launch overhead
(~28 layers x ~15 small kernels per token), not by the weights it reads. On a DAS-6 A10 every
width decodes at the same ~32 ms/token although reading the full width's weights takes ~2 ms
(validation job 28336), so eager timings would hide exactly the effect of width. Capturing one
decode step in a CUDA graph and replaying it removes that overhead, as deployment runtimes do
(torch.compile "reduce-overhead" does the same, but needs Triton, which the Jetson wheel lacks).

GraphDecoder(model, max_len)       one static cache + one captured decode step per model
GraphDecoder(model, max_len, share=wider)   the KV buffers are prefix views (first KV heads) of a
                                   wider width's buffers: all widths of an elastic model share one
                                   full-width KV allocation (only one width runs at a time)
Prefill runs eagerly (it is compute-bound). Positions beyond the current one are masked, so the
cache is never cleared between requests. Outputs equal eager greedy decoding (check_decode()).
"""
from __future__ import annotations

import time
from typing import List, Optional

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class GraphDecoder:
    def __init__(self, model, max_len: int, share: Optional["GraphDecoder"] = None, use_graph: bool = True):
        from transformers import StaticCache
        self.model = model
        p = next(model.parameters())
        self.device, self.dtype, self.max_len = p.device, p.dtype, int(max_len)
        if share is not None:
            if share.max_len != self.max_len:
                raise ValueError("a shared KV cache must have the same max_len")
            # Build the cache object for this width (its own small bookkeeping), then point its
            # buffers at prefix views of the wider width's buffers and free its own.
            self.cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=1,
                                     device=self.device, dtype=self.dtype)
            self.cache.max_cache_len = self.max_len
            g = model.config.num_key_value_heads
            for i in range(len(self.cache.key_cache)):
                self.cache.key_cache[i] = share.cache.key_cache[i][:, :g]
                self.cache.value_cache[i] = share.cache.value_cache[i][:, :g]
        else:
            self.cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=self.max_len,
                                     device=self.device, dtype=self.dtype)
        self.use_graph = use_graph and self.device.type == "cuda"
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.graph = None
        self.out = None

    def kv_bytes(self) -> int:
        return sum(t.untyped_storage().nbytes() for t in {id(t.untyped_storage()): t
                   for t in self.cache.key_cache + self.cache.value_cache}.values())

    def _forward_step(self):
        return self.model(input_ids=self.tok, past_key_values=self.cache, cache_position=self.pos,
                          use_cache=True).logits

    @torch.no_grad()
    def _step(self):
        if not self.use_graph:
            self.out = self._forward_step()
            return
        if self.graph is None:
            # warm-up on a side stream (required before capture); these runs write the same
            # token at the same position as the replay below, so the cache stays correct
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._forward_step()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.out = self._forward_step()
            self.graph = g
        self.graph.replay()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos: List[int]) -> dict:
        """Same record as workload.timed_greedy (new_ids, ttft_ms, decode_ms, ...)."""
        n = int(input_ids.shape[1])
        if n + max_new_tokens > self.max_len:
            raise ValueError(f"prompt {n} + {max_new_tokens} new tokens > static cache {self.max_len}")
        _sync()
        t0 = time.perf_counter()
        logits = self.model(input_ids=input_ids, past_key_values=self.cache,
                            cache_position=torch.arange(n, device=self.device), use_cache=True).logits
        nxt = logits[:, -1].argmax(-1)
        new = [nxt.item()]
        t1 = time.perf_counter()
        pos = n
        while len(new) < max_new_tokens and new[-1] not in eos:
            self.tok.copy_(nxt.view(1, 1))
            self.pos.fill_(pos)
            self._step()
            nxt = self.out[:, -1].argmax(-1)
            new.append(nxt.item())
            pos += 1
        _sync()
        t2 = time.perf_counter()
        n_dec = len(new) - 1
        return {"new_ids": new, "n_new": len(new), "prompt_tokens": n,
                "ttft_ms": (t1 - t0) * 1000.0, "decode_ms": (t2 - t1) * 1000.0, "total_ms": (t2 - t0) * 1000.0,
                "ms_per_decode_token": ((t2 - t1) * 1000.0 / n_dec) if n_dec else None,
                "stopped_on_eos": new[-1] in eos, "decoder": "graph" if self.use_graph else "static"}


BUCKETS = (512, 1024, 2048, 4096)


def _bucket(need: int, max_len: int, buckets=BUCKETS) -> int:
    """Smallest cache length >= need. Each decode step builds a mask over, and attends across, the
    whole static cache, so one cache sized for the longest prompt slows every short request
    (validation job 28339: +25% per answer); length buckets keep the cost near the request's own."""
    for b in sorted(buckets):
        if need <= b <= max_len:
            return b
    if need > max_len:
        raise ValueError(f"request needs {need} cache positions > max_len {max_len}")
    return max_len


class _Proxy:
    """What the benchmarks call: generate() picks the bucket for this request."""

    def __init__(self, parent, level):
        self.parent, self.level = parent, level

    def generate(self, input_ids, max_new_tokens, eos):
        need = int(input_ids.shape[1]) + max_new_tokens
        return self.parent.decoder(self.level, _bucket(need, self.parent.max_len)).generate(input_ids, max_new_tokens, eos)

    def kv_bytes(self) -> int:
        return self.parent.kv_bytes()


class ElasticDecoders:
    """GraphDecoders for every width of an ElasticModel, per cache-length bucket. Within a bucket all
    widths share ONE full-width KV allocation (prefix views of its KV heads). Caches and graphs are
    created on first use of a (width, bucket)."""

    def __init__(self, elastic, max_len: int, use_graph: bool = True):
        self.elastic, self.max_len, self.use_graph = elastic, int(max_len), use_graph
        self._dec = {}

    def decoder(self, level: int, bucket: int) -> GraphDecoder:
        top = max(self.elastic.levels)
        if (top, bucket) not in self._dec:
            self._dec[(top, bucket)] = GraphDecoder(self.elastic.width(top), bucket, use_graph=self.use_graph)
        if (level, bucket) not in self._dec:
            self._dec[(level, bucket)] = GraphDecoder(self.elastic.width(level), bucket, share=self._dec[(top, bucket)],
                                                      use_graph=self.use_graph)
        return self._dec[(level, bucket)]

    def get(self, level: int) -> _Proxy:
        return _Proxy(self, level)

    def kv_bytes(self) -> int:
        top = max(self.elastic.levels)
        return sum(d.kv_bytes() for (l, _), d in self._dec.items() if l == top)


class DenseDecoders:
    """The same bucketing for one ordinary model (baselines, reloaded separate models)."""

    def __init__(self, model, max_len: int, use_graph: bool = True):
        self.model, self.max_len, self.use_graph = model, int(max_len), use_graph
        self._dec = {}

    def decoder(self, level, bucket: int) -> GraphDecoder:
        if bucket not in self._dec:
            self._dec[bucket] = GraphDecoder(self.model, bucket, use_graph=self.use_graph)
        return self._dec[bucket]

    def get(self, level=None) -> _Proxy:
        return _Proxy(self, level)

    def generate(self, input_ids, max_new_tokens, eos):
        return self.get().generate(input_ids, max_new_tokens, eos)

    def kv_bytes(self) -> int:
        return sum(d.kv_bytes() for d in self._dec.values())


TIE_MARGIN = 0.25   # logits: 2 bf16 steps at |logit| in [16, 32), the range of answer-token logits here


@torch.no_grad()
def compare_decoders(model, tokenizer, rows, ids_list, static_dec: "GraphDecoder", graph_dec: "GraphDecoder",
                     max_new: int = 32) -> dict:
    """graph vs static (must be identical: capture/replay exactness) and static vs eager (may
    differ only where eager's top-2 logits are within TIE_MARGIN at the first divergence: bf16
    ties broken differently). Also: share of prompts with the same scored answer."""
    from . import workload as wl
    eos = wl.eos_ids(model, tokenizer)
    n_gs = n_se = n_ans = 0
    gaps, ms_e, ms_g = [], [], []
    for r, x in zip(rows, ids_list):
        a = wl.timed_greedy(model, x, max_new, eos)
        s_ = static_dec.generate(x, max_new, eos)["new_ids"]
        g = graph_dec.generate(x, max_new, eos)
        ms_e.append(a["ms_per_decode_token"])
        ms_g.append(g["ms_per_decode_token"])
        a, g = a["new_ids"], g["new_ids"]
        n_gs += g == s_
        n_se += s_ == a
        dec = lambda t: tokenizer.decode(t, skip_special_tokens=True)
        n_ans += wl.score(r, dec(a), "as_is")["boxed"] == wl.score(r, dec(g), "as_is")["boxed"]
        if s_ != a:
            k = next((i for i, (u, v) in enumerate(zip(a, s_)) if u != v), min(len(a), len(s_)))
            if k < min(len(a), len(s_)):
                full = torch.cat([x, torch.tensor([a[:k]], device=x.device, dtype=x.dtype)], 1) if k else x
                lo = model(input_ids=full).logits[0, -1].float()
                gaps.append({"pos": k, "gap": (lo.max() - lo[s_[k]]).item()})
    n = len(rows)
    med = lambda v: sorted(x for x in v if x)[len([x for x in v if x]) // 2] if any(v) else None
    return {"n": n, "graph_eq_static": n_gs / n, "static_eq_eager": n_se / n, "same_answer": n_ans / n,
            "divergence_gaps": gaps, "max_gap": max([d["gap"] for d in gaps], default=0.0),
            "eager_ms_tok": med(ms_e), "graph_ms_tok": med(ms_g)}


@torch.no_grad()
def check_decode(model, decoder: GraphDecoder, prompts_ids: List[torch.Tensor], max_new: int, eos: List[int]):
    """Graph decoding vs eager greedy decoding (workload.timed_greedy): same tokens?"""
    from .workload import timed_greedy
    same, rows = 0, []
    for ids in prompts_ids:
        a = timed_greedy(model, ids, max_new, eos)
        b = decoder.generate(ids, max_new, eos)
        ok = a["new_ids"] == b["new_ids"]
        same += ok
        rows.append({"same_tokens": ok, "eager_ms_tok": a["ms_per_decode_token"], "graph_ms_tok": b["ms_per_decode_token"],
                     "eager_total_ms": a["total_ms"], "graph_total_ms": b["total_ms"]})
    return {"same_tokens": same / len(prompts_ids), "rows": rows}


def max_len_for(prompt_lens, max_new: int, multiple: int = 256) -> int:
    need = max(prompt_lens) + max_new
    return ((need + multiple - 1) // multiple) * multiple
