"""CPU tests of the Jetson elastic runtime (jetson/mednest_jetson) on a tiny random Qwen3."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jetson"))
transformers = pytest.importorskip("transformers")
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from mednest_jetson import elastic as el  # noqa: E402
from mednest_jetson import measure, workload  # noqa: E402

WIDTHS = [dict(hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1),
          dict(hidden_size=48, intermediate_size=96, num_attention_heads=2, num_key_value_heads=1),
          dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2)]


def _full(tie=True):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=101, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                      tie_word_embeddings=tie, eos_token_id=100)
    m = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():  # random norms, so a wrong slice of a norm weight is caught
        for n, p in m.named_parameters():
            if n.endswith("norm.weight"):
                p.uniform_(0.5, 1.5)
    return m


def _dense_slice(full, w):
    """What training/export_flex_hf.py writes for a level: contiguous prefix slices."""
    cfg = full.config.__class__.from_dict(full.config.to_dict())
    for k, v in w.items():
        setattr(cfg, k, v)
    d = Qwen3ForCausalLM(cfg).eval()
    src = dict(full.named_parameters(remove_duplicate=False))
    sd = {n: el._prefix(src.get(n, src["model.embed_tokens.weight"]), p.shape).contiguous().clone()
          for n, p in d.named_parameters(remove_duplicate=False)}
    d.load_state_dict(sd, strict=False)
    return d


@pytest.mark.parametrize("tie", [True, False])
def test_views_equal_dense_slices_and_share_storage(tie):
    full = _full(tie)
    e = el.ElasticModel.from_widths(full, WIDTHS)
    ids = torch.randint(0, 100, (2, 9))
    for lvl, w in enumerate(WIDTHS):
        v = e.width(lvl)
        assert torch.allclose(v(ids).logits, _dense_slice(full, w)(ids).logits, atol=1e-5), lvl
        for p in v.parameters():  # every weight is a view into the full model's storage
            assert any(p.untyped_storage().data_ptr() == q.untyped_storage().data_ptr() for q in full.parameters())
    assert e.unique_param_bytes() == measure.param_bytes(full)  # nine views cost nothing extra
    assert torch.equal(e.width(2)(ids).logits, full(ids).logits)


def test_materialize_mode_equals_views():
    full = _full()
    ev, em = el.ElasticModel.from_widths(full, WIDTHS), el.ElasticModel.from_widths(full, WIDTHS, mode="materialize")
    ids = torch.randint(0, 100, (1, 7))
    for lvl in range(3):
        m = em.width(lvl)
        assert all(p.is_contiguous() for p in m.parameters())
        assert torch.allclose(m(ids).logits, ev.width(lvl)(ids).logits, atol=1e-6)
    assert em.switch(0)["switch_ms"] >= 0


def test_widest_level_must_be_the_loaded_model():
    with pytest.raises(ValueError):
        el.ElasticModel.from_widths(_full(), WIDTHS[:2])


def test_load_width_from_the_single_full_checkpoint(tmp_path):
    full = _full()
    full.save_pretrained(tmp_path)
    e = el.ElasticModel.from_widths(full, WIDTHS)
    ids = torch.randint(0, 100, (1, 11))
    for lvl in range(3):
        m = measure.load_width_from_full(str(tmp_path), e.spec(lvl).config, device="cpu", dtype=torch.float32)
        assert torch.allclose(m(ids).logits, e.width(lvl)(ids).logits, atol=1e-6)
        assert measure.param_bytes(m) < measure.param_bytes(full) or lvl == 2


def test_timed_greedy_equals_generate():
    full = _full()
    v = el.ElasticModel.from_widths(full, WIDTHS).width(1)
    ids = torch.randint(0, 100, (1, 6))
    r = workload.timed_greedy(v, ids, max_new_tokens=8, eos=[100])
    ref = v.generate(ids, max_new_tokens=8, do_sample=False, eos_token_id=100, pad_token_id=0)[0, 6:].tolist()
    assert r["new_ids"] == ref[:len(r["new_ids"])]
    assert r["ttft_ms"] > 0 and r["n_new"] == len(r["new_ids"]) and r["prompt_tokens"] == 6


def test_subset_is_stratified_and_keeps_test_indices(tmp_path):
    import json
    p = tmp_path / "test.jsonl"
    with open(p, "w") as f:
        for i in range(70):
            f.write(json.dumps({"source": workload.SOURCES[i % 7], "gt": "A", "prompt": f"q{i}"}) + "\n")
    rows = workload.make_subset(str(p), str(tmp_path / "s.jsonl"), per_source=3)
    assert len(rows) == 21 and all(sum(r["source"] == s for r in rows) == 3 for s in workload.SOURCES)
    orig = [json.loads(l) for l in open(p)]
    assert all(orig[r["test_index"]]["prompt"] == r["prompt"] for r in rows)
    assert workload.make_subset(str(p), str(tmp_path / "s2.jsonl"), per_source=3) == rows  # seeded


def test_mem_poller_starts_and_stops():
    import time as _t
    with measure.MemPoller(period_s=0.005) as mp:
        _t.sleep(0.03)
    assert not mp.is_alive()


def test_static_cache_decoder_equals_eager_also_with_a_shared_kv_buffer():
    from mednest_jetson.decode import ElasticDecoders, GraphDecoder
    e = el.ElasticModel.from_widths(_full(), WIDTHS)
    decs = ElasticDecoders(e, max_len=48, use_graph=False)       # CPU: same code path minus the graph
    torch.manual_seed(1)
    for _ in range(2):                                             # 2 requests: the cache is reused, never cleared
        for lvl in (0, 2, 1):
            ids = torch.randint(0, 100, (1, int(torch.randint(4, 12, (1,)))))
            ref = workload.timed_greedy(e.width(lvl), ids, 10, eos=[100])["new_ids"]
            assert decs.get(lvl).generate(ids, 10, eos=[100])["new_ids"] == ref, lvl
    top = decs.decoder(2, 48).cache.key_cache[0]
    assert decs.decoder(0, 48).cache.key_cache[0].untyped_storage().data_ptr() == top.untyped_storage().data_ptr()
    assert decs.kv_bytes() == top.untyped_storage().nbytes() * 2 * len(decs.decoder(2, 48).cache.key_cache)
    with pytest.raises(ValueError):
        GraphDecoder(e.width(0), 8, use_graph=False).generate(torch.zeros(1, 6, dtype=torch.long), 5, [100])


def test_length_buckets_pick_the_smallest_cache_that_fits():
    from mednest_jetson.decode import ElasticDecoders, _bucket
    assert _bucket(300, 4096) == 512 and _bucket(513, 4096) == 1024 and _bucket(700, 768) == 768
    with pytest.raises(ValueError):
        _bucket(900, 768)
    e = el.ElasticModel.from_widths(_full(), WIDTHS)
    decs = ElasticDecoders(e, max_len=1024, use_graph=False)
    ids = torch.randint(0, 100, (1, 8))
    ref = workload.timed_greedy(e.width(1), ids, 6, eos=[100])["new_ids"]
    assert decs.get(1).generate(ids, 6, eos=[100])["new_ids"] == ref
    assert set(decs._dec) == {(2, 512), (1, 512)}                    # only the bucket that was needed


def test_warmup_builds_everything_a_request_needs_and_leaves_outputs_unchanged():
    from mednest_jetson.decode import ElasticDecoders
    e = el.ElasticModel.from_widths(_full(), WIDTHS)
    decs = ElasticDecoders(e, max_len=1024, use_graph=False)
    ids = torch.randint(0, 100, (1, 9))
    refs = {l: workload.timed_greedy(e.width(l), ids, 6, eos=[100])["new_ids"] for l in range(3)}
    assert decs.prepare_all([9 + 6, 600], levels=[0, 1, 2]) >= 0
    built = set(decs._dec)
    assert built == {(l, b) for l in range(3) for b in (512, 1024)}      # every width x every needed length
    assert all(d.warm for d in decs._dec.values())
    assert decs.get(1).prepare(15) == 0.0                                  # already warm: nothing left to pay
    for l in range(3):                                                     # the warm-up step left no trace
        assert decs.get(l).generate(ids, 6, eos=[100])["new_ids"] == refs[l]
    assert set(decs._dec) == built                                         # and the requests built nothing new
