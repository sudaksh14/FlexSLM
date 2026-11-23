"""Weight-only fake int8 (eval/fake_quant.py): exact int8 * scale numerics, and slicing a quantised matrix = quantising-then-slicing."""
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.fake_quant import dequantise, is_linear, quantise


def test_scale_and_slice():
    torch.manual_seed(0)
    w = (torch.randn(64, 96) * 0.05).to(torch.bfloat16)
    q, s = quantise(w)
    assert q.dtype == torch.int8 and s.shape == (64, 1)
    d = dequantise(q, s)
    assert (d.float() - w.float()).abs().max() <= 0.5 * s.float().max() * 1.1 + 1e-3
    assert torch.equal(dequantise(q[:32, :48], s[:32]), d[:32, :48])
    assert is_linear("model.layers.0.self_attn.q_proj.weight") and not is_linear("model.embed_tokens.weight")


def test_cli(tmp_path):
    import subprocess
    torch.manual_seed(1)
    full, nar = tmp_path / "f", tmp_path / "n"
    for d in (full, nar):
        d.mkdir(); json.dump({}, open(d / "config.json", "w"))
    p = torch.randn(8, 12).to(torch.bfloat16)
    save_file({"model.layers.0.mlp.up_proj.weight": p, "model.embed_tokens.weight": torch.randn(4, 8).to(torch.bfloat16)}, full / "model.safetensors")
    save_file({"model.layers.0.mlp.up_proj.weight": p[:4, :6].clone(), "model.embed_tokens.weight": torch.randn(4, 4).to(torch.bfloat16)}, nar / "model.safetensors")
    root = os.path.join(os.path.dirname(__file__), "..")
    subprocess.run([sys.executable, "-m", "eval.fake_quant", str(full), str(tmp_path / "of"), "--slice", str(nar), str(tmp_path / "on")], check=True, cwd=root)
    from safetensors.torch import load_file
    a, b = load_file(tmp_path / "of" / "model.safetensors"), load_file(tmp_path / "on" / "model.safetensors")
    assert torch.equal(a["model.layers.0.mlp.up_proj.weight"][:4, :6], b["model.layers.0.mlp.up_proj.weight"])
