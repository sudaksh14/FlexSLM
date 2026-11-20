"""CPU tests of training/finetune_teacher_lora.py on a tiny Qwen2 model (no GPU, no download)."""
import json
import os

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("peft")
from transformers import Qwen2Config, Qwen2ForCausalLM

from training.finetune_teacher_lora import (
    AnswerLM, IGNORE_INDEX, check_export, lr_at, merge_and_trim, step_plan, wrap_lora)

V, TRIM = 160, 150          # padded vocab (like Qwen2.5's 152,064) and the "student" vocab (151,936)


def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=False)
    return Qwen2ForCausalLM(cfg).eval()


def batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, TRIM, (3, 12), generator=g)
    labels = torch.full_like(ids, IGNORE_INDEX)
    for row, start in enumerate((6, 8, 9)):            # answer = the tail of each row (right padded rows end earlier)
        labels[row, start:] = ids[row, start:]
    return ids, labels


def test_answer_loss_matches_full_model_cross_entropy():
    model = tiny()
    ids, labels = batch()
    with torch.no_grad():
        logits = model(input_ids=ids).logits[:, :-1].float()
    tgt = labels[:, 1:]
    sel = tgt != IGNORE_INDEX
    want = F.cross_entropy(logits[sel], tgt[sel], reduction="sum")
    peft = wrap_lora(model, rank=4, alpha=8, dropout=0.0)
    wrapper = AnswerLM(peft).eval()
    with torch.no_grad():
        ce, n, correct = wrapper(ids, labels)           # LoRA B = 0 at init -> identical to the base model
    assert n == int(sel.sum())
    assert torch.allclose(ce, want, atol=1e-4)
    assert int(correct) == int((logits[sel].argmax(-1) == tgt[sel]).sum())


def test_only_lora_params_train_and_loss_goes_down():
    peft = wrap_lora(tiny(), rank=4, alpha=8, dropout=0.0)
    wrapper = AnswerLM(peft).train()
    params = [p for p in peft.parameters() if p.requires_grad]
    assert params and all("lora_" in n for n, p in peft.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params, lr=2e-2)
    ids, labels = batch()
    first = None
    for _ in range(60):
        ce, n, _ = wrapper(ids, labels)
        loss = ce / n
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.9 * first


def test_merge_and_trim_keeps_logits_and_cuts_vocab(tmp_path):
    peft = wrap_lora(tiny(), rank=4, alpha=8, dropout=0.0)
    for n, p in peft.named_parameters():                 # make the adapter non-trivial
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)
    ids, _ = batch()
    peft.eval()
    with torch.no_grad():
        before = peft(input_ids=ids).logits[..., :TRIM]
    merged = merge_and_trim(peft, TRIM)
    assert merged.get_input_embeddings().weight.shape[0] == TRIM and merged.config.vocab_size == TRIM
    with torch.no_grad():
        after = merged.eval()(input_ids=ids).logits
    assert after.shape[-1] == TRIM
    assert torch.allclose(before, after, atol=1e-4)      # merging changes nothing on the kept ids
    merged.save_pretrained(tmp_path, safe_serialization=True)
    kind, shapes = check_export(str(tmp_path), TRIM)
    assert kind == "qwen2" and shapes["model.embed_tokens.weight"][0] == TRIM and shapes["lm_head.weight"][0] == TRIM
    assert json.load(open(os.path.join(tmp_path, "config.json")))["vocab_size"] == TRIM


def test_step_plan_and_schedule():
    lengths = list(range(100))
    plan = step_plan(100, 32, 1.0, seed=42)
    assert len(plan) == 3 and all(len(s) == 32 for s in plan)
    flat = [i for s in plan for i in s]
    assert len(set(flat)) == 96                          # drop_last, no repeats within an epoch
    assert plan == step_plan(100, 32, 1.0, seed=42)      # deterministic
    smoke = step_plan(100, 32, 1.0, seed=42, lengths=lengths, longest_first=True)
    assert min(lengths[i] for i in smoke[0]) >= max(lengths[i] for i in smoke[-1])   # longest sequences first
    assert lr_at(0, 100, 0.03, 1e-4) < lr_at(2, 100, 0.03, 1e-4) <= 1e-4     # warmup rises
    assert lr_at(99, 100, 0.03, 1e-4) < 1e-5                                   # cosine decays to ~0
