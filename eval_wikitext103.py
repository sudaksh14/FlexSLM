# Token perplexity on the WikiText-103 test set, using the same data pipeline
# as training. Use this to cross-check W&B validation numbers. They should be close.
# (For benchmark comparisons against other models, use eval_lm_harness.py instead.)
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from run_experiment import resolve_from_str
from utils import load_model, load_wikitext


@torch.no_grad()
def eval_ppl(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    vocab_size = model.config.vocab_size

    for input_ids, _ in tqdm(loader, leave=False):
        input_ids = input_ids.to(device)
        logits = model(input_ids)  # [B, S, vocab]
        # Sum loss over all tokens so we can average correctly at the end
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += input_ids[:, 1:].numel()

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def resolve_tokenizer_name(builder, override: str = None, default: str = "gpt2") -> str:
    """
    The dataset must be tokenized with whatever tokenizer the model was actually
    trained/vocab-sized for, or token ids land outside the embedding table and
    crash as an opaque out-of-bounds CUDA gather deep inside the model instead of
    a clear error here. Preference order: explicit --tokenizer override > KD
    teacher (training_context.teacher_hf_model) > warm-start source
    (model_config.pretrained_hf_model) > `default` (caller's architecture-specific
    fallback - "gpt2" for FlexGPT, but that's wrong for a FlexLLaMA config with
    neither a KD teacher nor a warm-start source, so callers must pass their own).
    """
    if override:
        return override
    teacher = getattr(builder.training_context, 'teacher_hf_model', None)
    if teacher:
        return teacher
    pretrained = getattr(builder.model_config, 'pretrained_hf_model', None)
    if pretrained:
        return pretrained
    return default


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="flexgpt,wikitext103.gpt2pretrained")
    parser.add_argument("--dataset", default="wikitext-103-raw-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tokenizer", default=None,
                        help="override the auto-detected tokenizer (KD teacher / warm-start source / gpt2)")
    args = parser.parse_args()

    builder = resolve_from_str(args.config)
    cfg = builder.model_config

    print(f"Loading model: {args.config}")
    model = load_model(args.config, cfg).to(args.device)

    default_tokenizer = "JackFram/llama-160m" if args.config.startswith("flexllama") else "gpt2"
    tokenizer_name = resolve_tokenizer_name(builder, args.tokenizer, default=default_tokenizer)
    from transformers import AutoTokenizer
    tokenizer_vocab_size = len(AutoTokenizer.from_pretrained(tokenizer_name))
    if tokenizer_vocab_size != cfg.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocab mismatch: {tokenizer_name!r} has "
            f"{tokenizer_vocab_size} tokens, but {args.config} expects "
            f"vocab_size={cfg.vocab_size}. Pass --tokenizer to override "
            f"(this would otherwise crash as an out-of-bounds CUDA gather)."
        )

    print(f"Loading dataset: {args.dataset}  (tokenizer={tokenizer_name})")
    _, _, test_loader = load_wikitext(
        dataset_name=args.dataset,
        max_seq_length=cfg.max_seq_length,
        batch_size=args.batch_size,
        tokenizer_name=tokenizer_name,
    )

    print(f"\n{'Level':<8} {'Token PPL':>10}")
    print("-" * 20)
    for level in range(model.max_level() + 1):
        model.set_level_use(level)
        ppl = eval_ppl(model, test_loader, args.device)
        print(f"{level:<8} {ppl:>10.2f}")
