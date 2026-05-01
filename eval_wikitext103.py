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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="flexgpt,wikitext103.gpt2pretrained")
    parser.add_argument("--dataset", default="wikitext-103-raw-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    builder = resolve_from_str(args.config)
    cfg = builder.model_config

    print(f"Loading model: {args.config}")
    model = load_model(args.config, cfg).to(args.device)

    print(f"Loading dataset: {args.dataset}")
    _, _, test_loader = load_wikitext(
        dataset_name=args.dataset,
        max_seq_length=cfg.max_seq_length,
        batch_size=args.batch_size,
    )

    print(f"\n{'Level':<8} {'Token PPL':>10}")
    print("-" * 20)
    for level in range(model.max_level() + 1):
        model.set_level_use(level)
        ppl = eval_ppl(model, test_loader, args.device)
        print(f"{level:<8} {ppl:>10.2f}")
