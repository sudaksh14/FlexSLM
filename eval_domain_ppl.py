# Perplexity of each trained checkpoint on the held-out test text of each
# domain (PMC, RefinedWeb, GneissWeb). The test split is rebuilt with the same
# 4-rank sharding as training, so it never overlaps the training data.
import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from run_experiment import resolve_from_str
from utils import load_model, load_pmc_open_access, load_refinedweb, load_gneissweb

DOMAIN_LOADERS = {
    "pmc": load_pmc_open_access,
    "refinedweb": load_refinedweb,
    "gneissweb": load_gneissweb,
}


@torch.no_grad()
def eval_ppl(model, batches, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    vocab_size = model.config.vocab_size
    for input_ids, _ in tqdm(batches, leave=False):
        input_ids = input_ids.to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            input_ids[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += input_ids[:, 1:].numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def build_domain_test_batches(loader_fn, tokenizer_name, max_seq_length, batch_size, world_size=4):
    batches = []
    for rank in range(world_size):
        os.environ['SLURM_PROCID'] = str(rank)
        os.environ['SLURM_NTASKS'] = str(world_size)
        _, _, test_loader = loader_fn(
            max_seq_length=max_seq_length, batch_size=batch_size,
            tokenizer_name=tokenizer_name,
        )
        rank_batches = list(test_loader)
        print(f"    rank {rank}: {len(rank_batches)} batches")
        batches.extend(rank_batches)
    os.environ.pop('SLURM_PROCID', None)
    os.environ.pop('SLURM_NTASKS', None)
    return batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True,
                        help="semicolon-separated list of config strings, e.g. "
                             "'flexllama,refinedweb.kd_lambda05_1p4B;flexllama,gneissweb.kd_lambda05_1p4B;flexllama,pmc.kd_lambda05_3B'")
    parser.add_argument("--domains", default="pmc,refinedweb,gneissweb",
                        help="comma-separated subset of pmc,refinedweb,gneissweb to test against")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tokenizer", default="JackFram/llama-160m")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(";") if c.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    for d in domains:
        if d not in DOMAIN_LOADERS:
            raise ValueError(f"Unknown domain '{d}', expected one of {list(DOMAIN_LOADERS)}")

    # Load each config's model once, keep all of them resident (they're tiny -
    # 16-186M params each) rather than reloading per domain.
    models = {}
    for config in configs:
        builder = resolve_from_str(config)
        cfg = builder.model_config
        models[config] = load_model(config, cfg).to(args.device)

    print(f"\n{'Domain':<12} {'Config':<45} {'Level':<8} {'PPL':>10}")
    print("-" * 80)
    for domain in domains:
        print(f"Building held-out test set for domain='{domain}' (4-rank split)...")
        test_batches = build_domain_test_batches(
            DOMAIN_LOADERS[domain], tokenizer_name=args.tokenizer,
            max_seq_length=1024, batch_size=args.batch_size,
        )
        print(f"  total batches: {len(test_batches)}")
        for config, model in models.items():
            for level in range(model.max_level() + 1):
                model.set_level_use(level)
                ppl = eval_ppl(model, test_batches, args.device)
                print(f"{domain:<12} {config:<45} {level:<8} {ppl:>10.2f}")
        del test_batches
        torch.cuda.empty_cache()
