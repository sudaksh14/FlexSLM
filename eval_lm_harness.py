# Run FlexGPT/FlexLLaMA through EleutherAI's lm-evaluation-harness.
import json
import argparse
import os

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, AutoTokenizer
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import evaluator
from lm_eval.utils import make_table


@register_model("flexgpt")
class FlexGPTLMEval(LM):

    def __init__(self, model, level: int, device: str = "cuda", tokenizer_name: str = "gpt2"):
        super().__init__()
        self.model = model.to(device)
        self.model.set_level_use(level)
        self.model.eval()
        self._device = device
        self._tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self.model.config.max_seq_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return 8

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(inps)

    def loglikelihood(self, requests):
        # Used for multiple-choice tasks: score each (context, continuation) pair
        # and also check whether the continuation is what greedy decoding would pick.
        results = []
        for ctx, cont in [r.args for r in requests]:
            ctx_ids = self.tok_encode(ctx)
            cont_ids = self.tok_encode(cont)
            all_ids = torch.tensor(
                [ctx_ids + cont_ids], dtype=torch.long, device=self._device
            )
            all_ids = all_ids[:, -self.max_length:]
            ctx_len = min(len(ctx_ids), all_ids.shape[1] - len(cont_ids))

            logits = self._model_call(all_ids)
            log_probs = F.log_softmax(logits, dim=-1)

            cont_slice = slice(ctx_len - 1, ctx_len - 1 + len(cont_ids))
            cont_toks = torch.tensor(cont_ids, device=self._device)
            ll = log_probs[0, cont_slice, :].gather(
                1, cont_toks.unsqueeze(1)
            ).sum().item()

            greedy_toks = logits[0, cont_slice, :].argmax(-1)
            is_greedy = (greedy_toks == cont_toks).all().item()
            results.append((ll, bool(is_greedy)))
        return results

    def loglikelihood_rolling(self, requests):
        # Used for perplexity tasks: score the full string in non-overlapping chunks.
        results = []
        for (string,) in [r.args for r in requests]:
            ids = self.tok_encode(string)
            total_ll = 0.0
            stride = self.max_length
            for start in range(0, max(1, len(ids) - 1), stride):
                chunk = ids[start : start + self.max_length + 1]
                inp = torch.tensor([chunk[:-1]], dtype=torch.long, device=self._device)
                tgt = torch.tensor(chunk[1:], dtype=torch.long, device=self._device)
                logits = self._model_call(inp)
                lp = F.log_softmax(logits[0], dim=-1)
                total_ll += lp.gather(1, tgt.unsqueeze(1)).sum().item()
            results.append(total_ll)
        return results

    def generate_until(self, requests):
        # Used for open-ended generation tasks: greedy decode until a stop string.
        results = []
        for ctx, gen_kwargs in [r.args for r in requests]:
            until = gen_kwargs.get("until", [self._tokenizer.eos_token])
            max_new = gen_kwargs.get("max_new_tokens", self.max_gen_toks)
            ids = self.tok_encode(ctx)
            inp = torch.tensor([ids], dtype=torch.long, device=self._device)
            generated = []
            with torch.no_grad():
                for _ in range(max_new):
                    logits = self.model(inp[:, -self.max_length:])
                    next_tok = logits[0, -1, :].argmax().item()
                    generated.append(next_tok)
                    inp = torch.cat(
                        [inp, torch.tensor([[next_tok]], device=self._device)], dim=1
                    )
                    decoded = self._tokenizer.decode(generated)
                    if any(decoded.endswith(s) for s in until):
                        break
            results.append(self._tokenizer.decode(generated))
        return results


@register_model("flexllama")
class FlexLLaMALMEval(FlexGPTLMEval):

    def __init__(self, model, level: int, device: str = "cuda", tokenizer_name: str = "JackFram/llama-160m"):
        # Call LM.__init__ directly to skip GPT2TokenizerFast setup in parent
        LM.__init__(self)
        self.model = model.to(device)
        self.model.set_level_use(level)
        self.model.eval()
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token


if __name__ == "__main__":
    from run_experiment import resolve_from_str
    from utils import load_model
    from eval_wikitext103 import resolve_tokenizer_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="flexgpt,wikitext103.gpt2pretrained")
    parser.add_argument("--tasks", default="wikitext,lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,openbookqa,triviaqa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", default="results/flexgpt_eval.json")
    parser.add_argument("--tokenizer", default=None,
                        help="override the auto-detected tokenizer (KD teacher / warm-start source / gpt2 or llama-160m)")
    args = parser.parse_args()

    builder = resolve_from_str(args.config)
    model = load_model(args.config, builder.model_config)
    model.eval()

    is_llama = args.config.startswith("flexllama")
    EvalClass = FlexLLaMALMEval if is_llama else FlexGPTLMEval
    default_tokenizer = "JackFram/llama-160m" if is_llama else "gpt2"
    tokenizer_name = resolve_tokenizer_name(builder, args.tokenizer, default=default_tokenizer)

    tokenizer_vocab_size = len(AutoTokenizer.from_pretrained(tokenizer_name))
    if tokenizer_vocab_size != builder.model_config.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocab mismatch: {tokenizer_name!r} has "
            f"{tokenizer_vocab_size} tokens, but {args.config} expects "
            f"vocab_size={builder.model_config.vocab_size}. Pass --tokenizer to "
            f"override (this would otherwise crash as an out-of-bounds CUDA gather)."
        )
    print(f"Loading model: {args.config}  (tokenizer={tokenizer_name})")

    all_results = {}
    for level in range(model.max_level() + 1):
        lm = EvalClass(model, level=level, device=args.device, tokenizer_name=tokenizer_name)
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=args.tasks.split(","),
            batch_size=args.batch_size,
            log_samples=False,
        )
        all_results[f"level_{level}"] = results["results"]
        print(f"\n=== Level {level} ===")
        print(make_table(results))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")
