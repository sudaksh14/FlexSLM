#!/usr/bin/env python3
"""Why do graph-decoded tokens differ from eager decoding (if they do)? For each width and prompt:
eager HF decoding (dynamic cache), static cache without a CUDA graph, static cache with the graph.

  graph == static     -> the CUDA-graph capture/replay is exact (any difference is the static cache)
  static vs eager     -> where they first diverge, the eager top-1/top-2 logit margin there: a
                         near-tie (bf16 resolution) means numerics, a large margin means a bug
  answer agreement    -> do the scored answers (boxed content) agree, which is what the paper uses

    python3 jetson/diag_decode.py --elastic models/mednest --levels 0 8 --n 40
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mednest_jetson.cli import base_parser, setup, torch_dtype  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--elastic", required=True)
    p.add_argument("--levels", type=int, nargs="*", default=None)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--subset", default=os.path.join(os.path.dirname(__file__), "questions", "subset.jsonl"))
    args = p.parse_args()
    from transformers import AutoTokenizer

    from mednest_jetson import elastic as el
    from mednest_jetson import workload as wl
    from mednest_jetson.decode import TIE_MARGIN, ElasticDecoders, compare_decoders, max_len_for

    w = setup(args, "diag_decode")
    dirs = el.level_dirs_of(args.elastic)
    e = el.ElasticModel.load(dirs[max(dirs)], [dirs[l] for l in sorted(dirs)], device=args.device,
                             dtype=torch_dtype(args.dtype))
    tok = AutoTokenizer.from_pretrained(dirs[max(dirs)])
    rows = wl.load_subset(args.subset)
    rows = rows[::max(1, len(rows) // args.n)][:args.n]
    ids = [wl.encode(tok, wl.prompt_for(r, "as_is"), "as_is", args.device) for r in rows]
    ml = max_len_for([i.shape[1] for i in ids], 32)
    graph, static = ElasticDecoders(e, ml, use_graph=True), ElasticDecoders(e, ml, use_graph=False)
    for level in (args.levels if args.levels is not None else sorted(dirs)):
        r = compare_decoders(e.width(level), tok, rows, ids, static.get(level), graph.get(level))
        w.write({"kind": "decode_diag", "level": level, **r})
        print(f"  level{level}: graph==static {r['graph_eq_static']:.0%}, static==eager {r['static_eq_eager']:.0%}, "
              f"same answer {r['same_answer']:.0%}; logit gaps at divergence "
              f"{[round(d['gap'], 3) for d in r['divergence_gaps']]} (tie margin {TIE_MARGIN}); "
              f"decode {r['eager_ms_tok']:.2f} -> {r['graph_ms_tok']:.2f} ms/token", flush=True)
    w.close()


if __name__ == "__main__":
    main()
