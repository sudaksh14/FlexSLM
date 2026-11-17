#!/usr/bin/env python3
"""Preflight on the board (run first, every session): software stack, GPU, rails, power modes,
passwordless sudo, model files; with --elastic also proves on the device that every elastic width
(a view of the full weights) gives the same logits as the exported dense checkpoint of that level.

    python3 jetson/check_env.py --elastic models/mednest --subset jetson/questions/subset.jsonl
Exit code 0 only if every required check passed.
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mednest_jetson.cli import base_parser, setup, torch_dtype  # noqa: E402

MIN_TRANSFORMERS = (4, 51)  # Qwen3


def _ver(s):
    return tuple(int(x) for x in s.split(".")[:2] if x.isdigit())


def main():
    p = base_parser(__doc__)
    p.add_argument("--elastic", default=None, help="export dir with level<i>/ subdirs (MedNest)")
    p.add_argument("--subset", default=os.path.join(os.path.dirname(__file__), "questions", "subset.jsonl"))
    p.add_argument("--check-prompts", type=int, default=8)
    p.add_argument("--models", nargs="*", default=[], help="other model dirs that must exist (baselines)")
    args = p.parse_args()
    w = setup(args, "env")
    ok = True

    def check(name, passed, detail="", required=True):
        nonlocal ok
        ok &= bool(passed) or not required
        tag = "PASS" if passed else ("FAIL" if required else "warn")
        print(f"  [{tag}] {name}: {detail}", flush=True)
        w.write({"kind": "check", "check": name, "passed": bool(passed), "required": required, "detail": str(detail)})

    from nestbench import device, power
    info = device.device_info()
    check("Jetson board", info["is_jetson"], f"{info['board_model']} | L4T {info['l4t']} | JetPack {info['jetpack']}",
          required=False)
    import torch
    check("torch + CUDA", torch.cuda.is_available(), f"torch {torch.__version__} cuda {torch.version.cuda} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU (CPU-only torch wheel?)'}")
    if torch.cuda.is_available():
        check("bf16 on GPU", torch.cuda.is_bf16_supported(), "", required=args.dtype == "bf16")
    import transformers
    check("transformers >= 4.51 (Qwen3)", _ver(transformers.__version__) >= MIN_TRANSFORMERS, transformers.__version__)
    for mod in ("safetensors", "tokenizers"):
        try:
            m = __import__(mod)
            check(mod, True, getattr(m, "__version__", "?"))
        except ImportError as e:
            check(mod, False, e)
    rails = power.discover_ina_channels()
    check("INA3221 rails (sysfs)", bool(rails), [r[0] for r in rails][:6], required=False)
    check("tegrastats", shutil.which("tegrastats") is not None, shutil.which("tegrastats"), required=False)
    check("power mode", True, info["power_mode"], required=False)
    check("power modes available", True, info["power_modes_available"], required=False)
    sudo = subprocess.run(["sudo", "-n", "nvpmodel", "-q"], capture_output=True, text=True).returncode == 0
    check("passwordless sudo nvpmodel", sudo, "needed for --power-mode and the power-mode trace", required=False)
    drop = subprocess.run("sudo -n tee /proc/sys/vm/drop_caches </dev/null >/dev/null", shell=True).returncode == 0
    check("passwordless sudo drop_caches", drop, "needed for cold-load timings in bench_switch", required=False)
    check("swap", (info.get("swap_mb") or 0) > 0, f"{info.get('swap_mb')} MB", required=False)
    check("max temperature", True, f"{device.max_temp_c()} C", required=False)
    check("subset file", os.path.exists(args.subset), args.subset)
    for m in args.models:
        check(f"model {m}", os.path.exists(os.path.join(m, "config.json")), m)

    if args.elastic and ok:
        from transformers import AutoTokenizer

        from mednest_jetson import elastic as el
        from mednest_jetson import workload as wl
        dirs = el.level_dirs_of(args.elastic)
        check("elastic export levels", len(dirs) >= 2, sorted(dirs))
        top = max(dirs)
        e = el.ElasticModel.load(dirs[top], [dirs[l] for l in sorted(dirs)], device=args.device,
                                 dtype=torch_dtype(args.dtype))
        tok = AutoTokenizer.from_pretrained(dirs[top])
        rows = wl.load_subset(args.subset)[::max(1, len(wl.load_subset(args.subset)) // args.check_prompts)]
        prompts = [wl.prompt_for(r, "as_is") for r in rows[:args.check_prompts]]
        res = el.check_views(e, dirs, prompts, tok)
        for r in res:
            # bf16 matmuls on strided vs contiguous weights may differ in the last bits; the answer
            # (argmax) must not.
            check(f"view == exported level{r['level']}", r["argmax_agree"] == 1.0 and r["same_params"],
                  f"max|dlogit| {r['max_abs_logit_diff']:.3g}, argmax agree {r['argmax_agree']:.0%}, "
                  f"params equal {r['same_params']}")
            w.write({"kind": "view_check", **r})
        w.write({"kind": "elastic", "levels": el.describe(e), "unique_param_bytes": e.unique_param_bytes()})
        # The benchmarks decode with a static KV cache + CUDA graphs (mednest_jetson/decode.py).
        # Required: graph == static cache exactly (capture/replay is exact), and every difference
        # from eager decoding sits at a bf16 near-tie (TIE_MARGIN). Info: same scored answer.
        from mednest_jetson.decode import TIE_MARGIN, ElasticDecoders, compare_decoders, max_len_for
        ids_list = [wl.encode(tok, p_, "as_is", args.device) for p_ in prompts]
        ml = max_len_for([i.shape[1] for i in ids_list], 32)
        on_gpu = str(args.device).startswith("cuda")
        gdec, sdec = ElasticDecoders(e, ml, use_graph=on_gpu), ElasticDecoders(e, ml, use_graph=False)
        for level in sorted({min(dirs), max(dirs)}):
            r = compare_decoders(e.width(level), tok, rows[:len(ids_list)], ids_list, sdec.get(level), gdec.get(level))
            check(f"graph decode == static cache, level{level}", r["graph_eq_static"] == 1.0,
                  f"{r['graph_eq_static']:.0%} of prompts; decode {r['eager_ms_tok'] and round(r['eager_ms_tok'], 2)} "
                  f"(eager) -> {r['graph_ms_tok'] and round(r['graph_ms_tok'], 2)} ms/token (graph)")
            check(f"static vs eager differs only at bf16 ties, level{level}", r["max_gap"] <= TIE_MARGIN,
                  f"same tokens {r['static_eq_eager']:.0%}, max logit gap at divergence {r['max_gap']:.3f} "
                  f"(<= {TIE_MARGIN}), same answer {r['same_answer']:.0%}")
            w.write({"kind": "decode_check", "level": level, **r})
    w.write({"kind": "device", **info})
    w.write({"kind": "result", "all_required_passed": ok})
    print("ALL REQUIRED CHECKS PASSED" if ok else "SOME REQUIRED CHECKS FAILED", flush=True)
    w.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
