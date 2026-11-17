"""Flags shared by every MedNest Jetson script, and the run setup (power mode, provenance, writer)."""
from __future__ import annotations

import argparse
import os
import sys
import time

_JETSON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _JETSON not in sys.path:
    sys.path.insert(0, _JETSON)

DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out-dir", default=os.environ.get("MEDNEST_OUT", os.path.expanduser("~/mednest-results")))
    p.add_argument("--tag", default=None, help="suffix of the output file name")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=sorted(DTYPES),
                   help="bf16 = the cluster evaluation's precision (Orin's Ampere GPU supports it)")
    p.add_argument("--power-mode", default=None,
                   help="nvpmodel mode to set first (e.g. 15W, 7W; needs passwordless sudo); "
                        "default: leave the board as it is and record the mode")
    p.add_argument("--power-source", default="auto", choices=["auto", "sysfs", "tegrastats", "none"])
    p.add_argument("--power-period-ms", type=int, default=5)
    p.add_argument("--cooldown-c", type=float, default=50.0,
                   help="wait until the hottest zone is below this before each measured block (0: never)")
    p.add_argument("--warmup", type=int, default=3, help="untimed questions before each measured block")
    return p


def torch_dtype(name: str):
    import torch
    return getattr(torch, DTYPES[name])


def setup(args, experiment: str):
    """Set the power mode (if asked), then open <out-dir>/<experiment>[_tag].jsonl with provenance."""
    from nestbench import device
    from nestbench.results import ResultWriter
    mode = None
    if args.power_mode:
        mode = device.set_power_mode(args.power_mode, require=True)
    os.makedirs(args.out_dir, exist_ok=True)
    name = experiment + (f"_{args.tag}" if args.tag else "") + ".jsonl"
    path = os.path.join(args.out_dir, name)
    if os.path.exists(path):
        path = path.replace(".jsonl", time.strftime("_%Y%m%d_%H%M%S.jsonl"))  # never overwrite a result
    w = ResultWriter(path, provenance_extra={"experiment": experiment, "args": vars(args), "power_mode_set": mode})
    print(f"[{experiment}] writing {path}", flush=True)
    return w


def cooldown(args):
    if args.cooldown_c and args.cooldown_c > 0:
        from nestbench.device import cool_down
        return cool_down(target_c=args.cooldown_c)
    return None
