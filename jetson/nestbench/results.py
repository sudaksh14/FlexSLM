"""Result records: provenance, JSON-lines output, and reloading for analysis.

One record per (experiment, backbone, token budget, power mode, repetition
block).  Records are appended to a single .jsonl per experiment so a crashed or
OOM-killed run keeps everything measured up to that point -- which matters here
because the 7B configuration is *expected* to die, and losing the 1.1B and 1.7B
numbers with it would be an own goal.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        out = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=15).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else None
    except (OSError, subprocess.SubprocessError):
        return None


def file_digest(path: str, limit_bytes: int = 1 << 20) -> Optional[str]:
    """Short digest of a file's head -- enough to tell two elastic_config.json
    apart without hashing a multi-GB checkpoint."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(limit_bytes))
        return h.hexdigest()[:16]
    except OSError:
        return None


def run_provenance(extra: Optional[dict] = None) -> dict:
    from . import __version__
    from .device import device_info
    prov = {
        "nestbench_version": __version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp_epoch": time.time(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
        "device": device_info(),
    }
    if extra:
        prov.update(extra)
    return prov


class ResultWriter:
    """Append-only JSON-lines writer.

    The first line of every file is a ``{"_provenance": ...}`` record; every
    later line is a measurement.  Reading back with :func:`load_results` skips
    the provenance line and hands it to you separately.
    """

    def __init__(self, path: str, provenance_extra: Optional[dict] = None,
                 append: bool = False):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        new = (not append) or (not os.path.exists(path)) or os.path.getsize(path) == 0
        self.fh = open(path, "a" if append else "w", buffering=1)
        if new:
            self._write({"_provenance": run_provenance(provenance_extra)})
        self.n = 0

    def _write(self, obj: Dict[str, Any]) -> None:
        self.fh.write(json.dumps(obj, default=str) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())     # a power-cut mid-sweep should not lose rows

    def write(self, record: Dict[str, Any]) -> None:
        record.setdefault("_t", time.time())
        self._write(record)
        self.n += 1

    def close(self) -> None:
        try:
            self.fh.close()
        except OSError:
            pass

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_results(path: str):
    """-> (provenance_dict_or_None, [record, ...]).  Tolerates a truncated
    final line, which is what a hard OOM kill leaves behind."""
    prov, rows = None, []
    if not os.path.exists(path):
        return prov, rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue                      # truncated tail
            if "_provenance" in obj and prov is None:
                prov = obj["_provenance"]
            else:
                rows.append(obj)
    return prov, rows


def load_many(paths: Iterable[str]) -> List[dict]:
    out: List[dict] = []
    for p in paths:
        _, rows = load_results(p)
        out.extend(rows)
    return out


def stat_block(samples: List[float]) -> Dict[str, float]:
    """mean / std / min / median / p90 / p95 / max over a list of measurements.

    Report the median as the headline and the p95 alongside it: on a thermally
    constrained passive board the mean is dragged by a small number of
    throttled or interrupt-delayed iterations, and hiding that in a mean is how
    an on-device latency table stops being reproducible.
    """
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / (n - 1) if n > 1 else 0.0

    def q(p: float) -> float:
        if n == 1:
            return s[0]
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    return {"n": n, "mean": mean, "std": var ** 0.5, "min": s[0], "max": s[-1],
            "median": q(0.5), "p90": q(0.90), "p95": q(0.95)}
