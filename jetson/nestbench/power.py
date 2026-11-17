"""Power / energy measurement on Jetson.

Two independent sources, both giving the same record shape so the rest of the
harness does not care which one ran:

``sysfs``
    Poll the on-module INA3221 shunt monitors directly through their hwmon
    nodes.  This is the lowest-overhead and highest-rate option (a few hundred
    microseconds per sample, so a 5 ms period is comfortable) and it is what
    you want for per-query energy on workloads in the tens of milliseconds --
    which is exactly the NEST-VLM prefill regime (34-121 ms).

``tegrastats``
    Spawn ``tegrastats --interval N`` and parse its lines.  Same rails, but the
    sampler is a separate process whose practical floor is ~10 ms and whose
    instantaneous column is itself an average over the driver's window.  Use it
    as the cross-check / for long runs, not for a 34 ms prefill.

Both integrate power over time with the trapezoid rule and return joules.  Both
also record the idle floor so the harness can report *net* energy (the energy
the query actually cost) next to *gross* energy (what the wall sees), because
on a 7 W part the idle floor is a large fraction of the total and quoting only
gross energy makes every operating point look the same.

Rail semantics on Orin Nano 8GB:
    VDD_IN            total module input power  <- this is the one to report
    VDD_CPU_GPU_CV    CPU + GPU + DLA/PVA complex
    VDD_SOC           SoC rail (memory controller, fabric, ...)
VDD_IN is not exactly VDD_CPU_GPU_CV + VDD_SOC; the remainder is I/O, display
and regulator loss.  Report VDD_IN as the headline number and keep the other
two for the attribution breakdown.

IMPORTANT: the INA3221 on the devkit measures the *module*, not the carrier
board, so a wall-plug measurement will read higher.  State this in the paper.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .device import read_sysfs

# ---------------------------------------------------------------------------
# tegrastats line parsing
# ---------------------------------------------------------------------------

# "VDD_IN 4034mW/4034mW"  ->  (rail, instantaneous mW, running-average mW)
_RAIL_RE = re.compile(r"\b([A-Z][A-Z0-9_]*(?:VDD|VIN|POM)[A-Z0-9_]*|VDD_[A-Z0-9_]+)\s+(\d+)mW/(\d+)mW")
# JetPack 5/6 also emit rails as "VDD_IN 4034mW/4034mW"; older L4T used
# "POM_5V_IN 4034/4034".  Accept the older shape too.
_RAIL_RE_OLD = re.compile(r"\b(POM_[A-Z0-9_]+)\s+(\d+)/(\d+)\b")
_RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
_SWAP_RE = re.compile(r"\bSWAP\s+(\d+)/(\d+)MB")
_GR3D_RE = re.compile(r"\bGR3D_FREQ\s+(\d+)%(?:@\[?(\d+))?")
_EMC_RE = re.compile(r"\bEMC_FREQ\s+(\d+)%")
_TEMP_RE = re.compile(r"\b([a-z0-9]+)@(-?\d+(?:\.\d+)?)C")
_CPU_RE = re.compile(r"\bCPU\s+\[([^\]]*)\]")


def parse_tegrastats_line(line: str) -> Optional[dict]:
    """Parse one tegrastats line into a dict. Returns None if it has no rails.

    Keys: ``rails`` {name: mW}, ``rails_avg`` {name: mW}, ``ram_mb``,
    ``ram_total_mb``, ``swap_mb``, ``gr3d_pct``, ``emc_pct``, ``temps_c``,
    ``cpu_pct`` (list, one per core).
    """
    rails: Dict[str, float] = {}
    rails_avg: Dict[str, float] = {}
    for m in _RAIL_RE.finditer(line):
        rails[m.group(1)] = float(m.group(2))
        rails_avg[m.group(1)] = float(m.group(3))
    for m in _RAIL_RE_OLD.finditer(line):
        rails.setdefault(m.group(1), float(m.group(2)))
        rails_avg.setdefault(m.group(1), float(m.group(3)))
    if not rails:
        return None

    out: dict = {"rails": rails, "rails_avg": rails_avg}

    m = _RAM_RE.search(line)
    if m:
        out["ram_mb"], out["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    m = _SWAP_RE.search(line)
    if m:
        out["swap_mb"] = int(m.group(1))
    m = _GR3D_RE.search(line)
    if m:
        out["gr3d_pct"] = int(m.group(1))
        if m.group(2):
            out["gr3d_mhz"] = int(m.group(2))
    m = _EMC_RE.search(line)
    if m:
        out["emc_pct"] = int(m.group(1))
    temps = {k: float(v) for k, v in _TEMP_RE.findall(line)}
    if temps:
        out["temps_c"] = temps
    m = _CPU_RE.search(line)
    if m:
        cores = []
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok in ("off", "-"):
                cores.append(0.0)
            else:
                cores.append(float(tok.split("%")[0]))
        out["cpu_pct"] = cores
    return out


def pick_total_rail(rails: Dict[str, float]) -> Optional[str]:
    """Choose the rail that represents total module input power."""
    for cand in ("VDD_IN", "POM_5V_IN", "VDD_SYS_5V0", "VIN_SYS_5V0"):
        if cand in rails:
            return cand
    # Fall back to the largest-magnitude rail, which on every Jetson board is
    # the input rail by construction.
    return max(rails, key=rails.get) if rails else None


# ---------------------------------------------------------------------------
# sample container
# ---------------------------------------------------------------------------

@dataclass
class PowerTrace:
    """A time series of rail power, in mW, with monotonic timestamps."""

    t: List[float] = field(default_factory=list)              # seconds, monotonic
    rails: Dict[str, List[float]] = field(default_factory=dict)
    extra: List[dict] = field(default_factory=list)           # per-sample dicts
    source: str = "unknown"
    period_s: float = 0.0

    def add(self, t: float, rails: Dict[str, float], extra: Optional[dict] = None) -> None:
        self.t.append(t)
        for k, v in rails.items():
            self.rails.setdefault(k, []).append(v)
        self.extra.append(extra or {})

    # -- derived ----------------------------------------------------------
    def n(self) -> int:
        return len(self.t)

    def duration_s(self) -> float:
        return (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def has_data(self, rail: Optional[str] = None) -> bool:
        """True when this trace can yield a real energy/power number.

        Two samples are the minimum for a trapezoidal integral; with fewer, or
        with no rail, every derived quantity below is *unmeasured*, and the
        summary reports it as null rather than as 0.
        """
        rail = rail or self.total_rail()
        return rail is not None and len(self.t) >= 2 and bool(self.rails.get(rail))

    def energy_j(self, rail: Optional[str] = None) -> float:
        """Trapezoidal integral of rail power (mW) over time -> joules.

        Returns 0.0 when there is nothing to integrate; callers that record the
        value must gate on :meth:`has_data` first so an unmeasured run is
        written as null, not as a measured zero (see :meth:`summary`).
        """
        rail = rail or self.total_rail()
        if rail is None or len(self.t) < 2:
            return 0.0
        p = self.rails[rail]
        e_mws = 0.0
        for i in range(1, len(self.t)):
            e_mws += 0.5 * (p[i] + p[i - 1]) * (self.t[i] - self.t[i - 1])
        return e_mws / 1000.0

    def mean_power_w(self, rail: Optional[str] = None) -> float:
        d = self.duration_s()
        return (self.energy_j(rail) / d) if d > 0 else 0.0

    def peak_power_w(self, rail: Optional[str] = None) -> float:
        rail = rail or self.total_rail()
        if rail is None or not self.rails.get(rail):
            return 0.0
        return max(self.rails[rail]) / 1000.0

    def total_rail(self) -> Optional[str]:
        if not self.rails:
            return None
        head = {k: (v[0] if v else 0.0) for k, v in self.rails.items()}
        return pick_total_rail(head)

    def peak_ram_mb(self) -> Optional[int]:
        vals = [e.get("ram_mb") for e in self.extra if e.get("ram_mb") is not None]
        return max(vals) if vals else None

    def peak_temp_c(self) -> Optional[float]:
        vals = []
        for e in self.extra:
            temps = e.get("temps_c") or {}
            if temps:
                vals.append(max(temps.values()))
        return max(vals) if vals else None

    def mean_gr3d_pct(self) -> Optional[float]:
        vals = [e.get("gr3d_pct") for e in self.extra if e.get("gr3d_pct") is not None]
        return (sum(vals) / len(vals)) if vals else None

    def summary(self) -> dict:
        """Derived numbers for a result record.

        ``energy_j`` / ``mean_power_w`` / ``peak_power_w`` are ``None`` when the
        trace holds no usable samples (no rail monitor on this board, the
        sampler never started, source="none"). A null there means "not
        measured"; 0.0 would claim the board drew no power, which is never a
        measurement. ``measured`` states which of the two it is.
        """
        rail = self.total_rail()
        measured = self.has_data(rail)
        out = {
            "source": self.source,
            "measured": measured,
            "n_samples": self.n(),
            "duration_s": self.duration_s(),
            "sample_period_s": self.period_s,
            "total_rail": rail,
            "energy_j": self.energy_j(rail) if measured else None,
            "mean_power_w": self.mean_power_w(rail) if measured else None,
            "peak_power_w": self.peak_power_w(rail) if measured else None,
            "per_rail_energy_j": ({k: self.energy_j(k) for k in self.rails}
                                  if measured else {}),
            "per_rail_mean_w": ({k: self.mean_power_w(k) for k in self.rails}
                                if measured else {}),
            "peak_ram_mb": self.peak_ram_mb(),
            "peak_temp_c": self.peak_temp_c(),
            "mean_gr3d_pct": self.mean_gr3d_pct(),
        }
        return out


# ---------------------------------------------------------------------------
# INA3221 sysfs sampler
# ---------------------------------------------------------------------------

def discover_ina_channels() -> List[Tuple[str, str]]:
    """Find (label, hwmon_dir_with_index_prefix) for every INA3221 channel.

    Returns a list of (rail_label, "<hwmon_dir>|<idx>") pairs.  Handles both the
    driver-bound path (/sys/bus/i2c/drivers/ina3221/...) and the generic
    /sys/class/hwmon enumeration, because which one exists depends on the L4T
    release and on whether the node was probed by device tree or by the
    userspace overlay.
    """
    found: List[Tuple[str, str]] = []
    roots: List[str] = []
    roots += glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*")
    roots += glob.glob("/sys/bus/i2c/drivers/ina3221x/*/hwmon/hwmon*")
    roots += glob.glob("/sys/class/hwmon/hwmon*")
    seen = set()
    for root in roots:
        real = os.path.realpath(root)
        if real in seen:
            continue
        seen.add(real)
        for idx in (1, 2, 3):
            lab_p = os.path.join(real, f"in{idx}_label")
            cur_p = os.path.join(real, f"curr{idx}_input")
            if not (os.path.exists(lab_p) and os.path.exists(cur_p)):
                continue
            # Through device.read_sysfs, not open(): this is a sysfs node on
            # the same i2c bus as the rail readings and can answer EAGAIN.
            label = read_sysfs(lab_p)
            if not label:
                continue
            found.append((label, f"{real}|{idx}"))
    return found


class _SysfsPoller(threading.Thread):
    def __init__(self, channels, trace: PowerTrace, period_s: float, read_ram: bool):
        super().__init__(daemon=True)
        self.channels = channels
        self.trace = trace
        self.period_s = period_s
        self.read_ram = read_ram
        self._stop_evt = threading.Event()
        self._fds: Dict[str, Tuple[Optional[int], Optional[int], Optional[int]]] = {}

    def _open(self):
        for label, spec in self.channels:
            d, idx = spec.split("|")
            def _o(p):
                try:
                    return os.open(p, os.O_RDONLY)
                except OSError:
                    return None
            self._fds[label] = (
                _o(os.path.join(d, f"in{idx}_input")),     # mV
                _o(os.path.join(d, f"curr{idx}_input")),   # mA
                _o(os.path.join(d, f"power{idx}_input")),  # uW (not always present)
            )

    @staticmethod
    def _read(fd) -> Optional[float]:
        if fd is None:
            return None
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 32).decode().strip()
            return float(raw) if raw else None
        except (OSError, ValueError):
            return None

    def run(self):
        self._open()
        t0 = time.monotonic()
        nxt = t0
        while not self._stop_evt.is_set():
            rails: Dict[str, float] = {}
            for label, (v_fd, i_fd, p_fd) in self._fds.items():
                p_uw = self._read(p_fd)
                if p_uw is not None and p_uw > 0:
                    rails[label] = p_uw / 1000.0            # uW -> mW
                else:
                    mv, ma = self._read(v_fd), self._read(i_fd)
                    if mv is not None and ma is not None:
                        rails[label] = (mv * ma) / 1000.0   # mV*mA = uW -> mW
            if rails:
                self.trace.add(time.monotonic(), rails)
            nxt += self.period_s
            sleep = nxt - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                nxt = time.monotonic()   # we fell behind; resynchronise
        for fds in self._fds.values():
            for fd in fds:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    def stop(self):
        self._stop_evt.set()


class _TegrastatsPoller(threading.Thread):
    def __init__(self, trace: PowerTrace, interval_ms: int, binary: str):
        super().__init__(daemon=True)
        self.trace = trace
        self.interval_ms = interval_ms
        self.binary = binary
        self.proc: Optional[subprocess.Popen] = None
        self._stop_evt = threading.Event()

    def run(self):
        try:
            self.proc = subprocess.Popen(
                [self.binary, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except OSError:
            return
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if self._stop_evt.is_set():
                break
            rec = parse_tegrastats_line(line)
            if rec:
                extra = {k: v for k, v in rec.items() if k not in ("rails", "rails_avg")}
                self.trace.add(time.monotonic(), rec["rails"], extra)

    def stop(self):
        self._stop_evt.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class PowerSampler:
    """Context manager that records a :class:`PowerTrace` over its body.

    >>> with PowerSampler(source="auto", period_ms=5) as ps:
    ...     run_the_query()
    >>> ps.trace.energy_j()

    ``source`` is one of ``auto`` | ``sysfs`` | ``tegrastats`` | ``none``.
    ``none`` records no samples at all, so the experiment scripts still run on a
    machine with no rail monitors (a dev laptop, a CI box) and every energy
    field in :meth:`summary` comes back ``None`` -- unmeasured, not zero.
    """

    def __init__(self, source: str = "auto", period_ms: int = 5,
                 tegrastats_bin: Optional[str] = None):
        self.requested = source
        self.period_ms = max(1, int(period_ms))
        self.tegrastats_bin = tegrastats_bin or shutil.which("tegrastats") or "/usr/bin/tegrastats"
        self.trace = PowerTrace()
        self._poller = None
        self.source = self._resolve(source)
        self.trace.source = self.source
        self.trace.period_s = self.period_ms / 1000.0

    def _resolve(self, source: str) -> str:
        if source == "none":
            return "none"
        have_sysfs = bool(discover_ina_channels())
        have_tegra = os.path.exists(self.tegrastats_bin)
        if source == "sysfs":
            return "sysfs" if have_sysfs else "none"
        if source == "tegrastats":
            return "tegrastats" if have_tegra else "none"
        if have_sysfs:
            return "sysfs"
        if have_tegra:
            return "tegrastats"
        return "none"

    # -- context manager --------------------------------------------------
    def __enter__(self) -> "PowerSampler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        if self.source == "sysfs":
            self._poller = _SysfsPoller(discover_ina_channels(), self.trace,
                                        self.period_ms / 1000.0, read_ram=False)
        elif self.source == "tegrastats":
            self._poller = _TegrastatsPoller(self.trace, self.period_ms, self.tegrastats_bin)
        else:
            self._poller = None
        if self._poller is not None:
            self._poller.start()
            # tegrastats needs a moment to emit its first line; sysfs is instant.
            deadline = time.monotonic() + (2.0 if self.source == "tegrastats" else 0.2)
            while self.trace.n() == 0 and time.monotonic() < deadline:
                time.sleep(0.002)

    def stop(self) -> None:
        if self._poller is not None:
            self._poller.stop()
            self._poller.join(timeout=5)
            self._poller = None

    def summary(self) -> dict:
        return self.trace.summary()


def measure_idle(source: str = "auto", seconds: float = 5.0,
                 period_ms: int = 5, settle_s: float = 2.0) -> dict:
    """Measure the idle power floor.

    ``settle_s`` of untimed sleep first, so the tail of whatever ran before
    (page cache writeback, the GPU dropping to its idle clock) is not counted.
    Subtract this from a measured run to report net query energy.
    """
    time.sleep(settle_s)
    with PowerSampler(source=source, period_ms=period_ms) as ps:
        time.sleep(seconds)
    s = ps.summary()
    s["idle_power_w"] = s["mean_power_w"]      # None when no rail could be sampled
    return s
