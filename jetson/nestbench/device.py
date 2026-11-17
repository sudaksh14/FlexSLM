"""Jetson device identification, power-mode control, clocks and thermals.

Every experiment record carries the output of :func:`device_info` so that a
result file is self-describing: months later you can tell whether a number was
taken at 7 W with DVFS on or at MAXN SUPER with jetson_clocks pinned, without
consulting a lab notebook.  On an Orin Nano that distinction is worth up to
2-3x on latency, so an unlabelled measurement is not a measurement.

Power-mode IDs are NOT hardcoded.  On JetPack 6.2+ an Orin Nano flashed with
the ``jetson-orin-nano-devkit-super.conf`` configuration exposes 7W / 15W /
25W / MAXN SUPER, while a board still on the pre-Super flash config exposes a
different, shorter list with the same integer IDs meaning different things.
We parse /etc/nvpmodel.conf at runtime and match by NAME, so a script that asks
for "MAXN SUPER" either gets it or fails loudly instead of silently measuring
the wrong envelope.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


def _run(cmd: List[str], timeout: float = 20.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _read(path: str) -> Optional[str]:
    """Read a sysfs/procfs file as text, or return None if it cannot be read.

    Uses the raw ``os.open``/``os.read`` syscalls rather than ``open()`` +
    ``f.read()``, and that is the whole point of this function.

    A Jetson thermal node can answer a read with ``EAGAIN`` -- the thermal
    driver returns ``-EAGAIN`` when the sensor has no reading ready yet. Through
    a buffered text stream that surfaces as the raw layer returning ``None``
    (io's "no data available right now"), which ``TextIOWrapper`` hands
    straight to the incremental decoder, where ``self.buffer + input`` raises::

        TypeError: can't concat NoneType to bytes

    That is a ``TypeError`` from inside the decoder, so ``except OSError`` and
    ``except UnicodeDecodeError`` both miss it, and it crashed exp01's preflight
    (and report_hardware.py) on the way through ``device_info()``.
    ``os.read`` cannot produce that shape: on ``EAGAIN`` it raises
    ``BlockingIOError``, an ``OSError`` subclass. This is the same pattern
    ``nestbench/power.py:_SysfsPoller._read`` has always used on the rail nodes,
    which is why the power sampler never hit this.

    Returns ``None`` -- never a partial or fabricated value -- for every way a
    /sys or /proc node can refuse to be read:

    * file absent, or the directory vanished between a glob and this read (a
      thermal zone unregistered by the driver, a hot-unplugged rail)
    * the driver errors on read: EACCES (root-only node), EINVAL/ENODATA
      (sensor powered down), EAGAIN/EWOULDBLOCK (sensor not ready)
    * the node is not valid UTF-8 (some sysfs attributes are raw binary)

    A bad *path* is still a programming error and still raises: that check runs
    before the I/O block, so nothing inside the block can be mistaken for it.
    """
    if not isinstance(path, str) or not path:
        raise TypeError(f"_read() needs a non-empty str path, got {path!r}")

    fd = None
    try:
        fd = os.open(path, os.O_RDONLY)
        chunks = []
        while True:
            # procfs in particular can hand back a short read; loop to EOF.
            block = os.read(fd, 65536)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks).decode("utf-8").strip()
    except (OSError, UnicodeDecodeError, ValueError):
        # OSError covers BlockingIOError (EAGAIN) and every other errno the
        # driver can return; UnicodeDecodeError covers a binary attribute;
        # ValueError covers a closed/invalid descriptor mid-read.
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


#: Public name for :func:`_read`. Every /sys and /proc read in this harness
#: should go through it: it is the one place that knows a Jetson sysfs node can
#: answer EAGAIN, and that reading such a node through a buffered TEXT stream
#: raises ``TypeError: can't concat NoneType to bytes`` from inside the codec
#: rather than any exception a caller would think to catch.
read_sysfs = _read


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------

def is_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release") or os.path.exists("/sys/firmware/devicetree/base/model")


def l4t_version() -> Optional[str]:
    """e.g. '36.4.4' from /etc/nv_tegra_release, else from the apt package."""
    raw = _read("/etc/nv_tegra_release")
    if raw:
        m = re.search(r"R(\d+).*?REVISION:\s*([\d.]+)", raw)
        if m:
            return f"{m.group(1)}.{m.group(2).rstrip('.')}"
    out = _run(["dpkg-query", "--showformat=${Version}", "--show", "nvidia-l4t-core"])
    return out.strip() or None


def jetpack_version() -> Optional[str]:
    out = _run(["dpkg-query", "--showformat=${Version}", "--show", "nvidia-jetpack"])
    return out.strip() or None


def board_model() -> Optional[str]:
    m = _read("/sys/firmware/devicetree/base/model")
    return m.replace("\x00", "").strip() if m else None


def module_ram_mb() -> Optional[int]:
    raw = _read("/proc/meminfo")
    if not raw:
        return None
    m = re.search(r"MemTotal:\s+(\d+) kB", raw)
    return int(m.group(1)) // 1024 if m else None


# ---------------------------------------------------------------------------
# nvpmodel
# ---------------------------------------------------------------------------

def list_power_modes(conf: str = "/etc/nvpmodel.conf") -> Dict[int, str]:
    """{mode_id: mode_name} parsed from the active nvpmodel config."""
    modes: Dict[int, str] = {}
    raw = _read(conf)
    if raw:
        for m in re.finditer(r"^\s*<\s*POWER_MODEL\s+ID\s*=\s*(\d+)\s+NAME\s*=\s*([^>]+?)\s*>",
                             raw, re.MULTILINE):
            modes[int(m.group(1))] = m.group(2).strip()
    if not modes:
        # nvpmodel -p --verbose prints the table too, in a different shape.
        out = _run(["nvpmodel", "-p", "--verbose"])
        for m in re.finditer(r"NV Power Mode:\s*(\S+)\s*\n?\s*(\d+)", out):
            modes[int(m.group(2))] = m.group(1)
    return modes


def current_power_mode() -> Dict[str, Optional[object]]:
    out = _run(["nvpmodel", "-q"])
    name = mode_id = None
    m = re.search(r"NV Power Mode:\s*(.+)", out)
    if m:
        name = m.group(1).strip()
    m = re.search(r"^\s*(\d+)\s*$", out, re.MULTILINE)
    if m:
        mode_id = int(m.group(1))
    return {"id": mode_id, "name": name}


def resolve_power_mode(spec: str) -> Optional[int]:
    """Turn '7W' / '15W' / 'MAXN SUPER' / '2' into a mode ID on THIS board."""
    spec = str(spec).strip()
    if spec.isdigit():
        return int(spec)
    modes = list_power_modes()
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    want = norm(spec)
    for mid, name in modes.items():
        if norm(name) == want:
            return mid
    for mid, name in modes.items():
        if want in norm(name) or norm(name) in want:
            return mid
    return None


def set_power_mode(spec: str, settle_s: float = 8.0, require: bool = True) -> Dict[str, object]:
    """Switch nvpmodel and wait for clocks to settle. Needs root (sudo)."""
    mid = resolve_power_mode(spec)
    if mid is None:
        msg = (f"power mode {spec!r} not available on this board. "
               f"Available: {list_power_modes()}. If you expected MAXN SUPER / 25W, "
               f"the board was flashed with the pre-Super nvpmodel config -- reflash "
               f"with jetson-orin-nano-devkit-super.conf (see jetson_setup.md Step 3).")
        if require:
            raise RuntimeError(msg)
        return {"ok": False, "error": msg}
    out = _run(["sudo", "-n", "nvpmodel", "-m", str(mid)], timeout=60)
    time.sleep(settle_s)
    cur = current_power_mode()
    ok = (cur["id"] == mid)
    if not ok and require:
        raise RuntimeError(
            f"nvpmodel -m {mid} did not take effect (now {cur}). Output: {out.strip()!r}. "
            "Passwordless sudo for nvpmodel is required -- see jetson_setup.md Step 6.")
    return {"ok": ok, "requested": spec, "id": mid, "current": cur, "output": out.strip()}


def set_jetson_clocks(enable: bool, settle_s: float = 5.0, require: bool = True) -> Dict[str, object]:
    """Pin (or release) all clocks to the max of the active power mode.

    Pinning removes DVFS ramp-up from the measurement, which is what you want
    for a clean latency number; it also raises idle power, which is what you do
    NOT want for an energy number.  Measure both and say which is which.

    ``require=True`` (the default) verifies the ``sudo -n jetson_clocks`` call
    actually succeeded and raises if not -- previously this always reported
    ``ok: True`` even when passwordless sudo wasn't configured, so an
    unprivileged run would silently record "pinned" latency numbers that were
    never pinned (contradicts this module's own "fail loudly" design -- see
    the module docstring). Pass ``require=False`` for best-effort
    releases/restores in cleanup paths, where a failure isn't load-bearing.
    """
    if not shutil.which("jetson_clocks"):
        msg = "jetson_clocks not found"
        if require:
            raise RuntimeError(msg)
        return {"ok": False, "error": msg}
    cmd = ["sudo", "-n", "jetson_clocks"] + ([] if enable else ["--restore"])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    if not ok and require:
        raise RuntimeError(
            f"sudo -n jetson_clocks{' --restore' if not enable else ''} failed "
            f"(rc={proc.returncode}): {out.strip()!r}. Passwordless sudo for "
            "jetson_clocks is required -- see jetson_setup.md Step 7.")
    time.sleep(settle_s)
    return {"ok": ok, "enabled": enable, "output": out.strip()}


def clocks_snapshot() -> Dict[str, Optional[int]]:
    """Current GPU / EMC / CPU clock, in Hz where the node reports Hz."""
    snap: Dict[str, Optional[int]] = {}

    def _int(p):
        v = _read(p)
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    for name, pats in {
        "gpu_cur_hz": ["/sys/devices/platform/bus@0/*.gpu/devfreq/*.gpu/cur_freq",
                       "/sys/class/devfreq/17000000.gpu/cur_freq",
                       "/sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq"],
        "gpu_max_hz": ["/sys/class/devfreq/17000000.gpu/max_freq",
                       "/sys/devices/gpu.0/devfreq/17000000.gv11b/max_freq"],
        "emc_cur_hz": ["/sys/kernel/debug/bpmp/debug/clk/emc/rate",
                       "/sys/kernel/nvpmodel_emc_cap/emc_iso_cap"],
    }.items():
        val = None
        for pat in pats:
            for p in glob.glob(pat):
                val = _int(p)
                if val:
                    break
            if val:
                break
        snap[name] = val

    cpu = _int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    snap["cpu0_cur_khz"] = cpu
    snap["cpu_governor"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    snap["online_cpus"] = _read("/sys/devices/system/cpu/online")
    return snap


# ---------------------------------------------------------------------------
# thermals
# ---------------------------------------------------------------------------

THERMAL_ROOT = "/sys/class/thermal"

# Plausible range for a Jetson thermal zone in millidegrees C. Values outside
# it are the driver's "no reading" sentinels (-256000 on several Orin zones)
# and are reported as unavailable rather than recorded as a temperature.
_TEMP_MIN_MC, _TEMP_MAX_MC = -40_000, 150_000


@dataclass
class ThermalZone:
    """One /sys/class/thermal zone. ``temp_c is None`` means unavailable."""

    zone: str                       # directory basename, e.g. "thermal_zone0"
    name: Optional[str]             # driver-reported type, e.g. "CPU-therm"
    temp_c: Optional[float]
    status: str                     # "ok" | "unavailable"
    detail: str = ""

    @property
    def label(self) -> str:
        return self.name or self.zone


def thermal_zone_report(root: Optional[str] = None) -> List[ThermalZone]:
    """Every thermal zone on the board, readable or not.

    Thermal telemetry is OPTIONAL for every experiment in this harness (no
    measurement's correctness depends on it; it gates cooldown waits and is
    recorded as context), so a zone that cannot be read is reported as
    unavailable and the run continues. What is *not* acceptable is inventing a
    number for it -- an unreadable zone never contributes a temperature.

    Every failure mode is handled per zone, and only the narrow exception types
    a sysfs read can legitimately raise are caught, so an unrelated bug in this
    function still surfaces instead of being silently swallowed:

    * ``type``/``temp`` missing, empty, or unreadable (permissions, driver
      error, zone unregistered between the glob and the read) -> ``_read``
      returns None
    * ``temp`` not an integer, or outside the plausible range (driver sentinel)
      -> ValueError / range check
    * the whole ``/sys/class/thermal`` tree absent (non-Jetson, container
      without /sys mounted) -> empty list
    """
    # Resolved at call time, not bound as a default at import time, so a test
    # (or a container with /sys mounted elsewhere) can repoint THERMAL_ROOT.
    root = THERMAL_ROOT if root is None else root
    zones: List[ThermalZone] = []
    try:
        paths = sorted(glob.glob(os.path.join(root, "thermal_zone*")))
    except OSError as exc:
        return [ThermalZone(zone=root, name=None, temp_c=None,
                            status="unavailable", detail=f"cannot list {root}: {exc}")]

    for zone_dir in paths:
        zone = os.path.basename(zone_dir)
        name = _read(os.path.join(zone_dir, "type")) or None
        raw = _read(os.path.join(zone_dir, "temp"))
        if raw is None:
            zones.append(ThermalZone(zone, name, None, "unavailable",
                                     "temp node missing or not readable"))
            continue
        if not raw:
            zones.append(ThermalZone(zone, name, None, "unavailable",
                                     "temp node is empty"))
            continue
        try:
            milli = int(raw.split()[0])
        except (ValueError, IndexError):
            zones.append(ThermalZone(zone, name, None, "unavailable",
                                     f"temp node is not an integer: {raw[:32]!r}"))
            continue
        if not (_TEMP_MIN_MC <= milli <= _TEMP_MAX_MC):
            zones.append(ThermalZone(zone, name, None, "unavailable",
                                     f"driver sentinel {milli} mC (outside "
                                     f"{_TEMP_MIN_MC}..{_TEMP_MAX_MC})"))
            continue
        zones.append(ThermalZone(zone, name, milli / 1000.0, "ok"))
    return zones


def thermal_zones(root: Optional[str] = None) -> Dict[str, float]:
    """{zone_label: temperature_c} for the zones that could actually be read.

    Unreadable zones are absent rather than zero. An empty dict means no
    thermal telemetry is available on this board -- see
    :func:`thermal_zone_report` for the per-zone reason.
    """
    out: Dict[str, float] = {}
    for z in thermal_zone_report(root):
        if z.status == "ok" and z.temp_c is not None:
            out[z.label] = z.temp_c
    return out


def max_temp_c() -> Optional[float]:
    """Hottest readable zone, or ``None`` when no zone can be read.

    Returns ``None``, not 0.0: 0 C is a temperature a caller can act on (it is
    below every cooldown target, so a fabricated zero would make ``cool_down``
    report a cooled board that was never measured) and would be written into
    result files as if it had been measured.
    """
    z = thermal_zones()
    return max(z.values()) if z else None


def cool_down(target_c: float = 45.0, timeout_s: float = 300.0,
              poll_s: float = 5.0, verbose: bool = True) -> Dict[str, object]:
    """Idle until the hottest zone drops below ``target_c``.

    Call this between power modes and between backbones.  Orin Nano's passive
    heatsink means a back-to-back sweep measures the *thermal state of the
    previous experiment* as much as the current one; the 25 W mode in
    particular will soak and start throttling within a couple of minutes of
    sustained prefill.  A run that did not cool down is not comparable to one
    that did, so the wait is recorded in the result.
    """
    t0 = time.monotonic()
    start = max_temp_c()
    if start is None:
        # No readable thermal zone. Waiting would be waiting on nothing, and
        # claiming "reached" would be claiming a measurement that was never
        # taken -- so say so and let the caller continue.
        if verbose:
            print("    [cooldown] no readable thermal zone -- skipping the cooldown wait "
                  "(temperatures are recorded as unavailable, not as 0 C).", flush=True)
        return {"start_c": None, "end_c": None, "waited_s": 0.0,
                "target_c": target_c, "reached": None,
                "thermal_available": False}

    cur = start
    while cur is not None and cur > target_c and (time.monotonic() - t0) < timeout_s:
        if verbose:
            print(f"    [cooldown] {cur:.1f}C > {target_c:.1f}C, waiting...", flush=True)
        time.sleep(poll_s)
        cur = max_temp_c()
    end = cur
    return {"start_c": start, "end_c": end,
            "waited_s": round(time.monotonic() - t0, 1),
            "target_c": target_c,
            "reached": (end is not None and end <= target_c),
            "thermal_available": end is not None}


# ---------------------------------------------------------------------------
# whole-device snapshot
# ---------------------------------------------------------------------------

def device_info(include_pip: bool = False) -> dict:
    info = {
        "hostname": socket.gethostname(),
        "is_jetson": is_jetson(),
        "board_model": board_model(),
        "l4t": l4t_version(),
        "jetpack": jetpack_version(),
        "ram_total_mb": module_ram_mb(),
        "power_mode": current_power_mode(),
        "power_modes_available": list_power_modes(),
        "clocks": clocks_snapshot(),
        # thermal_c holds ONLY zones that were actually read; thermal_zone_status
        # lists every zone including the unreadable ones (temp_c: null), so a
        # result file distinguishes "45.2 C" from "never measured".
        "thermal_c": thermal_zones(),
        "thermal_zone_status": [
            {"zone": z.zone, "name": z.name, "temp_c": z.temp_c,
             "status": z.status, "detail": z.detail}
            for z in thermal_zone_report()
        ],
        "kernel": _run(["uname", "-r"]).strip(),
        "swap_mb": None,
    }
    raw = _read("/proc/meminfo") or ""
    m = re.search(r"SwapTotal:\s+(\d+) kB", raw)
    if m:
        info["swap_mb"] = int(m.group(1)) // 1024
    try:
        import torch  # noqa
        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        }
    except Exception as e:  # torch missing or broken -- record it, do not crash
        info["torch"] = {"error": repr(e)}
    if include_pip:
        info["pip_freeze"] = _run(["python3", "-m", "pip", "freeze"]).splitlines()
    return info


if __name__ == "__main__":
    print(json.dumps(device_info(), indent=2, default=str))
