# Copied unchanged (apart from the import path) from Matryoshka Learning/jetson_eval_harness/tests (2026-11-17).
#!/usr/bin/env python3
"""Jetson telemetry robustness: every way a sysfs probe can fail, and what is recorded.

    python3 -m unittest discover -s tests -v        (from the harness root)

These reproduce the exp01 crash class -- a thermal zone whose ``temp`` node
cannot be read -- against a fake /sys/class/thermal tree, so they run on any
machine. The invariants under test are the two the harness cares about:

  1. an unreadable optional probe never aborts the run, and
  2. an unreadable optional probe never becomes a number.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jetson"))

from nestbench import device, power  # noqa: E402


def zone(root: str, n: int, *, ztype: str = None, temp: str = None,
         temp_bytes: bytes = None, no_temp_file: bool = False) -> str:
    d = os.path.join(root, f"thermal_zone{n}")
    os.makedirs(d, exist_ok=True)
    if ztype is not None:
        with open(os.path.join(d, "type"), "w", encoding="utf-8") as f:
            f.write(ztype + "\n")
    if not no_temp_file:
        if temp_bytes is not None:
            with open(os.path.join(d, "temp"), "wb") as f:
                f.write(temp_bytes)
        elif temp is not None:
            with open(os.path.join(d, "temp"), "w", encoding="utf-8") as f:
                f.write(temp)
    return d


class TestThermalZones(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def test_healthy_zone(self):
        zone(self.root, 0, ztype="CPU-therm", temp="42500\n")
        report = device.thermal_zone_report(self.root)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0].status, "ok")
        self.assertAlmostEqual(report[0].temp_c, 42.5)
        self.assertEqual(device.thermal_zones(self.root), {"CPU-therm": 42.5})

    def test_missing_temp_file_does_not_raise(self):
        """The exp01 crash: a zone directory whose temp node cannot be read."""
        zone(self.root, 0, ztype="CPU-therm", no_temp_file=True)
        report = device.thermal_zone_report(self.root)
        self.assertEqual(report[0].status, "unavailable")
        self.assertIsNone(report[0].temp_c)
        self.assertNotIn("CPU-therm", device.thermal_zones(self.root))

    def test_missing_type_file_still_reports_the_zone(self):
        zone(self.root, 3, temp="51000")
        report = device.thermal_zone_report(self.root)
        self.assertEqual(report[0].status, "ok")
        self.assertEqual(report[0].label, "thermal_zone3")   # falls back to the dir

    def test_empty_temp_file(self):
        zone(self.root, 0, ztype="GPU-therm", temp="")
        r = device.thermal_zone_report(self.root)[0]
        self.assertEqual(r.status, "unavailable")
        self.assertIn("empty", r.detail)

    def test_non_integer_temp(self):
        zone(self.root, 0, ztype="odd", temp="not-a-number")
        r = device.thermal_zone_report(self.root)[0]
        self.assertEqual(r.status, "unavailable")
        self.assertIn("not an integer", r.detail)

    def test_binary_temp_node(self):
        """A sysfs attribute that is not valid UTF-8 -> UnicodeDecodeError, not a crash."""
        zone(self.root, 0, ztype="weird", temp_bytes=b"\xff\xfe\x00\x01")
        r = device.thermal_zone_report(self.root)[0]
        self.assertEqual(r.status, "unavailable")
        self.assertIsNone(r.temp_c)

    def test_driver_sentinel_is_not_a_temperature(self):
        """Several Orin zones report -256000 mC when the sensor has no reading."""
        zone(self.root, 0, ztype="tj-therm", temp="-256000")
        r = device.thermal_zone_report(self.root)[0]
        self.assertEqual(r.status, "unavailable")
        self.assertIn("sentinel", r.detail)
        self.assertEqual(device.thermal_zones(self.root), {})

    def test_mixed_zones_partial_success(self):
        zone(self.root, 0, ztype="CPU-therm", temp="42500")
        zone(self.root, 1, ztype="GPU-therm", no_temp_file=True)
        zone(self.root, 2, ztype="SOC0-therm", temp="45100")
        report = device.thermal_zone_report(self.root)
        self.assertEqual(len(report), 3)
        self.assertEqual(sum(1 for z in report if z.status == "ok"), 2)
        temps = device.thermal_zones(self.root)
        self.assertEqual(set(temps), {"CPU-therm", "SOC0-therm"})

    def test_no_thermal_tree_at_all(self):
        missing = os.path.join(self.root, "nope")
        self.assertEqual(device.thermal_zone_report(missing), [])
        self.assertEqual(device.thermal_zones(missing), {})

    def test_trailing_whitespace_and_units(self):
        zone(self.root, 0, ztype="CPU-therm ", temp="  42500  \n")
        r = device.thermal_zone_report(self.root)[0]
        self.assertAlmostEqual(r.temp_c, 42.5)


class TestMaxTempAndCooldown(unittest.TestCase):
    """Unavailable thermals must be null, never 0 C (which reads as 'freezing')."""

    def setUp(self):
        """Repoint THERMAL_ROOT at an empty tree -- these must hold on a Jetson too."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        original = device.THERMAL_ROOT
        device.THERMAL_ROOT = os.path.join(self.tmp.name, "no-thermal-here")
        self.addCleanup(setattr, device, "THERMAL_ROOT", original)

    def test_the_root_override_actually_takes_effect(self):
        """Guards the default-argument trap: a default bound at import time
        would silently keep reading the real /sys and make the two tests below
        vacuous on Linux."""
        self.assertEqual(device.thermal_zones(), {})
        self.assertEqual(device.thermal_zone_report(), [])

    def test_max_temp_is_none_when_nothing_readable(self):
        self.assertIsNone(device.max_temp_c())

    def test_cooldown_reports_unavailable_rather_than_waiting_or_lying(self):
        out = device.cool_down(45.0, timeout_s=1.0, poll_s=0.01, verbose=False)
        self.assertIs(out["thermal_available"], False)
        self.assertIsNone(out["start_c"])
        self.assertIsNone(out["end_c"])
        self.assertIsNone(out["reached"])       # not True, and not False
        self.assertEqual(out["waited_s"], 0.0)


class TestReadHelper(unittest.TestCase):

    def test_read_returns_none_for_missing_file(self):
        self.assertIsNone(device._read(os.path.join(tempfile.gettempdir(), "nope-xyz")))

    def test_read_rejects_a_none_path_loudly(self):
        """A None path is a programming error and must not be swallowed."""
        with self.assertRaises(TypeError):
            device._read(None)

    def test_eagain_on_read_returns_none(self):
        """The real Jetson crash (2026-09-16): a temp node that answers EAGAIN.

        The thermal driver returns -EAGAIN when the sensor has no reading ready.
        Through a buffered TEXT stream that used to surface as the raw layer
        returning None, which TextIOWrapper fed to the incremental decoder:

            File "/usr/lib/python3.10/codecs.py", line 321, in decode
                data = self.buffer + input
            TypeError: can't concat NoneType to bytes

        os.read raises BlockingIOError instead, which is an OSError. This test
        fails against the old open()/f.read() implementation.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "temp")
        with open(path, "w", encoding="utf-8") as f:
            f.write("42500")

        real_read = os.read
        state = {"first": True}

        def eagain(fd, n):
            if state["first"]:
                state["first"] = False
                raise BlockingIOError(11, "Resource temporarily unavailable")
            return real_read(fd, n)

        os.read = eagain
        self.addCleanup(setattr, os, "read", real_read)
        self.assertIsNone(device._read(path))

    def test_a_typeerror_from_the_io_stack_is_not_mistaken_for_a_bad_path(self):
        """Any decoder-level failure must yield None, not propagate."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "temp")
        with open(path, "wb") as f:
            f.write(bytes([0xFF, 0xFE, 0x00, 0x01]))
        self.assertIsNone(device._read(path))

    def test_the_whole_thermal_report_survives_an_eagain_zone(self):
        """End to end: one EAGAIN zone must not take down device_info()."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        zone(tmp.name, 0, ztype="CPU-therm", temp="42500")
        zone(tmp.name, 1, ztype="tj-therm", temp="51000")

        real_read = os.read
        blocked = {}

        def eagain(fd, n):
            # Fail every read of the SECOND zone's temp node, once per open.
            try:
                name = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                name = ""
            if "thermal_zone1" in name and fd not in blocked:
                blocked[fd] = True
                raise BlockingIOError(11, "Resource temporarily unavailable")
            return real_read(fd, n)

        if not os.path.exists("/proc/self/fd"):
            self.skipTest("needs /proc to identify the descriptor under test")
        os.read = eagain
        self.addCleanup(setattr, os, "read", real_read)

        report = device.thermal_zone_report(tmp.name)
        self.assertEqual(len(report), 2)
        by_zone = {z.zone: z for z in report}
        self.assertEqual(by_zone["thermal_zone0"].status, "ok")
        self.assertEqual(by_zone["thermal_zone1"].status, "unavailable")
        self.assertIsNone(by_zone["thermal_zone1"].temp_c)


class TestDeviceInfoDegradesGracefully(unittest.TestCase):

    def test_device_info_succeeds_off_jetson(self):
        """Every probe unavailable -> a complete record of nulls, not an exception."""
        info = device.device_info()
        for key in ("hostname", "is_jetson", "board_model", "l4t", "jetpack",
                    "ram_total_mb", "power_mode", "power_modes_available",
                    "clocks", "thermal_c", "thermal_zone_status", "torch"):
            self.assertIn(key, info)
        # Sentinels, not fabricated zeroes.
        for key in ("board_model", "l4t", "jetpack", "ram_total_mb"):
            self.assertIsNot(info[key], 0, f"{key} must be null when unknown, not 0")
        for value in info["clocks"].values():
            self.assertNotEqual(value, 0)


class TestPowerNeverFabricatesZero(unittest.TestCase):

    def test_empty_trace_summary_is_null_not_zero(self):
        s = power.PowerTrace().summary()
        self.assertFalse(s["measured"])
        self.assertIsNone(s["energy_j"])
        self.assertIsNone(s["mean_power_w"])
        self.assertIsNone(s["peak_power_w"])
        self.assertEqual(s["n_samples"], 0)

    def test_single_sample_cannot_be_integrated(self):
        t = power.PowerTrace()
        t.add(1.0, {"VDD_IN": 4000.0})
        self.assertFalse(t.has_data())
        self.assertIsNone(t.summary()["energy_j"])

    def test_real_samples_are_measured(self):
        t = power.PowerTrace()
        t.add(0.0, {"VDD_IN": 4000.0})
        t.add(1.0, {"VDD_IN": 4000.0})
        s = t.summary()
        self.assertTrue(s["measured"])
        self.assertAlmostEqual(s["energy_j"], 4.0)          # 4 W for 1 s
        self.assertAlmostEqual(s["mean_power_w"], 4.0)

    def test_source_none_sampler_records_nothing(self):
        with power.PowerSampler(source="none", period_ms=5) as ps:
            pass
        s = ps.summary()
        self.assertEqual(s["source"], "none")
        self.assertFalse(s["measured"])
        self.assertIsNone(s["mean_power_w"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
