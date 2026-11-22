"""Shared benchmark helpers: timing with warm-up + median/spread, machine info."""
from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


def time_fn(fn, runs: int = 5, warmup: int = 1, sync=None) -> dict:
    """Run fn() `warmup` times untimed, then `runs` times timed. Returns seconds stats."""
    for _ in range(warmup):
        fn()
        if sync:
            sync()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t0)
    return {"median": statistics.median(ts), "min": min(ts), "max": max(ts), "runs": runs}


def time_interleaved(fns: dict, runs: int = 5, warmup: set | bool = True, sync=None) -> dict:
    """Time several configs round-robin: each round takes one sample of every config.

    On a laptop the CPU clock drifts (turbo, thermals, background load). Timing config A
    in one block and config B in a later block bakes that drift into the A-vs-B
    comparison; interleaving spreads it evenly across all configs.
    `warmup`: True = one untimed call of every config first; a set = only those names.
    """
    names = list(fns)
    warm = set(names) if warmup is True else set(warmup or ())
    for n in names:
        if n in warm:
            fns[n]()
            if sync:
                sync()
    ts = {n: [] for n in names}
    for _ in range(runs):
        for n in names:
            t0 = time.perf_counter()
            fns[n]()
            if sync:
                sync()
            ts[n].append(time.perf_counter() - t0)
    return {n: {"median": statistics.median(v), "min": min(v), "max": max(v), "runs": runs}
            for n, v in ts.items()}


def power_source() -> str:
    if sys.platform != "win32":
        return "unknown"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return "unknown"
    if not out:
        return "AC (no battery)"
    return "AC" if out.splitlines()[0].strip() in {"2", "6", "7", "8", "9"} else "battery"


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or "unknown"


def machine_info() -> dict:
    info = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "os": platform.platform(),
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "power": power_source(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        cfg = np.show_config(mode="dicts")
        info["blas"] = cfg["Build Dependencies"]["blas"]["name"]
    except Exception:
        pass
    return info


def machine_md(info: dict) -> str:
    return "\n".join(f"- **{k}**: {v}" for k, v in info.items())
