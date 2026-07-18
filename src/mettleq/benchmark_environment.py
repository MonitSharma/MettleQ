"""Privacy-conscious environment metadata for reproducible performance runs."""
from __future__ import annotations

from datetime import datetime, timezone
import os
import platform
import subprocess


def _command(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            arguments, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = (completed.stdout or completed.stderr).strip()
    return value or None


def capture_performance_environment() -> dict:
    """Return stable machine/load facts without usernames or process names."""
    uname = platform.uname()
    load = os.getloadavg()
    process_count = _command("/bin/ps", "-A", "-o", "pid=")
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": {"system": uname.system, "release": uname.release,
               "version": platform.mac_ver()[0], "machine": uname.machine},
        "hardware": {
            "model": _command("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
            "model_identifier": _command("/usr/sbin/sysctl", "-n", "hw.model"),
            "physical_memory_bytes": int(_command(
                "/usr/sbin/sysctl", "-n", "hw.memsize") or 0),
            "logical_cpu_count": os.cpu_count(),
        },
        "load": {
            "load_average_1m": load[0], "load_average_5m": load[1],
            "load_average_15m": load[2],
            "process_count": len(process_count.splitlines()) if process_count else None,
        },
        "thermal_state": _command("/usr/bin/pmset", "-g", "therm"),
        "power_mode": _command("/usr/bin/pmset", "-g", "custom"),
        "privacy": "No usernames, home paths, serial numbers, or process names are recorded.",
    }


__all__ = ["capture_performance_environment"]
