"""CPU temperature guard (Allen, 2026-10-17: --workers 2 while CPU > 85 °C). Not signal processing; not in ictra_lib.

    python src/thermal.py                 # prints the temperature in °C
    python src/thermal.py --workers 3     # prints 2 if the CPU is above LIMIT_C, else 3 (for launch scripts)

Source: the ACPI thermal zone \\_TZ.THRM from the Windows performance counter Win32_PerfFormattedData_Counters_
ThermalZoneInformation (no admin rights needed). It is the firmware's system/CPU zone, not the per-core sensor; the
per-core MSAcpi class needs admin. If no zone can be read, the guard assumes HOT (returns the reduced worker count).
"""
import argparse, subprocess

LIMIT_C, HOT_WORKERS = 85.0, 2


def cpu_temp_c():
    cmd = ("Get-CimInstance -Namespace root/cimv2 -ClassName Win32_PerfFormattedData_Counters_ThermalZoneInformation | "
           "ForEach-Object { $_.Name + '=' + ($_.Temperature - 273.15) }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    zones = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.rsplit("=", 1)
            try:
                zones[k.strip()] = float(v)
            except ValueError:
                pass
    if not zones:
        return None
    return next((v for k, v in zones.items() if k.upper().endswith("THRM")), max(zones.values()))


def workers(n):
    t = cpu_temp_c()
    return min(n, HOT_WORKERS) if (t is None or t > LIMIT_C) else n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    print(workers(a.workers) if a.workers else cpu_temp_c())
