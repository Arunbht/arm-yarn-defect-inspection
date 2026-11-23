#!/usr/bin/env python3
"""
Raspberry Pi 5 GenAI Operator Advisory Monitor (Terminal 2)
-----------------------------------------------------------
Dedicated real-time viewer for Llama 3.2 1B operator guidance.
Runs in Terminal 2 (side-by-side with inspect_live.py in Terminal 1).

Features:
  - High-visibility operator advisory cards (ACTION & CHECK)
  - Color-coded severity (Red for HIGH/KNOTTING, Yellow for MEDIUM, Magenta for WRONG_IMAGE)
  - Full independent terminal scrollback to review past shift advisories
  - Zero DL dependencies: ultra-lightweight, 0% CPU, 2MB RAM
"""

import os
import sys
import time
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_CSV = ROOT / "inspection_log.csv"

ALERT_COLORS = {
    "HIGH": "\033[1;41;97m",       # Bold White on Red Background
    "MEDIUM": "\033[1;43;30m",     # Bold Black on Yellow Background
    "INVALID": "\033[1;45;97m",    # Bold White on Magenta Background
    "CYAN": "\033[1;36m",
    "BOLD": "\033[1m",
    "GREEN": "\033[1;32m",
    "DIM": "\033[2m",
    "RESET": "\033[0m",
}


def print_header():
    print(f"\n{ALERT_COLORS['BOLD']}╔══════════════════════════════════════════════════════════════════════╗{ALERT_COLORS['RESET']}")
    print(f"{ALERT_COLORS['BOLD']}║     ON-DEVICE GenAI OPERATOR ADVISORY MONITOR (TERMINAL 2)           ║{ALERT_COLORS['RESET']}")
    print(f"{ALERT_COLORS['BOLD']}║     Target: Raspberry Pi 5 | Model: Llama 3.2 1B (Quantized Edge)    ║{ALERT_COLORS['RESET']}")
    print(f"{ALERT_COLORS['BOLD']}╚══════════════════════════════════════════════════════════════════════╝{ALERT_COLORS['RESET']}")
    print(f"{ALERT_COLORS['CYAN']}[*] Monitoring live advisory feed from: {LOG_CSV.name}{ALERT_COLORS['RESET']}")
    print(f"{ALERT_COLORS['DIM']}[*] Full terminal scrollback active. Press Ctrl+C to exit.{ALERT_COLORS['RESET']}\n")


def print_advisory_card(row):
    """
    row: [timestamp, file, defect, confidence, severity, action, check, vision_ms, source, tok_s]
    """
    if len(row) < 7:
        return

    ts = row[0]
    filename = row[1]
    defect = row[2]
    conf = row[3]
    severity = row[4].upper()
    action = row[5]
    check = row[6]
    vis_ms = row[7] if len(row) > 7 else ""
    source = row[8] if len(row) > 8 else "advisor"
    tok_s = row[9] if len(row) > 9 else ""

    # Don't clutter advisory monitor with nominal yarns
    if severity in ("NONE", "NORMAL") or defect.lower() == "normal":
        return

    reset = ALERT_COLORS["RESET"]
    bold = ALERT_COLORS["BOLD"]

    if severity == "HIGH":
        badge = f"{ALERT_COLORS['HIGH']} 🚨 CRITICAL HIGH-SEVERITY DEFECT: {defect.upper()} {reset}"
    elif severity == "INVALID":
        badge = f"{ALERT_COLORS['INVALID']} ⚠️ OUT-OF-DISTRIBUTION: {defect.upper()} {reset}"
    else:
        badge = f"{ALERT_COLORS['MEDIUM']} ⚠️ WARNING DEFECT: {defect.upper()} {reset}"

    perf_info = f"Source: {source}"
    if tok_s:
        perf_info += f" @ {tok_s} tok/s"
    if vis_ms:
        perf_info += f" | Vision: {vis_ms}ms"

    print(f"\n┌──────────────────────────────────────────────────────────────────────┐")
    print(f"│ [{ts}] {badge}")
    print(f"│ Frame: {bold}{filename}{reset} | Confidence: {conf} | {perf_info}")
    print(f"├──────────────────────────────────────────────────────────────────────┤")
    print(f"│ {bold}ACTION:{reset} {action}")
    if check:
        print(f"│ {bold}CHECK :{reset} {check}")
    print(f"└──────────────────────────────────────────────────────────────────────┘")
    sys.stdout.flush()


def tail_advisory_log():
    print_header()

    # Wait for CSV log to exist
    while not LOG_CSV.exists():
        time.sleep(0.5)

    processed_rows = 0

    # Initial scan of recent existing defect rows (last 5)
    try:
        with open(LOG_CSV, "r", newline="") as f:
            reader = list(csv.reader(f))
            if len(reader) > 1:
                header = reader[0]
                defect_rows = [r for r in reader[1:] if len(r) > 4 and r[4].upper() not in ("NONE", "NORMAL")]
                for r in defect_rows[-5:]:
                    print_advisory_card(r)
                processed_rows = len(reader)
    except Exception:
        pass

    # Follow new rows continuously
    while True:
        try:
            with open(LOG_CSV, "r", newline="") as f:
                reader = list(csv.reader(f))
                if len(reader) > processed_rows:
                    new_rows = reader[processed_rows:]
                    for r in new_rows:
                        print_advisory_card(r)
                    processed_rows = len(reader)
        except Exception:
            pass

        time.sleep(0.5)


def main():
    try:
        tail_advisory_log()
    except KeyboardInterrupt:
        print(f"\n{ALERT_COLORS['DIM']}Advisory monitor stopped.{ALERT_COLORS['RESET']}")


if __name__ == "__main__":
    main()
