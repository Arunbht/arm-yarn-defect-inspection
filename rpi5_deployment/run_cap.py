#!/usr/bin/env python3
"""
Interval Runner for cap.py (Raspberry Pi 5)
------------------------------------------
Executes cap.py in a continuous loop at a precise interval (default: 1.0s).

Features:
  - Precise Drift-Compensated Pacing: Accounts for cap.py execution time
    (e.g., if cap.py takes 0.4s to capture, it sleeps 0.6s so exactly one frame
    is captured every 1.0s).
  - Auto-detects cap.py in current directory, parent directory, or ~/arm.
  - Clean live counter with Ctrl+C graceful shutdown.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

BOLD = "\033[1m"
GREEN = "\033[1;32m"
CYAN = "\033[1;36m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"


def find_default_cap_script():
    candidates = [
        Path("cap.py"),
        Path(__file__).resolve().parent / "cap.py",
        Path.cwd() / "cap.py",
        Path(__file__).resolve().parent.parent / "cap.py",
        Path.home() / "arm/cap.py",
        Path.home() / "cap.py",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return Path("cap.py")


def main():
    parser = argparse.ArgumentParser(
        description="Runs cap.py continuously at a fixed interval (1s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--script", "-s",
        type=str,
        default=str(find_default_cap_script()),
        help="Path to cap.py",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Capture interval in seconds (time between each capture start)",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=0,
        help="Total captures (0 = infinite loop until Ctrl+C)",
    )
    args = parser.parse_args()

    script_path = Path(args.script).expanduser().resolve()

    if not script_path.is_file():
        print(f"\n❌ ERROR: Cannot find capture script at: {script_path}")
        print("Please place cap.py in this folder or pass: python3 run_cap.py --script /path/to/cap.py\n")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(f"{BOLD}📷 Automated Interval Camera Runner (cap.py loop){RESET}")
    print("=" * 65)
    print(f"  Target Script : {BOLD}{script_path}{RESET}")
    print(f"  Pacing        : {BOLD}{args.interval:.1f} second per capture{RESET}")
    print("=" * 65)
    print(f"\n{BOLD}[*] Starting automated capture loop (Press Ctrl+C to stop)...{RESET}\n")

    capture_num = 0
    start_time = time.time()

    try:
        while True:
            t0 = time.perf_counter()
            capture_num += 1

            # Run cap.py as a subprocess
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
            )

            t_exec = time.perf_counter() - t0

            if proc.returncode == 0:
                output_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "OK"
                print(f"#{capture_num:<5} | {GREEN}CAPTURED{RESET} in {t_exec:.2f}s | {output_line}")
            else:
                err_msg = proc.stderr.strip() or proc.stdout.strip() or f"Exit code {proc.returncode}"
                print(f"#{capture_num:<5} | {YELLOW}ERROR{RESET} in {t_exec:.2f}s | {err_msg}")

            if args.count > 0 and capture_num >= args.count:
                break

            # Drift compensation: subtract cap.py execution time
            sleep_time = max(0.0, args.interval - t_exec)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"{BOLD}🛑 Capture Stopped!{RESET}")
    print(f"  Total Frames Captured: {capture_num}")
    print(f"  Elapsed Time:          {elapsed:.1f}s")
    if capture_num > 0 and elapsed > 0:
        print(f"  Average Rate:          {capture_num / elapsed:.2f} captures/sec")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
