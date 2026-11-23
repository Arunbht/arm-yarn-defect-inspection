#!/usr/bin/env python3
"""
Continuous Webcam Queue Feeder for Raspberry Pi 5 (PS5)
------------------------------------------------------
Watches a "web_cam" folder continuously and transfers photos into "sequence_30"
at a controlled interval (default: 1.0s per image).

Handles two modes seamlessly:
  1. Pre-existing Mass-Dump: If "web_cam" already has a batch of photos, it queues
     them in chronological order (FIFO) and streams them 1-by-1 every second.
  2. Sequential Live Streaming: If photos arrive sequentially from a camera process,
     it stays actively listening in the background and forwards them as they drop in.

Features:
  - Active daemon loop: never exits when empty; stays idle waiting for new photos.
  - Safe Atomic Writes: writes to .tmp_... then renames so inspect_live.py never
    encounters a half-written file.
  - Zero Dependencies: Pure standard Python 3.
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path

# Terminal colors
BOLD = "\033[1m"
GREEN = "\033[1;32m"
CYAN = "\033[1;36m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
RESET = "\033[0m"

VALID_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp", ""}


def find_default_webcam_dir():
    candidates = [
        Path(__file__).resolve().parent / "web_cam",
        Path.cwd() / "web_cam",
        Path(__file__).resolve().parent.parent / "web_cam",
        Path.home() / "arm/web_cam",
        Path.home() / "web_cam",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return Path(__file__).resolve().parent / "web_cam"


def find_default_target_dir():
    candidates = [
        Path(__file__).resolve().parent / "sequence_30",
        Path.cwd() / "sequence_30",
        Path(__file__).resolve().parent.parent / "sequence_30",
        Path.home() / "arm/sequence_30",
    ]
    for c in candidates:
        if c.is_dir() or c.parent.is_dir():
            return c
    return Path(__file__).resolve().parent / "sequence_30"


def is_image_candidate(p: Path) -> bool:
    if p.name.startswith(".") or p.name.startswith("tmp"):
        return False
    if not p.is_file():
        return False
    if p.suffix.lower() in {".txt", ".csv", ".json", ".log", ".py", ".sh", ".tmp"}:
        return False
    try:
        # Minimum size check to ignore corrupt/zero-byte files
        return p.stat().st_size >= 128
    except Exception:
        return False


def get_pending_images(source_dir: Path, processed_set: set):
    """Scans source_dir for new images, sorted chronologically (oldest first)."""
    try:
        files = [p for p in source_dir.iterdir() if is_image_candidate(p)]
    except Exception:
        return []

    new_files = []
    for f in files:
        try:
            stat = f.stat()
            file_key = (f.name, stat.st_mtime, stat.st_size)
            if file_key not in processed_set:
                new_files.append((f, stat.st_mtime, file_key))
        except Exception:
            continue

    # Sort by modification time (FIFO order)
    new_files.sort(key=lambda x: x[1])
    return new_files


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Webcam Queue Feeder for Raspberry Pi 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=str(find_default_webcam_dir()),
        help="Path to incoming webcam watch folder",
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        default=str(find_default_target_dir()),
        help="Path to destination sequence_30 folder watched by inspect_live.py",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Time interval (seconds) between transferring each file",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (consumes source folder)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop mode: when all existing files are sent, restart from the beginning",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean destination sequence_30 folder before starting",
    )
    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    target_path = Path(args.target).expanduser().resolve()

    source_path.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"{BOLD}📸 PS5 Webcam Continuous Queue Feeder (Active Watcher){RESET}")
    print("=" * 70)
    print(f"  Source (web_cam):   {BOLD}{source_path}{RESET}")
    print(f"  Target (sequence):  {BOLD}{target_path}{RESET}")
    print(f"  Feed Rate:          {BOLD}{args.interval:.1f}s per image{RESET}")
    print(f"  Mode:               {BOLD}{"MOVE" if args.move else "COPY"}{RESET}" +
          (f" {YELLOW}(Looping Enabled){RESET}" if args.loop else ""))
    print("=" * 70)

    if args.clean:
        print(f"\n[*] Cleaning existing files from {target_path}...")
        cleaned = 0
        for f in target_path.iterdir():
            if f.is_file() and not f.name.startswith("."):
                try:
                    f.unlink()
                    cleaned += 1
                except Exception:
                    pass
        print(f"[✓] Removed {cleaned} old files.")

    processed = set()
    total_transferred = 0
    was_idle = False
    start_time = time.time()

    print(f"\n{BOLD}[*] Feeder active. Monitoring '{source_path.name}/' (Ctrl+C to stop)...{RESET}\n")
    print(f"{"#":<6} | {"STATUS":<12} | {"FORWARDED FILENAME":<42} | {"QUEUE LEFT"}")
    print("-" * 78)

    try:
        while True:
            pending = get_pending_images(source_path, processed)

            if not pending:
                if args.loop and processed:
                    print(f"{DIM}[LOOP] All files dispatched. Restarting queue...{RESET}")
                    processed.clear()
                    time.sleep(0.5)
                    continue

                if not was_idle:
                    print(f"{DIM}[IDLE] Waiting for incoming photos in {source_path}...{RESET}")
                    sys.stdout.flush()
                    was_idle = True
                time.sleep(0.2)
                continue

            was_idle = False

            # Take the oldest pending image
            src_file, _, file_key = pending[0]
            remaining_in_queue = len(pending) - 1

            total_transferred += 1
            processed.add(file_key)

            dest_filename = f"webcam_{total_transferred:04d}_{src_file.name}"
            dest_file = target_path / dest_filename
            temp_file = target_path / f".tmp_{dest_filename}"

            try:
                if args.move:
                    shutil.move(str(src_file), str(dest_file))
                    action_tag = f"{YELLOW}MOVED{RESET}"
                else:
                    shutil.copy2(src_file, temp_file)
                    os.replace(temp_file, dest_file)
                    action_tag = f"{GREEN}COPIED{RESET}"

                print(f"#{total_transferred:<5} | {action_tag:<21} | {dest_filename:<42} | {remaining_in_queue:>4} pending")
                sys.stdout.flush()

            except Exception as e:
                print(f"{YELLOW}[!] Error transferring {src_file.name}: {e}{RESET}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"{BOLD}🛑 Webcam Feeder Stopped!{RESET}")
        print(f"  Total Images Forwarded: {total_transferred}")
        print(f"  Elapsed Runtime:        {elapsed:.1f}s")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
