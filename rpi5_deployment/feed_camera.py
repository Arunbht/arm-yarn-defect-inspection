#!/usr/bin/env python3
"""
Camera Feed Simulator for Bharat AI-SoC Challenge (PS5)
------------------------------------------------------
Simulates a live industrial yarn inspection camera by dumping random BMP
photos from the 6 defect classes into the watch folder (sequence_30/) at a
configurable interval (default: 1.0 second per frame).
"""

import os
import sys
import time
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

COLORS = {
    'normal': '[1;32m',
    'knotting': '[1;31m',
    'long segment lack': '[1;35m',
    'short segment lack': '[1;33m',
    'irregularity': '[1;34m',
    'thinness': '[1;36m',
    'BOLD': '[1m',
    'RESET': '[0m',
}

def find_default_dataset():
    candidates = [
        Path(__file__).resolve().parent / 'dataset',
        Path(__file__).resolve().parent / 'Yarn Defect Dataset',
        Path(__file__).resolve().parent.parent / 'Yarn Defect Dataset',
        Path(__file__).resolve().parent.parent / 'dataset',
        Path.cwd() / 'dataset',
        Path.cwd() / 'Yarn Defect Dataset',
        Path.home() / 'dataset',
        Path.home() / 'arm/dataset',
        Path.home() / 'Yarn Defect Dataset',
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return Path(__file__).resolve().parent / 'dataset'

def find_default_target():
    candidates = [
        Path(__file__).resolve().parent / 'sequence_30',
        Path.cwd() / 'sequence_30',
        Path(__file__).resolve().parent.parent / 'sequence_30',
        Path.home() / 'arm/sequence_30',
    ]
    for c in candidates:
        if c.is_dir() or c.parent.is_dir():
            return c
    return Path(__file__).resolve().parent / 'sequence_30'

def discover_class_files(dataset_dir):
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not dataset_path.is_dir():
        return {}
    class_map = {}
    search_dirs = [dataset_path]
    for sub in ['train', 'val', 'test', 'images']:
        p = dataset_path / sub
        if p.is_dir():
            search_dirs.append(p)
    for d in search_dirs:
        for entry in d.iterdir():
            if entry.is_dir() and not entry.name.startswith('.'):
                cname = entry.name.lower().strip()
                bmps = list(entry.glob('*.bmp')) + list(entry.glob('*.BMP'))
                if bmps:
                    if cname not in class_map:
                        class_map[cname] = []
                    class_map[cname].extend(bmps)
    return class_map

def main():
    parser = argparse.ArgumentParser(description='Feeder Simulator: Dumps BMP yarn images to sequence_30 every second.')
    parser.add_argument('--dataset', '-d', type=str, default=str(find_default_dataset()), help='Path to dataset')
    parser.add_argument('--target', '-t', type=str, default=str(find_default_target()), help='Path to sequence_30')
    parser.add_argument('--interval', '-i', type=float, default=1.0, help='Delay in seconds')
    parser.add_argument('--count', '-n', type=int, default=0, help='Total frames (0 = infinite)')
    parser.add_argument('--clean', action='store_true', help='Clean sequence_30 first')
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    target_path = Path(args.target).expanduser().resolve()

    print('
' + '=' * 68)
    print(f"{COLORS['BOLD']}🚀 PS5 Camera Feed Simulator (Dataset Feeder){COLORS['RESET']}")
    print('=' * 68)
    print(f'  Source Dataset: {COLORS["BOLD"]}{dataset_path}{COLORS["RESET"]}')
    print(f'  Target Folder:  {COLORS["BOLD"]}{target_path}{COLORS["RESET"]}')
    print(f'  Feed Interval:  {COLORS["BOLD"]}{args.interval:.1f}s per image{COLORS["RESET"]}')
    print('=' * 68)

    class_map = discover_class_files(dataset_path)
    if not class_map:
        print(f'❌ ERROR: No class folders with BMP files found in {dataset_path}')
        sys.exit(1)

    print(f'
[✓] Discovered {len(class_map)} classes:')
    for cname, files in sorted(class_map.items()):
        color = COLORS.get(cname, '')
        print(f'    • {color}{cname:<20}{COLORS["RESET"]}: {len(files):>5} images')

    target_path.mkdir(parents=True, exist_ok=True)
    if args.clean:
        print(f'
[*] Cleaning existing files from {target_path}...')
        for f in target_path.iterdir():
            if f.is_file() and not f.name.startswith('.'):
                try: f.unlink()
                except: pass

    print(f"
{COLORS['BOLD']}[*] Starting feed (Press Ctrl+C to stop)...{COLORS['RESET']}
")
    print(f"{'#':<6} | {'CLASS':<20} | {'DUMPED FILENAME':<42} | {'INTERVAL'}")
    print('-' * 78)

    classes_list = list(class_map.keys())
    counts = Counter()
    total = 0
    start_time = time.time()

    try:
        while True:
            random.shuffle(classes_list)
            for cname in classes_list:
                src = random.choice(class_map[cname])
                total += 1
                counts[cname] += 1
                dest_name = f'frame_{total:04d}_{cname.replace(" ", "_")}_{src.name}'
                dest_p = target_path / dest_name
                tmp_p = target_path / f'.tmp_{dest_name}'
                shutil.copy2(src, tmp_p)
                os.replace(tmp_p, dest_p)
                color = COLORS.get(cname, '')
                print(f'#{total:<5} | {color}{cname:<20}{COLORS["RESET"]} | {dest_name:<42} | {args.interval:.1f}s')
                sys.stdout.flush()
                if args.count > 0 and total >= args.count:
                    raise KeyboardInterrupt
                time.sleep(args.interval)
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print('
' + '=' * 68)
        print(f"{COLORS['BOLD']}🛑 Feeder Stopped!{COLORS['RESET']}")
        print(f'  Total Images Dumped: {total}')
        print(f'  Elapsed Time:        {elapsed:.1f}s')
        print('=' * 68 + '
')

if __name__ == '__main__':
    main()
