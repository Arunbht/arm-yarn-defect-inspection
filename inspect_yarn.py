#!/usr/bin/env python3
import os
import sys
import csv
import time
import datetime
import json
import argparse
import subprocess
import atexit
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms, models

ROOT = Path(__file__).resolve().parent

SEVERITY = {
    "normal": "NONE",
    "irregularity": "MEDIUM",
    "thinness": "MEDIUM",
    "short segment lack": "MEDIUM",
    "knotting": "HIGH",
    "long segment lack": "HIGH",
}

ALERT_COLORS = {
    "HIGH": "\033[1;41;97m",    # Bold White on Red Background
    "MEDIUM": "\033[1;43;30m",  # Bold Black on Yellow Background
    "RESET": "\033[0m",
}


def crop_band(im):
    return im.crop((0, int(0.30 * im.height), im.width, int(0.70 * im.height)))


tf = transforms.Compose([
    transforms.Lambda(crop_band),
    transforms.Resize((96, 768)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model_path = ROOT / "yarn.pt"
if not model_path.exists():
    raise SystemExit(f"Vision model checkpoint not found: {model_path}")

ck = torch.load(model_path, map_location="cpu")
classes = ck["classes"]
m = models.mobilenet_v3_small()
m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(classes))
m.load_state_dict(ck["m"])
m.eval()

ADVISOR = None


def fallback_advice(label, severity, conf):
    if severity == "NONE":
        return "Yarn OK. Continue production."
    if conf < 0.70:
        return "Low confidence. Re-inspect this section manually."
    return f"{label} ({severity}). Mark section, check spinning/winding settings, and cut or re-wind if HIGH."


def start_advisor():
    """Start one Llama process for the whole inspection batch."""
    python = ROOT / "llmenv" / "bin" / "python"
    script = ROOT / "llm_advice.py"
    if not python.exists() or not script.exists():
        return None
    try:
        process = subprocess.Popen(
            [str(python), str(script), "--serve"],
            cwd=ROOT,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1,
        )
        return process
    except OSError:
        return None


def stop_advisor():
    global ADVISOR
    if ADVISOR and ADVISOR.poll() is None:
        ADVISOR.terminate()


def advise(label, severity, conf):
    """
    Queries Llama advisor subprocess.
    Returns: (advice_text, advice_source, tokens_per_s)
    """
    if ADVISOR and ADVISOR.poll() is None:
        try:
            ADVISOR.stdin.write(json.dumps({"label": label, "severity": severity, "confidence": conf}) + "\n")
            ADVISOR.stdin.flush()
            raw_line = ADVISOR.stdout.readline()
            response = json.loads(raw_line)
            if response.get("ok"):
                return response["advice"], "llm", f"{response.get('tok_s', 0.0):.1f}"
        except (BrokenPipeError, json.JSONDecodeError, OSError, KeyError):
            pass
    return fallback_advice(label, severity, conf), "fallback", ""


def trigger_alert(label, severity, conf, advice):
    if severity in ("HIGH", "MEDIUM"):
        color = ALERT_COLORS.get(severity, "")
        reset = ALERT_COLORS["RESET"]
        bell = "\a"
        print(f"\n{bell}{color} >>> [OPERATOR ALERT - {severity} SEVERITY] <<< {reset}")
        print(f"{color} DEFECT DETECTED: {label.upper()} (Confidence: {conf*100:.1f}%) {reset}")
        print(f"ADVICE:\n{advice}\n")


def inspect(path):
    t = time.time()
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(m(x), 1)[0]
    conf, i = p.max(0)
    label, conf = classes[i], conf.item()
    sev = SEVERITY[label]
    advice, source, tok_s = advise(label, sev, conf)
    latency_ms = (time.time() - t) * 1000
    return label, conf, sev, advice, source, tok_s, latency_ms


def main():
    global ADVISOR
    parser = argparse.ArgumentParser(description="Inspect yarn defect images with Edge Vision & Llama Advisory.")
    parser.add_argument("folder", help="Path to image folder or specific files")
    parser.add_argument("--stream", type=float, default=None, help="Replay frames at target FPS")
    args = parser.parse_args()

    ADVISOR = start_advisor()
    atexit.register(stop_advisor)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if os.path.isfile(args.folder):
        files = [args.folder]
    else:
        files = sorted(os.path.join(d, f) for d, _, fs in os.walk(args.folder)
                       for f in fs if f.lower().endswith(exts))

    if not files:
        sys.exit(f"No images found in {args.folder}")

    log_path = ROOT / "inspection_log.csv"
    new = not log_path.exists()
    times = []
    sources = {"llm": 0, "fallback": 0}

    # Stream replay tracking
    target_interval = (1.0 / args.stream) if args.stream and args.stream > 0 else 0
    dropped_frames = 0
    stream_start = time.perf_counter()

    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow([
                "timestamp", "file", "defect", "confidence",
                "severity", "advice", "latency_ms", "advice_source", "llm_tokens_s"
            ])

        for idx, p in enumerate(files):
            frame_start = time.perf_counter()
            label, conf, sev, adv, src, tok_s, ms = inspect(p)
            times.append(ms)
            sources[src] += 1

            w.writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                os.path.basename(p),
                label,
                f"{conf:.3f}",
                sev,
                adv,
                f"{ms:.1f}",
                src,
                tok_s,
            ])
            f.flush()

            # Operator Alert
            trigger_alert(label, sev, conf, adv)

            tok_str = f" | {tok_s} tok/s" if tok_s else ""
            print(f"[{idx+1}/{len(files)}] {os.path.basename(p)} -> {label} {conf:.2f} [{sev}] ({src}{tok_str}) {ms:.0f} ms")

            # Stream rate pacing
            if target_interval > 0:
                elapsed = time.perf_counter() - frame_start
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                else:
                    # Dropped frames due to processing delay exceeding target frame period
                    missed = int(elapsed // target_interval)
                    dropped_frames += missed

    total_stream_time = time.perf_counter() - stream_start
    sustained_fps = len(files) / total_stream_time if total_stream_time > 0 else 0

    print(f"\n--- Batch Summary ---")
    print(f"Processed: {len(files)} images | Avg Latency: {sum(times)/len(times):.1f} ms/image")
    print(f"Advisory Source: {sources['llm']} LLM ({sources['llm']/len(files)*100:.1f}%), {sources['fallback']} Fallback ({sources['fallback']/len(files)*100:.1f}%)")

    if args.stream:
        print(f"\n--- Stream Performance (Target: {args.stream:.1f} FPS) ---")
        print(f"Total Time: {total_stream_time:.2f} s")
        print(f"Sustained FPS: {sustained_fps:.2f} FPS")
        print(f"Dropped Frames: {dropped_frames} frames")


if __name__ == "__main__":
    main()
