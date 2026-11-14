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
import queue
import threading
from collections import deque, Counter
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
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
    """Start one Llama process for the inspection session."""
    for cand in [sys.executable, ROOT / "llmenv" / "bin" / "python", ROOT / "env" / "bin" / "python"]:
        cand = Path(cand)
        script = ROOT / "llm_advice.py"
        if cand.exists() and script.exists():
            try:
                process = subprocess.Popen(
                    [str(cand), str(script), "--serve"],
                    cwd=ROOT,
                    text=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=1,
                )
                ready_line = process.stdout.readline()
                data = json.loads(ready_line)
                if data.get("status") == "ready":
                    return process
            except (OSError, json.JSONDecodeError):
                continue
    return None


def stop_advisor():
    global ADVISOR
    if ADVISOR and ADVISOR.poll() is None:
        ADVISOR.terminate()


def query_llm_raw(label, severity, conf, history_str):
    if ADVISOR and ADVISOR.poll() is None:
        try:
            payload = {
                "label": label,
                "severity": severity,
                "confidence": conf,
                "history": history_str,
            }
            ADVISOR.stdin.write(json.dumps(payload) + "\n")
            ADVISOR.stdin.flush()
            raw_line = ADVISOR.stdout.readline()
            response = json.loads(raw_line)
            if response.get("ok"):
                return response["advice"], "llm", f"{response.get('tok_s', 0.0):.1f}"
        except (BrokenPipeError, json.JSONDecodeError, OSError, KeyError):
            pass
    return fallback_advice(label, severity, conf), "fallback", ""


def get_history_summary(history_window):
    if not history_window:
        return "first yarn inspected"
    total = len(history_window)
    defects = [x for x in history_window if x != "normal"]
    total_defects = len(defects)
    if total_defects == 0:
        return f"0 defects in last {total} yarns"
    counts = Counter(defects)
    dominant_defect, _ = counts.most_common(1)[0]
    return f"{total_defects} defects in last {total} yarns ({dominant_defect})"


def trigger_alert(label, severity, conf, extra_info=""):
    if severity in ("HIGH", "MEDIUM"):
        color = ALERT_COLORS.get(severity, "")
        reset = ALERT_COLORS["RESET"]
        bell = "\a"
        print(f"\n{bell}{color} >>> [OPERATOR ALERT - {severity} SEVERITY] <<< {reset}")
        print(f"{color} DEFECT: {label.upper()} (Confidence: {conf*100:.1f}%) {extra_info}{reset}")


def inspect_vision(path):
    t = time.time()
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(m(x), 1)[0]
    conf, i = p.max(0)
    label, conf = classes[i], conf.item()
    sev = SEVERITY[label]
    latency_ms = (time.time() - t) * 1000
    return label, conf, sev, latency_ms


def main():
    global ADVISOR
    parser = argparse.ArgumentParser(description="Inspect yarn defect images with Edge Vision & Llama Advisory.")
    parser.add_argument("folder", help="Path to image folder or specific file")
    parser.add_argument("--stream", type=float, default=None, help="Stream mode target FPS")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Per-class LLM cooldown in seconds (default: 10s)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to inspect")
    parser.add_argument("--sync", action="store_true", help="Synchronous advice generation for all frames")
    args = parser.parse_args()

    print("[Edge AI] Initializing Vision model & Llama 3.2 ExecuTorch Advisor...")
    ADVISOR = start_advisor()
    print(f"[Edge AI] Llama Advisor: {'READY' if ADVISOR else 'FAILED (fallback mode)'}")
    atexit.register(stop_advisor)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if os.path.isfile(args.folder):
        files = [args.folder]
    else:
        files = sorted(os.path.join(d, f) for d, _, fs in os.walk(args.folder)
                       for f in fs if f.lower().endswith(exts))

    if not files:
        sys.exit(f"No images found in {args.folder}")

    if args.max_frames and args.max_frames > 0:
        files = files[:args.max_frames]

    log_path = ROOT / "inspection_log.csv"
    new = not log_path.exists()
    f_log = open(log_path, "a", newline="")
    w_log = csv.writer(f_log)
    if new:
        w_log.writerow([
            "timestamp", "file", "defect", "confidence",
            "severity", "advice", "latency_ms", "advice_source", "llm_tokens_s"
        ])
        f_log.flush()

    history_window = deque(maxlen=20)
    class_cooldown = {}
    sources = {"llm": 0, "fallback": 0}
    advice_latencies = []
    llm_call_count = 0

    # Stream async worker setup
    is_stream = args.stream is not None and args.stream > 0 and not args.sync
    llm_queue = queue.Queue(maxsize=1)
    stop_worker = threading.Event()

    def llm_worker_func():
        nonlocal llm_call_count
        while not stop_worker.is_set():
            try:
                task = llm_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            alert_time = task["alert_time"]
            llm_call_count += 1
            adv, src, tok_s = query_llm_raw(task["label"], task["severity"], task["conf"], task["history"])
            adv_latency = time.time() - alert_time
            advice_latencies.append(adv_latency)
            sources[src] += 1

            # Log to CSV
            w_log.writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                task["file"],
                task["label"],
                f"{task['conf']:.3f}",
                task["severity"],
                adv,
                f"{task['vis_ms']:.1f}",
                src,
                tok_s,
            ])
            f_log.flush()

            print(f"\n  [ASYNC LLM ADVICE COMPLETED for {task['file']}]")
            print(f"  Latency from Alert: {adv_latency:.2f}s | Speed: {tok_s} tok/s | Source: {src}")
            print(f"  {adv}\n")
            llm_queue.task_done()

    if is_stream:
        worker_thread = threading.Thread(target=llm_worker_func, daemon=True)
        worker_thread.start()

    target_interval = (1.0 / args.stream) if is_stream else 0
    dropped_frames = 0
    vision_latencies = []
    stream_start = time.perf_counter()

    for idx, p in enumerate(files):
        frame_start = time.perf_counter()
        label, conf, sev, vis_ms = inspect_vision(p)
        vision_latencies.append(vis_ms)
        history_summary = get_history_summary(history_window)
        history_window.append(label)

        now = time.time()
        cooldown_rem = args.cooldown - (now - class_cooldown.get(label, 0))

        if is_stream:
            # Immediate Alert Banner
            if sev in ("HIGH", "MEDIUM"):
                trigger_alert(label, sev, conf, f"| Trend: {history_summary}")

            # Schedule LLM only for MEDIUM/HIGH if cooldown expired
            if sev in ("HIGH", "MEDIUM"):
                if cooldown_rem <= 0:
                    task = {
                        "file": os.path.basename(p),
                        "label": label,
                        "severity": sev,
                        "conf": conf,
                        "history": history_summary,
                        "vis_ms": vis_ms,
                        "alert_time": now,
                    }
                    try:
                        llm_queue.put_nowait(task)
                        class_cooldown[label] = now
                    except queue.Full:
                        # Queue size 1: drop stale request without blocking stream
                        pass

            print(f"[{idx+1}/{len(files)}] {os.path.basename(p)} -> {label} {conf:.2f} [{sev}] (vision {vis_ms:.1f}ms)")

            # Frame rate pacing
            elapsed = time.perf_counter() - frame_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            else:
                missed = int(elapsed // target_interval)
                dropped_frames += missed
        else:
            # Synchronous batch mode (for Task C and offline inspection)
            if sev in ("HIGH", "MEDIUM"):
                adv, src, tok_s = query_llm_raw(label, sev, conf, history_summary)
            else:
                adv, src, tok_s = fallback_advice(label, sev, conf), "fallback", ""
            sources[src] += 1
            w_log.writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                os.path.basename(p),
                label,
                f"{conf:.3f}",
                sev,
                adv,
                f"{vis_ms:.1f}",
                src,
                tok_s,
            ])
            f_log.flush()

            if sev in ("HIGH", "MEDIUM"):
                trigger_alert(label, sev, conf, f"| Trend: {history_summary}")
                print(f"ADVICE:\n{adv}\n")
            tok_str = f" | {tok_s} tok/s" if tok_s else ""
            print(f"[{idx+1}/{len(files)}] {os.path.basename(p)} -> {label} {conf:.2f} [{sev}] ({src}{tok_str}) {vis_ms:.1f}ms")

    stream_duration = time.perf_counter() - stream_start
    sustained_fps = len(files) / stream_duration if stream_duration > 0 else 0

    if is_stream:
        # Wait briefly for in-flight LLM advice to finish
        llm_queue.join()
        stop_worker.set()

    f_log.close()

    print(f"\n================================ SUMMARY ================================")
    print(f"Frames Processed: {len(files)}")
    print(f"Stream Duration: {stream_duration:.2f} s")
    print(f"Average Vision Latency: {sum(vision_latencies)/len(vision_latencies):.2f} ms (~{1000/(sum(vision_latencies)/len(vision_latencies)):.1f} FPS theoretical)")
    if is_stream:
        print(f"Stream Target: {args.stream:.1f} FPS | Sustained FPS: {sustained_fps:.2f} FPS")
        print(f"Dropped Frames: {dropped_frames}")
        print(f"LLM Calls Executed: {llm_call_count}")
        if advice_latencies:
            print(f"Average Advice Latency (Alert to Advice): {sum(advice_latencies)/len(advice_latencies):.2f} s")
            print(f"Advice Latency Range: {min(advice_latencies):.2f} s - {max(advice_latencies):.2f} s")
    else:
        print(f"Advisory Source: {sources['llm']} LLM ({sources['llm']/len(files)*100:.1f}%), {sources['fallback']} Fallback ({sources['fallback']/len(files)*100:.1f}%)")
    print(f"=========================================================================\n")


if __name__ == "__main__":
    main()
