#!/usr/bin/env python3
"""
Raspberry Pi 5 Live Single-Yarn Inspection & Edge Advisory Service
------------------------------------------------------------------
Watches a target folder (default: sequence_30) for yarn frames, paces inspection
at 1 frame every X seconds, and immediately reacts when new frames are dropped,
copied, or overwritten.

Supports:
  - Formats: .jpg, .JPG, .jpeg, .JPEG, .bmp, .BMP, .png, .PNG, .webp, or bare filenames (no extension!)
  - Out-of-Distribution (OOD) Guard: instantly rejects non-yarn / cluttered photos (<25ms)
  - MobileNetV3-Small classifier: defect detection (<30ms on Cortex-A76 CPU)
  - Edge ExecuTorch Llama 3.2 1B: async operator advisory generation (ACTION / CHECK)
  - Zero duplicate cycles, zero skipped files, 24/7 continuous operation

Outputs broadcast to:
  - Terminal Console: Live ANSI color alert banners & advisory readout
  - latest_output.json: Atomic machine-readable state for external UI / web dashboards
  - latest_advice.txt: Plain text 2-liner for terminal / OLED display
  - inspection_log.csv: Persistent audit trail spreadsheet
"""

import os
import sys
import time
import json
import csv
import signal
import datetime
import argparse
import subprocess
import threading
import queue
from pathlib import Path
from collections import deque, Counter
try:
    from PIL import Image
    import numpy as np
except ImportError:
    # Allows --view-advisory monitor to run on any python without requiring numpy/PIL
    Image = None
    np = None

# Silence harmless PyTorch internal compilation and Enum deprecation warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

ROOT = Path(__file__).resolve().parent
DEFAULT_INTERVAL_X = 1.5   # Default seconds between frame inspections
HISTORY_WINDOW_SIZE = 20  # Number of past frames tracked for defect clustering
LLM_COOLDOWN_SEC = 6.0    # Seconds between repeated LLM calls for identical defects

SEVERITY = {
    "normal": "NONE",
    "irregularity": "MEDIUM",
    "thinness": "MEDIUM",
    "short segment lack": "MEDIUM",
    "knotting": "HIGH",
    "long segment lack": "HIGH",
}

ALERT_COLORS = {
    "HIGH": "\033[1;41;97m",       # Bold White on Red Background
    "MEDIUM": "\033[1;43;30m",     # Bold Black on Yellow Background
    "NORMAL": "\033[1;42;97m",     # Bold White on Green Background
    "INVALID": "\033[1;45;97m",    # Bold White on Magenta Background
    "INFO": "\033[1;36m",          # Cyan
    "BOLD": "\033[1m",
    "RESET": "\033[0m",
}


def is_image_file(p):
    """
    Format-agnostic image detector.
    Works for:
      - .jpg, .JPG, .jpeg, .JPEG, .bmp, .BMP, .png, .PNG, .webp
      - Bare filenames without any extension (e.g., 'test', 'capture', 'raw_frame')
    Safely ignores non-image text, logs, and temporary files.
    """
    # 1. Skip system files, hidden files, and known text files
    ignored_suffixes = {".txt", ".json", ".csv", ".tmp", ".py", ".sh", ".md", ".log", ".pyc"}
    if p.name.startswith(".") or p.suffix.lower() in ignored_suffixes:
        return False

    try:
        # Minimum size check (corrupt or 0-byte file write guard)
        if p.stat().st_size < 512:
            return False

        # Fast extension match (case-insensitive)
        known_exts = {".bmp", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".dib"}
        if p.suffix.lower() in known_exts:
            return True

        # Header Magic Bytes check (handles files with NO extension or non-standard naming)
        with open(p, "rb") as f:
            header = f.read(4)
            # b'BM' for BMP, b'\xff\xd8' for JPEG, b'\x89PNG' for PNG, b'RIFF' for WEBP
            if header.startswith(b"BM") or header.startswith(b"\xff\xd8") or header.startswith(b"\x89PNG") or header.startswith(b"RIFF"):
                return True

        # Fallback: test with PIL directly
        with Image.open(p) as img:
            return img.format in ("BMP", "JPEG", "PNG", "WEBP", "TIFF")
    except Exception:
        return False


def crop_band(im):
    """Crops the central 40% vertical band of the yarn frame."""
    w, h = im.size
    y0 = int(0.30 * h)
    y1 = max(y0 + 1, int(0.70 * h))
    return im.crop((0, y0, w, y1))


def validate_yarn_frame(im):
    """
    Out-of-Distribution (OOD) Guard:
    Verifies geometric and optical properties of a yarn inspection frame.
    Rejects natural scenes, room photos, or non-yarn images before CNN inference.
    """
    w, h = im.size
    if w < 32 or h < 32:
        return False, f"Image dimensions too small ({w}x{h})."

    ar = w / h
    # Rule 1: Camera sensor format is a horizontal wide strip (aspect ratio >= 1.3:1)
    if ar < 1.3:
        return False, f"Invalid aspect ratio ({w}x{h}, {ar:.2f}:1). Sensor requires horizontal yarn strip."

    # Rule 2: Central 40% band extraction
    band = crop_band(im)
    gray = np.array(band.convert("L"))
    bh, bw = gray.shape

    top_strip = gray[:max(2, int(0.12 * bh)), :]
    bot_strip = gray[-max(2, int(0.12 * bh)):, :]
    top_mean = float(top_strip.mean())
    bot_mean = float(bot_strip.mean())

    # Rule 3: Optical backlight illumination check
    if top_mean < 60 or bot_mean < 60 or top_mean > 254 or bot_mean > 254:
        return False, f"No optical yarn backlight detected (average={min(top_mean, bot_mean):.0f})."

    # Rule 4: Clean background check (reject natural/cluttered scenes)
    min_std = min(top_strip.std(), bot_strip.std())
    if min_std > 30:
        return False, f"Cluttered/natural scene detected (clutter index={min_std:.1f} > 30)."

    return True, "VALID_YARN_FRAME"


def load_transforms():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Lambda(crop_band),
        transforms.Resize((96, 768)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_vision_model(custom_path=None):
    model_path = Path(custom_path) if custom_path else (ROOT / "yarn.pt")
    if not model_path.exists():
        sys.exit(
            f"\n[!] Error: Vision checkpoint not found at: {model_path}\n"
            "    Please place 'yarn.pt' in the deployment directory.\n"
        )

    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        try:
            ck = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(model_path, map_location="cpu")

        classes = ck["classes"]
        model = models.mobilenet_v3_small()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
        model.load_state_dict(ck["m"])
        model.eval()

        # Warm-up pass to eliminate initial cold-start latency spike on Cortex-A76
        with torch.no_grad():
            _ = model(torch.zeros(1, 3, 96, 768))

        return model, classes
    except Exception as err:
        sys.exit(
            f"\n[!] Failed to load vision model: {err}\n"
            "    Ensure PyTorch is installed: source rpi_env/bin/activate\n"
        )


def fallback_advice(label, severity, conf):
    """Deterministic fallback rules if LLM is disabled or offline."""
    if severity == "NONE":
        return "ACTION: Continue normal winding.\nCHECK: Spindle tension nominal."
    if conf < 0.70:
        return "ACTION: Pause line and inspect manually.\nCHECK: Sensor alignment and lighting."
    if severity == "HIGH":
        return f"ACTION: Stop spindle immediately for {label}.\nCHECK: Clean knotter mechanism and guide paths."
    return f"ACTION: Mark yarn segment for {label}.\nCHECK: Inspect winding traveler speed and guide tension."


class LLMBridge:
    """Manages persistent ExecuTorch Llama daemon in background via JSON-RPC."""
    def __init__(self, enabled=True):
        self.process = None
        self.enabled = enabled
        if self.enabled:
            self._start_daemon()

    def _start_daemon(self):
        model_dir = ROOT / "llama_model"
        if not (model_dir / "model.pte").exists():
            print(f"{ALERT_COLORS['INFO']}[*] Notice: 'llama_model/model.pte' not found.")
            print(f"[*] Operating in fast vision mode with deterministic operator rules.{ALERT_COLORS['RESET']}")
            return

        candidates = [
            sys.executable,
            ROOT / "rpi_env" / "bin" / "python3",
            ROOT / "rpi_env" / "bin" / "python",
            ROOT / "llmenv" / "bin" / "python",
            ROOT / "env" / "bin" / "python"
        ]
        script = ROOT / "llm_advice.py"

        for cand in candidates:
            cand_p = Path(cand)
            if cand_p.exists() and script.exists():
                try:
                    p = subprocess.Popen(
                        [str(cand_p), str(script), "--serve"],
                        cwd=ROOT,
                        text=True,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=1,
                    )
                    # Read until we receive JSON ready handshake
                    ready_received = False
                    while p.poll() is None:
                        line = p.stdout.readline()
                        if not line:
                            break
                        line_str = line.strip()
                        if line_str.startswith("{"):
                            try:
                                data = json.loads(line_str)
                                if data.get("status") == "ready":
                                    self.process = p
                                    ready_received = True
                                    print(f"{ALERT_COLORS['INFO']}[✓] ExecuTorch Llama 3.2 1B Daemon Resident in RAM & Ready.{ALERT_COLORS['RESET']}")
                                    return
                            except Exception:
                                pass

                    if not ready_received and p.poll() is not None:
                        err = p.stderr.read()[:200] if p.stderr else ""
                        if err:
                            print(f"[!] Daemon startup note: {err.strip()}")
                except Exception:
                    continue

        print(f"{ALERT_COLORS['INFO']}[*] Notice: LLM daemon offline. Operating with deterministic safety rules.{ALERT_COLORS['RESET']}")

    def query(self, label, severity, conf, history_str):
        if self.process and self.process.poll() is None:
            try:
                payload = json.dumps({
                    "label": label,
                    "severity": severity,
                    "confidence": conf,
                    "history": history_str
                }) + "\n"
                self.process.stdin.write(payload)
                self.process.stdin.flush()

                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        break
                    line_str = line.strip()
                    if line_str.startswith("{"):
                        resp = json.loads(line_str)
                        if resp.get("ok"):
                            return resp["advice"], "llm", resp.get("tok_s", 0.0)
                        break
            except Exception:
                pass
        return fallback_advice(label, severity, conf), "fallback", 0.0

    def terminate(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass


class InspectionEngine:
    def __init__(self, watch_dir, interval_x=DEFAULT_INTERVAL_X, enable_llm=True, no_ood=False):
        if Image is None or np is None:
            sys.exit(
                "\n[!] Error: PIL and numpy are required for live vision inspection.\n"
                "    Please activate your environment:\n"
                "      source ../llmenv/bin/activate (on laptop)\n"
                "      source rpi_env/bin/activate   (on Pi)\n"
            )

        self.watch_dir = Path(watch_dir).resolve()
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.interval_x = interval_x
        self.no_ood = no_ood

        self.tf = load_transforms()
        self.vision_model, self.classes = load_vision_model()
        self.llm_bridge = LLMBridge(enabled=enable_llm)

        self.history_window = deque(maxlen=HISTORY_WINDOW_SIZE)
        self.last_defect_time = {}

        # Tracking state
        self.processed_names = set()
        self.processed_signatures = {}

        # Thread safety lock for disk writes
        self.io_lock = threading.Lock()

        # Output paths
        self.csv_path = ROOT / "inspection_log.csv"
        self.json_path = ROOT / "latest_output.json"
        self.txt_path = ROOT / "latest_advice.txt"

        self._init_csv()

        # Asynchronous LLM advisory worker
        self.advisory_queue = queue.Queue(maxsize=2)
        self.stop_worker = threading.Event()
        self.worker_thread = threading.Thread(target=self._llm_worker, daemon=True)
        self.worker_thread.start()

    def _init_csv(self):
        with self.io_lock:
            if not self.csv_path.exists():
                try:
                    with open(self.csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "timestamp", "file", "defect", "confidence", "severity",
                            "action", "check", "vision_latency_ms", "advice_source", "llm_tok_s"
                        ])
                        f.flush()
                except Exception as err:
                    print(f"Warning: Could not initialize CSV log: {err}")

    def _llm_worker(self):
        """Background thread executing Llama 3.2 1B advisories asynchronously."""
        while not self.stop_worker.is_set():
            try:
                task = self.advisory_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t_start = time.perf_counter()
            advice, source, tok_s = self.llm_bridge.query(
                task["defect"], task["severity"], task["confidence"], task["history"]
            )
            elapsed_s = time.perf_counter() - t_start

            lines = advice.strip().splitlines()
            action = lines[0].replace("ACTION:", "").strip() if len(lines) > 0 else advice
            check = lines[1].replace("CHECK:", "").strip() if len(lines) > 1 else ""

            # Thread-safe log to CSV
            with self.io_lock:
                try:
                    with open(self.csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.datetime.now().isoformat(timespec="seconds"),
                            task["filename"],
                            task["defect"],
                            f"{task['confidence']:.3f}",
                            task["severity"],
                            action,
                            check,
                            f"{task['vis_ms']:.1f}",
                            source,
                            f"{tok_s:.1f}" if tok_s else ""
                        ])
                        f.flush()
                except Exception as err:
                    print(f"Warning: Failed writing to CSV: {err}")

            # Thread-safe update latest_output.json & latest_advice.txt
            self._write_outputs(task["filename"], task["defect"], task["confidence"],
                               task["severity"], action, check, task["vis_ms"])

            # Print clean 1-line notice in Terminal 1 (Terminal 2 displays full card)
            speed_txt = f"{tok_s:.1f} tok/s" if tok_s else f"{elapsed_s:.1f}s"
            print(f"  ↳ {ALERT_COLORS['BOLD']}[Llama 3.2 1B Advisory: {task['filename']} ({task['defect'].upper()})]{ALERT_COLORS['RESET']} Action: {action} ({speed_txt})")

            self.advisory_queue.task_done()

    def _write_outputs(self, filename, defect, confidence, severity, action, check, vis_ms):
        result_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "file": filename,
            "defect": defect,
            "confidence": round(confidence, 4),
            "severity": severity,
            "action": action,
            "check": check,
            "vision_latency_ms": round(vis_ms, 2)
        }
        with self.io_lock:
            try:
                tmp_json = self.json_path.with_suffix(".tmp")
                with open(tmp_json, "w") as f:
                    json.dump(result_data, f, indent=2)
                    f.flush()
                tmp_json.replace(self.json_path)

                tmp_txt = self.txt_path.with_suffix(".tmp")
                with open(tmp_txt, "w") as f:
                    f.write(f"ACTION: {action}\nCHECK: {check}\n")
                    f.flush()
                tmp_txt.replace(self.txt_path)
            except Exception as err:
                print(f"Warning: Failed updating live status files: {err}")

    def get_history_summary(self):
        if not self.history_window:
            return "initial frame"
        total = len(self.history_window)
        defects = [x for x in self.history_window if x not in ("normal", "WRONG_IMAGE")]
        if not defects:
            return f"0 defects in last {total} yarns"
        dominant, count = Counter(defects).most_common(1)[0]
        return f"{len(defects)} defects in last {total} yarns ({dominant})"

    def process_image(self, img_path):
        """Processes a single frame with progressive retry against copy race conditions."""
        import torch

        t0 = time.perf_counter()
        img_p = Path(img_path).resolve()
        filename = img_p.name

        # Progressive backoff retry to handle in-progress file copying/writing
        im = None
        for delay in (0.10, 0.25, 0.50):
            try:
                with Image.open(img_p) as raw_im:
                    im = raw_im.convert("RGB")
                break
            except Exception:
                time.sleep(delay)

        if im is None:
            print(f"[!] Warning: Cannot decode image '{filename}'. File may be corrupt or still copying.")
            return False

        # Record file as processed
        self.processed_names.add(filename)
        try:
            st = img_p.stat()
            self.processed_signatures[filename] = (st.st_size, st.st_mtime)
        except Exception:
            pass

        # Out-of-Distribution Guard: Reject non-yarn / wrong images unless disabled
        if not self.no_ood:
            is_valid, validation_msg = validate_yarn_frame(im)
            if not is_valid:
                vis_ms = (time.perf_counter() - t0) * 1000
                banner = ALERT_COLORS["INVALID"]
                reset = ALERT_COLORS["RESET"]
                print(f"\n{banner} >>> [REJECTED - WRONG / NON-YARN IMAGE] <<< {reset}")
                print(f"{banner} FILE: {filename} -> {validation_msg} [Check: {vis_ms:.1f}ms] {reset}\n")
                action = "Ignored non-yarn image. Verify camera alignment."
                check = "Remove obstruction and ensure camera focuses on yarn line."
                self._write_outputs(filename, "WRONG_IMAGE", 0.0, "INVALID", action, check, vis_ms)
                with self.io_lock:
                    try:
                        with open(self.csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.datetime.now().isoformat(timespec="seconds"),
                                filename, "WRONG_IMAGE", "0.000", "INVALID", action, check,
                                f"{vis_ms:.1f}", "input_validator", ""
                            ])
                            f.flush()
                    except Exception:
                        pass
                return True

        # Vision Inference through MobileNetV3
        try:
            tensor = self.tf(im).unsqueeze(0)
            with torch.no_grad():
                logits = self.vision_model(tensor)
                probs = torch.softmax(logits, dim=1)[0]

            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            defect = self.classes[pred_idx]
            severity = SEVERITY.get(defect, "MEDIUM")
            vis_ms = (time.perf_counter() - t0) * 1000
        except Exception as err:
            print(f"[!] Error during vision inference on {filename}: {err}")
            return False

        history_summary = self.get_history_summary()
        self.history_window.append(defect)

        now = time.time()

        # Immediate Terminal Alert (<30ms from frame arrival)
        reset = ALERT_COLORS["RESET"]
        if severity == "HIGH":
            banner = ALERT_COLORS["HIGH"]
            print(f"\n\a{banner} >>> [CRITICAL ALERT - HIGH SEVERITY DEFECT] <<< {reset}")
            print(f"{banner} DEFECT: {defect.upper()} ({confidence*100:.1f}%) | {filename} | {history_summary} {reset}")
        elif severity == "MEDIUM":
            banner = ALERT_COLORS["MEDIUM"]
            print(f"\n{banner} >>> [WARNING - MEDIUM DEFECT] <<< {reset}")
            print(f"{banner} DEFECT: {defect.upper()} ({confidence*100:.1f}%) | {filename} | {history_summary} {reset}")
        else:
            banner = ALERT_COLORS["NORMAL"]
            print(f"{banner} [NORMAL] {filename} -> Yarn Nominal ({confidence*100:.1f}%) [Vision: {vis_ms:.1f}ms] {reset}")

        # Dispatch operator advisory
        if severity == "NONE":
            action = "Continue winding. Line nominal."
            check = "Spindle tension in optimal range."
            self._write_outputs(filename, defect, confidence, severity, action, check, vis_ms)
            with self.io_lock:
                try:
                    with open(self.csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.datetime.now().isoformat(timespec="seconds"),
                            filename, defect, f"{confidence:.3f}", severity, action, check,
                            f"{vis_ms:.1f}", "fast_path", ""
                        ])
                        f.flush()
                except Exception:
                    pass
        else:
            last_called = self.last_defect_time.get(defect, 0)
            if (now - last_called) >= LLM_COOLDOWN_SEC:
                task = {
                    "filename": filename,
                    "defect": defect,
                    "confidence": confidence,
                    "severity": severity,
                    "history": history_summary,
                    "vis_ms": vis_ms
                }
                try:
                    self.advisory_queue.put_nowait(task)
                    self.last_defect_time[defect] = now
                    print(f"  -> Dispatched to on-device Llama 3.2 1B advisory worker...")
                except queue.Full:
                    pass
        return True

    def find_uninspected_images(self):
        """
        Scans sequence_30 for images that have not been inspected or were updated.
        Uses format-agnostic detection (JPEG, BMP, PNG, or bare extensionless files).
        """
        uninspected = []

        try:
            for p in sorted(self.watch_dir.iterdir()):
                if not p.is_file():
                    continue

                # Content-based image verification (works for jpg, JPG, bmp, or bare filenames)
                if not is_image_file(p):
                    continue

                try:
                    st = p.stat()
                    current_sig = (st.st_size, st.st_mtime)
                except (OSError, FileNotFoundError):
                    continue

                # Condition 1: Never inspected
                if p.name not in self.processed_names:
                    uninspected.append(p)
                # Condition 2: File was overwritten or touched (different size/mtime)
                elif current_sig != self.processed_signatures.get(p.name):
                    uninspected.append(p)
        except Exception as err:
            print(f"Notice: Directory scan issue: {err}")

        return uninspected

    def run_live(self):
        """
        Continuously watches target folder:
        1. Inspects existing frames one by one (paced by interval X).
        2. Stays alive forever: immediately picks up any new drops, copies, or camera streams!
        """
        print(f"\n=======================================================")
        print(f"  Raspberry Pi 5 Single-Yarn Live Inspection Service   ")
        print(f"  Watching Directory : {self.watch_dir}                ")
        print(f"  Inspection Pacing  : 1 frame every X = {self.interval_x:.1f}s        ")
        print(f"  Supports Formats   : JPG, BMP, PNG, or bare files    ")
        print(f"  Drop any image into '{self.watch_dir.name}' at any time!      ")
        print(f"=======================================================\n")

        try:
            while True:
                uninspected = self.find_uninspected_images()

                if uninspected:
                    for img_p in uninspected:
                        t_start = time.perf_counter()
                        self.process_image(img_p)
                        elapsed = time.perf_counter() - t_start
                        sleep_time = max(0.0, self.interval_x - elapsed)
                        time.sleep(sleep_time)
                else:
                    time.sleep(self.interval_x)

        except KeyboardInterrupt:
            print("\nStopping inspection service...")
        finally:
            self.stop_worker.set()
            self.llm_bridge.terminate()
            print("Clean shutdown complete.")


def print_advisory_card(row):
    """Formats and prints an operator advisory card for Terminal 2."""
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


def run_advisory_monitor(csv_path):
    """Dedicated Terminal 2 viewer mode: monitors live operator advisory cards."""
    print(f"\n{ALERT_COLORS['BOLD']}╔══════════════════════════════════════════════════════════════════════╗{ALERT_COLORS['RESET']}")
    print(f"{ALERT_COLORS['BOLD']}║     ON-DEVICE GenAI OPERATOR ADVISORY MONITOR (TERMINAL 2)           ║{ALERT_COLORS['RESET']}")
    print(f"{ALERT_COLORS['BOLD']}║     Target: Raspberry Pi 5 | Model: Llama 3.2 1B (Quantized Edge)    ║{ALERT_COLORS['RESET']}")
    print(f"{ALERT_COLORS['BOLD']}╚══════════════════════════════════════════════════════════════════════╝{ALERT_COLORS['RESET']}")
    print(f"{ALERT_COLORS['INFO']}[*] Monitoring live advisory feed from: {csv_path.name}{ALERT_COLORS['RESET']}")
    print(f"\033[2m[*] Full terminal scrollback active. Press Ctrl+C to exit.\033[0m\n")

    while not csv_path.exists():
        time.sleep(0.5)

    processed_rows = 0
    try:
        with open(csv_path, "r", newline="") as f:
            reader = list(csv.reader(f))
            if len(reader) > 1:
                defect_rows = [r for r in reader[1:] if len(r) > 4 and r[4].upper() not in ("NONE", "NORMAL")]
                for r in defect_rows[-5:]:
                    print_advisory_card(r)
                processed_rows = len(reader)
    except Exception:
        pass

    try:
        while True:
            try:
                with open(csv_path, "r", newline="") as f:
                    reader = list(csv.reader(f))
                    if len(reader) > processed_rows:
                        new_rows = reader[processed_rows:]
                        for r in new_rows:
                            print_advisory_card(r)
                        processed_rows = len(reader)
            except Exception:
                pass
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n\033[2mAdvisory monitor stopped.\033[0m")


def main():
    def handle_sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handle_sigterm)

    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Single-Yarn Live Folder Watcher & Advisory Service")
    parser.add_argument("-x", "--interval", type=float, default=DEFAULT_INTERVAL_X,
                        help=f"Seconds between frame checks (Default: {DEFAULT_INTERVAL_X}s)")
    parser.add_argument("--dir", "--watch-dir", dest="watch_dir", type=str, default=None,
                        help="Target folder to watch (Default: sequence_30)")
    parser.add_argument("--view-advisory", "--advisory", action="store_true",
                        help="Terminal 2 Mode: View dedicated live GenAI operator advisory feed")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable Llama 3.2 1B daemon (runs purely on ultra-fast deterministic rules)")
    parser.add_argument("--no-ood", action="store_true",
                        help="Disable Out-of-Distribution rejection filter")
    args = parser.parse_args()

    # Terminal 2 Mode: Dedicated live advisory viewer
    if args.view_advisory:
        run_advisory_monitor(ROOT / "inspection_log.csv")
        return

    # Terminal 1 Mode: Live camera inspection
    if args.watch_dir:
        target_dir = Path(args.watch_dir).resolve()
    else:
        target_dir = (ROOT / "sequence_30").resolve()

    engine = InspectionEngine(
        watch_dir=target_dir,
        interval_x=args.interval,
        enable_llm=not args.no_llm,
        no_ood=args.no_ood
    )
    engine.run_live()


if __name__ == "__main__":
    main()
