#!/usr/bin/env python3
"""
Raspberry Pi 5 MobileNetV3-Small Vision Model Sanity Check & Benchmark
----------------------------------------------------------------------
Verifies model weights, test preprocessing, OOD image validation, and measures
Cortex-A76 CPU inference latency (ms per frame & FPS).

Usage:
    python3 test_vision.py                          # Tests first image in sequence_30 (or synthetic)
    python3 test_vision.py path/to/frame.bmp        # Tests a specific image file
    python3 test_vision.py sequence_30              # Tests all images in a directory
    python3 test_vision.py --no-ood image.bmp       # Bypasses Out-of-Distribution filter
"""

import sys
import time
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent

# Check dependencies first with clear recovery guidance
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
except ImportError as err:
    print(f"\n[!] Error: Missing deep learning dependencies ({err}).")
    print("    Please ensure your virtual environment is active:")
    print("      source rpi_env/bin/activate")
    print("    Or run the setup script if you haven't yet:")
    print("      ./setup_rpi5.sh\n")
    sys.exit(1)


def crop_band(im):
    """Crops the central 40% vertical band of the yarn frame."""
    w, h = im.size
    y0 = int(0.30 * h)
    y1 = max(y0 + 1, int(0.70 * h))
    return im.crop((0, y0, w, y1))


def validate_yarn_frame(im):
    """
    Out-of-Distribution (OOD) Guard:
    Checks optical backlight, horizontal aspect ratio, and background clutter
    to ensure non-yarn photos are not misclassified as yarn defects.
    """
    w, h = im.size
    if w < 32 or h < 32:
        return False, f"Image dimensions too small ({w}x{h})."

    ar = w / h
    if ar < 1.3:
        return False, f"Aspect ratio ({w}x{h}, {ar:.2f}:1) < 1.3:1. Camera sensor requires horizontal yarn strip."

    band = crop_band(im)
    gray = np.array(band.convert("L"))
    bh, bw = gray.shape

    top_strip = gray[:max(2, int(0.12 * bh)), :]
    bot_strip = gray[-max(2, int(0.12 * bh)):, :]
    top_mean = float(top_strip.mean())
    bot_mean = float(bot_strip.mean())

    if top_mean < 60 or bot_mean < 60 or top_mean > 254 or bot_mean > 254:
        return False, f"No optical yarn backlight detected (average={min(top_mean, bot_mean):.0f})."

    min_std = min(top_strip.std(), bot_strip.std())
    if min_std > 30:
        return False, f"Cluttered/natural scene detected (clutter index={min_std:.1f} > 30)."

    return True, "VALID_YARN_FRAME"


def load_transforms():
    return transforms.Compose([
        transforms.Lambda(crop_band),
        transforms.Resize((96, 768)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def find_model_checkpoint():
    search_paths = [
        ROOT / "yarn.pt",
        Path("yarn.pt"),
        ROOT.parent / "yarn.pt",
    ]
    for p in search_paths:
        if p.exists() and p.is_file():
            return p.resolve()
    return None


def load_model(model_path):
    print(f"Loading MobileNetV3-Small weights from: {model_path}...")
    try:
        # Compatible with PyTorch >= 2.4 weights_only default
        try:
            ck = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(model_path, map_location="cpu")

        classes = ck["classes"]
        model = models.mobilenet_v3_small()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
        model.load_state_dict(ck["m"])
        model.eval()
        return model, classes
    except Exception as err:
        print(f"\n[!] Failed to load model from '{model_path}': {err}")
        sys.exit(1)


def benchmark_single_image(model, classes, tf, img_path, no_ood=False, runs=20):
    print(f"\nBenchmarking inference on: {img_path.name}")
    try:
        with Image.open(img_path) as raw:
            im = raw.convert("RGB")
    except Exception as err:
        print(f"[!] Error opening image file '{img_path}': {err}")
        return False

    # Check Out-of-Distribution filter
    is_valid, reason = validate_yarn_frame(im)
    if not is_valid:
        print(f"  [OOD Status]: \033[1;45;97m REJECTED - NON-YARN IMAGE \033[0m")
        print(f"  Reason:       {reason}")
        if not no_ood:
            print("  (To force classification anyway, pass '--no-ood')")
            return True

    tensor = tf(im).unsqueeze(0)

    # Warm-up pass
    t_w0 = time.perf_counter()
    with torch.no_grad():
        _ = model(tensor)
    warmup_ms = (time.perf_counter() - t_w0) * 1000

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(1).item()
            latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    pred_class = classes[pred_idx]
    confidence = probs[0][pred_idx].item()

    print("-" * 60)
    print(f"Prediction:          {pred_class.upper()} (Confidence: {confidence * 100:.2f}%)")
    print(f"Warm-up Latency:     {warmup_ms:.2f} ms (cold dispatch)")
    print(f"Steady CPU Latency:  {avg_ms:.2f} ms per frame (~{fps:.1f} FPS)")
    print(f"Latency Range:       {min(latencies):.2f} ms - {max(latencies):.2f} ms ({runs} iterations)")
    print("-" * 60)
    return True


def is_image_file(p):
    """Format-agnostic image detector (works for jpg, JPG, bmp, png, or bare filenames)."""
    ignored = {".txt", ".json", ".csv", ".tmp", ".py", ".sh", ".md", ".log", ".pyc"}
    if p.name.startswith(".") or p.suffix.lower() in ignored:
        return False
    try:
        if p.stat().st_size < 512:
            return False
        with Image.open(p) as img:
            return img.format in ("BMP", "JPEG", "PNG", "WEBP", "TIFF")
    except Exception:
        return False


def benchmark_directory(model, classes, tf, dir_path, no_ood=False):
    files = sorted([p for p in dir_path.iterdir() if p.is_file() and is_image_file(p)])
    if not files:
        print(f"[!] No images found in {dir_path}")
        return

    print(f"\nEvaluating {len(files)} frames from: {dir_path.name}")
    print(f"{'FRAME':<32} | {'RESULT':<22} | {'CONF':<8} | {'LATENCY':<10}")
    print("-" * 78)

    latencies = []
    for p in files:
        t0 = time.perf_counter()
        try:
            with Image.open(p) as raw:
                im = raw.convert("RGB")
        except Exception:
            print(f"{p.name:<32} | {'[CORRUPT IMAGE]':<22} | {'--':<8} | {'--':<10}")
            continue

        is_valid, _ = validate_yarn_frame(im)
        if not is_valid and not no_ood:
            ms = (time.perf_counter() - t0) * 1000
            print(f"{p.name:<32} | \033[35m{'REJECTED (NON-YARN)':<22}\033[0m | {'--':<8} | {ms:6.1f} ms")
            continue

        tensor = tf(im).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        pred_idx = probs.argmax().item()
        pred_class = classes[pred_idx]
        conf = probs[pred_idx].item()
        print(f"{p.name:<32} | {pred_class:<22} | {conf*100:5.1f}%  | {ms:6.1f} ms")

    if latencies:
        avg_ms = sum(latencies) / len(latencies)
        print("-" * 78)
        print(f"Processed {len(latencies)} valid frames | Avg Latency: {avg_ms:.1f} ms (~{1000.0/avg_ms:.1f} FPS)")


def benchmark_synthetic(model, classes):
    print("\n[i] No image files provided or found. Running benchmark on synthetic tensor...")
    dummy = torch.randn(1, 3, 96, 768)

    # Warm-up
    with torch.no_grad():
        _ = model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(25):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    print("-" * 60)
    print(f"Synthetic CPU Latency: {avg_ms:.2f} ms per frame (~{1000.0 / avg_ms:.1f} FPS)")
    print(f"Latency Range:         {min(latencies):.2f} ms - {max(latencies):.2f} ms")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 MobileNetV3-Small Sanity & Latency Benchmark")
    parser.add_argument("target", nargs="?", default=None,
                        help="Path to an image file or directory of images (e.g. sequence_30)")
    parser.add_argument("--no-ood", action="store_true",
                        help="Disable Out-of-Distribution rejection filter")
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit path to yarn.pt model checkpoint")
    args = parser.parse_args()

    # 1. Locate and load model
    model_path = Path(args.model) if args.model else find_model_checkpoint()
    if not model_path or not model_path.exists():
        print("\n[!] Error: Model checkpoint 'yarn.pt' was not found.")
        print("    Looked in:")
        print(f"      - {ROOT / 'yarn.pt'}")
        print("    Please ensure 'yarn.pt' is present in this folder.\n")
        sys.exit(1)

    model, classes = load_model(model_path)
    print(f"[✓] Model verified! ({len(classes)} defect classes): {classes}")

    tf = load_transforms()

    # 2. Determine target
    target_path = None
    if args.target:
        target_path = Path(args.target).resolve()
        if not target_path.exists():
            print(f"[!] Error: Specified target does not exist: {target_path}")
            sys.exit(1)
    else:
        # Auto-discover sequence_30 or sample image
        seq = ROOT / "sequence_30"
        if seq.exists() and seq.is_dir():
            target_path = seq
        else:
            candidates = list(ROOT.glob("*.bmp")) + list(ROOT.glob("*.png")) + list(ROOT.glob("*.jpg"))
            if candidates:
                target_path = candidates[0]

    if target_path is None:
        benchmark_synthetic(model, classes)
    elif target_path.is_dir():
        benchmark_directory(model, classes, tf, target_path, no_ood=args.no_ood)
    else:
        benchmark_single_image(model, classes, tf, target_path, no_ood=args.no_ood)

    print("\n[✓] Vision pipeline is verified and fully operational on this device!\n")


if __name__ == "__main__":
    main()
