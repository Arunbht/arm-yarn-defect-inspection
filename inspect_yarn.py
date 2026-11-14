import os, sys, csv, time, datetime, json, subprocess, atexit, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms, models

SEVERITY = {  # starting assumption, edit to suit your report
    "normal": "NONE", "irregularity": "MEDIUM", "thinness": "MEDIUM",
    "short segment lack": "MEDIUM", "knotting": "HIGH", "long segment lack": "HIGH"}

def crop_band(im):
    return im.crop((0, int(0.30 * im.height), im.width, int(0.70 * im.height)))

tf = transforms.Compose([
    transforms.Lambda(crop_band), transforms.Resize((96, 768)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ck = torch.load("yarn.pt")
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
    root = Path(__file__).resolve().parent
    python = root / "llmenv" / "bin" / "python"
    script = root / "llm_advice.py"
    if not python.exists() or not script.exists():
        return None
    try:
        process = subprocess.Popen(
            [str(python), str(script), "--serve"], cwd=root, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=1,
        )
        return process
    except OSError:
        return None

def stop_advisor():
    if ADVISOR and ADVISOR.poll() is None:
        ADVISOR.terminate()

def advise(label, severity, conf):
    if ADVISOR and ADVISOR.poll() is None:
        try:
            ADVISOR.stdin.write(json.dumps({"label": label, "severity": severity, "confidence": conf}) + "\n")
            ADVISOR.stdin.flush()
            response = json.loads(ADVISOR.stdout.readline())
            if response.get("ok"):
                return response["advice"]
        except (BrokenPipeError, json.JSONDecodeError, OSError):
            pass
    return fallback_advice(label, severity, conf)

def inspect(path):
    t = time.time()
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(m(x), 1)[0]
    conf, i = p.max(0)
    label, conf = classes[i], conf.item()
    sev = SEVERITY[label]
    return label, conf, sev, advise(label, sev, conf), (time.time() - t) * 1000

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python inspect_yarn.py "path/to/image_folder"')
    folder = sys.argv[1]
    ADVISOR = start_advisor()
    atexit.register(stop_advisor)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(folder)
                   for f in fs if f.lower().endswith(exts))
    if not files:
        sys.exit(f"No images found in {folder}")

    new = not os.path.exists("inspection_log.csv")
    times = []
    with open("inspection_log.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "file", "defect", "confidence",
                        "severity", "advice", "latency_ms"])
        for p in files:
            label, conf, sev, adv, ms = inspect(p)
            times.append(ms)
            w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                        os.path.basename(p), label, f"{conf:.3f}", sev, adv, f"{ms:.1f}"])
            print(f"{os.path.basename(p)} -> {label} {conf:.2f} [{sev}] {ms:.0f} ms")
    print(f"\n{len(times)} images, avg {sum(times)/len(times):.1f} ms/image")
