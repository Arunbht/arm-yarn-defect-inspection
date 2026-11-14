import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

dataset_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("YARN_DATASET_DIR")
if dataset_arg:
    R = Path(dataset_arg).resolve()
else:
    R = Path(__file__).resolve().parent / "Yarn Defect Dataset"

test_dir = R / "test"
if not test_dir.exists():
    sys.exit(f"Test dataset directory not found: {test_dir}. Provide path as argument or set YARN_DATASET_DIR.")

model_path = Path(__file__).resolve().parent / "yarn.pt"
if not model_path.exists():
    sys.exit(f"Model checkpoint not found: {model_path}")

def crop_band(im):
    return im.crop((0, int(0.30 * im.height), im.width, int(0.70 * im.height)))

tf = transforms.Compose([
    transforms.Lambda(crop_band),
    transforms.Resize((96, 768)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

te = datasets.ImageFolder(str(test_dir), tf)
ck = torch.load(model_path, map_location="cpu")
m = models.mobilenet_v3_small()
m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(ck["classes"]))
m.load_state_dict(ck["m"])
m.eval()

ys, ps = [], []
with torch.no_grad():
    for x, y in DataLoader(te, 64):
        ps += m(x).argmax(1).tolist()
        ys += y.tolist()

print(confusion_matrix(ys, ps))
print(classification_report(ys, ps, target_names=ck["classes"]))
