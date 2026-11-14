import os, torch, torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

R = Path(__file__).resolve().parent / "Yarn Defect Dataset"
def crop_band(im): return im.crop((0, int(0.30*im.height), im.width, int(0.70*im.height)))
tf = transforms.Compose([transforms.Lambda(crop_band), transforms.Resize((96, 768)),
     transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
te = datasets.ImageFolder(f"{R}/test", tf)
ck = torch.load("yarn.pt")
m = models.mobilenet_v3_small(); m.classifier[3] = nn.Linear(m.classifier[3].in_features, 6)
m.load_state_dict(ck["m"]); m.eval()
ys, ps = [], []
with torch.no_grad():
    for x, y in DataLoader(te, 64):
        ps += m(x).argmax(1).tolist(); ys += y.tolist()
print(confusion_matrix(ys, ps))
print(classification_report(ys, ps, target_names=ck["classes"]))
