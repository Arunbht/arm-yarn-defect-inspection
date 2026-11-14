import os, time, torch, torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

R = Path(__file__).resolve().parent / "Yarn Defect Dataset"
tf = transforms.Compose([
    transforms.Resize((128, 512)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

tr = datasets.ImageFolder(f"{R}/train", tf)
va = datasets.ImageFolder(f"{R}/val", tf)
te = datasets.ImageFolder(f"{R}/test", tf)
dl = DataLoader(tr, 32, shuffle=True, num_workers=2)
vl = DataLoader(va, 64)
tl = DataLoader(te, 64)

m = models.mobilenet_v3_small(weights="DEFAULT")
for p in m.features.parameters(): p.requires_grad = False
m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(tr.classes))
opt = torch.optim.Adam(m.classifier.parameters(), 1e-3)

def evaluate(loader):
    m.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += m(x).argmax(1).tolist(); ys += y.tolist()
    return ys, ps

for ep in range(6):
    m.train(); t = time.time()
    for x, y in dl:
        opt.zero_grad()
        nn.functional.cross_entropy(m(x), y).backward()
        opt.step()
    ys, ps = evaluate(vl)
    acc = sum(a == b for a, b in zip(ys, ps)) / len(ys)
    print(f"epoch {ep}  val_acc {acc:.3f}  ({time.time()-t:.0f}s)")

torch.save({"m": m.state_dict(), "classes": tr.classes}, "yarn.pt")

# final test report: run once, use these numbers in your report
ys, ps = evaluate(tl)
print("\nclasses:", tr.classes)
print(confusion_matrix(ys, ps))
print(classification_report(ys, ps, target_names=tr.classes))
print("model size (MB):", round(os.path.getsize("yarn.pt")/1e6, 2))
