import os, time, torch, torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

RUN_TEST = False   # set True only for the final run
R = Path(__file__).resolve().parent / "Yarn Defect Dataset"

def crop_band(im):  # keep the central 40% of height, where the yarn sits
    return im.crop((0, int(0.30 * im.height), im.width, int(0.70 * im.height)))

norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
tf_tr = transforms.Compose([transforms.Lambda(crop_band), transforms.Resize((96, 768)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
tf_ev = transforms.Compose([transforms.Lambda(crop_band), transforms.Resize((96, 768)),
        transforms.ToTensor(), norm])

tr = datasets.ImageFolder(f"{R}/train", tf_tr)
va = datasets.ImageFolder(f"{R}/val", tf_ev)
te = datasets.ImageFolder(f"{R}/test", tf_ev)
dl = DataLoader(tr, 32, shuffle=True, num_workers=2)
vl = DataLoader(va, 64); tl = DataLoader(te, 64)

m = models.mobilenet_v3_small(weights="DEFAULT")
m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(tr.classes))
for p in m.features.parameters(): p.requires_grad = False
for p in m.features[9:].parameters(): p.requires_grad = True   # fine-tune last blocks

EPOCHS = 10
opt = torch.optim.Adam([
    {"params": m.features[9:].parameters(), "lr": 3e-4},
    {"params": m.classifier.parameters(), "lr": 1e-3}])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

def set_train():
    m.eval()                      # frozen BN everywhere...
    m.features[9:].train()        # ...except the blocks being fine-tuned
    m.classifier.train()

def evaluate(loader):
    m.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += m(x).argmax(1).tolist(); ys += y.tolist()
    return ys, ps

best = 0
for ep in range(EPOCHS):
    set_train(); t = time.time()
    for x, y in dl:
        opt.zero_grad()
        nn.functional.cross_entropy(m(x), y).backward()
        opt.step()
    sched.step()
    ys, ps = evaluate(vl)
    acc = sum(a == b for a, b in zip(ys, ps)) / len(ys)
    print(f"epoch {ep}  val_acc {acc:.3f}  ({time.time()-t:.0f}s)")
    if acc > best:
        best = acc
        torch.save({"m": m.state_dict(), "classes": tr.classes}, "yarn.pt")

ck = torch.load("yarn.pt"); m.load_state_dict(ck["m"])
ys, ps = evaluate(vl)
print("\nBEST VAL", best, "\nclasses:", tr.classes)
print(confusion_matrix(ys, ps))
if RUN_TEST:
    ys, ps = evaluate(tl)
    print("\nTEST"); print(confusion_matrix(ys, ps))
    print(classification_report(ys, ps, target_names=tr.classes))
print("model size (MB):", round(os.path.getsize("yarn.pt")/1e6, 2))
