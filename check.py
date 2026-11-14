import os
from pathlib import Path
R = Path(__file__).resolve().parent / "Yarn Defect Dataset"
names = {}
for s in ["train", "val", "test"]:
    names[s] = sorted(os.listdir(f"{R}/{s}"))
    print(s)
    for c in names[s]:
        print(f"  {c}: {len(os.listdir(f'{R}/{s}/{c}'))}")
print("class names match across splits:", names["train"] == names["val"] == names["test"])
