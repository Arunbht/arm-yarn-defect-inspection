import os
import sys
from pathlib import Path

dataset_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("YARN_DATASET_DIR")
if dataset_arg:
    R = Path(dataset_arg).resolve()
else:
    R = Path(__file__).resolve().parent / "Yarn Defect Dataset"

if not R.exists():
    sys.exit(f"Dataset directory not found: {R}. Provide path as argument or set YARN_DATASET_DIR.")

names = {}
for s in ["train", "val", "test"]:
    split_dir = R / s
    if not split_dir.exists():
        sys.exit(f"Split directory not found: {split_dir}")
    names[s] = sorted(os.listdir(split_dir))
    print(s)
    for c in names[s]:
        print(f"  {c}: {len(os.listdir(split_dir / c))}")

match = names["train"] == names["val"] == names["test"]
print("class names match across splits:", match)
