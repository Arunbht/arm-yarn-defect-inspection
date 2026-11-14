#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${1:-$ROOT/llmenv}"

echo "=== Creating unified environment at $VENV_DIR ==="
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip

echo "=== Installing PyTorch and Dependencies ==="
"$VENV_DIR/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$VENV_DIR/bin/python" -m pip install scikit-learn Pillow huggingface_hub "accelerate>=0.26.0" "optimum~=2.0.0" "transformers==5.0.0rc1" "tokenizers==0.22.2"

echo "=== Installing Optimum-ExecuTorch ==="
if [ ! -d /tmp/optimum-executorch/.git ]; then
  git clone --depth 1 https://github.com/huggingface/optimum-executorch.git /tmp/optimum-executorch
fi
"$VENV_DIR/bin/python" -m pip install --no-deps /tmp/optimum-executorch

echo "=== Verifying Installation ==="
"$VENV_DIR/bin/python" -c "import torch, torchvision, optimum.executorch; print('Environment Ready. PyTorch:', torch.__version__, '| Torchvision:', torchvision.__version__)"
