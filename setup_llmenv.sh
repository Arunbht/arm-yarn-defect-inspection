#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
python3 -m venv "$ROOT/llmenv"
"$ROOT/llmenv/bin/python" -m pip install --upgrade pip
if [ ! -d /tmp/optimum-executorch/.git ]; then
  git clone --depth 1 https://github.com/huggingface/optimum-executorch.git /tmp/optimum-executorch
fi
"$ROOT/llmenv/bin/python" -m pip install /tmp/optimum-executorch
"$ROOT/llmenv/bin/optimum-cli" export executorch --help
