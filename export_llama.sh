#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/llmenv/bin/optimum-cli" export executorch \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --task text-generation \
  --recipe xnnpack \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --max_seq_len 512 \
  --dtype bfloat16 \
  --qlinear 8da4w \
  --qembedding 8w \
  --output_dir "$ROOT/llama_model"
