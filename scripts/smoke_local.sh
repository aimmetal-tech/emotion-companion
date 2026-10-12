#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Builds a tiny random model locally. First invocation downloads Python/CUDA dependencies,
# not competition model weights. Choose --device cuda to require the real QLoRA path.
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.cache/uv}"
exec uv run --locked --extra training python -m emotional_companion smoke "$@"
