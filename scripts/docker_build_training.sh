#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${TRAINING_IMAGE:-emotional-companion-training:7b-cu124}"

# Optional docker build arguments are passed as separate quoted arguments.
# Build-time proxy values must be supplied explicitly; host localhost is not
# the build container's localhost. Proxy arguments do not become image ENV.
exec docker build \
  --platform linux/amd64 \
  --file "$PROJECT_DIR/Dockerfile.training" \
  --tag "$IMAGE" \
  --progress plain \
  "$@" \
  "$PROJECT_DIR"
