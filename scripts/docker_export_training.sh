#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${TRAINING_IMAGE:-emotional-companion-training:7b-cu124}"
ARCHIVE="${1:-$PROJECT_DIR/artifacts/docker/emotional-companion-training-7b-cu124.tar}"

[[ $# -le 1 ]] || {
  echo "Usage: $0 [ARCHIVE_PATH]" >&2
  exit 2
}
[[ ! -e "$ARCHIVE" ]] || {
  echo "ERROR: refusing to overwrite existing archive: $ARCHIVE" >&2
  exit 1
}
docker image inspect "$IMAGE" --format 'Exporting {{.Id}}'
mkdir -p -- "$(dirname -- "$ARCHIVE")"
docker image save --output "$ARCHIVE" "$IMAGE"
sha256sum -- "$ARCHIVE"
