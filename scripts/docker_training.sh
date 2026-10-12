#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${TRAINING_IMAGE:-emotional-companion-training:7b-cu124}"
OUTPUT_ROOT="${TRAINING_WORKSPACE:-$PROJECT_DIR}"
DATA_DIR="${TRAINING_DATA_DIR:-$PROJECT_DIR/official_dl/train-val-dataset}"
CPU=0

if [[ "${1:-}" == "--cpu" ]]; then
  CPU=1
  shift
fi
if [[ $# -eq 0 ]]; then
  set -- --help
fi

OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")"
DATA_DIR="$(realpath -m -- "$DATA_DIR")"
[[ -d "$DATA_DIR" ]] || {
  echo "ERROR: dataset directory not found: $DATA_DIR" >&2
  exit 1
}

# Download is the only default command requiring network. Other commands run
# offline, including smoke tests. Explicit environment overrides are supported.
if [[ "$1" == "download" ]]; then
  NETWORK="${TRAINING_NETWORK:-bridge}"
  OFFLINE="${TRAINING_OFFLINE:-0}"
else
  NETWORK="${TRAINING_NETWORK:-none}"
  OFFLINE="${TRAINING_OFFLINE:-1}"
fi

COMMAND=(docker run --rm --init --stop-timeout 60 --shm-size 2g
  --user "$(id -u):$(id -g)"
  --workdir /workspace
  --network "$NETWORK"
  --env "HOME=/workspace/.cache"
  --env "HF_HUB_OFFLINE=$OFFLINE"
  --env "TRANSFORMERS_OFFLINE=$OFFLINE")

if ((CPU == 0)); then
  COMMAND+=(--gpus "${TRAINING_GPUS:-device=0}")
fi
for name in artifacts models runs; do
  COMMAND+=(--mount "type=bind,src=$OUTPUT_ROOT/$name,dst=/workspace/$name")
done
COMMAND+=(--mount "type=bind,src=$OUTPUT_ROOT/.cache/docker-training,dst=/workspace/.cache"
  --mount "type=bind,src=$DATA_DIR,dst=/workspace/official_dl/train-val-dataset,readonly")

# Pass Hub settings at runtime so mirrors do not need an image rebuild. Passing
# names rather than values also keeps optional tokens out of command-line values.
for name in HF_TOKEN HF_ENDPOINT HF_HUB_DISABLE_XET HF_HUB_DISABLE_IMPLICIT_TOKEN \
            HF_HUB_ETAG_TIMEOUT HF_HUB_DOWNLOAD_TIMEOUT; do
  if [[ -n "${!name:-}" ]]; then
    COMMAND+=(--env "$name")
  fi
done
COMMAND+=("$IMAGE" "$@")

if [[ "${TRAINING_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p -- "$OUTPUT_ROOT/artifacts" "$OUTPUT_ROOT/models" "$OUTPUT_ROOT/runs" \
  "$OUTPUT_ROOT/.cache/docker-training"
exec "${COMMAND[@]}"
