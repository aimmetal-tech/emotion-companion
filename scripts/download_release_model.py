"""Download a pinned public release model and verify every file without sending a token."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(path):
    source = json.loads(path.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", source["repo_id"]):
        raise ValueError("Invalid model repository ID")
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError("The model revision must be a pinned commit SHA")
    if not source["files"]:
        raise ValueError("Empty model file list")
    for name, checksum in source["files"].items():
        if Path(name).name != name or name in {".", ".."} or "\\" in name:
            raise ValueError("Model files must be plain filenames")
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"Invalid checksum: {name}")
    return source


def download(source_path, output, endpoint="https://huggingface.co", verify_only=False):
    if endpoint not in {"https://huggingface.co", "https://hf-mirror.com"}:
        raise ValueError("Use the official HF endpoint or the explicitly supported public mirror")
    source = load_source(source_path)
    if not verify_only:
        state_path = output / ".release-download.json"
        if (
            output.exists()
            and any(output.iterdir())
            and (not state_path.exists() or json.loads(state_path.read_text()) != source)
        ):
            raise ValueError("Nonempty output belongs to another download; choose a new directory")
        output.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=source["repo_id"],
            revision=source["revision"],
            local_dir=output,
            allow_patterns=list(source["files"]),
            token=False,
            endpoint=endpoint,
            max_workers=4,
        )
    for name, checksum in source["files"].items():
        if file_digest(output / name) != checksum:
            raise ValueError(f"Checksum mismatch: {name}")
        print(f"OK: {name}", flush=True)
    print(f"Verified standalone model: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "model_source.json")
    parser.add_argument("--output", type=Path, default=ROOT / "participant/model")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        download(args.source, args.output, args.endpoint, args.verify_only)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
