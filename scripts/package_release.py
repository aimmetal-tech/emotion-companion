"""Package the unmodified official inference entry with pinned HF model metadata, not weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def package(model, revision, output, version="v0.0.1", repo_id="aimmetal-tech/emotion-companion-7b", root=ROOT):
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Require a semantic release tag and a pinned HF commit SHA")
    if output.exists():
        raise ValueError(f"Refusing to overwrite release output: {output}")
    # SHA256SUMS was generated after full exported-weight verification. Exclude upload cache/state.
    checksums = {}
    for line in (model / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        if Path(name).name != name or "\\" in name or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("Invalid model checksum file")
        if not (model / name).is_file():
            raise ValueError(f"Missing model file: {name}")
        checksums[name] = checksum
    if (
        not {"config.json", "tokenizer.json", "model.safetensors.index.json", "LICENSE", "README.md"}
        <= checksums.keys()
    ):
        raise ValueError("Expected a standalone model including tokenizer, index, license and model card")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()) <= checksums.keys():
        raise ValueError("Weight shards missing from checksum manifest")
    checksums["SHA256SUMS"] = file_digest(model / "SHA256SUMS")
    name = f"emotion-companion-{version}"
    output.mkdir(parents=True)
    staging = output / name
    official = root / "official_dl/participant"
    shutil.copytree(official, staging / "participant", ignore=shutil.ignore_patterns("__pycache__", "model", ".cache"))
    shutil.copytree(
        root / "src/emotional_companion",
        staging / "src/emotional_companion",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (staging / "tools").mkdir()
    for script in ("download_release_model.py", "install_missing_requirements.py"):
        shutil.copy2(root / "scripts" / script, staging / "tools" / script)
    config_path = staging / "participant/configs/inference_config.json"
    config = json.loads(config_path.read_text())
    config.update(backend="openai_compatible", model_path="./model", api_model="model", hardware_label="2xRTX4090-24GB")
    write_json(config_path, config)
    (staging / "participant/start.sh").chmod(0o755)
    source = {"version": version, "repo_id": repo_id, "revision": revision, "files": checksums}
    write_json(staging / "model_source.json", source)
    shutil.copy2(root / "official_dl/test_inference_data.jsonl", staging / "test_inference_data.jsonl")
    shutil.copy2(root / "artifacts/data/val_quick.input.jsonl", staging / "validation_128.input.jsonl")
    shutil.copy2(root / "docs/releases" / version / "README.md", staging / "DEPLOY.md")
    shutil.copy2(model / "LICENSE", staging / "LICENSE.model")
    (staging / "reports").mkdir()
    reports = {
        "training_report.json": root / "remote-dl/runs/train/training_report.json",
        "baseline_metrics.json": root / "remote-dl/runs/baseline-quick/metrics.json",
        "candidate_metrics.json": root / "remote-dl/runs/candidate-quick/metrics.json",
        "local_export_checks.json": root / "artifacts/release-checks" / f"{version}.json",
    }
    for filename, path in reports.items():
        shutil.copy2(path, staging / "reports" / filename)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    for filename in ("start.sh", "run_inference.py"):
        if file_digest(staging / "participant" / filename) != file_digest(official / filename):
            raise ValueError(f"Official entry changed unexpectedly: {filename}")
    write_json(
        staging / "release_manifest.json",
        {
            "version": version,
            "source_commit": source_commit,
            "model_repo": repo_id,
            "model_revision": revision,
            "platform_entry": "/root/participant/start.sh",
            "weights_in_archive": False,
            "default_backend": "openai_compatible",
            "default_tensor_parallel_size": 2,
            "official_start_sha256": file_digest(official / "start.sh"),
            "official_runner_sha256": file_digest(official / "run_inference.py"),
            "platform_gpu_validation": "pending; required before competition upload",
        },
    )
    files = [path for path in sorted(staging.rglob("*")) if path.is_file()]
    (staging / "SHA256SUMS").write_text(
        "".join(f"{file_digest(path)}  {path.relative_to(staging).as_posix()}\n" for path in files), encoding="utf-8"
    )
    archive = output / f"{name}-deploy.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as handle:
        handle.add(staging, arcname=name)
    (output / f"{archive.name}.sha256").write_text(f"{file_digest(archive)}  {archive.name}\n", encoding="utf-8")
    print(f"Deployment archive: {archive}")
    print(f"Model: https://huggingface.co/{repo_id}/tree/{revision}")
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/companion-7b-v0.0.1")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-repo", default="aimmetal-tech/emotion-companion-7b")
    parser.add_argument("--version", default="v0.0.1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output if args.output else ROOT / "artifacts/releases" / args.version
    try:
        package(args.model, args.model_revision, output, args.version, args.model_repo)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
