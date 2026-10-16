from __future__ import annotations

import contextlib
import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import download_release_model as downloader
from scripts import package_release as packager

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ReleaseTests(unittest.TestCase):
    def fixture(self, root):
        official = root / "official_dl/participant"
        write(official / "start.sh", "#!/bin/bash\nexit 0\n")
        write(official / "run_inference.py", "# unchanged official fixture\n")
        write(official / "configs/inference_config.json", '{"backend":"transformers"}')
        write(official / "requirements.txt", "torch>=2.1\n")
        write(official / "__pycache__/ignored.pyc", "cache")
        write(official / "model/ignored.bin", "not a release model")
        write(root / "src/emotional_companion/__init__.py", "# source\n")
        write(root / "official_dl/test_inference_data.jsonl", "{}\n")
        write(root / "artifacts/data/val_quick.input.jsonl", "{}\n")
        write(root / "docs/releases/v0.0.1/README.md", "# deployment\n")
        for name in ("download_release_model.py", "install_missing_requirements.py"):
            (root / "scripts").mkdir(exist_ok=True)
            shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
        for name in (
            "remote-dl/runs/train/training_report.json",
            "remote-dl/runs/baseline-quick/metrics.json",
            "remote-dl/runs/candidate-quick/metrics.json",
            "artifacts/release-checks/v0.0.1.json",
        ):
            write(root / name, "{}")
        model = root / "model"
        for name in ("config.json", "tokenizer.json", "LICENSE", "README.md", "model-1.safetensors"):
            write(model / name, "fixture")
        write(model / "model.safetensors.index.json", '{"weight_map":{"weight":"model-1.safetensors"}}')
        write(
            model / "SHA256SUMS",
            "".join(f"{packager.file_digest(path)}  {path.name}\n" for path in sorted(model.iterdir())),
        )
        return model

    def test_package_preserves_official_entries_excludes_weights_and_pins_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self.fixture(root)
            with (
                patch.object(packager.subprocess, "check_output", return_value=REVISION),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                archive = packager.package(model, REVISION, root / "release", root=root)
            staged = root / "release/emotion-companion-v0.0.1"
            self.assertEqual(
                (staged / "participant/start.sh").read_bytes(), (root / "official_dl/participant/start.sh").read_bytes()
            )
            config = json.loads((staged / "participant/configs/inference_config.json").read_text())
            self.assertEqual(config["backend"], "openai_compatible")
            source = downloader.load_source(staged / "model_source.json")
            self.assertEqual(source["revision"], REVISION)
            self.assertEqual(
                source["files"]["model-1.safetensors"], packager.file_digest(model / "model-1.safetensors")
            )
            for line in (staged / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(packager.file_digest(staged / name), digest)
            with tarfile.open(archive) as handle:
                names = handle.getnames()
                self.assertFalse(any("__pycache__" in name or name.endswith(".safetensors") for name in names))
                self.assertIn("emotion-companion-v0.0.1/participant/start.sh", names)
            checksum = archive.with_name(archive.name + ".sha256").read_text().split()[0]
            self.assertEqual(packager.file_digest(archive), checksum)

    def test_refuses_existing_output_and_mutable_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                packager.package(root, REVISION, root)
            with self.assertRaises(ValueError):
                packager.package(root, "main", root / "output")

    def source(self, root):
        source = {
            "repo_id": "owner/model",
            "revision": REVISION,
            "files": {"config.json": downloader.file_digest(root / "config.json")},
        }
        path = root / "model_source.json"
        write(path, json.dumps(source))
        return path

    def test_downloader_verifies_files_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "config.json", "original")
            source = self.source(root)
            with contextlib.redirect_stdout(io.StringIO()):
                downloader.download(source, root, verify_only=True)
            write(root / "config.json", "changed")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                downloader.download(source, root, verify_only=True)

    def test_downloader_uses_pinned_revision_and_never_sends_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "config.json", "original")
            source = self.source(root)
            output = root / "download"

            def snapshot(**kwargs):
                shutil.copy2(root / "config.json", kwargs["local_dir"] / "config.json")

            with (
                patch("huggingface_hub.snapshot_download", side_effect=snapshot) as call,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                downloader.download(source, output, "https://hf-mirror.com")
                downloader.download(source, output, "https://hf-mirror.com")
            self.assertIs(call.call_args.kwargs["token"], False)
            self.assertEqual(call.call_args.kwargs["revision"], REVISION)
            self.assertEqual(call.call_args.kwargs["allow_patterns"], ["config.json"])
            self.assertEqual(call.call_count, 2)

    def test_downloader_refuses_other_nonempty_directory_and_unsafe_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "config.json", "original")
            source = self.source(root)
            with self.assertRaisesRegex(ValueError, "Nonempty output"):
                downloader.download(source, root)
            data = json.loads(source.read_text())
            data["files"] = {"../outside": "b" * 64}
            write(source, json.dumps(data))
            with self.assertRaisesRegex(ValueError, "plain filenames"):
                downloader.load_source(source)


if __name__ == "__main__":
    unittest.main()
