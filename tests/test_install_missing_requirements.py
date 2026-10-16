from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import install_missing_requirements as installer

ROOT = Path(__file__).resolve().parents[1]


class InstallMissingTests(unittest.TestCase):
    def requirements(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text(text, encoding="utf-8")
            return installer.load_requirements(path)

    def test_normalized_names_and_different_versions_are_skipped(self):
        requirements = self.requirements("Torch==2.6.0\nhuggingface_hub==0.36.2\npeft==0.18.1\n")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            missing = installer.plan_install(requirements, {"torch": "2.10.0+cu128", "huggingface-hub": "0.36.2"})
        self.assertEqual(missing, ["peft==0.18.1"])
        self.assertIn("kept unchanged", output.getvalue())

    def test_markers_and_comments(self):
        requirements = self.requirements(
            '# comment\n\npeft==0.18.1 # inline comment\ncolorama==0.4.6; sys_platform == "not-a-real-platform"\n'
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(installer.plan_install(requirements, {}), ["peft==0.18.1"])

    def test_invalid_or_external_requirements_rejected(self):
        for requirement in (
            "-r extra.txt",
            "--index-url https://example.org",
            "-e .",
            "pkg @ https://example.org/a.whl",
        ):
            with self.subTest(requirement=requirement), self.assertRaises(ValueError):
                self.requirements(requirement)

    def test_actual_requirements_parse(self):
        self.assertGreater(len(installer.load_requirements(ROOT / "requirements.txt")), 30)

    def invoke(self, arguments, installed):
        with (
            patch.object(installer, "installed_packages", return_value=installed),
            patch.object(installer.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = installer.main(arguments)
        return code, run

    def test_dry_run_never_invokes_pip(self):
        code, run = self.invoke(["--dry-run"], {})
        self.assertEqual(code, 0)
        run.assert_not_called()

    def test_only_missing_installed_without_dependencies_then_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("torch==2.6.0\npeft==0.18.1\n", encoding="utf-8")
            with (
                patch.object(installer, "installed_packages", return_value={"torch": "2.10.0"}),
                patch.object(installer.subprocess, "run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run.return_value = subprocess.CompletedProcess([], 0)
                code = installer.main(["--requirements", str(path), "--index-url", "https://example.org/simple"])
        self.assertEqual(code, 0)
        self.assertEqual(
            run.call_args_list[0].args[0],
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--index-url",
                "https://example.org/simple",
                "peft==0.18.1",
            ],
        )
        self.assertEqual(run.call_args_list[1].args[0], [sys.executable, "-m", "pip", "check"])

    def test_all_installed_only_checks_without_install(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("torch==2.6.0\n", encoding="utf-8")
            with (
                patch.object(installer, "installed_packages", return_value={"torch": "2.10.0"}),
                patch.object(installer.subprocess, "run") as run,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                run.return_value = subprocess.CompletedProcess([], 1)
                code = installer.main(["--requirements", str(path)])
        self.assertEqual(code, 1)
        run.assert_called_once_with([sys.executable, "-m", "pip", "check"], check=False)

    def test_install_failure_does_not_continue(self):
        with (
            patch.object(installer, "installed_packages", return_value={}),
            patch.object(installer.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run.return_value = subprocess.CompletedProcess([], 7)
            self.assertEqual(installer.main([]), 7)
        self.assertEqual(run.call_count, 1)

    def test_invalid_file_does_not_invoke_pip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("peft==0.18.1\n-r other.txt\n", encoding="utf-8")
            code, run = self.invoke(["--requirements", str(path)], {})
        self.assertEqual(code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
