from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/docker_training.sh"


class DockerLauncherTests(unittest.TestCase):
    def command(self, args, output_root):
        result = subprocess.run(
            ["bash", str(SCRIPT), *args],
            env={
                **os.environ,
                "TRAINING_DRY_RUN": "1",
                "TRAINING_WORKSPACE": str(output_root),
                "TRAINING_NETWORK": "none",
                "TRAINING_OFFLINE": "1",
                "TRAINING_GPUS": "device=0",
            },
            check=True,
            capture_output=True,
            text=True,
        )
        return shlex.split(result.stdout)

    def test_gpu_smoke_is_offline_and_outputs_persist(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "output with spaces"
            command = self.command(["smoke", "--device", "cuda"], output_root)
            self.assertEqual(command[:4], ["docker", "run", "--rm", "--init"])
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertEqual(command[command.index("--gpus") + 1], "device=0")
            self.assertIn("HF_HUB_OFFLINE=1", command)
            self.assertIn(f"type=bind,src={output_root}/runs,dst=/workspace/runs", command)
            self.assertTrue(any("train-val-dataset,readonly" in arg for arg in command))
            self.assertEqual(command[-3:], ["smoke", "--device", "cuda"])
            self.assertFalse(output_root.exists())

    def test_cpu_mode_does_not_request_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            command = self.command(["--cpu", "prepare"], Path(temp) / "output")
            self.assertNotIn("--gpus", command)
            self.assertEqual(command[-1], "prepare")

    def test_download_defaults_to_online_without_gpu(self):
        environment = {k: v for k, v in os.environ.items() if k not in {"TRAINING_NETWORK", "TRAINING_OFFLINE"}}
        result = subprocess.run(
            ["bash", str(SCRIPT), "--cpu", "download"],
            env={**environment, "TRAINING_DRY_RUN": "1"},
            check=True,
            capture_output=True,
            text=True,
        )
        command = shlex.split(result.stdout)
        self.assertEqual(command[command.index("--network") + 1], "bridge")
        self.assertIn("HF_HUB_OFFLINE=0", command)
        self.assertNotIn("--gpus", command)


if __name__ == "__main__":
    unittest.main()
