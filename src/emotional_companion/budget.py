"""Conservative GPU-hour ledger and a deadline-bound subprocess runner (Linux)."""

from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import subprocess
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path

from .common import write_json

LIMIT = 20.0
STAGES = {"baseline_smoke": 1.0, "train": 9.0, "eval": 3.0, "export": 1.0, "verify": 4.0, "recovery": 2.0}


@contextmanager
def locked_ledger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(path.read_text()) if path.exists() else {"limit_gpu_hours": LIMIT, "entries": []}
        yield ledger
        temp = path.with_suffix(".tmp")
        write_json(temp, ledger)
        temp.replace(path)


def charged(entry):
    return entry["reserved_gpu_hours"] if entry["status"] == "running" else entry["gpu_hours"]


def validate_allocation(ledger, stage: str, hours: float):
    if stage not in STAGES or not math.isfinite(hours) or hours <= 0:
        raise ValueError("Stage and positive finite GPU hours are required")
    used = sum(charged(e) for e in ledger["entries"])
    stage_used = sum(charged(e) for e in ledger["entries"] if e["stage"] == stage)
    if used + hours > LIMIT + 1e-8:
        raise ValueError(f"20 GPU-hour budget exceeded: used/reserved={used:.4f}, requested={hours:.4f}")
    if stage_used + hours > STAGES[stage] + 1e-8:
        raise ValueError(f"Stage {stage} exceeds its {STAGES[stage]} GPU-hour allowance")
    if used >= 18 and stage not in {"export", "verify", "recovery"}:
        raise ValueError("At 18 GPU hours only export, verification and recovery may start")


def status(path: Path):
    ledger = json.loads(path.read_text()) if path.exists() else {"limit_gpu_hours": LIMIT, "entries": []}
    used = sum(charged(e) for e in ledger["entries"])
    return {
        **ledger,
        "used_or_reserved_gpu_hours": used,
        "remaining_gpu_hours": LIMIT - used,
        "stage_remaining": {
            s: cap - sum(charged(e) for e in ledger["entries"] if e["stage"] == s) for s, cap in STAGES.items()
        },
    }


def record(path: Path, stage: str, gpu_hours: float, note: str):
    """Account for setup/idle time outside managed commands before further work."""
    with locked_ledger(path) as ledger:
        if any(e["status"] == "running" for e in ledger["entries"]):
            raise ValueError("Cannot add idle time while a managed command is active")
        validate_allocation(ledger, stage, gpu_hours)
        ledger["entries"].append(
            {
                "id": uuid.uuid4().hex,
                "stage": stage,
                "status": "recorded",
                "gpu_hours": gpu_hours,
                "note": note,
                "recorded_at": time.time(),
            }
        )


def reconcile(path: Path, entry_id: str, gpu_hours: float):
    """An interrupted wrapper keeps its reservation until actual usage is supplied."""
    if not math.isfinite(gpu_hours) or gpu_hours < 0:
        raise ValueError("GPU hours must be finite and nonnegative")
    with locked_ledger(path) as ledger:
        entry = next((e for e in ledger["entries"] if e["id"] == entry_id), None)
        if entry is None or entry["status"] != "running":
            raise ValueError("Expected an unresolved running entry")
        if gpu_hours > entry["reserved_gpu_hours"]:
            raise ValueError("Usage exceeded reservation; inspect instance billing before continuing")
        entry.update(status="reconciled", gpu_hours=gpu_hours, finished_at=time.time())


def stop_group(process, first_signal=signal.SIGINT):
    for sig, wait in ((first_signal, 40), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
        # A shell may exit before its child server; clean up remaining descendants.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return


def run(path: Path, stage: str, gpus: int, hours: float, command: list[str]):
    if gpus not in (1, 2) or not math.isfinite(hours) or hours <= 0 or not command:
        raise ValueError("Require 1/2 GPUs, positive hours, and a command after --")
    if hours * 3600 <= 60:
        raise ValueError("Reserve more than 60 seconds for graceful shutdown")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and len([part for part in visible.split(",") if part.strip()]) != gpus:
        raise ValueError("CUDA_VISIBLE_DEVICES must expose exactly the declared number of GPUs")
    entry_id = uuid.uuid4().hex
    started = time.time()
    reserved = gpus * hours
    with locked_ledger(path) as ledger:
        if any(e["status"] == "running" for e in ledger["entries"]):
            raise ValueError("Resolve the active reservation before launching another command")
        validate_allocation(ledger, stage, reserved)
        ledger["entries"].append(
            {
                "id": entry_id,
                "stage": stage,
                "status": "running",
                "gpus": gpus,
                "started_at": started,
                "reserved_gpu_hours": reserved,
                "command": command,
            }
        )
    process = None
    code = 1
    interrupted = False
    environment = {
        **os.environ,
        "PYTHONHASHSEED": "42",
        "TOKENIZERS_PARALLELISM": "false",
        "COMPANION_BUDGET_DEADLINE": str(started + hours * 3600 - 60),
        "COMPANION_BUDGET_GPUS": str(gpus),
    }
    environment.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(gpus)))
    previous_handlers = {}

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, interrupt)
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        try:
            code = process.wait(timeout=max(0, started + hours * 3600 - 60 - time.time()))
        except subprocess.TimeoutExpired:
            interrupted = True
            stop_group(process)
            code = 124
        except KeyboardInterrupt:
            interrupted = True
            stop_group(process)
            code = 130
    finally:
        if process is not None:
            stop_group(process, signal.SIGTERM)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        actual = (time.time() - started) * gpus / 3600
        with locked_ledger(path) as ledger:
            entry = next(e for e in ledger["entries"] if e["id"] == entry_id)
            entry.update(
                status="interrupted" if interrupted else "finished",
                gpu_hours=actual,
                exit_code=code,
                finished_at=time.time(),
            )
        print(f"GPU hours: {actual:.4f}; remaining: {status(path)['remaining_gpu_hours']:.4f}", flush=True)
    return code
