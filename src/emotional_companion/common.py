"""Dependency-free IO and an adapter to the unmodified participant framework."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def official_module(participant: Path):
    source = participant / "run_inference.py"
    spec = importlib.util.spec_from_file_location("companion_official_inference", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt_messages(sample: dict, official) -> list[dict[str, str]]:
    # Match the official runner exactly, including its emotion enumeration order.
    system = official.SYSTEM_PROMPT.format(
        emotions=", ".join(official.EMOTIONS),
        personality=", ".join(sorted(official.PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(official.PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(official.PROFILE_LABELS["style"])),
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": official.conversation_text(sample)}]


def target_valid(target: Any, official) -> bool:
    if not isinstance(target, dict) or set(target) != {"response_text", "emotion_label", "user_profile", "memory_refs"}:
        return False
    response = target["response_text"]
    emotion = target["emotion_label"]
    profile = target["user_profile"]
    if not isinstance(response, str) or not response.strip():
        return False
    if not isinstance(emotion, str) or emotion not in official.EMOTIONS:
        return False
    if not isinstance(profile, dict) or set(profile) != set(official.PROFILE_LABELS):
        return False
    for key, allowed in official.PROFILE_LABELS.items():
        values = profile[key]
        if (
            not isinstance(values, list)
            or not all(isinstance(v, str) for v in values)
            or len(values) != len(set(values))
            or not set(values) <= allowed
        ):
            return False
    memories = target["memory_refs"]
    return (
        isinstance(memories, list)
        and all(isinstance(v, str) and official.MEMORY_RE.fullmatch(v) for v in memories)
        and len(memories) == len(set(memories))
    )
