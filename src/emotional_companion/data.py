"""Conversation-level splitting, leakage checks and nested evaluation subsets."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from .common import digest, file_digest, official_module, read_jsonl, target_valid, write_json, write_jsonl


def expand(conversations: list[dict], prefix: str, official, issues: list[dict] | None = None) -> list[dict]:
    samples = []
    seen_ids = set()
    for index, conversation in enumerate(conversations, 1):
        turns = conversation["turns"]
        ids = [t["turn_id"] for t in turns]
        if ids != sorted(set(ids)) or any(type(i) is not int or i < 1 for i in ids):
            raise ValueError(f"Invalid turn ids: {conversation['conversation_id']}")
        if any(
            t.get("role") not in {"user", "assistant"}
            or not isinstance(t.get("content"), str)
            or not t["content"].strip()
            for t in turns
        ):
            raise ValueError(f"Invalid turns: {conversation['conversation_id']}")
        by_id = {t["turn_id"]: t for t in turns}
        for point in conversation["prediction_points"]:
            user_id = point["target_user_turn_id"]
            response_id = point["assistant_response_turn_id"]
            target = point["target"]
            if (
                user_id not in by_id
                or response_id not in by_id
                or response_id <= user_id
                or by_id[user_id]["role"] != "user"
                or by_id[response_id]["role"] != "assistant"
                or any(user_id < t["turn_id"] < response_id for t in turns)
                or not target_valid(target, official)
                or target["response_text"] != by_id[response_id]["content"]
            ):
                if issues is None:
                    raise ValueError(f"Inconsistent prediction point: {conversation['conversation_id']}")
                issues.append(
                    {
                        "conversation_id": conversation["conversation_id"],
                        "target_user_turn_id": user_id,
                        "assistant_response_turn_id": response_id,
                        "reason": "invalid target, mismatched reply, or response is not the next assistant turn",
                    }
                )
                continue
            sample_id = f"{prefix}_{index:06d}_t{user_id}"
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate prediction point: {sample_id}")
            seen_ids.add(sample_id)
            samples.append(
                {
                    "id": sample_id,
                    "conversation_id": conversation["conversation_id"],
                    "target_user_turn_id": user_id,
                    "history": [t for t in turns if t["turn_id"] <= user_id],
                    "target": {**target, "memory_refs": []},
                }
            )
    return samples


def history_signature(sample: dict) -> str:
    return digest([(t["role"], t["content"]) for t in sample["history"]])


def split_groups(samples: list[dict], seed: int = 42, fraction: float = 0.05):
    """Union conversations sharing any prediction history before splitting."""
    parent = {s["conversation_id"]: s["conversation_id"] for s in samples}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    histories = {}
    for sample in samples:
        cid = sample["conversation_id"]
        signature = history_signature(sample)
        if signature in histories:
            a, b = find(cid), find(histories[signature])
            parent[max(a, b)] = min(a, b)
        else:
            histories[signature] = cid
    groups = sorted({find(cid) for cid in parent})
    if len(groups) < 2:
        raise ValueError("Need at least two independent conversation groups")
    random.Random(seed).shuffle(groups)
    dev_groups = set(groups[: max(1, round(len(groups) * fraction))])
    train, dev = [], []
    for sample in samples:
        (dev if find(sample["conversation_id"]) in dev_groups else train).append(sample)
    return train, dev, len(groups)


def select_nested(samples: list[dict], size: int, seed: int = 42, initial: list[dict] | None = None):
    """Cover each available emotion, then interleave history-length quartiles."""
    chosen = list(initial or [])
    seen = {s["id"] for s in chosen}
    pool = sorted(samples, key=lambda s: s["id"])
    random.Random(seed).shuffle(pool)
    covered = {s["target"]["emotion_label"] for s in chosen}
    for label in sorted({s["target"]["emotion_label"] for s in pool} - covered):
        if len(chosen) >= size:
            break
        sample = next(s for s in pool if s["target"]["emotion_label"] == label and s["id"] not in seen)
        chosen.append(sample)
        seen.add(sample["id"])
    pool = sorted((s for s in pool if s["id"] not in seen), key=lambda s: sum(len(t["content"]) for t in s["history"]))
    bins = [pool[i * len(pool) // 4 : (i + 1) * len(pool) // 4] for i in range(4)]
    rng = random.Random(seed)
    for bucket in bins:
        rng.shuffle(bucket)
    while len(chosen) < min(size, len(samples)):
        for bucket in bins:
            if bucket and len(chosen) < size:
                chosen.append(bucket.pop())
    return chosen


def prepare(source: Path, output: Path, participant: Path, seed: int = 42):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty to preserve previous manifests: {output}")
    official = official_module(participant)
    train_path, val_path = source / "train/train_public.jsonl", source / "val/val_public.jsonl"
    raw_train, raw_val = read_jsonl(train_path), read_jsonl(val_path)
    for rows in (raw_train, raw_val):
        if len({r["conversation_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate conversation IDs")
    # Retain differing annotations as prediction points on the same conversation.
    unique = {}
    duplicates = 0
    annotation_conflicts = 0
    for row in raw_train:
        signature = digest([(t["role"], t["content"]) for t in row["turns"]])
        if signature not in unique:
            unique[signature] = {**row, "prediction_points": list(row["prediction_points"])}
            continue
        duplicates += 1
        existing = unique[signature]
        points = {p["target_user_turn_id"]: p for p in existing["prediction_points"]}
        for point in row["prediction_points"]:
            uid = point["target_user_turn_id"]
            if uid not in points:
                existing["prediction_points"].append(point)
                points[uid] = point
            elif digest({**points[uid]["target"], "memory_refs": []}) != digest({**point["target"], "memory_refs": []}):
                annotation_conflicts += 1
    train_issues, val_issues = [], []
    train_samples = expand(list(unique.values()), "train", official, train_issues)
    val = expand(raw_val, "test", official, val_issues)
    val_histories = {history_signature(s) for s in val}
    leaked_cids = {s["conversation_id"] for s in train_samples if history_signature(s) in val_histories}
    train_samples = [s for s in train_samples if s["conversation_id"] not in leaked_cids]
    train, dev, groups = split_groups(train_samples, seed)
    quick = select_nested(val, 128, seed)
    extended = select_nested(val, 512, seed, initial=quick)
    files = {
        "train": train,
        "dev": dev,
        "val_full": val,
        "val_quick": quick,
        "val_extended": extended,
        "dev_loss": select_nested(dev, 64, seed),
        "review": select_nested(quick, 30, seed),
    }
    manifest = {
        "seed": seed,
        "source_sha256": {"train": file_digest(train_path), "val": file_digest(val_path)},
        "source_conversations": {"train": len(raw_train), "val": len(raw_val)},
        "duplicate_train_conversations": duplicates,
        "duplicate_annotation_conflicts_first_wins": annotation_conflicts,
        "train_conversations_removed_for_val_history_overlap": len(leaked_cids),
        "independent_train_groups": groups,
        "memory_policy": "empty: no public memory bank",
        "excluded_prediction_points": {"train": train_issues, "val": val_issues},
        "participant_sha256": file_digest(participant / "run_inference.py"),
        "subsets": {},
    }
    for name, samples in files.items():
        path = output / f"{name}.jsonl"
        write_jsonl(path, samples)
        manifest["subsets"][name] = {
            "samples": len(samples),
            "conversations": len({s["conversation_id"] for s in samples}),
            "emotions": dict(sorted(Counter(s["target"]["emotion_label"] for s in samples).items())),
            "sha256": file_digest(path),
        }
        if name.startswith("val_"):
            write_jsonl(
                output / f"{name}.input.jsonl", [{k: v for k, v in s.items() if k != "target"} for s in samples]
            )
    write_json(output / "manifest.json", manifest)
    return manifest
