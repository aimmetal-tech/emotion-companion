"""Local diagnostics, deliberately distinct from the unpublished official score."""

from __future__ import annotations

import json
from pathlib import Path

from .common import digest, file_digest, official_module, read_jsonl, target_valid, write_json


def strict_prediction(value, official):
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        return False
    return target_valid({k: v for k, v in value.items() if k != "id"}, official) and value["memory_refs"] == []


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def score(gold: list[dict], predictions: list[dict], official):
    if not gold:
        raise ValueError("No gold samples")
    ids = [r["id"] for r in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction IDs")
    expected = {r["id"] for r in gold}
    if set(ids) - expected:
        raise ValueError("Prediction IDs not in gold set")
    by_id = {r["id"]: r for r in predictions}
    pairs = [(r["target"], official.normalize_prediction(r["id"], by_id.get(r["id"], {}))) for r in gold]
    emotion_scores = {}
    for label in sorted(official.EMOTIONS):
        tp = sum(t["emotion_label"] == label and p["emotion_label"] == label for t, p in pairs)
        fp = sum(t["emotion_label"] != label and p["emotion_label"] == label for t, p in pairs)
        fn = sum(t["emotion_label"] == label and p["emotion_label"] != label for t, p in pairs)
        emotion_scores[label] = {"f1": f1(tp, fp, fn), "support": tp + fn}
    profiles = {}
    for key, allowed in official.PROFILE_LABELS.items():
        counts = []
        labels = {}
        for label in sorted(allowed):
            tp = sum(label in t["user_profile"][key] and label in p["user_profile"][key] for t, p in pairs)
            fp = sum(label not in t["user_profile"][key] and label in p["user_profile"][key] for t, p in pairs)
            fn = sum(label in t["user_profile"][key] and label not in p["user_profile"][key] for t, p in pairs)
            counts.append((tp, fp, fn))
            labels[label] = {"f1": f1(tp, fp, fn), "support": tp + fn}
        totals = [sum(c[i] for c in counts) for i in range(3)]
        empties = [(t, p) for t, p in pairs if not t["user_profile"][key]]
        profiles[key] = {
            "micro_f1": f1(*totals),
            "macro_f1": sum(v["f1"] for v in labels.values()) / len(labels),
            "exact_match": sum(set(t["user_profile"][key]) == set(p["user_profile"][key]) for t, p in pairs)
            / len(pairs),
            "empty_gold_samples": len(empties),
            "empty_gold_exact_match": (sum(not p["user_profile"][key] for _, p in empties) / len(empties))
            if empties
            else None,
            "per_label": labels,
        }
    emotion_macro = sum(v["f1"] for v in emotion_scores.values()) / len(emotion_scores)
    profile_mean = sum(v["micro_f1"] for v in profiles.values()) / len(profiles)
    return {
        "samples": len(gold),
        "missing_predictions": len(expected - set(ids)),
        "field_compliance_rate": sum(strict_prediction(by_id.get(r["id"]), official) for r in gold) / len(gold),
        "emotion_accuracy": sum(t["emotion_label"] == p["emotion_label"] for t, p in pairs) / len(pairs),
        "emotion_macro_f1": emotion_macro,
        "emotion_per_label": emotion_scores,
        "profiles": profiles,
        "profile_mean_micro_f1": profile_mean,
        "local_selection_score": (emotion_macro + profile_mean) / 2,
        "note": "Local diagnostic score; official weights unknown. Memory and reply quality excluded.",
    }


def score_file(gold_path: Path, predictions_path: Path, output: Path, participant: Path):
    report = score(read_jsonl(gold_path), read_jsonl(predictions_path), official_module(participant))
    report.update(gold_sha256=file_digest(gold_path), predictions_sha256=file_digest(predictions_path))
    write_json(output, report)
    return report


def make_review(samples_path: Path, baseline_path: Path, candidate_path: Path, output: Path):
    import random

    if output.exists():
        raise ValueError(f"Review already exists: {output}")
    rows = read_jsonl(samples_path)
    baseline = {r["id"]: r for r in read_jsonl(baseline_path)}
    candidate = {r["id"]: r for r in read_jsonl(candidate_path)}
    rng = random.Random(42)
    sheet = []
    key = {}
    for sample in rows:
        sid = sample["id"]
        if sid not in baseline or sid not in candidate:
            raise ValueError(f"Review prediction missing: {sid}")
        order = ["baseline", "candidate"]
        rng.shuffle(order)
        choices = {"baseline": baseline[sid], "candidate": candidate[sid]}
        key[sid] = dict(zip(("A", "B"), order, strict=True))
        sheet.append(
            {
                "id": sid,
                "history": sample["history"],
                "A": choices[order[0]]["response_text"],
                "B": choices[order[1]]["response_text"],
                "ratings": {
                    label: dict.fromkeys(("empathy", "coherence", "preference_following", "factuality"))
                    for label in ("A", "B")
                },
                "notes": "",
            }
        )
    output.mkdir(parents=True)
    write_json(output / "review.json", sheet)
    write_json(output / "key.json", key)
    write_json(
        output / "sources.json",
        {
            "gold_sha256": file_digest(samples_path),
            "baseline_sha256": file_digest(baseline_path),
            "candidate_sha256": file_digest(candidate_path),
            "review_ids_sha256": digest([r["id"] for r in sheet]),
        },
    )


def compare(baseline_path: Path, candidate_path: Path, review_dir: Path, output: Path):
    baseline, candidate = (json.loads(p.read_text()) for p in (baseline_path, candidate_path))
    if baseline["gold_sha256"] != candidate["gold_sha256"]:
        raise ValueError("Compare the same gold evaluation subset")
    for setting in ("quantized", "max_input_tokens", "max_new_tokens", "participant_sha256", "python_hash_seed"):
        if baseline.get(setting) != candidate.get(setting):
            raise ValueError(f"Baseline and candidate evaluation settings differ: {setting}")
    rows = json.loads((review_dir / "review.json").read_text())
    key = json.loads((review_dir / "key.json").read_text())
    sources = json.loads((review_dir / "sources.json").read_text())
    if (
        sources["baseline_sha256"] != baseline["predictions_sha256"]
        or sources["candidate_sha256"] != candidate["predictions_sha256"]
    ):
        raise ValueError("Human review and metric reports must use the same predictions")
    if (
        len(rows) != 30
        or len({r["id"] for r in rows}) != 30
        or set(key) != {r["id"] for r in rows}
        or digest([r["id"] for r in rows]) != sources["review_ids_sha256"]
    ):
        raise ValueError("Human review must contain all 30 fixed samples")
    totals = {"baseline": 0, "candidate": 0}
    for row in rows:
        for choice in ("A", "B"):
            for dimension in ("empathy", "coherence", "preference_following", "factuality"):
                value = row["ratings"][choice][dimension]
                if type(value) is not int or not 1 <= value <= 5:
                    raise ValueError("Complete human ratings with integers from 1 to 5")
                totals[key[row["id"]][choice]] += value
    checks = {
        "compliance_at_least_99_percent": candidate["field_compliance_rate"] >= 0.99,
        "structured_score_improved": candidate["local_selection_score"] > baseline["local_selection_score"],
        "human_reply_quality_not_worse": totals["candidate"] >= totals["baseline"],
        "all_predictions_present": candidate["missing_predictions"] == 0,
    }
    report = {
        "accepted": all(checks.values()),
        "checks": checks,
        "human_totals": totals,
        "baseline_score": baseline["local_selection_score"],
        "candidate_score": candidate["local_selection_score"],
    }
    write_json(output, report)
    return report
