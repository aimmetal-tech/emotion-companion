from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emotional_companion import budget
from emotional_companion.common import official_module, read_jsonl, write_json, write_jsonl
from emotional_companion.data import expand, history_signature, prepare, select_nested, split_groups
from emotional_companion.metrics import compare, make_review, score
from emotional_companion.modeling import encode_sample, verify_results

PARTICIPANT = Path(__file__).resolve().parents[1] / "official_dl/participant"
OFFICIAL = official_module(PARTICIPANT)


def target(emotion="anxiety"):
    return {
        "response_text": "I hear you.",
        "emotion_label": emotion,
        "user_profile": {"personality_traits": ["sensitive"], "interests": [], "style": ["brief"]},
        "memory_refs": [],
    }


def sample(index, emotion="anxiety"):
    return {
        "id": f"test_{index:06d}_t3",
        "conversation_id": f"c{index}",
        "target_user_turn_id": 3,
        "history": [
            {"turn_id": 1, "role": "user", "content": f"Earlier {index}"},
            {"turn_id": 2, "role": "assistant", "content": "Old reply"},
            {"turn_id": 3, "role": "user", "content": f"Current {index}"},
        ],
        "target": target(emotion),
    }


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(f"{m['role']}:{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        return [ord(c) for c in text] if tokenize else text


class DataTests(unittest.TestCase):
    def test_only_history_before_response_and_memory_empty(self):
        row = {
            "conversation_id": "c1",
            "turns": [
                *sample(1)["history"],
                {"turn_id": 4, "role": "assistant", "content": "I hear you."},
                {"turn_id": 5, "role": "user", "content": "Future secret"},
            ],
            "prediction_points": [
                {
                    "target_user_turn_id": 3,
                    "assistant_response_turn_id": 4,
                    "target": {**target(), "memory_refs": ["mem_000001"]},
                }
            ],
        }
        result = expand([row], "test", OFFICIAL)[0]
        self.assertEqual([t["turn_id"] for t in result["history"]], [1, 2, 3])
        self.assertEqual(result["target"]["memory_refs"], [])

    def test_prior_reply_is_excluded_with_audit(self):
        row = {
            "conversation_id": "c1",
            "turns": sample(1)["history"],
            "prediction_points": [
                {
                    "target_user_turn_id": 3,
                    "assistant_response_turn_id": 2,
                    "target": {**target(), "response_text": "Old reply"},
                }
            ],
        }
        issues = []
        self.assertEqual(expand([row], "test", OFFICIAL, issues), [])
        self.assertEqual(len(issues), 1)
        with self.assertRaises(ValueError):
            expand([row], "test", OFFICIAL)

    def test_shared_history_groups_do_not_leak(self):
        samples = [sample(i) for i in range(1, 50)]
        samples[1]["history"] = samples[0]["history"]
        train, dev, _ = split_groups(samples)
        self.assertFalse({history_signature(s) for s in train} & {history_signature(s) for s in dev})
        self.assertFalse({s["conversation_id"] for s in train} & {s["conversation_id"] for s in dev})
        self.assertEqual(split_groups(samples), split_groups(samples))

    def test_nested_subsets_cover_all_emotions(self):
        emotions = sorted(OFFICIAL.EMOTIONS)
        rows = [sample(i + 1, emotions[i % len(emotions)]) for i in range(600)]
        quick = select_nested(rows, 128)
        extended = select_nested(rows, 512, initial=quick)
        self.assertEqual(len(quick), 128)
        self.assertEqual(len(extended), 512)
        self.assertTrue({s["id"] for s in quick} <= {s["id"] for s in extended})
        self.assertEqual({s["target"]["emotion_label"] for s in quick}, OFFICIAL.EMOTIONS)

    def test_loss_masks_only_prompt(self):
        encoded = encode_sample(sample(1), CharacterTokenizer(), OFFICIAL, 4096)
        labels = encoded["labels"]
        first = next(i for i, v in enumerate(labels) if v != -100)
        self.assertTrue(all(v == -100 for v in labels[:first]))
        decoded = "".join(chr(v) for v in labels[first:])
        self.assertEqual(json.loads(decoded), target())
        self.assertEqual(labels[first:], encoded["input_ids"][first:])

    def test_truncation_keeps_latest_user_and_complete_target(self):
        row = sample(1)
        row["history"][0]["content"] = "x" * 6000
        encoded = encode_sample(row, CharacterTokenizer(), OFFICIAL, 2048)
        self.assertEqual(encoded["removed_turns"], 2)
        self.assertIn("Current 1", "".join(chr(v) for v in encoded["input_ids"]))
        self.assertEqual(json.loads("".join(chr(v) for v in encoded["labels"] if v != -100)), target())

    def test_oversize_target_is_skipped(self):
        row = sample(1)
        row["target"]["response_text"] = "x" * 5000
        self.assertIsNone(encode_sample(row, CharacterTokenizer(), OFFICIAL, 2048))

    def test_full_official_dataset_audit(self):
        source = PARTICIPANT.parent / "train-val-dataset"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "data"
            report = prepare(source, output, PARTICIPANT)
            self.assertEqual(report["duplicate_train_conversations"], 202)
            self.assertEqual(len(report["excluded_prediction_points"]["train"]), 4)
            self.assertEqual(report["subsets"]["val_full"]["samples"], 2354)
            train_rows = read_jsonl(output / "train.jsonl")
            dev_rows = read_jsonl(output / "dev.jsonl")
            self.assertFalse({history_signature(s) for s in train_rows} & {history_signature(s) for s in dev_rows})
            self.assertTrue(all("target" not in row for row in read_jsonl(output / "val_quick.input.jsonl")))


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions_and_empty_profile_reporting(self):
        rows = [sample(i + 1, label) for i, label in enumerate(sorted(OFFICIAL.EMOTIONS))]
        predictions = [{"id": r["id"], **r["target"]} for r in rows]
        report = score(rows, predictions, OFFICIAL)
        self.assertEqual(report["emotion_macro_f1"], 1)
        self.assertEqual(report["field_compliance_rate"], 1)
        self.assertEqual(report["profiles"]["interests"]["micro_f1"], 0)
        self.assertEqual(report["profiles"]["interests"]["empty_gold_exact_match"], 1)

    def test_missing_and_invalid_predictions_fail_compliance(self):
        rows = [sample(1), sample(2)]
        report = score(rows, [{"id": rows[0]["id"], "response_text": "hi"}], OFFICIAL)
        self.assertEqual(report["missing_predictions"], 1)
        self.assertEqual(report["field_compliance_rate"], 0)
        self.assertEqual(report["emotion_accuracy"], 0)

    def test_duplicate_and_unknown_ids_rejected(self):
        with self.assertRaises(ValueError):
            score([sample(1)], [{"id": "other"}], OFFICIAL)
        with self.assertRaises(ValueError):
            score([sample(1)], [{"id": sample(1)["id"]}] * 2, OFFICIAL)

    def test_review_must_be_completed_and_quality_not_worse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [sample(i + 1) for i in range(30)]
            predictions = [{"id": r["id"], **r["target"]} for r in rows]
            write_jsonl(root / "gold.jsonl", rows)
            write_jsonl(root / "baseline.jsonl", predictions)
            write_jsonl(root / "candidate.jsonl", predictions)
            make_review(root / "gold.jsonl", root / "baseline.jsonl", root / "candidate.jsonl", root / "review")
            from emotional_companion.common import file_digest

            base = {
                "gold_sha256": "same",
                "predictions_sha256": file_digest(root / "baseline.jsonl"),
                "local_selection_score": 0.5,
                "missing_predictions": 0,
                "field_compliance_rate": 1,
            }
            candidate = {
                **base,
                "predictions_sha256": file_digest(root / "candidate.jsonl"),
                "local_selection_score": 0.6,
            }
            write_json(root / "base_metrics.json", base)
            write_json(root / "candidate_metrics.json", candidate)
            with self.assertRaises(ValueError):
                compare(
                    root / "base_metrics.json", root / "candidate_metrics.json", root / "review", root / "accept.json"
                )
            sheet = json.loads((root / "review/review.json").read_text())
            for row in sheet:
                for ratings in row["ratings"].values():
                    for key in ratings:
                        ratings[key] = 3
            write_json(root / "review/review.json", sheet)
            result = compare(
                root / "base_metrics.json", root / "candidate_metrics.json", root / "review", root / "accept.json"
            )
            self.assertTrue(result["accepted"])


class BudgetTests(unittest.TestCase):
    def test_stage_and_total_budget_limits(self):
        ledger = {"entries": [{"stage": "train", "status": "finished", "gpu_hours": 9}]}
        with self.assertRaises(ValueError):
            budget.validate_allocation(ledger, "train", 0.01)
        ledger["entries"].append({"stage": "eval", "status": "recorded", "gpu_hours": 11})
        with self.assertRaises(ValueError):
            budget.validate_allocation(ledger, "recovery", 0.01)
        with self.assertRaises(ValueError):
            budget.validate_allocation({"entries": []}, "train", float("nan"))

    def test_last_two_hours_cannot_start_new_experiments(self):
        ledger = {"entries": [{"stage": "verify", "status": "recorded", "gpu_hours": 18}]}
        with self.assertRaises(ValueError):
            budget.validate_allocation(ledger, "train", 1)
        budget.validate_allocation(ledger, "recovery", 1)

    def test_running_reservations_are_charged_until_reconciled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "budget.json"
            write_json(
                path,
                {"entries": [{"id": "interrupted", "stage": "train", "status": "running", "reserved_gpu_hours": 9}]},
            )
            self.assertEqual(budget.status(path)["remaining_gpu_hours"], 11)
            budget.reconcile(path, "interrupted", 2)
            self.assertEqual(budget.status(path)["remaining_gpu_hours"], 18)

    def test_runner_records_real_elapsed_time_and_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "budget.json"
            result = budget.run(path, "baseline_smoke", 1, 0.02, [sys.executable, "-c", "raise SystemExit(3)"])
            self.assertEqual(result, 3)
            entry = budget.status(path)["entries"][0]
            self.assertEqual(entry["status"], "finished")
            self.assertGreater(entry["gpu_hours"], 0)
            self.assertLess(entry["gpu_hours"], 0.02)

    def test_timeout_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            process = unittest.mock.Mock()
            process.wait.side_effect = subprocess.TimeoutExpired("test", 1)
            with patch.object(budget.subprocess, "Popen", return_value=process), patch.object(budget, "stop_group"):
                code = budget.run(Path(temp) / "budget.json", "baseline_smoke", 1, 0.02, ["test"])
            self.assertEqual(code, 124)
            self.assertEqual(budget.status(Path(temp) / "budget.json")["entries"][0]["status"], "interrupted")


class ContractTests(unittest.TestCase):
    def test_three_samples_cannot_complete_official_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [sample(i + 1) for i in range(3)]
            write_jsonl(root / "input.jsonl", rows)
            write_jsonl(root / "submission.jsonl", [{"id": r["id"], **r["target"]} for r in rows])
            write_json(root / "performance_report.json", {"required_rounds": 100, "rounds": 3, "complete": False})
            with self.assertRaises(ValueError):
                verify_results(root / "input.jsonl", root, PARTICIPANT)

    def test_invalid_profile_types_do_not_crash_scoring(self):
        row = sample(1)
        prediction = {"id": row["id"], **copy.deepcopy(row["target"])}
        prediction["user_profile"]["style"] = [None, {"bad": True}]
        self.assertEqual(score([row], [prediction], OFFICIAL)["field_compliance_rate"], 0)


if __name__ == "__main__":
    unittest.main()
