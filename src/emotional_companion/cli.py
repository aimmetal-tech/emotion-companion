"""Run with python -m emotional_companion; CPU commands need no third-party packages."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parser():
    cli = argparse.ArgumentParser(description="20 GPU-hour emotional-companion training workflow")
    cli.add_argument("--participant", type=Path, default=Path("official_dl/participant"))
    sub = cli.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="Deduplicate, split and build evaluation subsets (CPU)")
    prep.add_argument("--source", type=Path, default=Path("official_dl/train-val-dataset"))
    prep.add_argument("--output", type=Path, default=Path("artifacts/data"))
    dl = sub.add_parser("download", help="Download a pinned model snapshot before starting GPU")
    dl.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    dl.add_argument("--revision", default="main")
    dl.add_argument("--output", type=Path, default=Path("models/qwen2.5-7b-instruct"))
    tok = sub.add_parser("tokenize", help="Build completion-only tokenized datasets (CPU)")
    tok.add_argument("--data", type=Path, default=Path("artifacts/data"))
    tok.add_argument("--model", type=Path, default=Path("models/qwen2.5-7b-instruct"))
    tok.add_argument("--output", type=Path, default=Path("artifacts/tokens-4096"))
    tok.add_argument("--max-length", type=int, choices=(2048, 4096), default=4096)
    tr = sub.add_parser("train", help="Single-GPU QLoRA with resumable deadline")
    tr.add_argument("--model", type=Path, default=Path("models/qwen2.5-7b-instruct"))
    tr.add_argument("--data", type=Path, default=Path("artifacts/tokens-4096"))
    tr.add_argument("--output", type=Path, default=Path("runs/train"))
    tr.add_argument("--config", type=Path, default=Path("configs/train_7b.json"))
    tr.add_argument("--max-steps", type=int)
    tr.add_argument("--max-length", type=int, choices=(2048, 4096))
    tr.add_argument("--resume", help="latest or path to a full Trainer checkpoint")
    ev = sub.add_parser("evaluate", help="Generate and score baseline/adapter predictions offline")
    ev.add_argument("--model", type=Path, default=Path("models/qwen2.5-7b-instruct"))
    ev.add_argument("--adapter", type=Path)
    ev.add_argument("--data", type=Path, default=Path("artifacts/data/val_quick.jsonl"))
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--bf16", action="store_true", help="Use BF16 weights instead of 4-bit")
    ev.add_argument("--max-input-tokens", type=int, default=32768)
    ev.add_argument("--max-new-tokens", type=int, default=1200)
    sc = sub.add_parser("score", help="Score official normalized predictions (CPU)")
    sc.add_argument("--gold", type=Path, required=True)
    sc.add_argument("--predictions", type=Path, required=True)
    sc.add_argument("--output", type=Path, required=True)
    review = sub.add_parser("review", help="Create a blind human review of 30 fixed replies (CPU)")
    review.add_argument("--samples", type=Path, default=Path("artifacts/data/review.jsonl"))
    review.add_argument("--baseline", type=Path, required=True)
    review.add_argument("--candidate", type=Path, required=True)
    review.add_argument("--output", type=Path, default=Path("runs/review"))
    comparison = sub.add_parser("compare", help="Apply local acceptance gates after human ratings (CPU)")
    comparison.add_argument("--baseline", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--review", type=Path, default=Path("runs/review"))
    comparison.add_argument("--output", type=Path, default=Path("runs/acceptance.json"))
    ex = sub.add_parser("export", help="Merge selected adapter into original BF16 base weights")
    ex.add_argument("--model", type=Path, default=Path("models/qwen2.5-7b-instruct"))
    ex.add_argument("--adapter", type=Path, default=Path("runs/train/adapter"))
    ex.add_argument("--output", type=Path, default=Path("models/companion-7b"))
    bu = sub.add_parser("bundle", help="Copy official framework and exported model for platform validation")
    bu.add_argument("--model", type=Path, default=Path("models/companion-7b"))
    bu.add_argument("--output", type=Path, default=Path("runs/submission/participant"))
    bu.add_argument("--hardware-label", default="2xRTX4090-24GB")
    ve = sub.add_parser("verify-results", help="Validate official outputs and 100-round timing (CPU)")
    ve.add_argument("--inputs", type=Path, default=Path("artifacts/data/val_quick.input.jsonl"))
    ve.add_argument("--result", type=Path, required=True)
    smoke = sub.add_parser("smoke", help="Local tiny-model train/resume/export/official inference test")
    smoke.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    smoke.add_argument("--output", type=Path, default=Path("runs/local-smoke"))
    budget = sub.add_parser("budget", help="GPU-hour ledger and bounded command execution")
    budget.add_argument("--ledger", type=Path, default=Path("runs/budget.json"))
    bs = budget.add_subparsers(dest="budget_action", required=True)
    bs.add_parser("status")
    rec = bs.add_parser("record", help="Account for instance setup/idle GPU hours")
    rec.add_argument(
        "--stage", choices=("baseline_smoke", "train", "eval", "export", "verify", "recovery"), required=True
    )
    rec.add_argument("--gpu-hours", type=float, required=True)
    rec.add_argument("--note", required=True)
    recon = bs.add_parser("reconcile", help="Resolve an interrupted wrapper after confirming its GPU processes stopped")
    recon.add_argument("--id", required=True)
    recon.add_argument("--gpu-hours", type=float, required=True)
    run = bs.add_parser("run", help="Run a command with a GPU-hour reservation and timeout")
    run.add_argument(
        "--stage", choices=("baseline_smoke", "train", "eval", "export", "verify", "recovery"), required=True
    )
    run.add_argument("--gpus", type=int, choices=(1, 2), required=True)
    run.add_argument("--hours", type=float, required=True, help="Maximum wall-clock hours, including shutdown")
    run.add_argument("command", nargs=argparse.REMAINDER)
    return cli


def main():
    args = parser().parse_args()
    # The official prompt joins an unordered emotion set. Stabilize every Python entrypoint
    # without modifying the official runner; platform commands must use the same hash seed.
    if args.action in {"tokenize", "train", "evaluate", "smoke"} and os.environ.get("PYTHONHASHSEED") != "42":
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "emotional_companion", *sys.argv[1:]],
            {**os.environ, "PYTHONHASHSEED": "42"},
        )
    try:
        result = dispatch(args)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def dispatch(a):
    from . import budget, data, metrics, modeling

    match a.action:
        case "prepare":
            return data.prepare(a.source, a.output, a.participant)
        case "download":
            return modeling.download(a.model_id, a.revision, a.output)
        case "tokenize":
            return modeling.tokenize(a.data, a.model, a.output, a.participant, a.max_length)
        case "train":
            from .training import train

            return train(a.model, a.data, a.output, a.config, a.max_steps, a.max_length, a.resume)
        case "evaluate":
            return modeling.evaluate(
                a.model, a.adapter, a.data, a.output, a.participant, not a.bf16, a.max_input_tokens, a.max_new_tokens
            )
        case "score":
            return metrics.score_file(a.gold, a.predictions, a.output, a.participant)
        case "review":
            return metrics.make_review(a.samples, a.baseline, a.candidate, a.output)
        case "compare":
            return metrics.compare(a.baseline, a.candidate, a.review, a.output)
        case "export":
            return modeling.export(a.model, a.adapter, a.output)
        case "bundle":
            return modeling.bundle(a.model, a.participant, a.output, a.hardware_label)
        case "verify-results":
            return modeling.verify_results(a.inputs, a.result, a.participant)
        case "smoke":
            from .smoke import run

            return run(a.output, a.participant, a.device)
        case "budget":
            match a.budget_action:
                case "status":
                    return budget.status(a.ledger)
                case "record":
                    budget.record(a.ledger, a.stage, a.gpu_hours, a.note)
                    return budget.status(a.ledger)
                case "reconcile":
                    budget.reconcile(a.ledger, a.id, a.gpu_hours)
                    return budget.status(a.ledger)
                case "run":
                    command = a.command[1:] if a.command[:1] == ["--"] else a.command
                    raise SystemExit(budget.run(a.ledger, a.stage, a.gpus, a.hours, command))
    raise ValueError("Unknown action")
