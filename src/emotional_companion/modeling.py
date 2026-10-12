"""Offline tokenization, GPU inference and unquantized adapter export."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import time
from contextlib import suppress
from pathlib import Path

from .common import file_digest, official_module, prompt_messages, read_jsonl, target_valid, write_json, write_jsonl
from .metrics import score


def versions():
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface-hub"):
        with suppress(importlib.metadata.PackageNotFoundError):
            result[name] = importlib.metadata.version(name)
    return result


def require_empty(path: Path):
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output must be empty: {path}")


def download(model_id: str, revision: str, output: Path):
    from huggingface_hub import HfApi, snapshot_download

    if (output / "download_manifest.json").exists():
        previous = json.loads((output / "download_manifest.json").read_text())
        if previous["model_id"] != model_id or previous["requested_revision"] != revision:
            raise ValueError("Model directory already has a different source/revision")
        commit = previous["commit"]
    else:
        require_empty(output)
        commit = HfApi().model_info(model_id, revision=revision).sha
        if not commit:
            raise ValueError("Hub did not resolve a commit")
        write_json(
            output / "download_manifest.json",
            {
                "model_id": model_id,
                "requested_revision": revision,
                "commit": commit,
            },
        )
    snapshot_download(
        repo_id=model_id,
        revision=commit,
        local_dir=output,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "README.md", "LICENSE"],
    )
    print(f"Downloaded {model_id}@{commit} to {output}")


def shortened_history(history: list[dict]):
    # Remove the earliest complete exchange; never remove the latest user message.
    for index in range(1, len(history)):
        if history[index]["role"] == "user":
            return history[index:]
    return None


def encode_sample(sample: dict, tokenizer, official, max_length: int):
    current = {**sample, "history": list(sample["history"])}
    target_text = json.dumps(sample["target"], ensure_ascii=False, separators=(",", ":"))
    removed = 0
    while True:
        messages = prompt_messages(current, official)
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template([*messages, {"role": "assistant", "content": target_text}], tokenize=True)
        if full[: len(prefix)] != prefix:
            raise ValueError("Chat template does not have a stable assistant prefix; refusing incorrect loss masking")
        if len(full) <= max_length:
            return {
                "id": sample["id"],
                "input_ids": full,
                "attention_mask": [1] * len(full),
                "labels": [-100] * len(prefix) + full[len(prefix) :],
                "removed_turns": removed,
                "target_tokens": len(full) - len(prefix),
            }
        shorter = shortened_history(current["history"])
        if shorter is None:
            return None
        removed += len(current["history"]) - len(shorter)
        current["history"] = shorter


def tokenize(data: Path, model: Path, output: Path, participant: Path, max_length: int):
    from transformers import AutoTokenizer

    if max_length not in (2048, 4096):
        raise ValueError("Plan supports max_length=2048 or 4096")
    require_empty(output)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    official = official_module(participant)
    manifest = {
        "max_length": max_length,
        "model_path": str(model.resolve()),
        "model_config_sha256": file_digest(model / "config.json"),
        "participant_sha256": file_digest(participant / "run_inference.py"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "versions": versions(),
        "subsets": {},
    }
    for name in ("train", "dev_loss"):
        samples = read_jsonl(data / f"{name}.jsonl")
        rows, skipped, shortened = [], [], []
        for sample in samples:
            row = encode_sample(sample, tokenizer, official, max_length)
            if row is None:
                skipped.append(sample["id"])
            else:
                rows.append(row)
                if row["removed_turns"]:
                    shortened.append(sample["id"])
        if not rows:
            raise ValueError(f"No encodable samples in {name}")
        path = output / f"{name}.tokens.jsonl"
        write_jsonl(path, rows)
        manifest["subsets"][name] = {
            "samples": len(rows),
            "skipped_ids": skipped,
            "shortened_ids": shortened,
            "sha256": file_digest(path),
            "source_sha256": file_digest(data / f"{name}.jsonl"),
            "tokens": sum(len(r["input_ids"]) for r in rows),
            "target_tokens": sum(r["target_tokens"] for r in rows),
        }
        print(f"{name}: {len(rows)} samples, shortened={len(shortened)}, skipped={len(skipped)}")
    write_json(output / "tokenization.json", manifest)


def load_inference(model_path: Path, adapter: Path | None, quantized: bool, cpu_smoke: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() and not cpu_smoke:
        raise RuntimeError("GPU inference requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    options = {
        "torch_dtype": torch.float32 if cpu_smoke else torch.bfloat16,
        "device_map": "cpu" if cpu_smoke else "auto",
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if quantized and not cpu_smoke:
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **options)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    return model, tokenizer


def evaluate(
    model_path: Path,
    adapter: Path | None,
    data: Path,
    output: Path,
    participant: Path,
    quantized: bool,
    max_input_tokens: int,
    max_new_tokens: int,
    cpu_smoke: bool = False,
):
    import torch

    if max_input_tokens <= 0 or max_new_tokens <= 0:
        raise ValueError("Token limits must be positive")
    require_empty(output)
    official = official_module(participant)
    gold = read_jsonl(data)
    if not gold:
        raise ValueError("Empty evaluation subset")
    model, tokenizer = load_inference(model_path, adapter, quantized, cpu_smoke)
    output.mkdir(parents=True, exist_ok=True)
    raw_predictions, predictions, latencies = [], [], []
    shortened = []
    skipped = []
    raw_valid_count = 0
    started = time.time()
    deadline = float(os.environ.get("COMPANION_BUDGET_DEADLINE", "inf"))
    with (
        (output / "submission.jsonl").open("w", encoding="utf-8") as handle,
        (output / "raw.jsonl").open("w", encoding="utf-8") as raw_handle,
    ):
        for sample in gold:
            if time.time() >= deadline - 30:
                break
            current = {**sample, "history": list(sample["history"])}
            while True:
                prompt = tokenizer.apply_chat_template(
                    prompt_messages(current, official), tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt, return_tensors="pt")
                if inputs["input_ids"].shape[1] <= max_input_tokens:
                    break
                shorter = shortened_history(current["history"])
                if shorter is None:
                    break
                current["history"] = shorter
            if len(current["history"]) != len(sample["history"]):
                shortened.append(sample["id"])
            if inputs["input_ids"].shape[1] > max_input_tokens:
                skipped.append(sample["id"])
                decoded = ""
                elapsed = 0.0
            else:
                inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}
                if not cpu_smoke:
                    torch.cuda.synchronize()
                step_started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                decoded = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                if not cpu_smoke:
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - step_started) * 1000
            try:
                raw = json.loads(decoded)
                valid = target_valid(raw, official) and raw["memory_refs"] == []
            except (ValueError, TypeError):
                raw, valid = {}, False
            raw_valid_count += int(valid)
            parsed, failed = official.parse_prediction(sample["id"], decoded)
            predictions.append(parsed)
            raw_predictions.append({**raw, "id": sample["id"]} if isinstance(raw, dict) else {"id": sample["id"]})
            latencies.append(elapsed)
            handle.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            handle.flush()
            raw_handle.write(
                json.dumps(
                    {
                        "id": sample["id"],
                        "decoded": decoded,
                        "strict_valid": valid,
                        "parse_failed": failed,
                        "latency_ms": elapsed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_handle.flush()
            print(f"[{len(predictions)}/{len(gold)}] {sample['id']}: {elapsed:.0f} ms", flush=True)
    report = score(gold, raw_predictions, official)
    report.update(
        field_compliance_rate=raw_valid_count / len(gold),
        compliance_scope="strict raw JSON before official normalization",
        gold_sha256=file_digest(data),
        predictions_sha256=file_digest(output / "submission.jsonl"),
        participant_sha256=file_digest(participant / "run_inference.py"),
        model_path=str(model_path.resolve()),
        adapter_path=str(adapter.resolve()) if adapter else None,
        quantized=quantized,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        shortened_ids=shortened,
        skipped_ids=skipped,
        complete=len(predictions) == len(gold),
        evaluated_samples=len(predictions),
        elapsed_seconds=time.time() - started,
        versions=versions(),
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        average_latency_ms=sum(latencies) / len(latencies) if latencies else None,
        p95_latency_ms=official.percentile95(latencies) if latencies else None,
    )
    write_json(output / "metrics.json", report)
    return report


def export(model_path: Path, adapter: Path, output: Path, cpu_smoke: bool = False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_empty(output)
    # Reload original BF16 weights; do not merge into the 4-bit training weights.
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu" if cpu_smoke else "auto",
        local_files_only=True,
        trust_remote_code=False,
    )
    merged = PeftModel.from_pretrained(base, adapter, local_files_only=True).merge_and_unload(safe_merge=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenizer.save_pretrained(output)
    write_json(
        output / "export_manifest.json",
        {
            "base_model": str(model_path.resolve()),
            "adapter": str(adapter.resolve()),
            "dtype": "bfloat16",
            "versions": versions(),
            "files": {p.name: file_digest(p) for p in output.iterdir() if p.is_file()},
        },
    )
    del merged, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Check that the exported standalone directory can load offline, including every weight shard.
    check = AutoModelForCausalLM.from_pretrained(
        output, device_map="cpu", torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False
    )
    del check
    AutoTokenizer.from_pretrained(output, local_files_only=True, trust_remote_code=False)
    print(f"Export verified offline: {output}")


def bundle(model: Path, participant: Path, output: Path, hardware_label: str):
    require_empty(output)
    if not (model / "export_manifest.json").is_file():
        raise ValueError("Use the verified export directory")
    needed = sum(p.stat().st_size for p in model.rglob("*") if p.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < needed * 1.1:
        raise ValueError("Not enough disk space to copy model into the submission bundle")
    shutil.copytree(participant, output, dirs_exist_ok=True, ignore=shutil.ignore_patterns("model", "__pycache__"))
    shutil.copytree(model, output / "model")
    config_path = output / "configs/inference_config.json"
    config = json.loads(config_path.read_text())
    config.update(backend="openai_compatible", model_path="./model", hardware_label=hardware_label)
    write_json(config_path, config)
    write_json(
        output / "bundle_manifest.json",
        {
            "participant_source": str(participant.resolve()),
            "original_start_sha256": file_digest(participant / "start.sh"),
            "original_runner_sha256": file_digest(participant / "run_inference.py"),
            "export_manifest_sha256": file_digest(model / "export_manifest.json"),
            "python_hash_seed": "42",
            "platform_entry": "/root/participant/start.sh",
        },
    )


def verify_results(inputs: Path, result: Path, participant: Path):
    official = official_module(participant)
    samples, predictions = read_jsonl(inputs), read_jsonl(result / "submission.jsonl")
    if [s["id"] for s in samples] != [p["id"] for p in predictions]:
        raise ValueError("Predictions must cover every input ID once, in original order")
    if any(
        not target_valid({k: v for k, v in p.items() if k != "id"}, official) or p["memory_refs"] != []
        for p in predictions
    ):
        raise ValueError("Submission contains empty/invalid fields or unexpected memory references")
    report = json.loads((result / "performance_report.json").read_text())
    if len(samples) < 100 or report["required_rounds"] != 100 or report["rounds"] != 100 or not report["complete"]:
        raise ValueError("Need >=100 inputs and a complete official 100-round timing report")
    if (
        report["samples"] != len(samples)
        or report["parse_failure_count"] != 0
        or report["test_file_sha256"] != file_digest(inputs)
        or report["submission_sha256"] != file_digest(result / "submission.jsonl")
    ):
        raise ValueError("Official report, predictions and inputs do not match")
    label = report["hardware_label"]
    if not label or label == "fill-in-before-official-run":
        raise ValueError("Fill in hardware_label")
    return {
        "passed": True,
        "samples": len(samples),
        "average_latency_ms": report["average_latency_ms"],
        "p95_latency_ms": report["p95_latency_ms"],
    }
