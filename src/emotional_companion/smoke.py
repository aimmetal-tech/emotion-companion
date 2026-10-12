"""Offline end-to-end smoke test using a tiny random Qwen2, never a 7B download."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import file_digest, official_module, read_jsonl, write_json, write_jsonl
from .modeling import bundle, encode_sample, evaluate, export, require_empty, tokenize, versions
from .training import train


def make_tiny_model(output: Path):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    # Byte-level vocabulary supports Chinese and JSON without any remote tokenizer files.
    vocab = {token: index for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for token in ("<|im_start|>", "<|im_end|>", "<|pad|>"):
        vocab[token] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|im_start|>"],
        model_max_length=8192,
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
    )
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    return sum(p.numel() for p in model.parameters())


def make_samples():
    samples = []
    for index in range(1, 5):
        target = {
            "response_text": "I hear you. We can take this slowly.",
            "emotion_label": "anxiety",
            "user_profile": {"personality_traits": [], "interests": [], "style": ["brief"]},
            "memory_refs": [],
        }
        samples.append(
            {
                "id": f"test_{index:06d}_t3",
                "conversation_id": f"smoke_{index}",
                "target_user_turn_id": 3,
                "history": [
                    {"turn_id": 1, "role": "user", "content": "I have an exam tomorrow."},
                    {"turn_id": 2, "role": "assistant", "content": "How are you feeling?"},
                    {"turn_id": 3, "role": "user", "content": f"I am anxious. Test {index}."},
                ],
                "target": target,
            }
        )
    return samples


def run(output: Path, participant: Path, device: str):
    import torch
    from transformers import AutoTokenizer

    started = time.time()
    require_empty(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Inspect driver/WSL access or explicitly use --device cpu")
    cpu = device == "cpu"
    report = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if not cpu else None,
        "versions": versions(),
        "tests": {},
        "note": "Random tiny model tests execution only, not competition quality or 7B memory requirements.",
    }
    print(f"Local smoke test: {device}; no remote model weights", flush=True)
    model_path = output / "tiny-base"
    report["parameters"] = make_tiny_model(model_path)
    samples = make_samples()
    data = output / "data"
    write_jsonl(data / "train.jsonl", samples)
    write_jsonl(data / "dev_loss.jsonl", samples[:2])
    write_jsonl(data / "eval.jsonl", samples)
    tokens = output / "tokens"
    tokenize(data, model_path, tokens, participant, 2048)
    official = official_module(participant)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded = encode_sample(samples[0], tokenizer, official, 2048)
    if encoded is None or not any(v == -100 for v in encoded["labels"]):
        raise ValueError("Loss masking test failed")
    decoded_target = tokenizer.decode([v for v in encoded["labels"] if v != -100], skip_special_tokens=True)
    if samples[0]["target"]["response_text"] not in decoded_target:
        raise ValueError("Target was truncated or masked out")
    report["tests"]["tokenization_and_loss_masking"] = True
    config = json.loads(Path("configs/train_7b.json").read_text())
    config.update(
        max_length=2048,
        batch_size=1,
        gradient_accumulation_steps=1,
        lora_rank=4,
        lora_alpha=8,
        eval_steps=1,
        max_hours=0.1,
        shutdown_margin_seconds=20,
    )
    config_path = output / "train_config.json"
    write_json(config_path, config)
    run_path = output / "training"
    print("Train one step, save full checkpoint, then resume to step two", flush=True)
    train(model_path, tokens, run_path, config_path, max_steps=2, cpu_smoke=cpu, stop_after_steps=1)
    first = json.loads((run_path / "training_report.json").read_text())
    if first["global_step"] != 1 or not (run_path / "checkpoint-1/trainer_state.json").is_file():
        raise ValueError("First step did not save a resumable checkpoint")
    gc.collect()
    if not cpu:
        torch.cuda.empty_cache()
    train(model_path, tokens, run_path, config_path, max_steps=2, resume="latest", cpu_smoke=cpu)
    second = json.loads((run_path / "training_report.json").read_text())
    if second["global_step"] != 2 or second["steps_this_run"] != 1:
        raise ValueError("Checkpoint resume did not restore the correct step")
    report["tests"]["qlora_train_and_resume" if not cpu else "cpu_lora_train_and_resume"] = True
    report["peak_training_gpu_memory_gib"] = second["peak_gpu_memory_gib"]
    gc.collect()
    if not cpu:
        torch.cuda.empty_cache()
    adapter = run_path / "adapter"
    metrics = evaluate(
        model_path,
        adapter,
        data / "eval.jsonl",
        output / "evaluation",
        participant,
        quantized=not cpu,
        max_input_tokens=2048,
        max_new_tokens=8,
        cpu_smoke=cpu,
    )
    if not metrics["complete"] or metrics["evaluated_samples"] != 4:
        raise ValueError("Evaluation did not cover all samples")
    report["tests"]["adapter_generation_and_metrics"] = True
    gc.collect()
    if not cpu:
        torch.cuda.empty_cache()
    merged = output / "merged"
    export(model_path, adapter, merged, cpu_smoke=cpu)
    report["tests"]["bf16_merge_and_offline_reload"] = True
    packaged = output / "participant"
    bundle(merged, participant, packaged, f"local-smoke-{device}")
    config_path = packaged / "configs/inference_config.json"
    config = json.loads(config_path.read_text())
    config.update(
        backend="transformers", device_map="cpu" if cpu else "auto", torch_dtype="float32" if cpu else "bfloat16"
    )
    config["generation"]["max_new_tokens"] = 8
    write_json(config_path, config)
    if file_digest(packaged / "run_inference.py") != file_digest(participant / "run_inference.py"):
        raise ValueError("Official runner was modified during packaging")
    input_rows = []
    for i in range(100):
        sample = samples[i % 4]
        input_rows.append({"id": f"test_{i + 1:06d}_t3", "history": sample["history"]})
    inputs, results = output / "official-input.jsonl", output / "official-results"
    write_jsonl(inputs, input_rows)
    print("Run unmodified official Python entrypoint on 100 inputs", flush=True)
    subprocess.run(
        [sys.executable, str(packaged / "run_inference.py"), str(inputs), str(results)],
        check=True,
        timeout=300,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "42",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        },
    )
    predictions = read_jsonl(results / "submission.jsonl")
    performance = json.loads((results / "performance_report.json").read_text())
    if [p["id"] for p in predictions] != [s["id"] for s in input_rows]:
        raise ValueError("Official output IDs incomplete or out of order")
    if any(p != official.normalize_prediction(p["id"], p) for p in predictions):
        raise ValueError("Official outputs fail normalization contract")
    if (
        not performance["complete"]
        or performance["rounds"] != 100
        or performance["test_file_sha256"] != file_digest(inputs)
        or performance["submission_sha256"] != file_digest(results / "submission.jsonl")
    ):
        raise ValueError("Official timing/integrity report failed")
    report["tests"]["official_100_sample_offline_inference"] = True
    # start.sh requires the platform's conda_env. Test its syntax and contract locally,
    # but execute its Python target above without faking a platform conda environment.
    subprocess.run(["bash", "-n", str(packaged / "start.sh")], check=True)
    report["tests"]["official_shell_syntax"] = True
    report["platform_checks_remaining"] = [
        "conda_env activation",
        "vLLM startup",
        "two-GPU tensor parallelism",
        "7B training memory/throughput",
        "actual platform entry and image rules",
    ]
    report.update(
        passed=True,
        elapsed_seconds=time.time() - started,
        average_official_latency_ms=performance["average_latency_ms"],
    )
    write_json(output / "smoke_report.json", report)
    return report
