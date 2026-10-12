"""Single-GPU QLoRA Trainer with completion-only labels and resumable deadlines."""

from __future__ import annotations

import json
import math
import os
import signal
import time
from pathlib import Path

from .common import file_digest, read_jsonl, write_json
from .modeling import versions


def train(
    model_path: Path,
    data: Path,
    output: Path,
    config_path: Path,
    max_steps: int | None = None,
    max_length: int | None = None,
    resume: str | None = None,
    cpu_smoke: bool = False,
    stop_after_steps: int | None = None,
):
    started = time.time()
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    if not cpu_smoke and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Single-GPU plan: expose exactly one CUDA GPU (CUDA_VISIBLE_DEVICES=0)")
    if not cpu_smoke and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required; plan targets RTX 4090")
    config = json.loads(config_path.read_text())
    tokenization = json.loads((data / "tokenization.json").read_text())
    if max_length is not None:
        config["max_length"] = max_length
    if tokenization["max_length"] != config["max_length"]:
        raise ValueError("Retokenize data for the chosen max_length before training")
    if tokenization["model_config_sha256"] != file_digest(model_path / "config.json"):
        raise ValueError("Tokenized data and model configuration differ")
    if tokenization["python_hash_seed"] != os.environ.get("PYTHONHASHSEED"):
        raise ValueError("Tokenization/training PYTHONHASHSEED mismatch")
    for name in ("train", "dev_loss"):
        if tokenization["subsets"][name]["sha256"] != file_digest(data / f"{name}.tokens.jsonl"):
            raise ValueError(f"Tokenized {name} was modified after manifest creation")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Nonempty run directory requires --resume")
    if resume and not (output / "resolved_config.json").exists():
        raise ValueError("Resume requires an existing run and resolved configuration")
    resolved = {
        "config": config,
        "tokenization_sha256": file_digest(data / "tokenization.json"),
        "model_path": str(model_path.resolve()),
        "max_steps": max_steps,
        "versions": versions(),
    }
    if resume:
        previous = json.loads((output / "resolved_config.json").read_text())
        if any(previous[k] != resolved[k] for k in ("config", "tokenization_sha256", "model_path", "max_steps")):
            raise ValueError("Resume configuration/data changed; start a separate run instead")
    write_json(output / "resolved_config.json", resolved)
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=None if cpu_smoke else quantization,
        torch_dtype=torch.float32 if cpu_smoke else torch.bfloat16,
        device_map="cpu" if cpu_smoke else {"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if not cpu_smoke:
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, path):
            self.rows = read_jsonl(path)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return {k: self.rows[index][k] for k in ("input_ids", "attention_mask", "labels")}

    train_data = TokenDataset(data / "train.tokens.jsonl")
    dev_data = TokenDataset(data / "dev_loss.tokens.jsonl")

    def collate(features):
        length = max(len(f["input_ids"]) for f in features)
        pads = {"input_ids": tokenizer.pad_token_id, "attention_mask": 0, "labels": -100}
        return {
            key: torch.tensor([f[key] + [pad] * (length - len(f[key])) for f in features]) for key, pad in pads.items()
        }

    stop_requested = False

    def request_stop(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True

    margin = config["shutdown_margin_seconds"]
    deadline = (
        min(started + config["max_hours"] * 3600, float(os.environ.get("COMPANION_BUDGET_DEADLINE", "inf"))) - margin
    )
    step_times = []
    global_start_step = 0
    steps_per_epoch = math.ceil(len(train_data) / (config["batch_size"] * config["gradient_accumulation_steps"]))

    class DeadlineCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            nonlocal global_start_step
            global_start_step = state.global_step

        def on_step_end(self, args, state, control, **kwargs):
            step_times.append(time.time())
            if len(step_times) >= 2:
                seconds_per_step = (step_times[-1] - step_times[0]) / (len(step_times) - 1)
                if len(step_times) == 50 or len(step_times) % 250 == 0:
                    write_json(
                        output / "throughput.json",
                        {
                            "measured_optimizer_steps": len(step_times),
                            "seconds_per_optimizer_step": seconds_per_step,
                            "estimated_epoch_hours": steps_per_epoch * seconds_per_step / 3600,
                            "steps_per_epoch": steps_per_epoch,
                            "estimated_remaining_steps_before_deadline": max(
                                0, int((deadline - time.time()) / seconds_per_step)
                            ),
                            "peak_gpu_memory_gib": 0 if cpu_smoke else torch.cuda.max_memory_allocated() / 1024**3,
                        },
                    )
            if stop_requested or time.time() >= deadline:
                control.should_training_stop = True
                control.should_save = True
                # Preserve a resumable last checkpoint without a costly last evaluation.
                control.should_evaluate = False
            elif stop_after_steps is not None and state.global_step - global_start_step >= stop_after_steps:
                control.should_training_stop = True
                control.should_save = True
                control.should_evaluate = True
            return control

    arguments = TrainingArguments(
        output_dir=str(output),
        seed=config["seed"],
        data_seed=config["seed"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        bf16=not cpu_smoke,
        use_cpu=cpu_smoke,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["eval_steps"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=collate,
        processing_class=tokenizer,
        callbacks=[DeadlineCallback()],
    )
    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = trainer.train(resume_from_checkpoint=True if resume == "latest" else resume)
        # Best evaluated checkpoint is selected by Trainer; retain latest full optimizer checkpoint too.
        trainer.save_model(str(output / "adapter"))
        tokenizer.save_pretrained(output / "adapter")
        trainer.save_state()
        write_json(
            output / "training_report.json",
            {
                "metrics": result.metrics,
                "global_step": trainer.state.global_step,
                "steps_this_run": trainer.state.global_step - global_start_step,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_eval_loss": trainer.state.best_metric,
                "elapsed_hours": (time.time() - started) / 3600,
                "stopped_for_deadline_or_signal": stop_requested or time.time() >= deadline,
                "peak_gpu_memory_gib": 0 if cpu_smoke else torch.cuda.max_memory_allocated() / 1024**3,
            },
        )
    except torch.cuda.OutOfMemoryError:
        write_json(
            output / "failure.json",
            {
                "reason": "CUDA out of memory",
                "action": "Retokenize at 2048 and start a new run; do not resume mismatched data",
            },
        )
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
