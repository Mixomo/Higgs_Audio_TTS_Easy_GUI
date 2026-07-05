from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_samples(dataset_dir: str | Path) -> tuple[Path, list[dict]]:
    root = Path(dataset_dir)
    metadata = root / "metadata.json"
    if not metadata.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata}")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    samples = payload.get("samples") or []
    if not samples:
        raise ValueError(f"No samples found in {metadata}")
    return root, samples


def stop_requested() -> bool:
    stop_file = os.environ.get("HIGGS_TRAINING_STOP_FILE", "")
    return bool(stop_file and Path(stop_file).exists())


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def training_progress_line(prefix: str, step: int, total: int, loss: float, started_at: float) -> str:
    total = max(int(total), 1)
    step = max(int(step), 0)
    elapsed = max(time.time() - started_at, 1e-6)
    speed = step / elapsed if step else 0.0
    remaining = (total - step) / speed if speed > 0 else 0.0
    pct = 100.0 * min(step, total) / total
    width = 28
    filled = int(width * min(step, total) / total)
    bar = "█" * filled + "░" * (width - filled)
    return (
        f"{prefix} |{bar}| {pct:5.1f}% "
        f"optimizer_step={step}/{total} loss={loss:.4f} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)} "
        f"speed={speed:.3f} step/s"
    )


def make_conversation(dataset_root: Path, sample: dict) -> list[dict]:
    transcript = (dataset_root / sample["transcript_file"]).read_text(encoding="utf-8", errors="ignore").strip()
    audio_path = dataset_root / sample["audio_file"]
    return [
        {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
        {"role": "scene", "content": [{"type": "text", "text": sample.get("scene") or "Audio is recorded from a quiet room."}]},
        {"role": "user", "content": [{"type": "text", "text": transcript}]},
        {"role": "assistant", "content": [{"type": "audio", "url": str(audio_path)}]},
    ]


def prepare_inputs(processor, model, conversation):
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        processor_kwargs={"sampling_rate": 24000, "output_labels": True},
        return_tensors="pt",
    )
    return {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}


def evaluate(model, processor, dataset_root: Path, samples: list[dict], limit: int = 1) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for sample in samples[:limit]:
            inputs = prepare_inputs(processor, model, make_conversation(dataset_root, sample))
            outputs = model(**inputs)
            if outputs.loss is not None:
                losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(model, processor, out_dir: Path, lora_only: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if lora_only and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out_dir / "lora_adapter"))
    else:
        model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))


def save_training_state(out_dir: Path, optimizer, scheduler, global_step: int, args) -> None:
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": int(global_step),
        "step_semantics": "optimizer_step",
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "args": vars(args),
    }
    torch.save(state, out_dir / "trainer_state.pt")


def load_training_state(checkpoint_dir: Path, optimizer, scheduler=None) -> int:
    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.exists():
        print(f"[v2-hf-trainer] No trainer_state.pt found in {checkpoint_dir}; model weights only.", flush=True)
        return 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    global_step = int(state.get("global_step", 0))
    if state.get("step_semantics") != "optimizer_step":
        scheduler_step = int((state.get("scheduler") or {}).get("last_epoch") or 0)
        if scheduler_step and scheduler_step != global_step:
            print(
                f"[v2-hf-trainer] Legacy checkpoint detected: global_step={global_step} was a micro-batch count; "
                f"using scheduler.last_epoch={scheduler_step} as optimizer_step.",
                flush=True,
            )
            global_step = scheduler_step
    print(f"[v2-hf-trainer] Restored optimizer/scheduler/RNG/optimizer_step={global_step} from {state_path}", flush=True)
    return global_step


def generate_eval_audio(model, processor, text: str, out_wav: Path, max_new_tokens: int = 1000) -> tuple[int, np.ndarray]:
    model.eval()
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
        {"role": "scene", "content": [{"type": "text", "text": "Audio is recorded from a quiet room."}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    with torch.no_grad():
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            processor_kwargs={"sampling_rate": 24000},
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decoded = processor.batch_decode(outputs)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        processor.save_audio(decoded, str(out_wav))
    audio, sr = sf.read(str(out_wav), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    model.train()
    return int(sr), np.asarray(audio, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Native Transformers Higgs Audio V2 single-speaker trainer.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--eval_data_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--enable_eval_audio", action="store_true")
    parser.add_argument("--eval_text", default="This is my voice evolution during training.")
    parser.add_argument("--eval_audio_max_new_tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_dir = Path(args.resume_checkpoint) if args.resume_checkpoint else None

    from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    print(f"[v2-hf-trainer] Loading processor/model from {args.model_path}", flush=True)
    processor_path = resume_dir if resume_dir and (resume_dir / "preprocessor_config.json").exists() else args.model_path
    model_path = resume_dir if resume_dir and not args.use_lora and (resume_dir / "config.json").exists() else args.model_path
    processor = AutoProcessor.from_pretrained(str(processor_path), device_map="auto", trust_remote_code=True)
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=dtype,
        use_text_head=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.train()

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        resume_adapter = resume_dir / "lora_adapter" if resume_dir else None
        if resume_adapter and resume_adapter.exists():
            from peft import PeftModel

            print(f"[v2-hf-trainer] Resuming LoRA adapter from {resume_adapter}", flush=True)
            model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
        else:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
        model.print_trainable_parameters()
    elif args.resume_checkpoint:
        print(
            "[v2-hf-trainer] resume_checkpoint is currently applied only to LoRA adapter checkpoints.",
            flush=True,
        )

    train_root, train_samples = load_samples(args.train_data_dir)
    eval_root, eval_samples = (None, [])
    if args.eval_data_dir:
        eval_root, eval_samples = load_samples(args.eval_data_dir)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    optimizer_step = load_training_state(resume_dir, optimizer, scheduler) if resume_dir else 0
    batches_per_epoch = math.ceil(len(train_samples) / max(args.batch_size, 1))
    total_steps_by_epochs = math.ceil(batches_per_epoch / max(args.gradient_accumulation_steps, 1)) * max(args.num_train_epochs, 1)
    target_steps = args.max_steps if args.max_steps > 0 else total_steps_by_epochs
    max_steps = max(target_steps, optimizer_step)
    running_loss = 0.0
    running_batches = 0
    writer = None
    if args.enable_eval_audio or eval_samples:
        from torch.utils.tensorboard import SummaryWriter

        tb_run_dir = out_dir / "tensorboard" / f"{time.strftime('%Y%m%d_%H%M%S')}_step{optimizer_step:07d}"
        writer = SummaryWriter(log_dir=str(tb_run_dir))
        print(f"[v2-hf-trainer] TensorBoard run: {tb_run_dir}", flush=True)

    print(
        f"[v2-hf-trainer] Starting single-speaker training: samples={len(train_samples)} "
        f"eval={len(eval_samples)} start_optimizer_step={optimizer_step} max_optimizer_steps={max_steps} "
        f"batch={args.batch_size} grad_accum={args.gradient_accumulation_steps} "
        f"batches_per_epoch={batches_per_epoch}",
        flush=True,
    )
    optimizer.zero_grad(set_to_none=True)
    last_eval_audio_step = -1
    train_started_at = time.time()
    while optimizer_step < max_steps:
        random.shuffle(train_samples)
        batch_size = max(args.batch_size, 1)
        accum_batches = 0
        for batch_start in range(0, len(train_samples), batch_size):
            if optimizer_step >= max_steps:
                break
            batch = train_samples[batch_start : batch_start + batch_size]
            batch_loss = None
            for sample in batch:
                inputs = prepare_inputs(processor, model, make_conversation(train_root, sample))
                outputs = model(**inputs)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model forward did not return loss. Check output_labels=True support.")
                batch_loss = loss if batch_loss is None else batch_loss + loss
            batch_loss = batch_loss / max(len(batch), 1)
            (batch_loss / max(args.gradient_accumulation_steps, 1)).backward()
            running_loss += float(batch_loss.detach().cpu())
            running_batches += 1
            accum_batches += 1
            is_epoch_tail = batch_start + batch_size >= len(train_samples)
            if accum_batches >= max(args.gradient_accumulation_steps, 1) or is_epoch_tail:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                accum_batches = 0
            else:
                continue
            if optimizer_step % max(args.logging_steps, 1) == 0:
                loss_value = running_loss / max(running_batches, 1)
                print(
                    training_progress_line("[v2-train]", optimizer_step, max_steps, loss_value, train_started_at),
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar("train/loss", loss_value, optimizer_step)
                running_loss = 0.0
                running_batches = 0
            if eval_root is not None and eval_samples and optimizer_step % max(args.eval_steps, 1) == 0:
                eval_loss = evaluate(model, processor, eval_root, eval_samples)
                print(f"[v2-hf-trainer] optimizer_step={optimizer_step} eval_loss={eval_loss:.4f}", flush=True)
                if writer is not None:
                    writer.add_scalar("eval/loss", eval_loss, optimizer_step)
            if args.enable_eval_audio and optimizer_step % max(args.eval_steps, 1) == 0:
                try:
                    wav_path = out_dir / "eval_audio" / f"step_{optimizer_step:07d}.wav"
                    sr, audio = generate_eval_audio(
                        model,
                        processor,
                        args.eval_text,
                        wav_path,
                        max_new_tokens=args.eval_audio_max_new_tokens,
                    )
                    last_eval_audio_step = optimizer_step
                    print(f"[v2-hf-trainer] optimizer_step={optimizer_step} eval_audio={wav_path}", flush=True)
                    if writer is not None:
                        writer.add_audio("eval/generated_audio", torch.from_numpy(audio).unsqueeze(0), optimizer_step, sample_rate=sr)
                except Exception as exc:
                    print(f"[v2-hf-trainer] eval audio generation failed at optimizer_step {optimizer_step}: {exc}", flush=True)
            if optimizer_step % max(args.save_steps, 1) == 0:
                checkpoint_dir = out_dir / f"checkpoint-{optimizer_step}"
                save_checkpoint(model, processor, checkpoint_dir, args.use_lora)
                save_training_state(checkpoint_dir, optimizer, scheduler, optimizer_step, args)
                save_checkpoint(model, processor, out_dir, args.use_lora)
                save_training_state(out_dir, optimizer, scheduler, optimizer_step, args)
            if stop_requested():
                print(f"[v2-hf-trainer] stop file detected at optimizer_step={optimizer_step}; saving final state.", flush=True)
                save_checkpoint(model, processor, out_dir, args.use_lora)
                save_training_state(out_dir, optimizer, scheduler, optimizer_step, args)
                if writer is not None:
                    writer.flush()
                    writer.close()
                print(f"[v2-hf-trainer] Graceful stop saved to {out_dir}", flush=True)
                return 0
            if optimizer_step >= max_steps:
                break

    save_checkpoint(model, processor, out_dir, args.use_lora)
    save_training_state(out_dir, optimizer, scheduler, optimizer_step, args)
    if args.enable_eval_audio and last_eval_audio_step != optimizer_step:
        try:
            wav_path = out_dir / "eval_audio" / f"step_{optimizer_step:07d}_final.wav"
            sr, audio = generate_eval_audio(
                model,
                processor,
                args.eval_text,
                wav_path,
                max_new_tokens=args.eval_audio_max_new_tokens,
            )
            print(f"[v2-hf-trainer] optimizer_step={optimizer_step} final_eval_audio={wav_path}", flush=True)
            if writer is not None:
                writer.add_audio("eval/generated_audio_final", torch.from_numpy(audio).unsqueeze(0), optimizer_step, sample_rate=sr)
        except Exception as exc:
            print(f"[v2-hf-trainer] final eval audio generation failed at optimizer_step {optimizer_step}: {exc}", flush=True)
    if writer is not None:
        writer.flush()
        writer.close()
    print(f"[v2-hf-trainer] Saved to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
