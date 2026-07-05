from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
import torchaudio
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from higgs_local.training_utils import normalize_transcript
from higgs_local.v3_training_loss import compute_v3_audio_code_loss


SUPPORTED_TASKS = {"single_speaker_smart_voice", "zero_shot_voice_cloning"}


def _load_samples(data_dir: Path) -> list[dict]:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = metadata.get("samples") or []
    if not samples:
        raise RuntimeError(f"No samples in metadata: {metadata_path}")
    return samples


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


def training_progress_line(prefix: str, step: int, total: int, loss: float, started_at: float, suffix: str = "") -> str:
    total = max(int(total), 1)
    step = max(int(step), 0)
    elapsed = max(time.time() - started_at, 1e-6)
    speed = step / elapsed if step else 0.0
    remaining = (total - step) / speed if speed > 0 else 0.0
    pct = 100.0 * min(step, total) / total
    width = 28
    filled = int(width * min(step, total) / total)
    bar = "█" * filled + "░" * (width - filled)
    extra = f" {suffix}" if suffix else ""
    return (
        f"{prefix} |{bar}| {pct:5.1f}% "
        f"optimizer_step={step}/{total} loss={loss:.5f} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)} "
        f"speed={speed:.3f} step/s{extra}"
    )


def _load_waveform(path: Path, target_sr: int = 24000) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), int(sr)


def _sample_paths(data_dir: Path, sample: dict) -> tuple[Path, Path]:
    audio = data_dir / sample.get("audio_file", "")
    transcript = data_dir / sample.get("transcript_file", "")
    if not audio.exists():
        raise FileNotFoundError(f"audio file not found: {audio}")
    if not transcript.exists():
        raise FileNotFoundError(f"transcript file not found: {transcript}")
    return audio, transcript


def _reference_for_sample(data_dir: Path, sample: dict, task_type: str) -> tuple[torch.Tensor | None, int | None, str | None]:
    if task_type != "zero_shot_voice_cloning":
        return None, None, None
    ref_audio = sample.get("ref_audio_file")
    if not ref_audio:
        raise ValueError(f"zero_shot_voice_cloning sample has no ref_audio_file: {sample.get('id', 'unknown')}")
    ref_path = data_dir / ref_audio
    ref_wav, ref_sr = _load_waveform(ref_path)
    return ref_wav, ref_sr, sample.get("ref_transcript") or ""


def _apply_lora(model, rank: int, alpha: int, dropout: float):
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.model = get_peft_model(model.model, config)
    return model


def _load_lora(model, adapter_dir: Path):
    print(f"[v3-train] Resuming Qwen3 LoRA adapter from {adapter_dir}", flush=True)
    model.model = PeftModel.from_pretrained(model.model, str(adapter_dir), is_trainable=True)
    return model


def _trainable_parameter_report(model) -> str:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    pct = 100.0 * trainable / max(total, 1)
    return f"trainable params: {trainable:,} / {total:,} ({pct:.4f}%)"


def save_training_state(out_dir: Path, optimizer, scheduler, optimizer_step: int, micro_step: int, args: argparse.Namespace) -> None:
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "optimizer_step": int(optimizer_step),
        "micro_step": int(micro_step),
        "step_semantics": "optimizer_step",
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "args": vars(args),
    }
    torch.save(state, out_dir / "trainer_state.pt")


def load_training_state(checkpoint_dir: Path, optimizer, scheduler=None) -> tuple[int, int]:
    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.exists():
        print(f"[v3-train] No trainer_state.pt found in {checkpoint_dir}; adapter/model weights only.", flush=True)
        return 0, 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    optimizer_step = int(state.get("optimizer_step", state.get("global_step", 0)))
    micro_step = int(state.get("micro_step", 0))
    print(
        f"[v3-train] Restored optimizer/scheduler/RNG optimizer_step={optimizer_step} micro_step={micro_step} from {state_path}",
        flush=True,
    )
    return optimizer_step, micro_step


def generate_eval_audio(model, tokenizer, text: str, out_wav: Path, max_new_tokens: int = 1000) -> tuple[int, np.ndarray]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        audio = model.generate_speech(
            text.strip() or "This is a Higgs V3 training preview.",
            tokenizer,
            max_new_tokens=int(max_new_tokens),
            temperature=0.8,
            top_p=0.95,
            top_k=50,
        )
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    audio_np = audio.detach().float().cpu().numpy().astype(np.float32)
    sr = int(getattr(model.config, "sample_rate", 24000))
    sf.write(str(out_wav), audio_np, sr)
    if was_training:
        model.train()
    return sr, audio_np


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dir = Path(args.train_data_dir)
    eval_dir = Path(args.eval_data_dir) if args.eval_data_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_dir = Path(args.resume_checkpoint) if args.resume_checkpoint else None

    if args.task_type not in SUPPORTED_TASKS:
        raise ValueError(f"Custom V3 trainer supports {sorted(SUPPORTED_TASKS)}, got {args.task_type}")

    train_samples = _load_samples(train_dir)
    eval_samples = _load_samples(eval_dir) if eval_dir else []
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_eval_samples > 0:
        eval_samples = eval_samples[: args.max_eval_samples]

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and args.bf16 else torch.float32
    torch.set_float32_matmul_precision("high")
    print(f"[v3-train] device={device} dtype={dtype}", flush=True)
    print(f"[v3-train] train={len(train_samples)} eval={len(eval_samples)} task={args.task_type}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, dtype=dtype)
    model.config.audio_tokenizer_id = args.audio_tokenizer_path
    model = model.to(device)
    model.train()
    model.get_audio_codec()

    if args.freeze_audio_head:
        for param in model.audio_head.parameters():
            param.requires_grad_(False)
        for param in model.audio_embedding.parameters():
            param.requires_grad_(False)
    if args.use_lora:
        resume_adapter = resume_dir / "qwen3_lora" if resume_dir else None
        if resume_adapter and resume_adapter.exists():
            model = _load_lora(model, resume_adapter)
        else:
            model = _apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    print(f"[v3-train] {_trainable_parameter_report(model)}", flush=True)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    optimizer_step = 0
    micro_step = 0
    if resume_dir:
        optimizer_step, micro_step = load_training_state(resume_dir, optimizer, scheduler)
    optimizer.zero_grad(set_to_none=True)
    batches_per_epoch = math.ceil(len(train_samples) / max(args.batch_size, 1))
    total_updates = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(batches_per_epoch * args.num_train_epochs / max(args.gradient_accumulation_steps, 1))
    )
    total_updates = max(total_updates, optimizer_step)
    writer = None
    if args.enable_eval_audio or eval_samples:
        from torch.utils.tensorboard import SummaryWriter

        tb_run_dir = output_dir / "tensorboard" / f"{time.strftime('%Y%m%d_%H%M%S')}_step{optimizer_step:07d}"
        writer = SummaryWriter(log_dir=str(tb_run_dir))
        print(f"[v3-train] TensorBoard run: {tb_run_dir}", flush=True)
    print(
        f"[v3-train] start_optimizer_step={optimizer_step} max_optimizer_steps={total_updates} "
        f"batch={args.batch_size} grad_accum={args.gradient_accumulation_steps} batches_per_epoch={batches_per_epoch}",
        flush=True,
    )
    pbar = tqdm(total=total_updates, initial=optimizer_step, desc="[v3-train] updates", unit="step")
    running_loss = 0.0
    running_batches = 0
    last_stats = {}
    last_eval_audio_step = -1
    train_started_at = time.time()

    while optimizer_step < total_updates:
        random.shuffle(train_samples)
        batch_size = max(args.batch_size, 1)
        accum_batches = 0
        for batch_start in range(0, len(train_samples), batch_size):
            if optimizer_step >= total_updates:
                break
            batch = train_samples[batch_start : batch_start + batch_size]
            is_epoch_tail = batch_start + batch_size >= len(train_samples)
            batch_loss = None
            for sample in batch:
                audio_path, transcript_path = _sample_paths(train_dir, sample)
                text = normalize_transcript(transcript_path.read_text(encoding="utf-8", errors="ignore"), convert_v2_tags=False)
                wav, sr = _load_waveform(audio_path, target_sr=int(model.config.sample_rate))
                ref_wav, ref_sr, ref_text = _reference_for_sample(train_dir, sample, args.task_type)
                loss, stats = compute_v3_audio_code_loss(
                    model,
                    tokenizer,
                    text=text,
                    target_audio=wav,
                    target_sample_rate=sr,
                    reference_audio=ref_wav,
                    reference_sample_rate=ref_sr,
                    reference_text=ref_text,
                )
                last_stats = stats
                batch_loss = loss if batch_loss is None else batch_loss + loss
            batch_loss = batch_loss / max(len(batch), 1)
            (batch_loss / max(args.gradient_accumulation_steps, 1)).backward()
            running_loss += float(batch_loss.detach().cpu())
            running_batches += 1
            micro_step += 1
            accum_batches += 1
            if accum_batches < max(args.gradient_accumulation_steps, 1) and not is_epoch_tail:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            accum_batches = 0
            pbar.update(1)
            if optimizer_step % max(args.logging_steps, 1) == 0:
                loss_value = running_loss / max(running_batches, 1)
                print(
                    training_progress_line(
                        "[v3-train]",
                        optimizer_step,
                        total_updates,
                        loss_value,
                        train_started_at,
                        suffix=f"stats={last_stats}",
                    ),
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar("train/loss", loss_value, optimizer_step)
                running_loss = 0.0
                running_batches = 0
            if eval_samples and args.eval_steps > 0 and optimizer_step % args.eval_steps == 0:
                eval_loss = evaluate(model, tokenizer, eval_dir, eval_samples, args)
                print(f"[v3-train] optimizer_step={optimizer_step} eval_loss={eval_loss:.5f}", flush=True)
                if writer is not None:
                    writer.add_scalar("eval/loss", eval_loss, optimizer_step)
            if args.enable_eval_audio and args.eval_steps > 0 and optimizer_step % args.eval_steps == 0:
                try:
                    wav_path = output_dir / "eval_audio" / f"step_{optimizer_step:07d}.wav"
                    sr, audio = generate_eval_audio(model, tokenizer, args.eval_text, wav_path, args.eval_audio_max_new_tokens)
                    last_eval_audio_step = optimizer_step
                    print(f"[v3-train] optimizer_step={optimizer_step} eval_audio={wav_path}", flush=True)
                    if writer is not None:
                        writer.add_audio("eval/generated_audio", torch.from_numpy(audio).unsqueeze(0), optimizer_step, sample_rate=sr)
                except Exception as exc:
                    print(f"[v3-train] eval audio generation failed at optimizer_step {optimizer_step}: {exc}", flush=True)
            if args.save_steps > 0 and optimizer_step % args.save_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{optimizer_step}"
                save_checkpoint(model, tokenizer, checkpoint_dir, args.use_lora)
                save_training_state(checkpoint_dir, optimizer, scheduler, optimizer_step, micro_step, args)
                save_checkpoint(model, tokenizer, output_dir, args.use_lora)
                save_training_state(output_dir, optimizer, scheduler, optimizer_step, micro_step, args)
            if stop_requested():
                print(f"[v3-train] stop file detected at optimizer_step={optimizer_step}; saving final state.", flush=True)
                save_checkpoint(model, tokenizer, output_dir, args.use_lora)
                save_training_state(output_dir, optimizer, scheduler, optimizer_step, micro_step, args)
                if writer is not None:
                    writer.flush()
                    writer.close()
                pbar.close()
                print(f"[v3-train] Graceful stop saved to {output_dir}", flush=True)
                return

    pbar.close()
    save_checkpoint(model, tokenizer, output_dir, args.use_lora)
    save_training_state(output_dir, optimizer, scheduler, optimizer_step, micro_step, args)
    if args.enable_eval_audio and last_eval_audio_step != optimizer_step:
        try:
            wav_path = output_dir / "eval_audio" / f"step_{optimizer_step:07d}_final.wav"
            sr, audio = generate_eval_audio(model, tokenizer, args.eval_text, wav_path, args.eval_audio_max_new_tokens)
            print(f"[v3-train] optimizer_step={optimizer_step} final_eval_audio={wav_path}", flush=True)
            if writer is not None:
                writer.add_audio("eval/generated_audio_final", torch.from_numpy(audio).unsqueeze(0), optimizer_step, sample_rate=sr)
        except Exception as exc:
            print(f"[v3-train] final eval audio generation failed at optimizer_step {optimizer_step}: {exc}", flush=True)
    if writer is not None:
        writer.flush()
        writer.close()


@torch.no_grad()
def evaluate(model, tokenizer, eval_dir: Path, eval_samples: list[dict], args: argparse.Namespace) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for sample in eval_samples:
        audio_path, transcript_path = _sample_paths(eval_dir, sample)
        text = normalize_transcript(transcript_path.read_text(encoding="utf-8", errors="ignore"))
        wav, sr = _load_waveform(audio_path, target_sr=int(model.config.sample_rate))
        ref_wav, ref_sr, ref_text = _reference_for_sample(eval_dir, sample, args.task_type)
        loss, _ = compute_v3_audio_code_loss(
            model,
            tokenizer,
            text=text,
            target_audio=wav,
            target_sample_rate=sr,
            reference_audio=ref_wav,
            reference_sample_rate=ref_sr,
            reference_text=ref_text,
        )
        losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(model, tokenizer, output_dir: Path, lora_only: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if lora_only and hasattr(model.model, "save_pretrained"):
        model.model.save_pretrained(str(output_dir / "qwen3_lora"))
        tokenizer.save_pretrained(str(output_dir))
        (output_dir / "adapter_note.txt").write_text(
            "Custom Higgs V3 adapter: load the base V3 model, then attach qwen3_lora to model.model.\n",
            encoding="utf-8",
        )
    else:
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    print(f"[v3-train] saved: {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom Higgs V3 audio-code trainer")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--audio_tokenizer_path", required=True)
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--eval_data_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_type", default="single_speaker_smart_voice", choices=sorted(SUPPORTED_TASKS))
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--freeze_audio_head", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--enable_eval_audio", action="store_true")
    parser.add_argument("--eval_text", default="This is my Higgs V3 voice evolution during training.")
    parser.add_argument("--eval_audio_max_new_tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    started = time.time()
    train(parse_args())
    print(f"[v3-train] done in {time.time() - started:.1f}s", flush=True)
