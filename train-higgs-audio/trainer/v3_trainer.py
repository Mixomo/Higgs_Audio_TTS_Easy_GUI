from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
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
from higgs_local.v3_training_loss import V3Example, compute_v3_batch_loss


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


def training_progress_line(
    prefix: str, step: int, total: int, loss: float, started_at: float, suffix: str = "", start_step: int = 0
) -> str:
    total = max(int(total), 1)
    step = max(int(step), 0)
    elapsed = max(time.time() - started_at, 1e-6)
    done = max(step - int(start_step), 0)  # steps of this session (a resumed run starts mid-way)
    speed = done / elapsed if done else 0.0
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


def _audio_code_cache_path(cache_dir: Path, audio_path: Path, sample_rate: int) -> Path:
    key = f"{audio_path.resolve()}|{audio_path.stat().st_mtime_ns}|{audio_path.stat().st_size}|{int(sample_rate)}"
    return cache_dir / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.pt"


def _cached_audio_codes(model, audio_path: Path, sample_rate: int, cache_dir: Path) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _audio_code_cache_path(cache_dir, audio_path, sample_rate)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        codes = payload.get("codes") if isinstance(payload, dict) else payload
        if isinstance(codes, torch.Tensor):
            return codes.to(torch.long)
    wav, sr = _load_waveform(audio_path, target_sr=sample_rate)
    codes = model._encode_reference(wav, sr).to(torch.long).cpu()
    torch.save({"codes": codes}, cache_path)
    return codes


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
    modules_path = adapter_dir / "higgs_v3_audio_modules.pt"
    if modules_path.exists():
        modules = torch.load(modules_path, map_location="cpu", weights_only=False)
        if "audio_embedding" in modules:
            model.audio_embedding.load_state_dict(modules["audio_embedding"], strict=False)
        if "audio_head" in modules:
            model.audio_head.load_state_dict(modules["audio_head"], strict=False)
        print(f"[v3-train] Restored V3 audio modules from {modules_path}", flush=True)
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


def save_training_state(
    out_dir: Path,
    optimizer,
    scheduler,
    optimizer_step: int,
    micro_step: int,
    args: argparse.Namespace,
    extra: dict | None = None,
) -> None:
    """Everything needed to continue exactly: optimizer (incl. Prodigy's d), LR schedule position,
    RNG, early-stop guard, running loss average and the planned length of the run."""
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "optimizer_step": int(optimizer_step),
        "micro_step": int(micro_step),
        "step_semantics": "optimizer_step",
        "optimizer_name": getattr(args, "optimizer", "adamw"),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "args": vars(args),
        "lr_multiplier": _lr_multiplier(scheduler),
        **(extra or {}),
    }
    torch.save(state, out_dir / "trainer_state.pt")


def _lr_multiplier(scheduler) -> float | None:
    """Current LR as a fraction of the base LR (where the schedule is right now)."""
    try:
        base = float(scheduler.base_lrs[0])
        return float(scheduler.get_last_lr()[0]) / base if base > 0 else None
    except Exception:
        return None


def load_training_state(checkpoint_dir: Path, optimizer, scheduler=None) -> tuple[int, int, dict]:
    """Restore optimizer / scheduler / RNG. Returns (optimizer_step, micro_step, full saved state)."""
    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.exists():
        print(f"[v3-train] No trainer_state.pt found in {checkpoint_dir}; adapter/model weights only.", flush=True)
        return 0, 0, {}
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state["_optimizer_restored"] = False
    saved_name = state.get("optimizer_name", "adamw")
    current_name = type(optimizer).__name__.lower()
    expected = {"adamw": "adamw", "prodigy": "prodigy", "prodigy_sf": "prodigyplusschedulefree"}.get(saved_name, saved_name)
    if expected != current_name:
        print(
            f"[v3-train] Checkpoint was trained with {saved_name}; optimizer is now {current_name}. "
            "Keeping the weights and step count, starting fresh optimizer state.",
            flush=True,
        )
    else:
        optimizer.load_state_dict(state["optimizer"])
        state["_optimizer_restored"] = True
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
    return optimizer_step, micro_step, state


def _continue_schedule(args: argparse.Namespace, optimizer, scheduler, state: dict, start_step: int, total_updates: int,
                       base_lrs: list[float], warmup: int, lr_fn) -> str:
    """Keep the learning rate continuous across a resume.

    Same run length and schedule: the original curve simply goes on. Otherwise (more epochs / steps,
    another schedule or LR): the LR continues from where it was and the remaining schedule is fitted
    to the remaining steps - cosine / linear keep decaying from the current value to the floor,
    constant stays there. No new warmup. Returns a short description for the log."""
    saved_args = state.get("args") or {}
    multiplier = state.get("lr_multiplier")
    if multiplier is None:
        try:
            multiplier = float(state["scheduler"]["_last_lr"][0]) / float(state["scheduler"]["base_lrs"][0])
        except Exception:
            multiplier = lr_fn(start_step)
    multiplier = float(multiplier)
    lr_changed = abs(float(saved_args.get("learning_rate", args.learning_rate)) - float(args.learning_rate)) > 1e-12
    same_plan = (
        int(state.get("total_updates") or -1) == int(total_updates)
        and saved_args.get("lr_scheduler") == args.lr_scheduler
        and not lr_changed
    )
    if same_plan and state.get("scheduler") is not None:
        fn = lr_fn
        how = "same plan: the original schedule continues"
    else:
        floor = min(max(args.min_lr_ratio, 0.0), 1.0)
        remaining = max(total_updates - start_step, 1)
        end = min(floor, multiplier)
        schedule = args.lr_scheduler

        def fn(step: int, m0=multiplier, end=end, s0=start_step, n=remaining, schedule=schedule) -> float:
            if schedule == "constant":
                return m0
            progress = min(max((step - s0) / float(n), 0.0), 1.0)
            if schedule == "linear":
                return end + (m0 - end) * (1.0 - progress)
            return end + (m0 - end) * 0.5 * (1.0 + math.cos(math.pi * progress))

        how = f"continues from {multiplier:.3f}x base LR, {schedule} over the {remaining} remaining steps (no new warmup)"
        if schedule != "constant" and multiplier <= floor * 1.05:
            how += "; the previous schedule had already decayed to its floor"
    # Base LR: the one chosen now (lets you change it on resume); the multiplier keeps continuity.
    scheduler.base_lrs = list(base_lrs)
    scheduler.lr_lambdas = [fn] * len(optimizer.param_groups)
    scheduler.last_epoch = start_step
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["initial_lr"] = base
        group["lr"] = base * fn(start_step)
    scheduler._last_lr = [g["lr"] for g in optimizer.param_groups]
    return how


def generate_eval_audio(
    model, tokenizer, text: str, out_wav: Path, max_new_tokens: int = 1000, seed: int = 1234
) -> tuple[int, np.ndarray]:
    """Same text + same seed at every eval, so previews differ only by training."""
    was_training = model.training
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed))
    try:
        with torch.no_grad():
            audio = model.generate_speech(
                text.strip() or "This is a Higgs V3 training preview.",
                tokenizer,
                max_new_tokens=int(max_new_tokens),
                temperature=0.8,
                top_p=0.95,
                top_k=50,
            )
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    audio_np = audio.detach().float().cpu().numpy().astype(np.float32)
    sr = int(getattr(model.config, "sample_rate", 24000))
    sf.write(str(out_wav), audio_np, sr)
    if was_training:
        model.train()
    return sr, audio_np


def _sample_duration(sample: dict) -> float:
    try:
        return float(sample.get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _filter_by_duration(samples: list[dict], args: argparse.Namespace, label: str) -> list[dict]:
    kept = [s for s in samples if args.min_audio_seconds <= _sample_duration(s) <= args.max_audio_seconds or _sample_duration(s) == 0.0]
    dropped = len(samples) - len(kept)
    if dropped:
        print(
            f"[v3-train] {label}: skipped {dropped} clip(s) outside {args.min_audio_seconds:g}-{args.max_audio_seconds:g}s "
            "(long clips risk OOM and Higgs V3 reads ~25 s per pass at most)",
            flush=True,
        )
    return kept


def _length_grouped_batches(samples: list[dict], batch_size: int, max_batch_seconds: float = 0.0) -> list[list[dict]]:
    """Shuffle, then sort inside mega-batches by duration so batches pad little.

    ``max_batch_seconds`` caps the audio per micro-batch: groups of long clips get
    fewer samples, so length grouping can never stack e.g. 4 x 15 s into one OOM.
    """
    order = samples[:]
    random.shuffle(order)
    mega = max(batch_size * 16, batch_size)
    batches: list[list[dict]] = []
    for i in range(0, len(order), mega):
        chunk = sorted(order[i : i + mega], key=_sample_duration)
        current: list[dict] = []
        for sample in chunk:
            longest = max([_sample_duration(s) for s in current] + [_sample_duration(sample)])
            too_much = max_batch_seconds > 0 and current and longest * (len(current) + 1) > max_batch_seconds
            if len(current) >= batch_size or too_much:
                batches.append(current)
                current = []
            current.append(sample)
        if current:
            batches.append(current)
    random.shuffle(batches)
    return batches


def _make_example(model, data_dir: Path, sample: dict, args: argparse.Namespace, cache_dir: Path) -> V3Example:
    audio_path, transcript_path = _sample_paths(data_dir, sample)
    # Same normalisation for train and eval: V3 keeps its own inline tags.
    text = normalize_transcript(transcript_path.read_text(encoding="utf-8", errors="ignore"), convert_v2_tags=False)
    sr = int(model.config.sample_rate)
    codes = _cached_audio_codes(model, audio_path, sr, cache_dir)
    ref_codes = None
    ref_text = None
    if args.task_type == "zero_shot_voice_cloning":
        ref_name = sample.get("ref_audio_file")
        if not ref_name:
            raise ValueError(f"zero_shot_voice_cloning sample has no ref_audio_file: {sample.get('id', 'unknown')}")
        ref_codes = _cached_audio_codes(model, data_dir / ref_name, sr, cache_dir)
        ref_text = sample.get("ref_transcript") or ""
    return V3Example(text=text, target_codes=codes, reference_codes=ref_codes, reference_text=ref_text)


def _lr_lambda(args: argparse.Namespace, total_updates: int):
    warmup = int(math.ceil(total_updates * max(args.warmup_ratio, 0.0))) if args.warmup_steps < 0 else int(args.warmup_steps)
    floor = min(max(args.min_lr_ratio, 0.0), 1.0)

    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / float(warmup)
        if args.lr_scheduler == "constant":
            return 1.0
        progress = min(max((step - warmup) / float(max(total_updates - warmup, 1)), 0.0), 1.0)
        if args.lr_scheduler == "linear":
            return floor + (1.0 - floor) * (1.0 - progress)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn, warmup


def _param_groups(model, args: argparse.Namespace) -> list[dict]:
    audio_ids = {id(p) for m in (model.audio_embedding, model.audio_head) for p in m.parameters()}
    lora, audio, other = [], [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in audio_ids:
            audio.append(p)
        elif "lora_" in name:
            lora.append(p)
        else:
            other.append(p)
    groups = []
    if lora:
        groups.append({"params": lora, "lr": args.learning_rate, "weight_decay": args.weight_decay, "name": "lora"})
    if other:
        groups.append({"params": other, "lr": args.learning_rate, "weight_decay": args.weight_decay, "name": "backbone"})
    if audio:
        # The tied codec embedding/head is shared by every voice: move it gently.
        groups.append({"params": audio, "lr": args.learning_rate * args.audio_lr_scale, "weight_decay": 0.0, "name": "audio_modules"})
    for g in groups:
        print(f"[v3-train] param group {g['name']}: {sum(p.numel() for p in g['params']):,} params, lr={g['lr']:.2e}, wd={g['weight_decay']}", flush=True)
    return groups


def _build_optimizer(model, args: argparse.Namespace):
    """AdamW (measured recipe), Prodigy, or Prodigy Plus Schedule-Free.

    Prodigy estimates the LR scale ``d`` online (``lr`` stays 1.0 and the cosine
    schedule only shapes it); it needs one shared lr across groups, so a trained
    codec embedding uses the same scale. Schedule-Free needs no LR schedule and
    keeps averaged weights that must be swapped in (``optimizer.eval()``) before
    evaluating or saving.
    """
    groups = _param_groups(model, args)
    name = args.optimizer
    if name == "adamw":
        return AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    for g in groups:
        g["lr"] = 1.0
    if name == "prodigy":
        from prodigyopt import Prodigy

        if any(g.get("name") == "audio_modules" for g in groups):
            print("[v3-train] Prodigy uses one LR scale for all groups: --audio_lr_scale is ignored.", flush=True)
        return Prodigy(
            groups,
            lr=1.0,
            betas=(0.9, 0.99),
            weight_decay=args.weight_decay,
            decouple=True,
            use_bias_correction=True,
            safeguard_warmup=True,
            d_coef=args.prodigy_d_coef,
            d0=args.prodigy_d0,
        )
    if name == "prodigy_sf":
        from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

        return ProdigyPlusScheduleFree(
            groups,
            lr=1.0,
            betas=(0.9, 0.99),
            weight_decay=args.weight_decay,
            d_coef=args.prodigy_d_coef,
            d0=args.prodigy_d0,
            split_groups=True,
            use_stableadamw=True,
            use_schedulefree=True,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def _effective_lr(optimizer) -> float:
    """lr x d for Prodigy-family optimizers (their lr is a 1.0 multiplier)."""
    group = optimizer.param_groups[0]
    return float(group["lr"]) * float(group.get("d", 1.0))


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dir = Path(args.train_data_dir)
    eval_dir = Path(args.eval_data_dir) if args.eval_data_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_cache_dir = output_dir / "audio_code_cache"
    (output_dir / "live_settings.json").unlink(missing_ok=True)  # live changes belong to one run only
    resume_dir = Path(args.resume_checkpoint) if args.resume_checkpoint else None

    if args.task_type not in SUPPORTED_TASKS:
        raise ValueError(f"Custom V3 trainer supports {sorted(SUPPORTED_TASKS)}, got {args.task_type}")

    train_samples = _filter_by_duration(_load_samples(train_dir), args, "train")
    eval_samples = _filter_by_duration(_load_samples(eval_dir), args, "eval") if eval_dir else []
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_eval_samples > 0:
        eval_samples = eval_samples[: args.max_eval_samples]
    if not train_samples:
        raise RuntimeError("No train samples left after the duration filter.")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and args.bf16 else torch.float32
    torch.set_float32_matmul_precision("high")
    total_minutes = sum(_sample_duration(s) for s in train_samples) / 60.0
    print(f"[v3-train] device={device} dtype={dtype}", flush=True)
    print(f"[v3-train] train={len(train_samples)} clips ({total_minutes:.1f} min) eval={len(eval_samples)} task={args.task_type}", flush=True)
    print(f"[v3-train] audio_code_cache={code_cache_dir}", flush=True)

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
    elif dtype != torch.float32:
        # Trainable audio modules keep fp32 master weights (bf16 drops tiny updates).
        model.audio_embedding.float()
        model.audio_head.weight = model.audio_embedding.weight
    if args.use_lora:
        resume_adapter = resume_dir / "qwen3_lora" if resume_dir else None
        if resume_adapter and resume_adapter.exists():
            model = _load_lora(model, resume_adapter)
        else:
            model = _apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    if args.gradient_checkpointing:
        backbone = model.model.get_base_model() if hasattr(model.model, "get_base_model") else model.model
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[v3-train] gradient checkpointing: on", flush=True)
    print(f"[v3-train] {_trainable_parameter_report(model)}", flush=True)

    batch_size = max(int(args.batch_size), 1)
    accum = max(int(args.gradient_accumulation_steps), 1)
    batches_per_epoch = len(_length_grouped_batches(train_samples, batch_size, args.max_batch_seconds))
    total_updates = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(batches_per_epoch * args.num_train_epochs / accum)
    )
    optimizer = _build_optimizer(model, args)
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    schedule_free = args.optimizer == "prodigy_sf"
    if schedule_free:
        # Schedule-Free replaces the LR schedule; Prodigy's own safeguard handles warmup.
        args.lr_scheduler, args.warmup_ratio, args.warmup_steps = "constant", 0.0, 0
    elif args.optimizer == "prodigy":
        args.warmup_ratio, args.warmup_steps = 0.0, 0  # safeguard_warmup instead of a ramp
    lr_fn, warmup = _lr_lambda(args, total_updates)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
    if schedule_free:
        optimizer.train()
    optimizer_step = 0
    micro_step = 0
    resume_state: dict = {}
    if resume_dir:
        optimizer_step, micro_step, resume_state = load_training_state(resume_dir, optimizer, scheduler)
    optimizer.zero_grad(set_to_none=True)
    if resume_state and optimizer_step >= total_updates:
        print(
            f"[v3-train] The checkpoint is already at step {optimizer_step} of the planned {total_updates}: "
            "raise Epochs (or Max steps) to train further.",
            flush=True,
        )
    total_updates = max(total_updates, optimizer_step)
    start_step = optimizer_step
    if resume_state:
        if schedule_free and resume_state.get("_optimizer_restored"):
            # Checkpoints are written with the averaged (eval) weights: swap back to training weights.
            optimizer.train()
        if not schedule_free:
            how = _continue_schedule(args, optimizer, scheduler, resume_state, optimizer_step, total_updates, base_lrs, warmup, lr_fn)
            print(f"[v3-train] LR on resume: {_effective_lr(optimizer):.3e} ({how})", flush=True)
        else:
            print(f"[v3-train] LR on resume: Schedule-Free keeps its own state (d={_effective_lr(optimizer):.3e})", flush=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_run_dir = output_dir / "tensorboard" / f"{time.strftime('%Y%m%d_%H%M%S')}_step{optimizer_step:07d}"
        writer = SummaryWriter(log_dir=str(tb_run_dir))
        print(f"[v3-train] TensorBoard run: {tb_run_dir}", flush=True)
    except Exception as exc:
        print(f"[v3-train] TensorBoard disabled: {exc}", flush=True)
    epochs_equiv = total_updates * accum * batch_size / max(len(train_samples), 1)
    print(
        f"[v3-train] start_optimizer_step={optimizer_step} max_optimizer_steps={total_updates} (~{epochs_equiv:.1f} epochs) "
        f"batch={batch_size} grad_accum={accum} effective_batch={batch_size * accum} batches_per_epoch={batches_per_epoch} "
        f"optimizer={args.optimizer} "
        + (f"lr={args.learning_rate:.2e} " if args.optimizer == "adamw" else f"lr=auto(d_coef={args.prodigy_d_coef}) ")
        + f"schedule={args.lr_scheduler} warmup={warmup} min_lr_ratio={args.min_lr_ratio}",
        flush=True,
    )
    pbar = tqdm(total=total_updates, initial=optimizer_step, desc="[v3-train] updates", unit="step")
    metrics = MetricsLog(output_dir / "metrics.jsonl", fresh=resume_dir is None)
    if resume_dir is not None:
        # Drop what a later, abandoned branch logged after this checkpoint, so the charts read as one run.
        previous = metrics.truncate_after(optimizer_step)
        metrics.write(
            "resume",
            step=optimizer_step,
            checkpoint=resume_dir.name if resume_dir != output_dir else "last state",
            previous_reason=previous,
            lr=_effective_lr(optimizer),
        )
    running_loss = 0.0
    running_batches = 0
    loss_ema: float | None = resume_state.get("loss_ema") if resume_state else None
    last_stats: dict = {}
    nan_steps = 0
    stop_reason: str | None = None
    train_started_at = time.time()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if dtype == torch.bfloat16 else _nullcontext()
    best_dir = output_dir / "checkpoint-best"

    def epoch_at(step: int) -> float:
        return round(step * accum / max(batches_per_epoch, 1), 3)

    metrics.write(
        "config",
        step=optimizer_step,
        total_steps=total_updates,
        steps_per_epoch=round(batches_per_epoch / accum, 3),
        epochs=round(epochs_equiv, 2),
        optimizer=args.optimizer,
        lr=args.learning_rate,
        schedule=args.lr_scheduler,
        train_clips=len(train_samples),
        train_minutes=round(total_minutes, 2),
        eval_clips=len(eval_samples),
        batch=batch_size,
        accum=accum,
        rank=args.lora_rank,
        patience=args.early_stop_patience,
        eval_steps=args.eval_steps,
    )

    # --- eval-audio setup: one fixed unseen sentence + seed for the whole run --
    eval_text = (args.eval_text or "").strip()
    previous_text = metrics.last_eval_text() if resume_dir is not None else None
    if args.enable_eval_audio and not eval_text and previous_text:
        eval_text = previous_text  # same preview sentence as before the resume: previews stay comparable
    if args.enable_eval_audio and not eval_text and eval_samples:
        ref = min(eval_samples, key=lambda s: abs(_sample_duration(s) - 6.0))
        ref_audio, ref_txt = _sample_paths(eval_dir, ref)
        eval_text = normalize_transcript(ref_txt.read_text(encoding="utf-8", errors="ignore"), convert_v2_tags=False)
        ref_out = output_dir / "eval_audio" / "reference.wav"
        ref_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref_audio, ref_out)
        metrics.write("reference", step=optimizer_step, path=str(ref_out), text=eval_text)
    if args.enable_eval_audio and not eval_text:
        eval_text = "This is my voice evolution during training."
    if args.enable_eval_audio and eval_text != previous_text:
        metrics.write("eval_text", step=optimizer_step, text=eval_text, seed=args.eval_seed)

    preview_ref = {"seconds": metrics.base_audio_seconds() if resume_dir is not None else None}

    def make_eval_audio(step: int, tag: str = "") -> None:
        if not args.enable_eval_audio:
            return
        try:
            wav_path = output_dir / "eval_audio" / f"step_{step:07d}{tag}.wav"
            sr, audio = generate_eval_audio(
                model, tokenizer, eval_text, wav_path, args.eval_audio_max_new_tokens, seed=args.eval_seed
            )
            print(f"[v3-train] optimizer_step={step} eval_audio={wav_path}", flush=True)
            seconds = round(len(audio) / max(sr, 1), 2)
            if preview_ref["seconds"] is None and step == 0:
                preview_ref["seconds"] = seconds
            runaway = bool(preview_ref["seconds"] and seconds > 2.5 * preview_ref["seconds"])
            metrics.write("audio", step=step, path=str(wav_path), seconds=seconds, runaway=runaway)
            if runaway:
                warn(
                    step,
                    f"eval preview ran {seconds:.1f} s for a sentence the base model said in {preview_ref['seconds']:.1f} s: "
                    "the model failed to stop (unstable end-of-speech). Keep an eye on later previews.",
                )
            if writer is not None:
                writer.add_audio("eval/generated_audio", torch.from_numpy(audio).unsqueeze(0), step, sample_rate=sr)
        except Exception as exc:
            print(f"[v3-train] eval audio generation failed at optimizer_step {step}: {exc}", flush=True)

    def extra_state(reason: str | None = None) -> dict:
        return {
            "guard": guard.state_dict(),
            "loss_ema": loss_ema,
            "total_updates": total_updates,
            "stop_reason": reason,
        }

    def warn(step: int, message: str) -> None:
        print(f"[v3-warn] step {step}: {message}", flush=True)
        metrics.write("warn", step=step, message=message)

    # --- baseline: the model before any update (base + zero LoRA) ---------------
    best_loss, best_step = _read_best(output_dir)
    guard = OverfitGuard(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        min_evals=args.early_stop_min_evals,
        base_eval=metrics.previous_base_eval(),
        best=best_loss,
        total_steps=total_updates,
    )
    guard.best_step = best_step
    if resume_state.get("guard"):
        guard.load_state(resume_state["guard"], resume_step=optimizer_step)
        if guard.best is None and best_loss is not None:
            guard.best, guard.best_step = best_loss, best_step
        if str(resume_state.get("stop_reason") or "").startswith("no eval improvement"):
            print(
                f"[v3-train] The previous run stopped on patience ({guard.bad_evals}/{resume_state.get('args', {}).get('early_stop_patience')}); "
                f"the counter restarts at 0/{guard.patience} because you chose to continue.",
                flush=True,
            )
            guard.bad_evals = 0
    elif resume_state:
        guard.load_history(metrics.eval_history(optimizer_step))
    if resume_state and best_step is not None and best_step > optimizer_step:
        # checkpoint-best came from a later branch that is being discarded: point it back at this branch's best.
        source = output_dir / f"checkpoint-{guard.best_step}" if guard.best_step else None
        shutil.rmtree(best_dir, ignore_errors=True)
        if source is not None and (source / "qwen3_lora").exists():
            shutil.copytree(source, best_dir, ignore=shutil.ignore_patterns("trainer_state.pt"))
            (best_dir / "best_info.json").write_text(
                json.dumps({"step": guard.best_step, "eval_loss": round(guard.best, 6)}), encoding="utf-8"
            )
            print(f"[v3-train] checkpoint-best reset to step {guard.best_step} (the later branch is discarded).", flush=True)
        else:
            print("[v3-train] checkpoint-best belonged to a discarded branch and was removed.", flush=True)
    if resume_state:
        for step_, loss_ in [(st, ls) for st, ls, _tr in guard.history]:
            # Seed the UI progress bar with this run's eval history.
            print(f"[v3-train] optimizer_step={step_} eval_loss={loss_:.5f} (history)", flush=True)
        print(
            f"[v3-train] Resumed guard: best={guard.best if guard.best is None else round(guard.best, 5)} "
            f"@ step {guard.best_step}, patience {guard.bad_evals}/{guard.patience or '-'}, "
            f"{len(guard.history)} previous evals",
            flush=True,
        )
    if resume_dir is None and eval_samples and args.eval_steps > 0:
        base_eval = evaluate(model, tokenizer, eval_dir, eval_samples, args)
        guard.base_eval = base_eval
        print(f"[v3-train] optimizer_step=0 base_eval_loss={base_eval:.5f}", flush=True)
        metrics.write("eval", step=0, epoch=0.0, eval_loss=round(base_eval, 6), base=True, status="baseline")
        if writer is not None:
            writer.add_scalar("eval/loss", base_eval, 0)
        make_eval_audio(0, "_base")

    while optimizer_step < total_updates and stop_reason is None:
        epoch_batches = _length_grouped_batches(train_samples, batch_size, args.max_batch_seconds)
        accum_batches = 0
        for b_idx, batch in enumerate(epoch_batches):
            if optimizer_step >= total_updates or stop_reason is not None:
                break
            is_epoch_tail = b_idx == len(epoch_batches) - 1
            examples = [_make_example(model, train_dir, s, args, code_cache_dir) for s in batch]
            with autocast:
                batch_loss, last_stats = compute_v3_batch_loss(model, tokenizer, examples)
            loss_value_now = float(batch_loss.detach().cpu())
            if not math.isfinite(loss_value_now):
                nan_steps += 1
                optimizer.zero_grad(set_to_none=True)
                accum_batches = 0
                warn(optimizer_step, f"non-finite loss ({loss_value_now}); batch skipped ({nan_steps}/3)")
                if nan_steps >= 3:
                    stop_reason = "loss became NaN/inf repeatedly (diverged): lower the learning rate"
                continue
            (batch_loss / accum).backward()
            running_loss += loss_value_now
            running_batches += 1
            loss_ema = loss_value_now if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_value_now
            micro_step += 1
            accum_batches += 1
            if accum_batches < accum and not is_epoch_tail:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                float("inf") if schedule_free else args.max_grad_norm,  # StableAdamW scales updates itself
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            accum_batches = 0
            pbar.update(1)
            grad_norm_value = float(grad_norm)
            if math.isfinite(grad_norm_value) and guard.note_grad_norm(grad_norm_value):
                warn(optimizer_step, f"gradient spike: grad_norm {grad_norm_value:.2f} (>5x the recent median)")
            if optimizer_step % max(args.logging_steps, 1) == 0:
                loss_value = running_loss / max(running_batches, 1)
                current_lr = _effective_lr(optimizer)
                print(
                    training_progress_line(
                        "[v3-train]",
                        optimizer_step,
                        total_updates,
                        loss_value,
                        train_started_at,
                        suffix=f"lr={current_lr:.2e} grad_norm={grad_norm_value:.3f} cb0={last_stats.get('cb0')} cb_rest={last_stats.get('cb_rest')}",
                        start_step=start_step,
                    ),
                    flush=True,
                )
                metrics.write(
                    "train",
                    step=optimizer_step,
                    epoch=epoch_at(optimizer_step),
                    loss=round(loss_value, 5),
                    lr=current_lr,
                    grad_norm=round(grad_norm_value, 4),
                    cb0=last_stats.get("cb0"),
                    cb_rest=last_stats.get("cb_rest"),
                )
                if writer is not None:
                    writer.add_scalar("train/loss", loss_value, optimizer_step)
                    writer.add_scalar("train/lr", current_lr, optimizer_step)
                    writer.add_scalar("train/grad_norm", grad_norm_value, optimizer_step)
                running_loss = 0.0
                running_batches = 0
            eval_due = args.eval_steps > 0 and optimizer_step % args.eval_steps == 0
            save_due = args.save_steps > 0 and optimizer_step % args.save_steps == 0
            stop_file = stop_requested()
            boundary = eval_due or save_due or stop_file or optimizer_step >= total_updates
            if schedule_free and boundary:
                optimizer.eval()  # swap in the averaged weights for eval / checkpoints
            if eval_samples and eval_due:
                live = _live_settings(output_dir)
                if "early_stop_patience" in live and int(live["early_stop_patience"]) != guard.patience:
                    guard.patience = max(int(live["early_stop_patience"]), 0)
                    args.early_stop_patience = guard.patience
                    print(f"[v3-train] patience changed live to {guard.patience or 'off'}", flush=True)
                    metrics.write("patience", step=optimizer_step, patience=guard.patience)
                eval_loss = evaluate(model, tokenizer, eval_dir, eval_samples, args)
                verdict = guard.update(optimizer_step, eval_loss, loss_ema)
                print(
                    f"[v3-train] optimizer_step={optimizer_step} eval_loss={eval_loss:.5f}"
                    f"{' (best)' if verdict['new_best'] else ''} status={verdict['status']} "
                    f"patience={verdict['bad_evals']}/{guard.patience or '-'}",
                    flush=True,
                )
                metrics.write(
                    "eval",
                    step=optimizer_step,
                    epoch=epoch_at(optimizer_step),
                    eval_loss=round(eval_loss, 6),
                    train_loss=round(loss_ema, 6) if loss_ema is not None else None,
                    best=round(guard.best, 6),
                    best_step=guard.best_step,
                    bad_evals=verdict["bad_evals"],
                    patience=guard.patience,
                    status=verdict["status"],
                )
                if writer is not None:
                    writer.add_scalar("eval/loss", eval_loss, optimizer_step)
                for message in verdict["warnings"]:
                    warn(optimizer_step, message)
                if verdict["new_best"] and args.save_best and args.use_lora:
                    save_checkpoint(model, tokenizer, best_dir, args.use_lora)
                    (best_dir / "best_info.json").write_text(
                        json.dumps({"step": optimizer_step, "eval_loss": round(eval_loss, 6)}), encoding="utf-8"
                    )
                if verdict["stop"]:
                    stop_reason = verdict["stop"]
            if eval_due:
                make_eval_audio(optimizer_step)
            if save_due:
                checkpoint_dir = output_dir / f"checkpoint-{optimizer_step}"
                save_checkpoint(model, tokenizer, checkpoint_dir, args.use_lora)
                save_training_state(checkpoint_dir, optimizer, scheduler, optimizer_step, micro_step, args, extra_state(stop_reason))
                save_checkpoint(model, tokenizer, output_dir, args.use_lora)
                save_training_state(output_dir, optimizer, scheduler, optimizer_step, micro_step, args, extra_state(stop_reason))
            if stop_file:
                stop_reason = stop_reason or "stopped by user"
            if schedule_free and boundary and stop_reason is None:
                optimizer.train()

    pbar.close()
    if schedule_free:
        optimizer.eval()
    finished_normally = stop_reason is None
    if stop_reason:
        print(f"[v3-train] Early stop at optimizer_step={optimizer_step}: {stop_reason}", flush=True)
    # The last weights stay resumable as a regular checkpoint.
    last_dir = output_dir / f"checkpoint-{optimizer_step}"
    if optimizer_step > 0 and not (last_dir / "trainer_state.pt").exists():
        save_checkpoint(model, tokenizer, last_dir, args.use_lora)
        save_training_state(last_dir, optimizer, scheduler, optimizer_step, micro_step, args, extra_state(stop_reason))
    elif optimizer_step > 0:
        # Already saved at this step before the stop decision: refresh it with the final guard state.
        save_training_state(last_dir, optimizer, scheduler, optimizer_step, micro_step, args, extra_state(stop_reason))
    save_checkpoint(model, tokenizer, output_dir, args.use_lora)
    save_training_state(output_dir, optimizer, scheduler, optimizer_step, micro_step, args, extra_state(stop_reason))
    final_source = "last"
    if args.restore_best and args.use_lora and (best_dir / "qwen3_lora").exists() and guard.best_step not in (None, optimizer_step):
        # Final adapter = best eval (anti-overfitting). Root is no longer a resume point:
        # continue from checkpoint-<last step> instead.
        shutil.rmtree(output_dir / "qwen3_lora", ignore_errors=True)
        shutil.copytree(best_dir / "qwen3_lora", output_dir / "qwen3_lora")
        (output_dir / "trainer_state.pt").unlink(missing_ok=True)
        final_source = f"best (step {guard.best_step})"
        print(f"[v3-train] Final adapter restored from checkpoint-best (step {guard.best_step}, eval {guard.best:.5f}).", flush=True)
    if finished_normally and guard.last_eval_step != optimizer_step:
        make_eval_audio(optimizer_step, "_final")
    metrics.write(
        "stop",
        step=optimizer_step,
        epoch=epoch_at(optimizer_step),
        reason=stop_reason or "finished",
        final=final_source,
        best=round(guard.best, 6) if guard.best is not None else None,
        best_step=guard.best_step,
        base_eval=round(guard.base_eval, 6) if guard.base_eval is not None else None,
    )
    if writer is not None:
        writer.flush()
        writer.close()


class MetricsLog:
    """Append-only JSONL of everything the UI dashboard plots."""

    def __init__(self, path: Path, fresh: bool):
        self.path = path
        if fresh:
            path.unlink(missing_ok=True)

    def write(self, kind: str, **data) -> None:
        record = {"type": kind, "time": round(time.time(), 2), **data}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _records(self) -> list[dict]:
        records = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
        return records

    def truncate_after(self, step: int) -> str | None:
        """Remove records of steps after ``step`` and the stop record of the run being continued.
        Returns how that run ended (for the 'resume' marker)."""
        records = self._records()
        previous = None
        kept = []
        for rec in records:
            if rec.get("type") == "stop" and int(rec.get("step", 0)) >= step:
                previous = rec.get("reason")
                if int(rec.get("step", 0)) == step:
                    continue
            if int(rec.get("step", 0)) > step:
                continue
            kept.append(rec)
        if previous is None:
            previous = next((r.get("reason") for r in reversed(kept) if r.get("type") == "stop"), None)
        if len(kept) != len(records):
            self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
        return previous

    def last_eval_text(self) -> str | None:
        return next((r.get("text") for r in reversed(self._records()) if r.get("type") == "eval_text"), None)

    def base_audio_seconds(self) -> float | None:
        return next((r.get("seconds") for r in self._records() if r.get("type") == "audio" and r.get("step") == 0), None)

    def eval_history(self, up_to: int) -> list[tuple[int, float, float | None]]:
        return [
            (int(r["step"]), float(r["eval_loss"]), r.get("train_loss"))
            for r in self._records()
            if r.get("type") == "eval" and not r.get("base") and int(r.get("step", 0)) <= up_to
        ]

    def previous_base_eval(self) -> float | None:
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("type") == "eval" and rec.get("base"):
                    return float(rec["eval_loss"])
        except Exception:
            pass
        return None


class OverfitGuard:
    """Early stopping + overfitting diagnostics from the eval curve.

    * new best: any eval below the best so far (saved to checkpoint-best);
    * patience counts evals that fail to beat the best by ``min_delta``
      (relative), so tiny wiggles do not reset it;
    * stops after ``patience`` such evals (once ``min_evals`` exist), or at
      once when eval gets worse than the untrained base model after having
      clearly beaten it (the LoRA is now hurting unseen text);
    * warns when eval rises twice in a row or the train/eval gap keeps growing.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float,
        min_evals: int,
        base_eval: float | None,
        best: float | None,
        total_steps: int = 0,
    ):
        self.total_steps = max(int(total_steps), 0)
        self.patience = max(int(patience), 0)
        self.min_delta = max(float(min_delta), 0.0)
        self.min_evals = max(int(min_evals), 1)
        self.base_eval = base_eval
        self.best = best
        self.best_step: int | None = None
        self.bad_evals = 0
        self.history: list[tuple[int, float, float | None]] = []
        self.gap_at_best: float | None = None
        self.last_eval_step: int | None = None
        self._grad_norms: list[float] = []
        self.warming_up = True

    def state_dict(self) -> dict:
        return {
            "best": self.best,
            "best_step": self.best_step,
            "bad_evals": self.bad_evals,
            "history": [list(h) for h in self.history],
            "gap_at_best": self.gap_at_best,
            "last_eval_step": self.last_eval_step,
            "grad_norms": self._grad_norms[-50:],
            "warming_up": self.warming_up,
            "base_eval": self.base_eval,
        }

    def load_state(self, state: dict, resume_step: int) -> None:
        self.best = state.get("best", self.best)
        self.best_step = state.get("best_step", self.best_step)
        self.bad_evals = int(state.get("bad_evals", 0))
        self.history = [tuple(h) for h in state.get("history", []) if int(h[0]) <= resume_step]
        self.gap_at_best = state.get("gap_at_best")
        self.last_eval_step = state.get("last_eval_step")
        self._grad_norms = list(state.get("grad_norms") or [])
        self.warming_up = bool(state.get("warming_up", True))
        if self.base_eval is None:
            self.base_eval = state.get("base_eval")

    def load_history(self, history: list[tuple[int, float, float | None]]) -> None:
        """Older checkpoints without guard state: rebuild it from metrics.jsonl."""
        for step, eval_loss, train_loss in history:
            self.update(step, eval_loss, train_loss)

    def note_grad_norm(self, value: float) -> bool:
        recent = self._grad_norms[-50:]
        self._grad_norms.append(value)
        if len(recent) < 20:
            return False
        median = sorted(recent)[len(recent) // 2]
        return median > 0 and value > 5 * median

    def update(self, step: int, eval_loss: float, train_loss: float | None) -> dict:
        warnings: list[str] = []
        self.history.append((step, eval_loss, train_loss))
        self.last_eval_step = step
        gap = eval_loss - train_loss if train_loss is not None else None
        new_best = self.best is None or eval_loss < self.best
        meaningful = self.best is None or eval_loss < self.best * (1.0 - self.min_delta)
        if new_best:
            self.best, self.best_step = eval_loss, step
            self.gap_at_best = gap
        # Warm-up: until the LoRA clearly beats the untrained model (or half the
        # planned steps are done) a flat eval means "not learning yet", not
        # "overfitting", so patience is not spent.
        beat_base = self.base_eval is None or (self.best is not None and self.best < self.base_eval * (1.0 - self.min_delta))
        past_half = self.total_steps > 0 and step >= self.total_steps * 0.5
        self.warming_up = not (beat_base or past_half)
        if meaningful or self.warming_up:
            self.bad_evals = 0
        else:
            self.bad_evals += 1

        rising = len(self.history) >= 3 and self.history[-1][1] > self.history[-2][1] > self.history[-3][1]
        if self.warming_up:
            status = "warming up"
        elif new_best:
            status = "improving"
        elif rising and eval_loss > (self.best or eval_loss) * 1.01:
            status = "overfitting"
        else:
            status = "plateau"
        if rising and not self.warming_up:
            warnings.append(f"eval loss rose twice in a row ({self.history[-3][1]:.3f} -> {eval_loss:.3f}); best was step {self.best_step}")
        if gap is not None and self.gap_at_best is not None and not new_best and gap > self.gap_at_best + 0.25:
            warnings.append(f"train/eval gap widened to {gap:.2f} (was {self.gap_at_best:.2f} at best): the model is memorising the training clips")

        stop = None
        regressed = (
            self.base_eval is not None
            and self.best is not None
            and self.best < self.base_eval * 0.99
            and eval_loss > self.base_eval
        )
        if regressed:
            status = "regressed"
            stop = f"eval {eval_loss:.3f} is now worse than the untrained model ({self.base_eval:.3f}); best was step {self.best_step}"
        elif self.patience and self.bad_evals >= self.patience and len(self.history) >= self.min_evals:
            stop = f"no eval improvement for {self.bad_evals} evals (patience {self.patience}); best was step {self.best_step}"
        return {"new_best": new_best, "status": status, "bad_evals": self.bad_evals, "warnings": warnings, "stop": stop}


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _live_settings(output_dir: Path) -> dict:
    """Settings the UI may change while training runs (exp/<run>/live_settings.json)."""
    try:
        return dict(json.loads((output_dir / "live_settings.json").read_text(encoding="utf-8")))
    except Exception:
        return {}


def _read_best(output_dir: Path) -> tuple[float | None, int | None]:
    info = output_dir / "checkpoint-best" / "best_info.json"
    try:
        data = json.loads(info.read_text(encoding="utf-8"))
        return float(data["eval_loss"]), int(data["step"]) if data.get("step") is not None else None
    except Exception:
        return None, None


@torch.no_grad()
def evaluate(model, tokenizer, eval_dir: Path, eval_samples: list[dict], args: argparse.Namespace) -> float:
    was_training = model.training
    model.eval()
    cache_dir = Path(args.output_dir) / "audio_code_cache"
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if model.device.type == "cuda" and args.bf16 else _nullcontext()
    total, tokens = 0.0, 0
    ordered = sorted(eval_samples, key=_sample_duration)
    bs = max(int(args.batch_size), 1)
    for i in range(0, len(ordered), bs):
        examples = [_make_example(model, eval_dir, s, args, cache_dir) for s in ordered[i : i + bs]]
        with autocast:
            loss, stats = compute_v3_batch_loss(model, tokenizer, examples)
        total += float(loss) * stats["tokens"]
        tokens += stats["tokens"]
    if was_training:
        model.train()
    return total / max(tokens, 1)


def save_checkpoint(model, tokenizer, output_dir: Path, lora_only: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if lora_only and hasattr(model.model, "save_pretrained"):
        adapter_dir = output_dir / "qwen3_lora"
        model.model.save_pretrained(str(adapter_dir))
        torch.save(
            {
                "audio_embedding": model.audio_embedding.state_dict(),
                "audio_head": model.audio_head.state_dict(),
            },
            adapter_dir / "higgs_v3_audio_modules.pt",
        )
        tokenizer.save_pretrained(str(output_dir))
        (output_dir / "adapter_note.txt").write_text(
            "Custom Higgs V3 adapter: load the base V3 model, attach qwen3_lora to model.model, then restore higgs_v3_audio_modules.pt when present.\n",
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
    parser.add_argument("--max_eval_samples", type=int, default=24)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--freeze_audio_head", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", choices=["cosine", "linear", "constant"], default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=-1, help="-1 = use --warmup_ratio")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--audio_lr_scale", type=float, default=0.1, help="LR multiplier for the tied audio embedding/head")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--max_batch_seconds", type=float, default=40.0, help="Cap on padded audio seconds per micro-batch (0 = off)")
    parser.add_argument("--min_audio_seconds", type=float, default=0.5)
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "prodigy", "prodigy_sf"], default="adamw")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Evals without improvement before stopping (0 = off)")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.002, help="Relative improvement that resets patience")
    parser.add_argument("--early_stop_min_evals", type=int, default=3)
    parser.add_argument("--restore_best", action="store_true", help="Final adapter = best eval checkpoint")
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--prodigy_d_coef", type=float, default=1.0, help="Scales Prodigy's LR estimate (0.5 = half, 2 = double)")
    parser.add_argument("--prodigy_d0", type=float, default=1e-5, help="Prodigy's starting LR estimate (raise if it stalls)")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--enable_eval_audio", action="store_true")
    parser.add_argument("--eval_text", default="", help="Empty = an unseen sentence from the eval set (its recording becomes the reference)")
    parser.add_argument("--eval_audio_max_new_tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    started = time.time()
    train(parse_args())
    print(f"[v3-train] done in {time.time() - started:.1f}s", flush=True)
