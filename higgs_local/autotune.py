"""Dataset-driven training plan for Higgs V3 LoRA (single speaker).

Every value is derived from the dataset that will actually be trained:

* size (minutes, clips) -> LoRA capacity (rank), epoch budget and LR;
* clip length distribution (p90 / max) -> micro-batch size that fits VRAM,
  gradient checkpointing, the long-clip filter;
* clip count -> effective batch (enough optimizer steps per epoch);
* eval set -> eval cadence, early-stopping patience, and whether the
  anti-overfitting guard can run at all.

The epoch budget is an upper bound: the overfitting guard stops earlier when
held-out loss stops improving, and the final adapter is the best-eval one.
Grounding: on 20 min / 193 clips, LR 1e-4, rank 32, effective batch 8 and
cosine reached the best held-out loss around epoch 5; on 83 min / 774 clips a
run at LR 8e-5 / effective batch 18 had its best eval at ~10 epochs or less
and overfit steadily afterwards.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# fwd+bwd cost measured on an RTX 3090 (bf16, LoRA r32): ~0.46 s for 4 clips of
# ~7 s -> ~0.017 s per second of training audio. Gradient checkpointing ~ +35 %.
SEC_PER_AUDIO_SEC = 0.017
EVAL_SEC_PER_AUDIO_SEC = 0.006
EVAL_AUDIO_SEC = 12.0  # one fixed-seed preview with the training (PEFT) model
VRAM_AUDIO_BUDGET_24GB = 30.0  # padded seconds of audio per micro-batch at 24 GB


@dataclass
class DatasetProfile:
    clips: int
    seconds: float
    durations: list[float]
    eval_clips: int = 0
    eval_seconds: float = 0.0
    transcript_coverage: float | None = None

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0

    def pct(self, q: float) -> float:
        if not self.durations:
            return 6.0
        ordered = sorted(self.durations)
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


@dataclass
class TrainingPlan:
    values: dict
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _durations(split_dir: Path) -> tuple[list[float], int]:
    meta = split_dir / "metadata.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    samples = data.get("samples") or []
    durs = [float(s.get("duration") or 0.0) for s in samples]
    with_text = sum(1 for s in samples if (split_dir / (s.get("transcript_file") or "")).is_file())
    return [d for d in durs if d > 0], with_text


def profile_dataset(train_dir: Path, eval_dir: Path | None) -> DatasetProfile:
    durs, with_text = _durations(train_dir)
    prof = DatasetProfile(clips=len(durs), seconds=sum(durs), durations=durs)
    prof.transcript_coverage = with_text / max(len(durs), 1)
    if eval_dir and (eval_dir / "metadata.json").exists():
        edurs, _ = _durations(eval_dir)
        prof.eval_clips, prof.eval_seconds = len(edurs), sum(edurs)
    return prof


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h} h {m:02d} min" if h else f"{max(m, 1)} min"


def plan_training(prof: DatasetProfile, vram_gb: int = 24, optimizer: str = "adamw") -> TrainingPlan:
    reasons: list[str] = []
    warnings: list[str] = []
    minutes = prof.minutes
    p90 = prof.pct(0.9)
    longest = max(prof.durations) if prof.durations else p90

    # 1) capacity + epoch budget + base LR from the amount of speech
    if minutes < 3:
        rank, epochs, base_lr = 8, 15, 1e-4
        warnings.append(f"Only {minutes:.1f} min of speech: voice cloning quality will be limited; 10-60 min is the sweet spot.")
    elif minutes < 10:
        rank, epochs, base_lr = 16, 12, 1e-4
    elif minutes < 30:
        rank, epochs, base_lr = 32, 8, 1e-4
    elif minutes < 90:
        rank, epochs, base_lr = 32, 6, 8e-5
    elif minutes < 240:
        rank, epochs, base_lr = 64, 4, 7e-5
    else:
        rank, epochs, base_lr = 64, 3, 6e-5
    reasons.append(
        f"{minutes:.1f} min of speech -> LoRA rank {rank} and a budget of {epochs} epochs "
        "(an upper bound: early stopping usually ends sooner)."
    )

    # 2) micro-batch from clip lengths and VRAM; checkpointing for long clips
    budget = VRAM_AUDIO_BUDGET_24GB * vram_gb / 24.0
    grad_ckpt = longest > 20 and vram_gb <= 24
    if grad_ckpt:
        budget *= 1.6
        reasons.append(f"Clips up to {longest:.0f} s on {vram_gb} GB -> gradient checkpointing on (more audio per batch, ~35 % slower).")
    batch = int(max(1, min(16, budget // max(p90, 1.0))))
    reasons.append(f"90 % of clips are <= {p90:.1f} s -> {batch} clips per micro-batch fit in ~{budget:.0f} s of padded audio.")

    # 3) effective batch from clip count (enough optimizer steps per epoch)
    effective = int(min(32, max(4, 2 ** round(math.log2(max(prof.clips / 25.0, 4.0))))))
    accum = int(max(1, math.ceil(effective / batch)))
    effective = batch * accum
    steps_per_epoch = max(1, math.ceil(prof.clips / effective))
    total_steps = steps_per_epoch * epochs
    reasons.append(f"{prof.clips} clips -> effective batch {effective} ({batch} x {accum} accumulation), {steps_per_epoch} steps per epoch.")

    # 4) LR scaled by sqrt(effective batch / 8), kept in a safe band
    lr = min(1.5e-4, max(5e-5, base_lr * math.sqrt(effective / 8.0)))
    if optimizer == "adamw":
        reasons.append(f"AdamW LR {lr:.1e} (base {base_lr:.0e} scaled for batch {effective}), 5 % warmup + cosine to 10 %.")
    else:
        reasons.append(f"{'Prodigy' if optimizer == 'prodigy' else 'Prodigy Schedule-Free'} estimates the LR itself (d_coef 1.0); the LR field is ignored.")

    # 5) evaluation, anti-overfitting and checkpoints
    has_eval = prof.eval_clips >= 2
    if has_eval:
        eval_steps = max(5, round(steps_per_epoch / 2))
        if total_steps / eval_steps < 10:
            eval_steps = max(5, total_steps // 10)
        patience = int(min(6, max(3, math.ceil(1.5 * steps_per_epoch / eval_steps))))
        reasons.append(
            f"Eval every {eval_steps} steps (~{eval_steps / steps_per_epoch:.1f} epoch) on {prof.eval_clips} held-out clips; "
            f"stop after {patience} evals without improvement, keep the best as Final."
        )
        if prof.eval_clips < 12:
            warnings.append(f"Only {prof.eval_clips} eval clips: eval loss is noisy. A validation split of 0.1 with >= 12 clips is better.")
    else:
        eval_steps, patience = 0, 0
        epochs = max(2, round(epochs * 0.6))
        total_steps = steps_per_epoch * epochs
        warnings.append(
            "No validation set: the overfitting guard cannot run. Epochs were reduced to "
            f"{epochs}; rebuild the dataset with a validation split (0.1) to enable early stopping."
        )
    save_steps = max(steps_per_epoch, (eval_steps or steps_per_epoch) * 2)

    # 6) data quality
    if prof.transcript_coverage is not None and prof.transcript_coverage < 0.95:
        warnings.append(f"{(1 - prof.transcript_coverage) * 100:.0f} % of clips have no transcript file; they will fail to load.")
    if longest > budget:
        warnings.append(f"Longest clip is {longest:.0f} s, more than the ~{budget:.0f} s of audio a micro-batch holds on "
                        f"{vram_gb} GB: it is trained too, but it may run out of VRAM. Batch 1 or segmenting long clips helps.")
    short = sum(1 for d in prof.durations if d < 1.5)
    if short > 0.2 * max(prof.clips, 1):
        warnings.append(f"{short} clips are shorter than 1.5 s: they teach little prosody; consider min segment 2 s.")

    # 7) time estimate (training + evals + previews)
    train_cost = prof.seconds * epochs * SEC_PER_AUDIO_SEC * (1.35 if grad_ckpt else 1.0)
    n_evals = (total_steps // eval_steps + 1) if eval_steps else 0
    eval_cost = n_evals * (prof.eval_seconds * EVAL_SEC_PER_AUDIO_SEC + EVAL_AUDIO_SEC)
    reasons.append(
        f"Estimated time at most ~{_fmt_time(train_cost + eval_cost + 60)} on a 3090-class GPU "
        "(less when early stopping triggers)."
    )

    values = {
        "epochs": epochs,
        "max_steps": -1,
        "batch": batch,
        "grad_accum": accum,
        "effective_batch": effective,
        "lr": float(f"{lr:.2g}"),
        "lora_rank": rank,
        "lora_alpha": 0,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "train_audio_modules": False,
        "grad_ckpt": grad_ckpt,
        "max_clip": 30,
        "max_batch_seconds": int(math.ceil(budget)),
        "optimizer": optimizer,
        "prodigy_d_coef": 1.0,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "logging_steps": max(1, min(10, steps_per_epoch // 4 or 1)),
        "patience": patience,
        "min_delta": 0.002,
        "restore_best": True,
        "eval_audio": has_eval,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }
    return TrainingPlan(values=values, reasons=reasons, warnings=warnings)


def plan_markdown(prof: DatasetProfile, plan: TrainingPlan, source: str) -> str:
    v = plan.values
    lines = [
        "### Auto-tune plan",
        "",
        f"**Dataset:** `{source}` · {prof.clips} train clips · {prof.minutes:.1f} min · "
        f"median {prof.pct(0.5):.1f} s · p90 {prof.pct(0.9):.1f} s · max {max(prof.durations or [0]):.1f} s · "
        f"{prof.eval_clips} eval clips",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Optimizer | {v['optimizer']}{'' if v['optimizer'] != 'adamw' else f' · LR {v['lr']:.1e}'} |",
        f"| LoRA rank / alpha | {v['lora_rank']} / {v['lora_rank'] * 2} |",
        f"| Batch × accumulation | {v['batch']} × {v['grad_accum']} = {v['effective_batch']} |",
        f"| Epoch budget | {v['epochs']} ({v['total_steps']} steps, {v['steps_per_epoch']}/epoch) |",
        f"| Eval / save every | {v['eval_steps'] or '—'} / {v['save_steps']} steps |",
        f"| Early stop | {('patience ' + str(v['patience']) + ' evals · Final = best eval') if v['patience'] else 'off (no eval set)'} |",
        f"| Gradient checkpointing | {'on' if v['grad_ckpt'] else 'off'} |",
        "",
        "**Why**",
        "",
        *[f"- {r}" for r in plan.reasons],
    ]
    if plan.warnings:
        lines += ["", "**Warnings**", "", *[f"- ⚠️ {w}" for w in plan.warnings]]
    lines += ["", AUTOTUNE_DISCLAIMER]
    return "\n".join(lines)


AUTOTUNE_DISCLAIMER = (
    "> ℹ️ **This plan is only a starting point and does not guarantee a successful training.** "
    "It is estimated from the dataset's size and clip lengths, not from how the voice actually sounds. "
    "If the result does not sound like the dataset, adjust the values by ear and retrain:\n"
    ">\n"
    "> - **Voice not close enough / generic** → more epochs, a higher LoRA rank, a slightly higher LR "
    "(or Prodigy d_coef 1.5–2), or raise early-stop patience.\n"
    "> - **Artifacts, mumbling, words skipped, or it doesn't stop talking** → it overfitted: use an earlier "
    "checkpoint, fewer epochs, a lower LR (or d_coef 0.5), or a lower rank.\n"
    "> - **Results vary a lot between previews** → a larger effective batch (more gradient accumulation).\n"
    ">\n"
    "> Change one thing at a time and compare the eval previews in the 📈 Training monitor."
)
