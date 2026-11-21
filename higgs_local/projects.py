"""Projects shared by Dataset Preparation and Training.

A project is one name used everywhere:

    data/<name>/{train,eval}/       dataset built by the Dataset tab
    exp/<name>/                     training runs, checkpoints, LoRA
    projects/<name>/project.json    saved settings of both stages

``project.json`` keeps the dataset settings (source folder, ASR, clean-up) and
the training choices so selecting a project restores them, and the status card
summarises the whole pipeline: dataset -> training -> voice ready.
"""

from __future__ import annotations

import html
import json
import os
import time
from datetime import datetime
from pathlib import Path

from .adapters import current_run_checkpoints
from .paths import ROOT
from .training_utils import slugify

DATA_DIR = ROOT / "data"
EXP_DIR = ROOT / "exp"
PROJECTS_DIR = ROOT / "projects"
SCHEMA_VERSION = 1


def safe_project_name(name: str | None) -> str:
    return slugify(name or "", "")


def list_projects() -> list[str]:
    """Project names across data/, exp/ and projects/ (case-insensitive, like Windows).

    When two folders differ only in case (data/fagnola_test vs exp/Fagnola_test)
    they are the same project; the training folder's spelling wins.
    """
    names: dict[str, str] = {}
    for root in (PROJECTS_DIR, DATA_DIR, EXP_DIR):  # later roots override spelling
        if root.exists():
            for p in root.iterdir():
                if p.is_dir() and not p.name.startswith((".", "_")):
                    names[p.name.lower()] = p.name
    return sorted(names.values(), key=str.lower)


def _stored_training_dirs(safe: str) -> tuple[str, str]:
    """train/eval dirs saved by a previous training of this project (may differ in name)."""
    cfg = EXP_DIR / safe / "project_config.json"
    try:
        args = json.loads(cfg.read_text(encoding="utf-8")).get("args") or {}
    except Exception:
        return "", ""

    def rel(value) -> str:
        if not value:
            return ""
        path = Path(str(value))
        try:
            return path.resolve().relative_to(ROOT.resolve()).as_posix() if path.is_absolute() else path.as_posix()
        except ValueError:
            return path.as_posix()

    return rel(args.get("train_data_dir")), rel(args.get("eval_data_dir"))


# ------------------------------------------------------------------ metadata
def _project_file(name: str) -> Path:
    return PROJECTS_DIR / safe_project_name(name) / "project.json"


def load_project(name: str | None) -> dict:
    if not safe_project_name(name):
        return {}
    try:
        return json.loads(_project_file(name).read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_project_section(name: str, section: str, values: dict) -> None:
    safe = safe_project_name(name)
    if not safe:
        return
    data = load_project(safe)
    data.update({"schema_version": SCHEMA_VERSION, "project": safe, "updated_at": datetime.now().isoformat(timespec="seconds")})
    data.setdefault("created_at", data["updated_at"])
    data[section] = {**data.get(section, {}), **values}
    path = _project_file(safe)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------- stats
def _dataset_info(split_dir: Path) -> dict | None:
    meta = split_dir / "metadata.json"
    if not meta.exists():
        return None
    try:
        info = json.loads(meta.read_text(encoding="utf-8")).get("dataset_info") or {}
        return {"clips": int(info.get("total_samples", 0)), "seconds": float(info.get("total_duration", 0.0)), "avg": float(info.get("avg_duration", 0.0))}
    except Exception:
        return None


def dataset_dirs(name: str) -> tuple[str, str]:
    """(train_dir, eval_dir) relative to ROOT for the project's dataset.

    Prefers data/<name>; falls back to the dataset a previous training of this
    project used (projects created before datasets shared the project name).
    """
    safe = safe_project_name(name)
    train = DATA_DIR / safe / "train"
    evald = DATA_DIR / safe / "eval"
    rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
    if (train / "metadata.json").exists():
        return rel(train), rel(evald) if (evald / "metadata.json").exists() else ""
    stored_train, stored_eval = _stored_training_dirs(safe)
    if stored_train and (ROOT / stored_train / "metadata.json").exists():
        return stored_train, stored_eval if stored_eval and (ROOT / stored_eval / "metadata.json").exists() else ""
    return "", ""


def project_stats(name: str | None) -> dict:
    safe = safe_project_name(name)
    stats: dict = {"name": safe}
    if not safe:
        return stats
    train_dir, eval_dir = DATA_DIR / safe / "train", DATA_DIR / safe / "eval"
    if not (train_dir / "metadata.json").exists():
        stored_train, stored_eval = _stored_training_dirs(safe)
        if stored_train:
            train_dir = ROOT / stored_train
            eval_dir = ROOT / stored_eval if stored_eval else eval_dir
    train = _dataset_info(train_dir)
    evals = _dataset_info(eval_dir)
    if train:
        stats["dataset"] = {
            "path": train_dir.parent.relative_to(ROOT).as_posix() if train_dir.is_relative_to(ROOT) else str(train_dir.parent),
            "train_clips": train["clips"],
            "eval_clips": evals["clips"] if evals else 0,
            "minutes": (train["seconds"] + (evals["seconds"] if evals else 0.0)) / 60.0,
            "avg": train["avg"],
        }
    exp = EXP_DIR / safe
    if exp.exists():
        kind = "V3" if (exp / "qwen3_lora").exists() or any(exp.glob("checkpoint-*/qwen3_lora")) else (
            "V2" if (exp / "lora_adapter").exists() or any(exp.glob("checkpoint-*/lora_adapter")) else None
        )
        steps = sorted(int(n.split("-")[1]) for n in current_run_checkpoints(exp))
        training = {"kind": kind, "checkpoints": len(steps), "last_step": steps[-1] if steps else None,
                    "final": (exp / "qwen3_lora").exists() or (exp / "lora_adapter").exists()}
        try:
            best = json.loads((exp / "checkpoint-best" / "best_info.json").read_text(encoding="utf-8"))
            training["best"] = (int(best["step"]), float(best["eval_loss"]))
        except Exception:
            pass
        stamps = [p.stat().st_mtime for p in exp.glob("checkpoint-*")] + [
            (exp / f).stat().st_mtime for f in ("trainer_state.pt",) if (exp / f).exists()
        ]
        if stamps:
            training["last_trained"] = datetime.fromtimestamp(max(stamps)).strftime("%d/%m %H:%M")
        stats["training"] = training
    return stats


def project_card_html(name: str | None, running: str | None = None) -> str:
    """Pipeline card: 1 Dataset -> 2 Training -> 3 Voice ready."""
    stats = project_stats(name)
    safe = stats.get("name")
    if not safe:
        return (
            '<div class="proj-card proj-empty"><div class="proj-title">No project selected</div>'
            '<div class="proj-sub">Type a new name or pick one: the same project is shared by Dataset Preparation and Training.</div></div>'
        )
    ds = stats.get("dataset")
    tr = stats.get("training") or {}
    has_training = bool(tr.get("checkpoints") or tr.get("final"))
    voice_ready = bool(tr.get("final") or tr.get("best"))

    def step(num: int, title: str, state: str, detail: str) -> str:
        icon = {"done": "✓", "active": "●", "todo": str(num)}[state]
        return (
            f'<div class="proj-step {state}"><div class="proj-dot">{icon}</div>'
            f'<div><div class="proj-step-title">{html.escape(title)}</div><div class="proj-step-detail">{html.escape(detail)}</div></div></div>'
        )

    if ds:
        ds_detail = f"{ds['train_clips']} train + {ds['eval_clips']} eval clips · {ds['minutes']:.1f} min · avg {ds['avg']:.1f}s"
        if ds.get("path") and ds["path"].lower() != f"data/{safe}".lower():
            ds_detail += f" · {ds['path']}"
    else:
        ds_detail = "Not built yet: build it in Dataset Preparation"
    if running == safe:
        tr_state, tr_detail = "active", "Training now…"
    elif has_training:
        parts = [f"{tr.get('kind') or ''} LoRA".strip()]
        if tr.get("last_step") is not None:
            parts.append(f"{tr['checkpoints']} checkpoints · last step {tr['last_step']}")
        if tr.get("last_trained"):
            parts.append(tr["last_trained"])
        tr_state, tr_detail = "done", " · ".join(parts)
    else:
        tr_state, tr_detail = "todo", "Not trained yet" if ds else "Needs a dataset first"
    if voice_ready:
        best = tr.get("best")
        v_detail = f"Best eval {best[1]:.3f} @ step {best[0]}" if best else "Final adapter available"
        v_detail += " · pick it as LoRA voice in Inference"
        v_state = "done"
    else:
        v_state, v_detail = "todo", "Appears in Inference once training saves a checkpoint"
    return (
        f'<div class="proj-card"><div class="proj-head"><div class="proj-title">{html.escape(safe)}</div>'
        f'<div class="proj-sub">data/{html.escape(safe)} · exp/{html.escape(safe)}</div></div>'
        f'<div class="proj-steps">{step(1, "Dataset", "done" if ds else "todo", ds_detail)}'
        f'{step(2, "Training", tr_state, tr_detail)}{step(3, "Voice ready", v_state, v_detail)}</div></div>'
    )


def open_project_folder(name: str | None) -> str:
    safe = safe_project_name(name)
    if not safe:
        return "Select a project first."
    for folder in (EXP_DIR / safe, DATA_DIR / safe, PROJECTS_DIR / safe):
        if folder.exists():
            try:
                os.startfile(str(folder))  # type: ignore[attr-defined]  # Windows
            except AttributeError:
                import subprocess

                subprocess.Popen(["xdg-open", str(folder)])
            return f"Opened {folder}"
    return f"Project {safe} has no folders yet."


def touch_project(name: str) -> None:
    """Create projects/<name>/project.json for a brand-new project."""
    if safe_project_name(name) and not _project_file(name).exists():
        save_project_section(name, "meta", {"created": time.strftime("%Y-%m-%d %H:%M")})
