from __future__ import annotations

from pathlib import Path

from .paths import ROOT


NONE_ADAPTER = "None"


def _adapter_choices(patterns: list[str]) -> list[str]:
    exp_root = ROOT / "exp"
    exp_root.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for adapter_config in sorted(exp_root.glob(pattern)):
            rel = adapter_config.parent.relative_to(ROOT).as_posix()
            if rel not in seen:
                seen.add(rel)
                candidates.append(adapter_config.parent)

    def sort_key(path: Path):
        rel_parts = path.relative_to(exp_root).parts
        project = rel_parts[0] if rel_parts else ""
        checkpoint = rel_parts[1] if len(rel_parts) > 2 and rel_parts[1].startswith("checkpoint-") else ""
        try:
            checkpoint_num = int(checkpoint.rsplit("-", 1)[1]) if checkpoint else 10**9
        except ValueError:
            checkpoint_num = -1
        # Project-root adapters are "latest" and should appear before snapshots.
        root_rank = 0 if not checkpoint else 1
        return (project.lower(), root_rank, -checkpoint_num, path.as_posix().lower())

    return [path.relative_to(ROOT).as_posix() for path in sorted(candidates, key=sort_key)]


def list_v2_lora_adapters() -> list[str]:
    return [NONE_ADAPTER, *_adapter_choices(["*/lora_adapter/adapter_config.json", "*/*/lora_adapter/adapter_config.json"])]


def list_v3_lora_adapters() -> list[str]:
    return [NONE_ADAPTER, *_adapter_choices(["*/qwen3_lora/adapter_config.json", "*/*/qwen3_lora/adapter_config.json"])]


def list_lora_adapters() -> list[str]:
    choices = [NONE_ADAPTER]
    seen = {NONE_ADAPTER}
    for adapter in [*list_v2_lora_adapters()[1:], *list_v3_lora_adapters()[1:]]:
        if adapter not in seen:
            seen.add(adapter)
            choices.append(adapter)
    return choices


_ADAPTER_DIR = {"v2": "lora_adapter", "v3": "qwen3_lora"}
_FINGERPRINTS: dict[tuple[str, int, int], str] = {}


def normalize_adapter_choice(choice: str | None) -> str:
    """One canonical form for an adapter: ``exp/...`` relative to ROOT, posix slashes."""
    if not choice or choice == NONE_ADAPTER:
        return NONE_ADAPTER
    path = Path(str(choice).strip().strip('"'))
    if path.is_absolute():
        try:
            return path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix().lstrip("./")


def _fingerprint(adapter_dir: Path) -> str:
    """Cheap content id (size + sampled bytes) to spot the final copy of a checkpoint."""
    weights = adapter_dir / "adapter_model.safetensors"
    if not weights.exists():
        weights = adapter_dir / "adapter_model.bin"
    try:
        st = weights.stat()
    except OSError:
        return ""
    key = (str(weights), st.st_size, int(st.st_mtime))
    if key in _FINGERPRINTS:
        return _FINGERPRINTS[key]
    import hashlib

    h = hashlib.sha1(str(st.st_size).encode())
    with open(weights, "rb") as f:
        for offset in (0, st.st_size // 3, (2 * st.st_size) // 3, max(0, st.st_size - 65536)):
            f.seek(offset)
            h.update(f.read(65536))
    _FINGERPRINTS[key] = h.hexdigest()
    return _FINGERPRINTS[key]


def _step_of(name: str) -> int | None:
    try:
        return int(name.rsplit("-", 1)[1]) if name.startswith("checkpoint-") else None
    except ValueError:
        return None


RUN_INFO = "run_info.json"


def _checkpoint_dirs(project_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for path in project_dir.glob("checkpoint-*"):
        step = _step_of(path.name)
        if step is not None and path.is_dir():
            out.append((step, path))
    return out


def current_run_checkpoints(project_dir: Path) -> set[str]:
    """Names of ``checkpoint-N`` folders that belong to the latest training run.

    New runs write ``run_info.json`` with their start time. Older projects that
    were retrained "from scratch" in the same folder can still hold checkpoints
    of a previous run; there the latest run is the newest unbroken sequence:
    walking back in time from the newest checkpoint, steps must keep
    decreasing (a resumed run keeps counting up, so it stays in one sequence);
    a larger step further back in time means an older run.
    """
    ckpts = _checkpoint_dirs(project_dir)
    if not ckpts:
        return set()

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    info = project_dir / RUN_INFO
    if info.exists():
        try:
            import json

            started = float(json.loads(info.read_text(encoding="utf-8"))["started"])
            return {path.name for _, path in ckpts if mtime(path) >= started - 5}
        except Exception:
            pass
    ordered = sorted(ckpts, key=lambda item: mtime(item[1]), reverse=True)
    keep = [ordered[0]]
    for step, path in ordered[1:]:
        prev_step, _prev_path = keep[-1]
        if step >= prev_step:
            break
        keep.append((step, path))
    return {path.name for _, path in keep}


def mark_new_run(project_dir: Path) -> None:
    import json
    import time

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / RUN_INFO).write_text(json.dumps({"started": time.time()}), encoding="utf-8")


def lora_projects(kind: str) -> list[str]:
    """Training projects under ``exp/`` that contain at least one ``kind`` adapter."""
    folder = _ADAPTER_DIR[kind]
    exp_root = ROOT / "exp"
    exp_root.mkdir(parents=True, exist_ok=True)
    projects = []
    for project in sorted((p for p in exp_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        if (project / folder / "adapter_config.json").exists() or any(project.glob(f"checkpoint-*/{folder}/adapter_config.json")):
            projects.append(project.name)
    return projects


def lora_checkpoints(kind: str, project: str | None) -> list[tuple[str, str]]:
    """``(label, value)`` choices for one project: final first, then steps newest first.

    The project-root adapter is the final export and is byte-identical to the
    last checkpoint, so that checkpoint is folded into the "Final" entry.
    """
    if not project or project == NONE_ADAPTER:
        return []
    from datetime import datetime

    folder = _ADAPTER_DIR[kind]
    base = ROOT / "exp" / project
    steps: list[tuple[int, Path]] = []
    current = current_run_checkpoints(base)  # hide snapshots of older runs
    for cfg in base.glob(f"checkpoint-*/{folder}/adapter_config.json"):
        step = _step_of(cfg.parent.parent.name)
        if step is not None and cfg.parent.parent.name in current:
            steps.append((step, cfg.parent))
    steps.sort(key=lambda item: -item[0])

    def when(path: Path) -> str:
        try:
            return datetime.fromtimestamp((path / "adapter_config.json").stat().st_mtime).strftime("%d/%m %H:%M")
        except OSError:
            return ""

    choices: list[tuple[str, str]] = []
    final = base / folder
    if (final / "adapter_config.json").exists():
        final_fp = _fingerprint(final)
        same_step = next((step for step, path in steps if final_fp and _fingerprint(path) == final_fp), None)
        label = f"Final · step {same_step}" if same_step is not None else "Final"
        if (base / "best_info.json").exists() and not (base / "checkpoint-best").exists():
            # a best-only run: the final adapter is its best eval
            try:
                import json

                info = json.loads((base / "best_info.json").read_text(encoding="utf-8"))
                label = f"Final (best) · step {info['step']} · loss {float(info['eval_loss']):.3f}"
            except Exception:
                label = "Final (best)"
        choices.append((f"{label} · {when(final)}", final.relative_to(ROOT).as_posix()))
        if same_step is not None:
            steps = [(s, p) for s, p in steps if s != same_step]
    best = base / "checkpoint-best" / folder
    if (best / "adapter_config.json").exists():
        label = "Best eval"
        try:
            import json

            info = json.loads((base / "checkpoint-best" / "best_info.json").read_text(encoding="utf-8"))
            label = f"Best eval · step {info['step']} · loss {float(info['eval_loss']):.3f}"
        except Exception:
            pass
        choices.insert(1 if choices else 0, (label, best.relative_to(ROOT).as_posix()))
    for step, path in steps:
        choices.append((f"Step {step} · {when(path)}", path.relative_to(ROOT).as_posix()))
    return choices


def adapter_project(choice: str | None) -> str:
    """Project name of an adapter value (``exp/<project>/...``), or None."""
    norm = normalize_adapter_choice(choice)
    parts = Path(norm).parts
    if norm == NONE_ADAPTER or len(parts) < 3 or parts[0] != "exp":
        return NONE_ADAPTER
    return parts[1]


def adapter_kind(choice: str | None) -> str | None:
    if not choice or choice == NONE_ADAPTER:
        return None
    normalized = Path(choice).as_posix()
    if normalized.endswith("/lora_adapter"):
        return "v2"
    if normalized.endswith("/qwen3_lora"):
        return "v3"
    return None


def resolve_lora_adapter(choice: str | None, expected_kind: str | None = None) -> Path | None:
    if not choice or choice == NONE_ADAPTER:
        return None
    choice = normalize_adapter_choice(choice)
    kind = adapter_kind(choice)
    if expected_kind and kind and kind != expected_kind:
        raise ValueError(f"Selected LoRA adapter is for Higgs {kind.upper()}, not Higgs {expected_kind.upper()}: {choice}")
    path = ROOT / choice if not Path(choice).is_absolute() else Path(choice)
    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {path}")
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"LoRA adapter_config.json not found: {path}")
    return path


def resolve_v2_lora_adapter(choice: str | None) -> Path | None:
    return resolve_lora_adapter(choice, "v2")


def resolve_v3_lora_adapter(choice: str | None) -> Path | None:
    return resolve_lora_adapter(choice, "v3")
