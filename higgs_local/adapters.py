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
