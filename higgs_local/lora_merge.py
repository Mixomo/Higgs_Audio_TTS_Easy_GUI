"""Merge PEFT LoRA adapters straight into the loaded Higgs V3 weights.

Merging (instead of wrapping the backbone with PEFT) means inference runs at
exactly base-model speed and the fast decoder's CUDA graphs keep working: the
weights are updated *in place*, so every tensor keeps its address.

Switching adapters restores the touched tensors from the base checkpoint first
(bit-exact, no RAM copy needed) and then merges the new adapter.

Supported: plain LoRA / rsLoRA (incl. ``rank_pattern`` / ``alpha_pattern``) on
the Qwen3 backbone linears, plus the optional ``higgs_v3_audio_modules.pt``
(full audio embedding/head weights) written by the training scripts.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch

_BACKBONE_PREFIX = "body."  # checkpoint name of ``model.model``
_AUDIO_EMBED_KEY = "tied.embedding.modality_embeddings.0.embedding.weight"


class UnsupportedAdapter(RuntimeError):
    pass


def _pattern_value(patterns: dict, module_name: str, default):
    for key, value in (patterns or {}).items():
        if module_name == key or module_name.endswith("." + key) or re.fullmatch(key, module_name):
            return value
    return default


class LoraMerger:
    def __init__(self, model, model_dir: str | Path):
        self.model = model
        self.model_dir = Path(model_dir)
        self.backbone = model.model
        self.active: str | None = None
        self._touched: list[str] = []  # backbone-relative module names
        self._audio_touched = False
        self._index = self._load_index()

    # ----------------------------------------------------------- checkpoint
    def _load_index(self) -> dict[str, str]:
        index_file = self.model_dir / "model.safetensors.index.json"
        if index_file.exists():
            data = json.loads(index_file.read_text(encoding="utf-8"))
            return {k: str(self.model_dir / v) for k, v in data.get("weight_map", {}).items()}
        single = self.model_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"Base checkpoint not found in {self.model_dir}")
        from safetensors import safe_open

        with safe_open(str(single), "pt") as f:
            return {k: str(single) for k in f.keys()}

    def _read_base(self, keys: list[str]) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        by_file: dict[str, list[str]] = {}
        for key in keys:
            if key not in self._index:
                raise KeyError(f"Base weight {key} not found in checkpoint index")
            by_file.setdefault(self._index[key], []).append(key)
        out: dict[str, torch.Tensor] = {}
        for file, file_keys in by_file.items():
            with safe_open(file, "pt") as f:
                for key in file_keys:
                    out[key] = f.get_tensor(key)
        return out

    # ------------------------------------------------------------- restore
    @torch.no_grad()
    def restore_base(self) -> None:
        if not self._touched and not self._audio_touched:
            self.active = None
            return
        keys = [f"{_BACKBONE_PREFIX}{name}.weight" for name in self._touched]
        if self._audio_touched:
            keys.append(_AUDIO_EMBED_KEY)
        base = self._read_base(keys)
        for name in self._touched:
            weight = self.backbone.get_submodule(name).weight
            weight.copy_(base[f"{_BACKBONE_PREFIX}{name}.weight"].to(weight.device, weight.dtype))
        if self._audio_touched:
            emb = self.model.audio_embedding.weight
            emb.copy_(base[_AUDIO_EMBED_KEY].to(emb.device, emb.dtype))
            head = self.model.audio_head.weight
            if head.data_ptr() != emb.data_ptr():
                head.copy_(emb)
        self._touched = []
        self._audio_touched = False
        self.active = None

    # --------------------------------------------------------------- apply
    @torch.no_grad()
    def apply(self, adapter_dir: str | Path | None) -> str | None:
        """Make ``adapter_dir`` the merged adapter (``None`` = base model)."""
        target = str(Path(adapter_dir).resolve()) if adapter_dir else None
        if target == self.active:
            return self.active
        self.restore_base()
        if target is None:
            return None
        adapter_dir = Path(target)
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        if cfg.get("peft_type", "LORA") != "LORA":
            raise UnsupportedAdapter(f"Only LoRA adapters can be merged (got {cfg.get('peft_type')}).")
        if cfg.get("use_dora"):
            raise UnsupportedAdapter("DoRA adapters are not supported by the fast merge path.")
        if cfg.get("modules_to_save"):
            raise UnsupportedAdapter("Adapters with modules_to_save are not supported by the fast merge path.")
        if cfg.get("fan_in_fan_out"):
            raise UnsupportedAdapter("fan_in_fan_out adapters are not supported by the fast merge path.")
        weights_file = adapter_dir / "adapter_model.safetensors"
        if weights_file.exists():
            from safetensors.torch import load_file

            state = load_file(str(weights_file))
        else:
            state = torch.load(adapter_dir / "adapter_model.bin", map_location="cpu", weights_only=True)

        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in state.items():
            m = re.match(r"^(?:base_model\.model\.)?(.+)\.lora_([AB])(?:\.default)?\.weight$", key)
            if m:
                pairs.setdefault(m.group(1), {})[m.group(2)] = tensor
        if not pairs:
            raise UnsupportedAdapter(f"No LoRA weights found in {weights_file}")

        r_default = int(cfg.get("r", 8))
        alpha_default = float(cfg.get("lora_alpha", r_default))
        rslora = bool(cfg.get("use_rslora", False))
        try:
            for name, ab in sorted(pairs.items()):
                if "A" not in ab or "B" not in ab:
                    raise UnsupportedAdapter(f"Incomplete LoRA pair for {name}")
                module = self.backbone.get_submodule(name)
                r = int(_pattern_value(cfg.get("rank_pattern"), name, r_default))
                alpha = float(_pattern_value(cfg.get("alpha_pattern"), name, alpha_default))
                scale = alpha / (math.sqrt(r) if rslora else r)
                w = module.weight
                a = ab["A"].to(w.device, torch.float32)
                b = ab["B"].to(w.device, torch.float32)
                delta = (b @ a) * scale
                if delta.shape != w.shape:
                    raise UnsupportedAdapter(f"LoRA shape {tuple(delta.shape)} != weight {tuple(w.shape)} for {name}")
                w.copy_((w.float() + delta).to(w.dtype))
                self._touched.append(name)
            modules_path = adapter_dir / "higgs_v3_audio_modules.pt"
            if modules_path.exists():
                modules = torch.load(modules_path, map_location="cpu", weights_only=False)
                if "audio_embedding" in modules:
                    self.model.audio_embedding.load_state_dict(modules["audio_embedding"], strict=False)
                    self._audio_touched = True
                if "audio_head" in modules:
                    self.model.audio_head.load_state_dict(modules["audio_head"], strict=False)
                    self._audio_touched = True
        except Exception:
            self.restore_base()
            raise
        self.active = target
        return self.active
