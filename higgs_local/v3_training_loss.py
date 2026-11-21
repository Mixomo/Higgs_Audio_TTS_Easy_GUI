from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

BOC_ID = 1024  # codec "begin of codes" padding of the delay pattern


@dataclass
class V3Example:
    """One training example: transcript + codec codes ([T, N] long, CPU)."""

    text: str
    target_codes: torch.Tensor
    reference_codes: torch.Tensor | None = None
    reference_text: str | None = None


def _modeling(model):
    return __import__(type(model).__module__, fromlist=["apply_delay_pattern"])


def _build_inputs(model, tokenizer, ex: V3Example):
    """Prompt embeds + teacher-forced audio embeds and the delayed target rows."""
    apply_delay_pattern = _modeling(model).apply_delay_pattern
    device = model.device
    delayed_ref = None
    if ex.reference_codes is not None:
        delayed_ref = apply_delay_pattern(ex.reference_codes.to(torch.long)).to(device)
    target = apply_delay_pattern(ex.target_codes.to(torch.long)).to(device)  # [L, N]
    prompt_ids = model._build_prompt_ids(
        tokenizer,
        ex.text.strip(),
        num_ref_tokens=0 if delayed_ref is None else delayed_ref.shape[0],
        reference_text=ex.reference_text if delayed_ref is not None else None,
    )
    prompt = model._prefill_embeds(prompt_ids, delayed_ref)[0]  # [P, D]
    if target.shape[0] > 1:
        prev = model.audio_embedding(target[:-1]).to(prompt.dtype)  # [L-1, D]
        embeds = torch.cat([prompt, prev], dim=0)
    else:
        embeds = prompt
    return embeds, target, prompt.shape[0]


def compute_v3_batch_loss(model, tokenizer, examples: list[V3Example]) -> tuple[torch.Tensor, dict]:
    """Teacher-forced Higgs V3 loss over a padded batch.

    Mirrors generation exactly: prompt (text [+ reference]) then the delayed
    8-codebook rows; position ``P-1+t`` predicts delayed row ``t``. Text and
    reference tokens are inputs only. BOC rows of the delay ramp are masked
    because inference forces them, so the model never has to predict them;
    EOC (end of codes) stays supervised so the model learns when to stop.
    The loss is a token-level mean across the batch and codebooks.
    """
    if not examples:
        raise ValueError("empty batch")
    built = [_build_inputs(model, tokenizer, ex) for ex in examples]
    max_len = max(e.shape[0] for e, _, _ in built)
    hidden_size = built[0][0].shape[-1]
    dtype = built[0][0].dtype
    device = model.device
    batch = torch.zeros(len(built), max_len, hidden_size, device=device, dtype=dtype)
    mask = torch.zeros(len(built), max_len, device=device, dtype=torch.long)
    for i, (embeds, _, _) in enumerate(built):
        batch[i, : embeds.shape[0]] = embeds
        mask[i, : embeds.shape[0]] = 1

    out = model.model(inputs_embeds=batch, attention_mask=mask, use_cache=False)
    hidden = out.last_hidden_state

    rows_h, rows_y = [], []
    for i, (_, target, prompt_len) in enumerate(built):
        rows_h.append(hidden[i, prompt_len - 1 : prompt_len - 1 + target.shape[0]])
        rows_y.append(target)
    logits = model.audio_head(torch.cat(rows_h, dim=0)).float()  # [sum L, N, V]
    labels = torch.cat(rows_y, dim=0).clone()  # [sum L, N]
    labels[labels == BOC_ID] = -100
    n_cb = labels.shape[1]
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).view(-1, n_cb)
    valid = (labels != -100).float()
    loss = (per_token * valid).sum() / valid.sum().clamp_min(1.0)
    with torch.no_grad():
        cb_loss = (per_token * valid).sum(0) / valid.sum(0).clamp_min(1.0)
    stats = {
        "frames": int(labels.shape[0]),
        "tokens": int(valid.sum().item()),
        "cb0": round(float(cb_loss[0]), 4),
        "cb_rest": round(float(cb_loss[1:].mean()), 4) if n_cb > 1 else None,
    }
    return loss, stats


def compute_v3_audio_code_loss(
    model,
    tokenizer,
    *,
    text: str,
    target_audio: torch.Tensor | None = None,
    target_sample_rate: int | None = None,
    target_codes: torch.Tensor | None = None,
    reference_audio: torch.Tensor | None = None,
    reference_sample_rate: int | None = None,
    reference_codes: torch.Tensor | None = None,
    reference_text: str | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict]:
    """Single-example wrapper kept for backwards compatibility."""
    if not text or not text.strip():
        raise ValueError("text is required for V3 training loss")
    if target_codes is None:
        if target_audio is None:
            raise ValueError("target_audio or target_codes is required for V3 training loss")
        target_codes = model._encode_reference(target_audio, target_sample_rate or getattr(model.config, "sample_rate", 24000))
    if reference_codes is None and reference_audio is not None:
        reference_codes = model._encode_reference(reference_audio, reference_sample_rate or getattr(model.config, "sample_rate", 24000))
    return compute_v3_batch_loss(
        model,
        tokenizer,
        [V3Example(text=text, target_codes=target_codes, reference_codes=reference_codes, reference_text=reference_text)],
    )
