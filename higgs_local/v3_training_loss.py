from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_v3_audio_code_loss(
    model,
    tokenizer,
    *,
    text: str,
    target_audio: torch.Tensor,
    target_sample_rate: int,
    reference_audio: torch.Tensor | None = None,
    reference_sample_rate: int | None = None,
    reference_text: str | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict]:
    """Teacher-forced Higgs V3 audio-code loss.

    This is the core patch needed for a V3 trainer. It mirrors generation:

    - build the text/reference prompt;
    - encode target audio to codec codes;
    - apply the V3 delay pattern;
    - feed the prompt plus previous delayed target rows;
    - project hidden states through ``audio_head``;
    - compute CE against the next delayed target row for each codebook.

    The codec stays frozen because the model's encode/decode helpers are already
    no-grad. Gradients flow through the Qwen3 backbone, audio embedding, and
    audio head unless callers freeze or LoRA-wrap those modules.
    """
    if not text or not text.strip():
        raise ValueError("text is required for V3 training loss")
    if target_audio is None:
        raise ValueError("target_audio is required for V3 training loss")

    device = model.device
    modeling_module = __import__(type(model).__module__, fromlist=["apply_delay_pattern"])
    apply_delay_pattern = modeling_module.apply_delay_pattern

    delayed_ref = None
    if reference_audio is not None:
        ref_sr = reference_sample_rate or getattr(model.config, "sample_rate", 24000)
        ref_codes = model._encode_reference(reference_audio, ref_sr)
        delayed_ref = apply_delay_pattern(ref_codes.to(torch.long)).to(device)

    target_codes = model._encode_reference(target_audio, target_sample_rate)
    target_delayed = apply_delay_pattern(target_codes.to(torch.long)).to(device)
    if target_delayed.ndim != 2:
        raise ValueError(f"expected delayed codes [L, N], got {tuple(target_delayed.shape)}")

    prompt_ids = model._build_prompt_ids(
        tokenizer,
        text.strip(),
        num_ref_tokens=0 if delayed_ref is None else delayed_ref.shape[0],
        reference_text=reference_text,
    )
    prompt_embeds = model._prefill_embeds(prompt_ids, delayed_ref)

    if target_delayed.shape[0] > 1:
        prev_audio_embeds = model.audio_embedding(target_delayed[:-1]).unsqueeze(0)
        inputs_embeds = torch.cat([prompt_embeds, prev_audio_embeds.to(prompt_embeds.dtype)], dim=1)
    else:
        inputs_embeds = prompt_embeds

    outputs = model.model(inputs_embeds=inputs_embeds, use_cache=False)
    prompt_len = prompt_embeds.shape[1]
    prediction_hidden = outputs.last_hidden_state[:, prompt_len - 1 : prompt_len - 1 + target_delayed.shape[0], :]
    logits = model.audio_head(prediction_hidden[0]).to(torch.float32)

    labels = target_delayed
    if ignore_index >= 0:
        labels = labels.clone()
        labels[labels < 0] = ignore_index
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
    stats = {
        "frames": int(target_delayed.shape[0]),
        "num_codebooks": int(target_delayed.shape[1]),
        "prompt_tokens": int(prompt_len),
        "codebook_vocab_size": int(logits.shape[-1]),
    }
    return loss, stats
