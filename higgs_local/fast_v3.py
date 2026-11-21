"""Fast Higgs Audio V3 decoder: static KV cache + CUDA graphs + on-GPU sampling.

The reference decode loop (``modeling_higgs_multimodal_qwen3.generate_speech``)
runs the HF Qwen3 forward eagerly for every 40 ms audio frame and syncs with
the CPU each frame (``codes.cpu()`` / ``.item()``). That makes it launch-bound:
hundreds of tiny kernels per frame, each paying Python + driver overhead.

This module keeps the exact same math but:

* re-implements the single-token Qwen3 step directly on the model's own
  modules/weights (so merged LoRA weights are used transparently),
* writes K/V into a preallocated static cache,
* runs head -> sampling -> delay mask -> re-embedding -> backbone entirely on
  the GPU and captures one full frame as a CUDA graph (one graph per cache
  length bucket, so attention only reads the used part of the cache),
* replays the graph many times back to back and only syncs every few frames to
  look for the end-of-codes marker and to hand rows to the streaming decoder.

Sampling is distributionally identical to the reference ``_sample`` (temperature,
top-k, top-p, multinomial) but uses the exponential-race trick so it is graph
safe; sequences for a given seed therefore differ from the legacy engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
import torch.nn.functional as F

BOC_ID = 1024
EOC_ID = 1025

_MIN_BUCKET = 512


def _bucket_for(length: int) -> int:
    return max(_MIN_BUCKET, 1 << max(0, math.ceil(math.log2(max(1, length)))))


def _rms(norm: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Same math as Qwen3RMSNorm.forward, kept inline so it is obviously graph safe.
    dtype = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * h.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@dataclass
class StepStats:
    frames: int = 0
    seconds: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


class FastV3Decoder:
    """Graph-captured AR decoder bound to one loaded Higgs V3 model."""

    def __init__(self, model, sync_every: int = 16):
        if not torch.cuda.is_available() or model.device.type != "cuda":
            raise RuntimeError("FastV3Decoder needs the model on a CUDA device.")
        self.model = model
        self.backbone = model.model
        # PEFT wraps the backbone; the fast path needs the merged plain modules.
        base = getattr(self.backbone, "get_base_model", None)
        if callable(base):
            self.backbone = base()
        self.layers = list(self.backbone.layers)
        cfg = self.backbone.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.hidden = cfg.hidden_size
        self.max_positions = int(cfg.max_position_embeddings)
        self.N = int(model.num_codebooks)
        self.V = int(model.codebook_vocab_size)
        self.device = model.device
        self.dtype = self.backbone.embed_tokens.weight.dtype
        self.sync_every = max(1, int(sync_every))

        self.cache_len = 0
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []
        self.cos_tab: torch.Tensor | None = None
        self.sin_tab: torch.Tensor | None = None
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = None

        dev = self.device
        self.hidden_buf = torch.zeros(1, self.hidden, device=dev, dtype=self.dtype)
        self.step_t = torch.zeros((), device=dev, dtype=torch.long)
        self.pos_t = torch.zeros(1, device=dev, dtype=torch.long)
        self.temp_t = torch.ones((), device=dev, dtype=torch.float32)
        self.topk_t = torch.full((), self.V, device=dev, dtype=torch.long)
        self.topp_t = torch.ones((), device=dev, dtype=torch.float32)
        self.cb_idx = torch.arange(self.N, device=dev)
        self.cb_offsets = self.cb_idx * self.V
        self.out_rows: torch.Tensor | None = None
        self.stats = StepStats()
        self.noise = torch.ones(self.sync_every, self.N, self.V, device=dev, dtype=torch.float32)
        self.rng = torch.Generator(device=dev)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv * self.head_dim
        self.group = self.num_heads // self.num_kv
        self.fuse_weights()

    # ------------------------------------------------------------- weights
    def fuse_weights(self) -> None:
        """Store q/k/v and gate/up as one matrix per layer (fewer, larger reads).

        The original Linear modules keep working: their ``weight`` becomes a
        view into the fused tensor, so memory use does not grow and in-place
        updates (LoRA merge/unmerge) stay visible to both.
        """
        self.w_qkv: list[torch.Tensor] = []
        self.w_gu: list[torch.Tensor] = []
        for layer in self.layers:
            attn, mlp = layer.self_attn, layer.mlp
            mods = (attn.q_proj, attn.k_proj, attn.v_proj)
            if any(getattr(m, "bias", None) is not None for m in (*mods, mlp.gate_proj, mlp.up_proj)):
                raise RuntimeError("FastV3Decoder expects bias-free Qwen3 projections.")
            self.w_qkv.append(self._fuse(mods))
            self.w_gu.append(self._fuse((mlp.gate_proj, mlp.up_proj)))
        torch.cuda.empty_cache()

    @staticmethod
    def _fuse(mods) -> torch.Tensor:
        # Already fused by an earlier decoder on the same model?
        prev = getattr(mods[0], "_higgs_fused", None)
        if prev is not None and all(getattr(m, "_higgs_fused", None) is prev for m in mods):
            return prev
        fused = torch.cat([m.weight.detach() for m in mods], dim=0).contiguous()
        start = 0
        for m in mods:
            rows = m.weight.shape[0]
            m.weight = torch.nn.Parameter(fused[start:start + rows], requires_grad=False)
            m._higgs_fused = fused
            start += rows
        return fused

    # ------------------------------------------------------------------ setup
    def _ensure_cache(self, needed: int) -> None:
        needed = min(_bucket_for(needed), _bucket_for(self.max_positions))
        if needed <= self.cache_len:
            return
        self.graphs.clear()
        self.pool = None
        self.k_cache = []
        self.v_cache = []
        torch.cuda.empty_cache()
        shape = (1, self.num_kv, needed, self.head_dim)
        for _ in self.layers:
            self.k_cache.append(torch.zeros(shape, device=self.device, dtype=self.dtype))
            self.v_cache.append(torch.zeros(shape, device=self.device, dtype=self.dtype))
        pos = torch.arange(needed, device=self.device).unsqueeze(0)
        probe = torch.zeros(1, 1, self.hidden, device=self.device, dtype=self.dtype)
        cos, sin = self.backbone.rotary_emb(probe, pos)
        self.cos_tab = cos[0].to(self.dtype).contiguous()  # [L, head_dim]
        self.sin_tab = sin[0].to(self.dtype).contiguous()
        self.cache_len = needed

    def _ensure_rows(self, max_new_tokens: int) -> None:
        need = int(max_new_tokens) + self.sync_every + 1
        if self.out_rows is None or self.out_rows.shape[0] < need:
            self.graphs.clear()
            self.pool = None
            self.out_rows = torch.full((need, self.N), BOC_ID, device=self.device, dtype=torch.long)

    # --------------------------------------------------------------- backbone
    def _layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               *, kv_len: int | None, pos: torch.Tensor | None) -> torch.Tensor:
        """One Qwen3 decoder layer. Prefill when ``pos`` is None (writes [0, S))."""
        layer = self.layers[i]
        attn = layer.self_attn
        B, S, _ = x.shape
        h = _rms(layer.input_layernorm, x)
        qkv = F.linear(h, self.w_qkv[i])
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = _rms(attn.q_norm, q.view(B, S, self.num_heads, self.head_dim)).transpose(1, 2)
        k = _rms(attn.k_norm, k.view(B, S, self.num_kv, self.head_dim)).transpose(1, 2)
        v = v.reshape(B, S, self.num_kv, self.head_dim).transpose(1, 2)
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s
        if pos is None:
            self.k_cache[i][:, :, :S].copy_(k)
            self.v_cache[i][:, :, :S].copy_(v)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            o = o.transpose(1, 2).reshape(B, S, self.q_size)
        else:
            self.k_cache[i].index_copy_(2, pos, k)
            self.v_cache[i].index_copy_(2, pos, v)
            keys = self.k_cache[i][0, :, :kv_len]  # [KV, L, D]
            vals = self.v_cache[i][0, :, :kv_len]
            # Single query, grouped heads: [KV, group, D] @ [KV, D, L] -> no K/V repeat.
            qg = q.view(self.num_kv, self.group, self.head_dim)
            scores = torch.bmm(qg, keys.transpose(1, 2)).float() * (self.head_dim ** -0.5)
            mask = torch.arange(kv_len, device=x.device) > pos
            scores = scores.masked_fill(mask.view(1, 1, kv_len), float("-inf"))
            probs = scores.softmax(dim=-1).to(vals.dtype)
            o = torch.bmm(probs, vals).reshape(B, S, self.q_size)
        x = x + attn.o_proj(o)
        h = _rms(layer.post_attention_layernorm, x)
        gate, up = F.linear(h, self.w_gu[i]).chunk(2, dim=-1)
        return x + layer.mlp.down_proj(F.silu(gate) * up)

    @torch.no_grad()
    def prefill(self, inputs_embeds: torch.Tensor) -> int:
        S = int(inputs_embeds.shape[1])
        x = inputs_embeds.to(self.dtype)
        cos, sin = self.cos_tab[:S], self.sin_tab[:S]
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, kv_len=None, pos=None)
        x = _rms(self.backbone.norm, x)
        self.hidden_buf.copy_(x[:, -1, :])
        return S

    # ---------------------------------------------------------------- one frame
    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Graph-safe equivalent of the reference ``_sample`` ([N, V] -> [N])."""
        greedy = logits.argmax(dim=-1)
        scaled = logits / self.temp_t.clamp_min(1e-5)
        sorted_vals, sorted_idx = scaled.sort(dim=-1, descending=True)
        k = self.topk_t.clamp(1, self.V)
        kth = sorted_vals.gather(-1, (k - 1).view(1, 1).expand(self.N, 1))
        sorted_vals = sorted_vals.masked_fill(sorted_vals < kth, float("-inf"))
        probs = sorted_vals.softmax(dim=-1)
        before = probs.cumsum(dim=-1) - probs  # mass strictly above each token
        drop = (before > self.topp_t) & (self.topp_t < 1.0)
        probs = probs.masked_fill(drop, 0.0)
        # Exp(1) noise comes from a buffer refilled outside the graph by a
        # dedicated generator, so seeds are reproducible (graph capture/warmup
        # would otherwise consume the global RNG stream).
        race = self.noise.index_select(0, self.step_t.remainder(self.sync_every).view(1))[0]
        pick = (probs / race).argmax(dim=-1, keepdim=True)
        sampled = sorted_idx.gather(-1, pick).squeeze(-1)
        return torch.where(self.temp_t <= 1e-5, greedy, sampled)

    def _frame(self, kv_len: int) -> None:
        logits = F.linear(self.hidden_buf, self.model.audio_head.weight).float().view(self.N, self.V)
        codes = self._sample(logits).long()
        codes = torch.where(self.cb_idx > self.step_t, torch.full_like(codes, BOC_ID), codes)
        self.out_rows.index_copy_(0, self.step_t.view(1), codes.view(1, self.N))
        emb = F.embedding(codes + self.cb_offsets, self.model.audio_embedding.weight).sum(0)
        x = emb.view(1, 1, self.hidden).to(self.dtype)
        cos = self.cos_tab.index_select(0, self.pos_t)
        sin = self.sin_tab.index_select(0, self.pos_t)
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, kv_len=kv_len, pos=self.pos_t)
        x = _rms(self.backbone.norm, x)
        self.hidden_buf.copy_(x[:, -1, :])
        self.step_t.add_(1)
        self.pos_t.add_(1)

    def _graph(self, kv_len: int) -> torch.cuda.CUDAGraph:
        g = self.graphs.get(kv_len)
        if g is not None:
            return g
        # Snapshot mutable state: warmup + capture run the frame for real.
        saved = [t.clone() for t in (self.hidden_buf, self.step_t, self.pos_t, self.out_rows)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._frame(kv_len)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=self.pool):
            self._frame(kv_len)
        for dst, src in zip((self.hidden_buf, self.step_t, self.pos_t, self.out_rows), saved):
            dst.copy_(src)
        # The warmup wrote K/V at the current position; the real replay overwrites it.
        self.graphs[kv_len] = g
        return g

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def generate_rows(
        self,
        inputs_embeds: torch.Tensor,
        *,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        should_stop: Callable[[], bool] | None = None,
        stop_on_eoc: bool = True,
    ) -> Iterator[torch.Tensor]:
        """Yield blocks of delayed code rows ([n, N] long, on GPU) as they are made.

        Mirrors ``_sampler_step``: forced BOC ramp for the first N frames, then
        after the first EOC on codebook 0 keeps ``N - 3`` more rows and stops.
        """
        import time

        S = int(inputs_embeds.shape[1])
        max_new_tokens = max(1, min(int(max_new_tokens), self.max_positions - S - self.sync_every - 2))
        self._ensure_cache(S + max_new_tokens + self.sync_every + 1)
        self._ensure_rows(max_new_tokens)
        self.temp_t.fill_(float(temperature))
        self.topk_t.fill_(int(top_k) if top_k and int(top_k) > 0 else self.V)
        self.topp_t.fill_(float(top_p) if top_p is not None else 1.0)

        # Seed the private sampler RNG from the global CPU RNG, so
        # ``torch.manual_seed(seed)`` before a call makes it reproducible.
        self.rng.manual_seed(int(torch.randint(0, 2**62, (1,)).item()))

        self.prefill(inputs_embeds)
        self.step_t.zero_()
        self.pos_t.fill_(S)

        N = self.N
        end: int | None = None  # number of rows to keep once EOC is seen
        emitted = 0
        step = 0
        t0 = time.perf_counter()
        while step < max_new_tokens:
            if should_stop is not None and should_stop():
                return
            burst = min(self.sync_every, max_new_tokens - step)
            self.noise.exponential_(generator=self.rng)  # bursts start at multiples of sync_every
            for _ in range(burst):
                self._graph(_bucket_for(S + step + 1)).replay()
                step += 1
            block = self.out_rows[emitted:step].cpu()  # one sync per burst
            if stop_on_eoc and end is None:
                first = max(N, emitted)
                hits = (block[first - emitted:, 0] == EOC_ID).nonzero()
                if hits.numel():
                    eoc_step = first + int(hits[0, 0])
                    end = eoc_step + N - 2
            stop_at = step if end is None else min(step, end)
            if stop_at > emitted:
                yield block[: stop_at - emitted]
                emitted = stop_at
            if end is not None and step >= end:
                break
        self.stats = StepStats(frames=step, seconds=time.perf_counter() - t0)
