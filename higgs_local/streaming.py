"""Incremental Higgs codec decoding for live playback.

The decoder emits *delayed* code rows (codebook ``c`` lags ``c`` frames). Raw
frame ``t`` is complete once delayed row ``t + N - 1`` exists. The audio codec is
non-causal, so each window is decoded with some left context and a small
look-ahead; measured against a full decode, 25 frames of context and 10 of
look-ahead are indistinguishable (max abs error ~2e-3, i.e. fp32 noise).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


class StreamingCodecDecoder:
    def __init__(
        self,
        decode_codes: Callable[[torch.Tensor], torch.Tensor],
        num_codebooks: int,
        samples_per_frame: int = 960,
        context_frames: int = 25,
        lookahead_frames: int = 10,
        first_block_frames: int = 12,
        block_frames: int = 25,
    ):
        self.decode_codes = decode_codes
        self.N = int(num_codebooks)
        self.spf = int(samples_per_frame)
        self.ctx = int(context_frames)
        self.la = int(lookahead_frames)
        self.first_block = int(first_block_frames)
        self.block = int(block_frames)
        self.rows: list[torch.Tensor] = []
        self.num_rows = 0
        self.emitted_frames = 0  # raw frames already turned into audio
        self._raw: torch.Tensor | None = None

    def _raw_codes(self) -> torch.Tensor:
        delayed = torch.cat(self.rows, dim=0) if len(self.rows) > 1 else self.rows[0]
        self.rows = [delayed]
        L = delayed.shape[0]
        T = L - (self.N - 1)
        out = torch.empty((max(T, 0), self.N), dtype=delayed.dtype)
        for c in range(self.N):
            out[:, c] = delayed[c : c + T, c]
        return out

    def _decode_range(self, raw: torch.Tensor, start: int, end: int, final: bool) -> np.ndarray:
        a = max(0, start - self.ctx)
        b = raw.shape[0] if final else min(raw.shape[0], end + self.la)
        wav = self.decode_codes(raw[a:b])
        lo = (start - a) * self.spf
        hi = wav.shape[-1] if final else (end - a) * self.spf
        return wav[lo:hi].numpy().astype(np.float32, copy=False)

    def push(self, delayed_rows: torch.Tensor) -> list[np.ndarray]:
        """Add delayed rows (CPU) and return any audio blocks now decodable."""
        if delayed_rows is None or delayed_rows.shape[0] == 0:
            return []
        self.rows.append(delayed_rows.cpu())
        self.num_rows += int(delayed_rows.shape[0])
        complete = self.num_rows - (self.N - 1)  # raw frames fully known
        ready = complete - self.la
        need = self.first_block if self.emitted_frames == 0 else self.block
        if ready - self.emitted_frames < need:
            return []
        raw = self._raw_codes()
        end = ready
        out = [self._decode_range(raw, self.emitted_frames, end, final=False)]
        self.emitted_frames = end
        return out

    def flush(self) -> list[np.ndarray]:
        """Decode everything left (call once the decoder stops)."""
        if not self.rows:
            return []
        raw = self._raw_codes()
        if raw.shape[0] <= self.emitted_frames:
            return []
        out = [self._decode_range(raw, self.emitted_frames, raw.shape[0], final=True)]
        self.emitted_frames = raw.shape[0]
        return out

    def raw_codes(self) -> torch.Tensor | None:
        return self._raw_codes() if self.rows else None
