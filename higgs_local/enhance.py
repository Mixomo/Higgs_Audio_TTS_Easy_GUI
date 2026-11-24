"""Voice enhancement: LavaSR bandwidth extension (restores the high frequencies, 48 kHz out).

LavaSR (Apache-2.0, https://github.com/ysharma3501/LavaSR) is a Vocos-based model of ~56 MB that
predicts the missing upper band; its Linkwitz-style merge keeps the original audio below the cutoff,
so the voice itself is untouched and only the air above it is added. Higgs speaks at 24 kHz, so the
cutoff sits just under 12 kHz. The optional denoiser works at 16 kHz (it drops the band above 8 kHz,
which the enhancer then rebuilds): only for noisy recordings.
"""

from __future__ import annotations

import gc
import threading

import numpy as np

from .paths import MODELS_DIR

LAVA_DIR = MODELS_DIR / "lavasr"
LAVA_REPO = "YatharthS/LavaSR"
LAVA_FILES = ["enhancer_v2/*", "denoiser/*", "config.json"]
OUT_SR = 48000


def lava_installed() -> bool:
    return (LAVA_DIR / "enhancer_v2" / "pytorch_model.bin").is_file() and (LAVA_DIR / "denoiser" / "denoiser.bin").is_file()


def ensure_lava(progress=None):
    if lava_installed():
        return LAVA_DIR
    from huggingface_hub import snapshot_download

    if progress:
        progress(None, "Downloading the LavaSR voice enhancer (~57 MB)…")
    snapshot_download(LAVA_REPO, allow_patterns=LAVA_FILES, local_dir=str(LAVA_DIR))
    return LAVA_DIR


class VoiceEnhancer:
    def __init__(self):
        self.model = None
        self.device = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return self.model
        import torch
        from LavaSR.model import LavaEnhance2

        ensure_lava()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[enhance] loading LavaSR on {self.device}", flush=True)
        self.model = LavaEnhance2(str(LAVA_DIR), self.device)
        return self.model

    def unload(self):
        with self.lock:
            if self.model is None:
                return
            self.model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def enhance(self, wav: np.ndarray, sr: int, denoise: bool = False, chunk_s: float = 20.0) -> tuple[np.ndarray, int]:
        """``wav`` mono float at any rate → 48 kHz with its upper band restored."""
        import torch
        import torchaudio
        from LavaSR.enhancer.linkwitz_merge import FastLRMerge

        with self.lock:
            model = self.load()
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim == 2:  # (n, channels) from soundfile, or (channels, n)
                wav = wav.mean(axis=1) if wav.shape[0] > wav.shape[1] else wav.mean(axis=0)
            x = torch.from_numpy(np.ascontiguousarray(wav.reshape(-1))).to(self.device)
            peak = float(x.abs().max()) if x.numel() else 0.0
            if denoise:
                x16 = torchaudio.functional.resample(x, sr, 16000) if sr != 16000 else x
                with torch.inference_mode():
                    x16 = model.denoiser_model.infer(x16.unsqueeze(0)).reshape(-1)
                x48 = torchaudio.functional.resample(x16, 16000, OUT_SR)
                cutoff = 7800
            else:
                x48 = torchaudio.functional.resample(x, sr, OUT_SR) if sr != OUT_SR else x
                cutoff = int(min(sr, OUT_SR) / 2 * 0.96)
            model.bwe_model.lr_refiner = FastLRMerge(device=self.device, cutoff=cutoff, transition_bins=1024)
            n = x48.numel()
            size, hop = int(chunk_s * OUT_SR), int((chunk_s - 0.5) * OUT_SR)
            if n <= size:
                out = model.bwe_model.infer(x48.unsqueeze(0)).reshape(-1)[:n]
            else:
                # long audio in overlapping pieces, cross-faded over half a second
                out = torch.zeros(n, device=self.device)
                weight = torch.zeros(n, device=self.device)
                fade = int(0.5 * OUT_SR)
                for a in range(0, n, hop):
                    piece = x48[a : a + size]
                    y = model.bwe_model.infer(piece.unsqueeze(0)).reshape(-1)[: piece.numel()]
                    w = torch.ones(y.numel(), device=self.device)
                    if a > 0:
                        w[:fade] = torch.linspace(0, 1, fade, device=self.device)
                    if a + size < n:
                        w[-fade:] = torch.linspace(1, 0, fade, device=self.device)
                    out[a : a + y.numel()] += y * w
                    weight[a : a + y.numel()] += w
                    if a + size >= n:
                        break
                out = out / weight.clamp_min(1e-6)
            y = out.float().cpu().numpy()
            want = int(round(len(wav) / sr * OUT_SR))
            if len(y) < want:
                y = np.pad(y, (0, want - len(y)))
        top = float(np.max(np.abs(y))) if y.size else 0.0
        ceiling = max(peak, 0.0) if peak > 0 else 0.98
        if top > 0.98 or (top > ceiling * 1.12 and ceiling > 0):
            y *= min(0.98, ceiling * 1.12) / top
        return y.astype(np.float32), OUT_SR


ENHANCER = VoiceEnhancer()
