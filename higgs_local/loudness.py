"""One loudness for every audio of the app.

References that go into a generation (library voices, uploads, dialogue / podcast / timeline references),
samples saved to the voice library, separated voices and every output (each block, the final mix of every tab)
are brought to the same integrated loudness (EBU R128 / LUFS) with a peak ceiling, so nothing is louder or
quieter than the rest. The files themselves are normalized (references once, when they are first used).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


SETTINGS = {"on": True, "target": -20.0, "ceiling": -1.0}


def configure(on: bool | None = None, target: float | None = None) -> None:
    if on is not None:
        SETTINGS["on"] = bool(on)
    if target is not None:
        SETTINGS["target"] = float(min(-6.0, max(-36.0, float(target))))


def enabled() -> bool:
    return bool(SETTINGS["on"])


def measure(x: np.ndarray, sr: int) -> float | None:
    """Integrated loudness in LUFS (mono mix); RMS-based for clips too short for the meter."""
    x = np.asarray(x, dtype=np.float32)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if not len(mono) or float(np.max(np.abs(mono))) < 1e-5:
        return None
    if len(mono) >= int(0.5 * sr):
        try:
            import pyloudnorm as pyln

            value = pyln.Meter(sr).integrated_loudness(mono.astype(np.float64))
            if np.isfinite(value):
                return float(value)
        except Exception:
            pass
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    return 20 * np.log10(max(rms, 1e-9)) - 0.691 if rms > 0 else None


def normalize(x: np.ndarray, sr: int, target: float | None = None, ceiling_db: float | None = None,
              force: bool = False) -> np.ndarray:
    """``x`` at the app's loudness (any shape), peaks under the ceiling. A no-op when switched off (unless forced)."""
    if not (force or enabled()):
        return x
    x = np.asarray(x, dtype=np.float32)
    value = measure(x, sr)
    if value is None:
        return x
    target = SETTINGS["target"] if target is None else float(target)
    ceiling = 10 ** ((SETTINGS["ceiling"] if ceiling_db is None else float(ceiling_db)) / 20)
    gain = 10 ** ((target - value) / 20)
    peak = float(np.max(np.abs(x))) * gain
    if peak > ceiling:
        gain *= ceiling / peak
    return (x * gain).astype(np.float32)


def normalize_file(path: str | Path, force: bool = False) -> bool:
    """Normalize a wav in place; True when it changed by more than 0.5 dB."""
    import soundfile as sf

    if not (force or enabled()):
        return False
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        out = normalize(data, sr, force=True)
        peak_in = float(np.max(np.abs(data))) if len(data) else 0.0
        peak_out = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak_in <= 0 or abs(20 * np.log10(max(peak_out, 1e-9) / peak_in)) < 0.5:
            return False
        info = sf.info(str(path))
        subtype = info.subtype if info.subtype in ("PCM_16", "PCM_24", "FLOAT") else "PCM_16"
        sf.write(str(path), out, sr, subtype=subtype)
        return True
    except Exception as exc:
        print(f"[loudness] {Path(str(path)).name}: {exc}", flush=True)
        return False


_CHECKED: dict[str, tuple] = {}  # file -> (size, mtime, target) already at the app's loudness


def reference(path: str | None) -> str | None:
    """A reference audio at the app's loudness: the file itself is normalized (in place, once)."""
    if not path or not enabled():
        return path
    src = Path(str(path))
    try:
        st = src.stat()
    except OSError:
        return path
    if src.suffix.lower() not in (".wav", ".flac"):
        return path  # compressed formats are left as they are (rewriting them would re-encode)
    stamp = (st.st_size, int(st.st_mtime), SETTINGS["target"], SETTINGS["ceiling"])
    key = str(src.resolve())
    if _CHECKED.get(key) == stamp:
        return path
    if normalize_file(src):
        print(f"[loudness] {src.name} → {SETTINGS['target']:g} LUFS", flush=True)
        st = src.stat()
        stamp = (st.st_size, int(st.st_mtime), SETTINGS["target"], SETTINGS["ceiling"])
    _CHECKED[key] = stamp
    return path
