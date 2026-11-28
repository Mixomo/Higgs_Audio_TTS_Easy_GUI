"""The one speech pipeline every tab uses (single voice, dialogue, podcast, fixer).

A render is a list of blocks, each with its own text, reference voice and LoRA.
Per block: the text is prepared (numbers in words), takes are generated and
checked (by ear with ASR, voice similarity, sound checks), a failed take is
retried, and the best one is kept. Takes are cached by what made them, so with
"voice only what changed" a block whose text, voice and settings did not change
reuses its take. The kept takes are finished (even pauses, one loudness, room
tone) and joined with the pause of the same speaker or of another one.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import loudness
from . import takecheck as tc
from .adapters import NONE_ADAPTER
from .audio_utils import SAMPLE_RATE
from .live import LIVE
from .paths import CACHE_DIR

TAKES_DIR = CACHE_DIR / "takes"

LOUDNESS_TARGETS = {
    "Off (as the model made it)": None,
    "Podcast · −16 LUFS, peaks −1 dB": (-16.0, -1.0),
    "Video · −14 LUFS, peaks −1 dB": (-14.0, -1.0),
    "Audiobook ACX · −20 LUFS, peaks −3 dB": (-20.0, -3.0),
}

NUMBER_LANGS = {
    "Auto-detect": None, "Spanish": "es", "English": "en", "Portuguese": "pt", "French": "fr", "German": "de",
    "Italian": "it", "Russian": "ru", "Polish": "pl", "Dutch": "nl", "Turkish": "tr", "Ukrainian": "uk",
    "Japanese": "ja", "Korean": "ko", "Chinese": "zh",
}

READ_AS_WORDS = ["numbers", "years", "times", "money", "ordinals", "percent"]


@dataclass
class Advanced:
    """The advanced speech settings shared by every tab."""

    asr_check: bool = False
    asr_model: str = "CrisperWhisper 2.0 turbo"
    language: str = "Auto-detect"
    similarity: bool = False
    similarity_min: float = 0.5
    strictness: str = "Normal"
    takes: int = 1
    retries: int = 2
    sound_checks: list[str] = field(default_factory=lambda: list(tc.CHEAP_CHECKS))
    pause_same_ms: int = 300
    pause_other_ms: int = 600
    tidy_pauses: bool = False
    level_pieces: bool = False
    room_tone: bool = False
    final_loudness: str = "Off (as the model made it)"
    words: list[str] = field(default_factory=list)
    live: bool = True
    only_changed: bool = False  # Dialogue and Podcast turn it on with their own switch
    keep_check_models: bool = True
    # LavaSR: restore the high frequencies and save at 48 kHz (optionally denoised first)
    enhance: bool = False
    enhance_denoise: bool = False

    @property
    def checking(self) -> bool:
        return self.asr_check or self.similarity or bool(self.sound_checks) or self.takes > 1


@dataclass
class Block:
    text: str
    ref_audio: str | None = None
    ref_text: str = ""
    lora: str | None = None
    speaker: str = ""
    # seconds of silence before this block; None = the same/other speaker pause
    pause: float | None = None
    # the block's own sampling when it has one (a podcast line, a fixer patch)
    seed: int | None = None
    # the library voice the block was made with (for the montage to voice it again)
    voice: str = ""


@dataclass
class Sampling:
    model_version: str
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    max_frames: int = 0  # 0 = auto from the text length
    seed: int = -1
    reuse_seed: bool = True
    chunk_mode: str = "Seamless (auto)"
    carry_tags: bool = True
    ras_win_len: int = 7
    ras_max_repeat: int = 2


@dataclass
class BlockResult:
    index: int
    text: str
    speaker: str
    file: str
    seconds: float
    report: str
    cached: bool
    ok: bool
    start: float = 0.0


# ---------------------------------------------------------------- text prep
def _num_lang(text: str, language: str) -> str:
    code = NUMBER_LANGS.get(language)
    if code:
        return code
    try:
        import langid

        return langid.classify(tc.visible(text))[0]
    except Exception:
        return "en"


def numbers_in_words(text: str, language: str, kinds: list[str]) -> str:
    """Numbers, years, times and money said in words before voicing (num2words)."""
    if not kinds:
        return text
    try:
        from num2words import num2words
    except ImportError as exc:
        raise RuntimeError("Reading numbers as words needs the num2words package (run install.bat)") from exc
    lang = _num_lang(text, language)

    def say(n, to="cardinal"):
        try:
            return num2words(n, lang=lang, to=to)
        except (NotImplementedError, OverflowError, ValueError):
            try:
                return num2words(n, lang=lang)
            except Exception:
                return str(n)

    # tags stay untouched: only the spoken parts are rewritten
    parts = re.split(r"(<\|[^|>]*\|>|</?SE[^>]*>)", text)
    for i in range(0, len(parts), 2):
        s = parts[i]
        if "times" in kinds:
            s = re.sub(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", lambda m: f"{say(int(m.group(1)))} {say(int(m.group(2)))}" if m.group(2) != "00" else say(int(m.group(1))), s)
        if "money" in kinds:
            cur = {"$": {"es": "dólares", "en": "dollars", "pt": "dólares", "fr": "dollars"}, "€": {"es": "euros", "en": "euros", "pt": "euros", "fr": "euros"}}
            s = re.sub(r"([$€])\s?(\d+(?:[.,]\d{1,2})?)", lambda m: f"{say(float(m.group(2).replace(',', '.')) if '.' in m.group(2) or ',' in m.group(2) else int(m.group(2)))} {cur[m.group(1)].get(lang, cur[m.group(1)]['en'])}", s)
        if "percent" in kinds:
            pct = {"es": "por ciento", "pt": "por cento", "fr": "pour cent", "it": "per cento", "de": "Prozent"}.get(lang, "percent")
            s = re.sub(r"(\d+(?:[.,]\d+)?)\s?%", lambda m: f"{say(float(m.group(1).replace(',', '.')) if any(c in m.group(1) for c in '.,') else int(m.group(1)))} {pct}", s)
        if "ordinals" in kinds:
            s = re.sub(r"\b(\d+)(?:º|ª|°|st|nd|rd|th|er|e|o\.|a\.)\b", lambda m: say(int(m.group(1)), "ordinal"), s)
        if "years" in kinds:
            s = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: say(int(m.group(1)), "year"), s)
        if "numbers" in kinds:
            s = re.sub(r"\b\d{1,3}(?:[.,]\d{3})+\b|\b\d+(?:[.,]\d+)?\b", lambda m: _say_number(m.group(0), say, lang), s)
        parts[i] = s
    return "".join(parts)


def _say_number(raw: str, say, lang: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw):
        return say(int(re.sub(r"[.,]", "", raw)))
    if "," in raw or "." in raw:
        try:
            return say(float(raw.replace(",", ".")))
        except ValueError:
            return raw
    return say(int(raw))


# ---------------------------------------------------------------- finishing
def tidy(x: np.ndarray, sr: int, max_inner: float = 1.2, keep_inner: float = 0.6) -> np.ndarray:
    """No silence at the edges; an inner pause longer than ``max_inner`` is shortened to ``keep_inner``."""
    a, b = tc.trim_edges(x, sr)
    x = x[a:b]
    if len(x) == 0:
        return x
    w = sr // 100
    n = len(x) // w
    if n < 3:
        return x
    rms = np.sqrt(np.mean(x[: n * w].reshape(n, w).astype(np.float64) ** 2, axis=1))
    quiet = rms < rms.max() * 10 ** (-40 / 20)
    out: list[np.ndarray] = []
    pos = 0
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * w / sr > max_inner:
                keep = int(keep_inner * sr)
                start, end = i * w, j * w
                out.append(x[pos : start + keep // 2])
                pos = end - keep // 2
            i = j
        else:
            i += 1
    out.append(x[pos:])
    return np.concatenate(out)


def level(x: np.ndarray, sr: int, target: float = -20.0, ceiling_db: float = -1.0) -> np.ndarray:
    value = tc.lufs(x, sr)
    if value is None:
        return x
    gain = 10 ** ((target - value) / 20)
    peak = float(np.max(np.abs(x))) * gain
    ceiling = 10 ** (ceiling_db / 20)
    if peak > ceiling:
        gain *= ceiling / peak
    return (x * gain).astype(np.float32)


def room_tone(seconds: float, sr: int, rng: np.random.Generator, level_db: float = -62.0) -> np.ndarray:
    n = int(seconds * sr)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    white = rng.standard_normal(n)
    # a soft low-pass (one pole) makes it sound like a room, not hiss
    b = 0.97
    tone = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = b * acc + (1 - b) * white[i]
        tone[i] = acc
    tone /= max(float(np.sqrt(np.mean(tone**2))), 1e-9)
    fade = min(n // 2, int(0.01 * sr))
    if fade:
        ramp = np.linspace(0, 1, fade)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
    return (tone * 10 ** (level_db / 20)).astype(np.float32)


def final_loudness(x: np.ndarray, sr: int, choice: str) -> np.ndarray:
    target = LOUDNESS_TARGETS.get(choice)
    if not target:
        return x
    return level(x, sr, target[0], target[1])


# ---------------------------------------------------------------- the cache
def _file_id(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        st = p.stat()
        return f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return str(p)


def take_key(block: Block, text: str, sampling: Sampling, seed: int | None) -> str:
    payload = {
        "model": sampling.model_version,
        "text": text,
        "ref": _file_id(block.ref_audio),
        "ref_text": (block.ref_text or "").strip(),
        "lora": block.lora or NONE_ADAPTER,
        "t": round(float(sampling.temperature), 3),
        "p": round(float(sampling.top_p), 3),
        "k": int(sampling.top_k),
        "max": int(sampling.max_frames),
        "mode": sampling.chunk_mode,
        "carry": bool(sampling.carry_tags),
        "seed": seed,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]


def cached_take(key: str) -> tuple[np.ndarray, dict] | None:
    wav = TAKES_DIR / f"{key}.wav"
    meta = TAKES_DIR / f"{key}.json"
    if not wav.is_file():
        return None
    data, _sr = sf.read(str(wav), dtype="float32")
    info = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
    return data.astype(np.float32), info


def store_take(key: str, wav: np.ndarray, info: dict) -> str:
    TAKES_DIR.mkdir(parents=True, exist_ok=True)
    path = TAKES_DIR / f"{key}.wav"
    sf.write(str(path), wav, SAMPLE_RATE, subtype="FLOAT")
    (TAKES_DIR / f"{key}.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return str(path)


def clear_take_cache() -> str:
    if not TAKES_DIR.is_dir():
        return "The take cache is empty."
    n = 0
    size = 0
    for f in TAKES_DIR.iterdir():
        if f.is_file():
            size += f.stat().st_size
            f.unlink()
            n += 1
    return f"Deleted {n // 2} cached takes ({size / 1e6:.0f} MB)."


# ---------------------------------------------------------------- the renderer
class Cancelled(Exception):
    pass


class Renderer:
    """Renders blocks one after another; ``events`` receives progress dicts for the UI."""

    def __init__(
        self,
        manager,
        sampling: Sampling,
        adv: Advanced,
        segmenter: Callable[[str, str, bool], list[dict]],
        asr=None,
        events: Callable[[dict], None] | None = None,
    ):
        self.manager = manager
        self.s = sampling
        self.adv = adv
        self.segmenter = segmenter
        self.asr = asr
        self.events = events or (lambda ev: None)
        self.v3 = str(sampling.model_version).startswith("Higgs V3")
        self.report: list[str] = []
        self.cancelled = False

    # ---------------------------------------------------------- engine
    def cancel(self):
        self.cancelled = True
        self.manager.v3.cancel()

    def _check_cancel(self):
        if self.cancelled or getattr(self.manager.v3, "cancel_requested", False):
            raise Cancelled()

    def _synth(self, block: Block, text: str, seed: int, live: bool) -> np.ndarray:
        self.manager.set_lora_adapter(block.lora or NONE_ADAPTER)
        segments = self.segmenter(text, self.s.chunk_mode, self.s.carry_tags)
        if not segments:
            return np.zeros(0, dtype=np.float32)
        max_frames = int(self.s.max_frames) or max(tc.auto_max_frames(seg["text"]) for seg in segments)

        def frame_progress(frame, total, desc):
            self.events({"type": "frames", "frames": int(frame)})

        if self.v3:
            self.manager.unload_other(self.s.model_version)
            self.manager.v3.load()
        if self.v3 and self.manager.v3.fast_active:
            wav = None
            for ev in self.manager.v3.stream(
                text,
                ref_audio=block.ref_audio,
                ref_text=block.ref_text or "",
                temperature=self.s.temperature,
                top_p=self.s.top_p,
                top_k=self.s.top_k,
                max_new_tokens=max_frames,
                seed=seed,
                progress_cb=frame_progress,
                segments=segments,
                emit_audio=live,
            ):
                if ev["type"] == "audio" and live:
                    LIVE.push(ev["audio"])
                elif ev["type"] == "segment":
                    self.events({"type": "segment", "index": ev["index"], "total": ev["total"]})
                elif ev["type"] == "done":
                    wav = ev["audio"]
            self._check_cancel()
            if wav is None:
                raise Cancelled()
            return np.asarray(wav, dtype=np.float32)
        payloads = [{"text": seg["text"], "ref_audio": block.ref_audio, "ref_text": block.ref_text or ""} for seg in segments]
        sr, wav, _log = self.manager.generate_many(
            self.s.model_version,
            payloads,
            float(segments[1]["gap"]) if len(segments) > 1 else 0.0,
            temperature=self.s.temperature,
            top_p=self.s.top_p,
            top_k=self.s.top_k,
            max_new_tokens=max_frames,
            ras_win_len=self.s.ras_win_len,
            ras_win_max_num_repeat=self.s.ras_max_repeat,
            seed=seed,
            chunk_mode="None",
            progress_cb=frame_progress,
        )
        self._check_cancel()
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != SAMPLE_RATE:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
        if live:
            LIVE.push(wav)
        return wav

    # ---------------------------------------------------------- checks
    def _check(self, block: Block, text: str, wav: np.ndarray, tmp_path: str) -> tc.TakeReport:
        rep = tc.TakeReport()
        adv = self.adv
        voice = block.speaker or (Path(block.ref_audio).stem if block.ref_audio else "") or (block.lora or "")
        if "defect" in adv.sound_checks:
            kind = tc.defect(wav, SAMPLE_RATE, text)
            if kind:
                rep.issues.append(kind)
        want = [k for k in adv.sound_checks if k != "defect"]
        suspects: list[str] = []
        if want or adv.asr_check:
            f = tc.measure(wav, SAMPLE_RATE, whistle="whistle" in want, repeat="audio_repeat" in want)
            rep.features = f
            for flag in tc.sound_flags(f, want, text, tc.voice_norm(voice)):
                {"defect": rep.issues, "note": rep.notes, "suspect": suspects}[tc.role(flag)].append(flag)
        # the take is only heard when "Check every new take by ear" is on: off, a suspect stays a note
        if adv.asr_check and self.asr is not None and not rep.issues:
            self.events({"type": "phase", "phase": "Listening to the take"})
            heard, _status = self.asr.transcribe_whisper_direct(tmp_path, adv.asr_model, adv.language)
            rep.heard = heard
            res = tc.check_text(text, heard, adv.strictness)
            rep.wer = res["wer"]
            if adv.asr_check:
                rep.passed_text = bool(res["passed"])
                if not rep.passed_text:
                    rep.issues.append("text")
                if res["repeat"]:
                    rep.issues.append("repeat")
            if suspects:
                # recognition decides: a suspect the words confirm is an issue, otherwise a note
                (rep.issues if not res["passed"] or res["repeat"] else rep.notes).extend(suspects)
        elif suspects:
            rep.notes.extend(suspects)
        if adv.similarity and block.ref_audio:
            self.events({"type": "phase", "phase": "Comparing the voice"})
            ref = tc.EMBEDDER.embed_file(block.ref_audio)
            emb = tc.EMBEDDER.embed(wav, SAMPLE_RATE)
            rep.similarity = float(ref @ emb)
            rep.similar = rep.similarity >= float(adv.similarity_min)
            if not rep.similar:
                rep.issues.append("unlike")
        return rep

    # ---------------------------------------------------------- one block
    def render_block(self, index: int, total: int, block: Block, base_seed: int) -> BlockResult:
        adv = self.adv
        text = numbers_in_words(block.text, adv.language, adv.words) if adv.words else block.text
        fixed = block.seed if block.seed is not None and block.seed >= 0 else (base_seed if base_seed >= 0 else None)
        key = take_key(block, text, self.s, fixed)
        label = (block.speaker + ": " if block.speaker else "") + tc.visible(text).strip()[:60]
        if adv.only_changed:
            hit = cached_take(key)
            if hit is not None:
                wav, info = hit
                self.events({"type": "block", "index": index, "total": total, "text": label, "cached": True})
                if adv.live:
                    LIVE.push(wav)
                return BlockResult(index, block.text, block.speaker, str(TAKES_DIR / f"{key}.wav"), len(wav) / SAMPLE_RATE,
                                   info.get("report", "cached"), True, bool(info.get("ok", True)))
        self.events({"type": "block", "index": index, "total": total, "text": label, "cached": False})
        best: tuple[tuple, np.ndarray, tc.TakeReport] | None = None
        planned = max(1, int(adv.takes))
        limit = planned + max(0, int(adv.retries))
        made = 0
        import random

        while made < limit:
            self._check_cancel()
            seed = (fixed + made) if fixed is not None else random.randint(0, 2_147_483_647)
            if made:
                self.events({"type": "phase", "phase": f"Take {made + 1}"})
                if adv.live:
                    LIVE.restart()
            wav = self._synth(block, text, seed, adv.live)
            made += 1
            tc.soften_tail_click(wav, SAMPLE_RATE)
            if adv.checking:
                tmp = TAKES_DIR / "_check.wav"
                TAKES_DIR.mkdir(parents=True, exist_ok=True)
                sf.write(str(tmp), wav, SAMPLE_RATE)
                rep = self._check(block, text, wav, str(tmp))
            else:
                rep = tc.TakeReport()
            print(f"[render] block {index}/{total} take {made} (seed {seed}): {rep.summary()}", flush=True)
            self.events({"type": "take", "index": index, "take": made, "report": rep.summary(), "ok": rep.ok})
            if best is None or rep.rank() < best[0]:
                best = (rep.rank(), wav, rep)
            # enough takes made and the best one passes; or out of planned takes with no check asked
            if made >= planned and (best[2].ok or not adv.checking):
                break
        assert best is not None
        _rank, wav, rep = best
        if rep.ok and rep.features is not None:
            tc.remember_take(block.speaker or (Path(block.ref_audio).stem if block.ref_audio else ""), rep.features, text)
        summary = rep.summary() + (f" · {made} takes" if made > 1 else "")
        path = store_take(key, wav, {"text": block.text, "speaker": block.speaker, "report": summary, "ok": rep.ok})
        return BlockResult(index, block.text, block.speaker, path, len(wav) / SAMPLE_RATE, summary, False, rep.ok)

    # ---------------------------------------------------------- all blocks
    def render(self, blocks: list[Block], live_label: str = "") -> tuple[np.ndarray, list[BlockResult]]:
        adv = self.adv
        blocks = [b for b in blocks if tc.visible(b.text).strip() or b.text.strip()]
        if not blocks:
            raise ValueError("Nothing to voice: every block is empty.")
        base_seed = int(self.s.seed)
        if base_seed < 0 and self.s.reuse_seed:
            import random

            base_seed = random.randint(0, 2_147_483_647) if not adv.only_changed else -1
        self.base_seed = base_seed
        self.manager.v3.cancel_requested = False
        if adv.live:
            LIVE.start(live_label)
        results: list[BlockResult] = []
        anchor: tuple[str, str, str] | None = None
        try:
            for i, block in enumerate(blocks, 1):
                self._check_cancel()
                if block.ref_audio and loudness.enabled():  # every reference at the app's one loudness (a cached copy)
                    block = Block(**{**block.__dict__, "ref_audio": loudness.reference(block.ref_audio)})
                if not block.ref_audio and (not block.lora or block.lora == NONE_ADAPTER) and anchor is not None and block.speaker == anchor[2]:
                    # no voice given: the first block's take is the voice for the rest of that speaker
                    block = Block(**{**block.__dict__, "ref_audio": anchor[0], "ref_text": anchor[1]})
                res = self.render_block(i, len(blocks), block, base_seed)
                results.append(res)
                if anchor is None and not blocks[0].ref_audio and (not blocks[0].lora or blocks[0].lora == NONE_ADAPTER) and res.seconds >= 1.0:
                    anchor = (res.file, tc.visible(blocks[0].text).strip(), blocks[0].speaker)
        finally:
            if adv.live:
                LIVE.end()
        return self.join(blocks, results), results

    def join(self, blocks: list[Block], results: list[BlockResult]) -> np.ndarray:
        adv = self.adv
        rng = np.random.default_rng(1234)
        out: list[np.ndarray] = []
        t = 0.0
        prev_speaker = None
        for block, res in zip(blocks, results):
            if loudness.enabled():  # each block at the one loudness: also the clips sent to the montage
                loudness.normalize_file(res.file)
            data, _sr = sf.read(res.file, dtype="float32")
            x = np.asarray(data, dtype=np.float32)
            if adv.tidy_pauses:
                x = tidy(x, SAMPLE_RATE)
            if adv.level_pieces:
                x = level(x, SAMPLE_RATE, -20.0, -1.0)
            if out:
                if block.pause is not None:
                    gap = float(block.pause)
                else:
                    same = block.speaker == prev_speaker
                    gap = (adv.pause_same_ms if same else adv.pause_other_ms) / 1000.0
                silence = room_tone(gap, SAMPLE_RATE, rng) if adv.room_tone else np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32)
                out.append(silence)
                t += len(silence) / SAMPLE_RATE
            res.start = t
            res.seconds = len(x) / SAMPLE_RATE
            out.append(x)
            t += res.seconds
            prev_speaker = block.speaker
        full = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
        if LOUDNESS_TARGETS.get(adv.final_loudness):  # an explicit delivery target wins
            return final_loudness(full, SAMPLE_RATE, adv.final_loudness)
        return loudness.normalize(full, SAMPLE_RATE)


def results_report(results: list[BlockResult]) -> str:
    lines = []
    for r in results:
        mark = "♻" if r.cached else ("✓" if r.ok else "✗")
        who = f"{r.speaker}: " if r.speaker else ""
        lines.append(f"{mark} {r.index}. {who}{tc.visible(r.text).strip()[:70]} — {r.report}")
    cached = sum(r.cached for r in results)
    if cached:
        lines.append(f"♻ {cached} of {len(results)} block(s) reused unchanged (voice only what changed).")
    return "\n".join(lines)
