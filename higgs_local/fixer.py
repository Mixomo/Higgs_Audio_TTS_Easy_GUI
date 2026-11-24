"""Dialogue fixer: correct a recorded voice in place, Descript-style.

1. The recording is transcribed with word timings (CrisperWhisper) and, when
   asked, SAVA's voice curve marks where speech is.
2. The person edits the transcript. The words that changed are found by a word
   diff. Each change grows by itself to the nearest place where a cut is inaudible
   (a punctuation mark or a pause, measured against the recording's own pace), and
   to enough words for a natural intonation: nothing to set by hand.
3. Each changed phrase is voiced again with the recording itself as the voice
   reference (the speech around the phrase, its words as the reference text) and
   checked by ear. Words the take says beyond its text (a word of the reference
   said again at an edge) are trimmed off, so nothing is said twice. The take is
   matched in loudness to its neighbours and spliced in with crossfades sized to
   the gap at each cut.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from .paths import CACHE_DIR

FIX_DIR = CACHE_DIR / "fixer"
WORD_RE = re.compile(r"\S+")


@dataclass
class Word:
    text: str
    start: float
    end: float


def norm(w: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFD", w.lower())
    return "".join(c for c in s if c.isalnum() and unicodedata.category(c) != "Mn")


def words_from_result(result) -> list[Word]:
    out = []
    for w in result.words or []:
        if w.start is None or w.end is None or not (w.word or "").strip():
            continue
        out.append(Word(w.word.strip(), float(w.start), float(w.end)))
    return out


def transcript_text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


def refine_with_sava(words: list[Word], audio: np.ndarray, sr: int, processor) -> list[Word]:
    """Word edges pulled to where SAVA hears voice: a cut that would clip a word's onset or tail is moved."""
    curve = processor.speech_curve(audio, sr)  # 40 ms frames
    if curve is None or not len(curve):
        return words
    frame = 0.04
    speech = curve >= 0.5
    # never past the middle of the gap to a neighbour: in running speech the voice does not stop between
    # words, and an edge pushed into the next word would cut its onset off
    raw = [(w.start, w.end) for w in words]
    lo = [0.0] + [(raw[i - 1][1] + raw[i][0]) / 2 for i in range(1, len(raw))]
    hi = [(raw[i][1] + raw[i + 1][0]) / 2 for i in range(len(raw) - 1)] + [len(audio) / sr]
    for k, w in enumerate(words):
        a = int(w.start / frame)
        while a > 0 and speech[min(a - 1, len(speech) - 1)] and w.start - (a - 1) * frame < 0.12:
            a -= 1
        b = int(w.end / frame)
        while b < len(speech) - 1 and speech[b] and b * frame - w.end < 0.12:
            b += 1
        w.start = max(min(w.start, a * frame), min(lo[k], w.start))
        w.end = min(max(w.end, b * frame), max(hi[k], w.end))
    return words


@dataclass
class Patch:
    # word indexes in the original [a, b) replaced by new words
    a: int
    b: int
    new: list[str]
    start: float = 0.0  # seconds in the original that are cut out
    end: float = 0.0
    before: str = ""
    after: str = ""
    report: str = ""
    seconds: float = 0.0
    ja: int = 0  # the patch's words in the edited transcript [ja, jb)
    jb: int = 0


CLAUSE_END = re.compile(r"[.,;:!?…»”\")\]]$")


def _boundary(words: list[Word], i: int, pause_gap: float) -> bool:
    """A natural cut between word i and word i+1: the first ends a clause or a pause follows it."""
    if i < 0 or i + 1 >= len(words):
        return True
    return bool(CLAUSE_END.search(words[i].text)) or words[i + 1].start - words[i].end >= pause_gap


def pause_threshold(words: list[Word]) -> float:
    """What counts as a pause in this recording: about twice its usual gap between words (0.12–0.35 s)."""
    gaps = [words[i + 1].start - words[i].end for i in range(len(words) - 1)]
    gaps = [g for g in gaps if g > 0.01]
    if not gaps:
        return 0.18
    return float(min(0.35, max(0.12, 2.0 * float(np.median(gaps)))))


AUTO_MAX_SIDE = 6  # unchanged words voiced again on a side, at most
AUTO_MIN_WORDS = 3  # a phrase shorter than this is voiced with its neighbours (one word alone sounds read)
AUTO_MIN_SECONDS = 0.7
SENTENCE_END = re.compile(r"[.!?…]['\"»”)\]]*$")


def _gap(words: list[Word], i: int) -> float:
    """Silence between word i and word i+1 (the edges of the recording count as long pauses)."""
    if i < 0 or i + 1 >= len(words):
        return 9.0
    return max(0.0, words[i + 1].start - words[i].end)


def _grow(words: list[Word], a: int, b: int, pause_gap: float, limit: int) -> tuple[int, int]:
    """[a, b) grown to the nearest natural cut on each side, at most ``limit`` words; where there is none
    that close, the cut goes in the widest gap within reach."""

    def left(a):
        if a <= 0 or _boundary(words, a - 1, pause_gap):
            return a
        best, best_gap = a, _gap(words, a - 1)
        for k in range(1, limit + 1):
            i = a - k
            if i <= 0 or _boundary(words, i - 1, pause_gap):
                return max(i, 0)
            if _gap(words, i - 1) > best_gap:
                best, best_gap = i, _gap(words, i - 1)
        return best

    def right(b):
        if b >= len(words) or _boundary(words, b - 1, pause_gap):
            return b
        best, best_gap = b, _gap(words, b - 1)
        for k in range(1, limit + 1):
            j = b + k
            if j >= len(words) or _boundary(words, j - 1, pause_gap):
                return min(j, len(words))
            if _gap(words, j - 1) > best_gap:
                best, best_gap = j, _gap(words, j - 1)
        return best

    return left(a), right(b)


def _auto_span(words: list[Word], i1: int, i2: int, n_new: int, pause_gap: float) -> tuple[int, int]:
    a, b = _grow(words, i1, i2, pause_gap, AUTO_MAX_SIDE)
    for _ in range(2):  # too little to say well: take in the shorter neighbouring phrase, then check again
        said = n_new + (i1 - a) + (b - i2)
        secs = (words[b - 1].end - words[a].start) if b > a else 0.0
        if said >= AUTO_MIN_WORDS and (secs >= AUTO_MIN_SECONDS or said >= AUTO_MIN_WORDS + 2):
            break
        # a sentence end is a hard edge: the minimum is reached inside the sentence, never across it
        la = _grow(words, a - 1, b, pause_gap, AUTO_MAX_SIDE)[0] if a > 0 and not SENTENCE_END.search(words[a - 1].text) else a
        rb = _grow(words, a, b + 1, pause_gap, AUTO_MAX_SIDE)[1] \
            if b < len(words) and not SENTENCE_END.search(words[b - 1].text) else b
        grow_l, grow_r = a - la, rb - b
        if not grow_l and not grow_r:
            break
        if grow_r and (not grow_l or grow_r <= grow_l):
            b = rb
        else:
            a = la
    return a, b


def find_patches(words: list[Word], edited: str, context_words: int | None = None, smart: bool = True,
                 pause_gap: float | None = None) -> list[Patch]:
    """Changed spans of the transcript.

    ``context_words`` None (the default): each change grows by itself to the nearest natural cut (a
    punctuation mark or a pause of this recording) and to enough words for a natural intonation.
    A number: exactly up to that many unchanged words on each side (``smart``: never past a cut).
    """
    old = [norm(w.text) for w in words]
    new_tokens = WORD_RE.findall(edited or "")
    new = [norm(t) for t in new_tokens]
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    raw = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    if not raw:
        return []
    if pause_gap is None:
        pause_gap = pause_threshold(words)
    patches: list[list[int]] = []
    if context_words is None:
        for i1, i2, j1, j2 in raw:
            if j1 == j2 and i2 > i1 and max(_gap(words, i1 - 1), _gap(words, i2 - 1)) >= 0.05:
                patches.append([i1, i2, j1, j1])  # words only taken out, with air around them: a clean cut, nothing voiced
                continue
            if j1 == j2:  # taken out in the middle of running speech: the two neighbours are voiced again
                a, b = _grow(words, i1, i2, pause_gap, 2)
                a, b = (a, b) if (i1 - a) + (b - i2) >= 2 else (max(i1 - 1, 0), min(i2 + 1, len(words)))
                patches.append([a, b, max(j1 - (i1 - a), 0), min(j2 + (b - i2), len(new_tokens))])
                continue
            a, b = _auto_span(words, i1, i2, j2 - j1, pause_gap)
            patches.append([a, b, max(j1 - (i1 - a), 0), min(j2 + (b - i2), len(new_tokens))])
        raw = []
    n = max(0, int(context_words or 0))
    for i1, i2, j1, j2 in raw:
        a, b = i1, i2
        for _ in range(n):  # grow left
            if a <= 0 or (smart and _boundary(words, a - 1, pause_gap)):
                break
            a -= 1
        for _ in range(n):  # grow right
            if b >= len(words) or (smart and _boundary(words, b - 1, pause_gap)):
                break
            b += 1
        if a == b and words:  # a pure insertion between two words: voice it with one neighbour for a clean join
            if b < len(words) and not (smart and _boundary(words, b - 1, pause_gap)):
                b += 1
            elif a > 0:
                a -= 1
        ja = j1 - (i1 - a)
        jb = j2 + (b - i2)
        patches.append([a, min(b, len(words)), max(ja, 0), min(jb, len(new_tokens))])
    merged: list[list[int]] = []
    for p in sorted(patches):
        if merged and p[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], p[1])
            merged[-1][3] = max(merged[-1][3], p[3])
        else:
            merged.append(p)
    out = []
    for a, b, ja, jb in merged:
        p = Patch(a=a, b=b, new=new_tokens[ja:jb], ja=ja, jb=jb)
        p.before = " ".join(w.text for w in words[a:b])
        p.after = " ".join(p.new)
        out.append(p)
    return out


def snap_to_valley(audio: np.ndarray, sr: int, t: float, lo: float, hi: float) -> float:
    """The quietest 5 ms point in [lo, hi] near ``t``: where one sound really ends and the next begins
    (word timings are often a few tens of ms off; a cut in the valley does not clip either word)."""
    lo, hi = max(0.0, lo), min(len(audio) / sr, hi)
    if hi - lo < 0.012:
        return t
    hop = max(1, int(0.005 * sr))
    win = 2 * hop
    a, b = int(lo * sr), int(hi * sr)
    seg = audio[a:b].astype(np.float64)
    if len(seg) < win:
        return t
    starts = range(0, len(seg) - win + 1, hop)
    energy = np.array([np.sqrt(np.mean(seg[s : s + win] ** 2)) for s in starts])
    times = np.array([lo + (s + win / 2) / sr for s in starts])
    # quietest point, with a slight preference for staying close to the timing we were given
    score = energy / (energy.max() + 1e-9) + 0.15 * np.abs(times - t) / max(hi - lo, 1e-3)
    return float(times[int(np.argmin(score))])


def cut_points(words: list[Word], p: Patch, total: float, audio: np.ndarray | None = None,
               sr: int | None = None) -> tuple[float, float]:
    """The cut is in the middle of the pauses around the phrase (at most 0.15 s into each); with the audio,
    each cut moves to the quietest point at the boundary between the two words."""
    if p.a < len(words):
        left_word_start = words[p.a].start
        prev_end = words[p.a - 1].end if p.a > 0 else 0.0
        # never into the word itself, even when the neighbours' edges overlap it (words said without a gap)
        start = min(left_word_start, max(prev_end, left_word_start - min(0.15, max(0.0, left_word_start - prev_end) / 2)))
    else:
        start = words[-1].end if words else 0.0
    if p.b > p.a:
        right_word_end = words[p.b - 1].end
        next_start = words[p.b].start if p.b < len(words) else total
        end = max(right_word_end, min(next_start, right_word_end + min(0.15, max(0.0, next_start - right_word_end) / 2)))
    else:
        end = start
    if audio is not None and sr:
        if 0 < p.a < len(words):
            start = snap_to_valley(audio, sr, start, start - 0.025, start + 0.025)
        if p.b > p.a and p.b < len(words):
            end = snap_to_valley(audio, sr, end, end - 0.025, end + 0.025)
        end = max(end, start)
    return start, end


def reference_excerpt(audio: np.ndarray, sr: int, words: list[Word], p: Patch, seconds: float = 15.0) -> tuple[np.ndarray, str]:
    """Speech of the same recording around the phrase (not the phrase itself) and its words, for the voice clone."""
    keep = [i for i in range(len(words)) if i < p.a or i >= p.b]
    if not keep:
        return audio[: int(seconds * sr)], ""
    # the words nearest to the patch first, until there are enough seconds
    centre = (p.a + p.b) / 2
    chosen: list[int] = []
    total = 0.0
    for i in sorted(keep, key=lambda i: abs(i - centre)):
        chosen.append(i)
        total += words[i].end - words[i].start
        if total >= seconds:
            break
    chosen.sort()
    # contiguous runs of chosen words, each run with a little air
    pieces: list[np.ndarray] = []
    texts: list[str] = []
    run = [chosen[0]]
    for i in chosen[1:] + [None]:
        if i is not None and i == run[-1] + 1:
            run.append(i)
            continue
        s = max(0.0, words[run[0]].start - 0.05)
        e = min(len(audio) / sr, words[run[-1]].end + 0.08)
        pieces.append(audio[int(s * sr) : int(e * sr)])
        pieces.append(np.zeros(int(0.25 * sr), dtype=np.float32))
        texts.append(" ".join(words[k].text for k in run))
        if i is not None:
            run = [i]
    return np.concatenate(pieces[:-1]) if pieces else audio[: int(seconds * sr)], " ".join(texts)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def seam_fades(words: list[Word], p: Patch) -> tuple[float, float]:
    """Crossfade at each cut, sized to the gap there: a smooth one in a silence; inside running speech only what
    keeps a click away (a longer fade there would swallow the onset of the next word)."""

    def size(gap: float) -> float:
        return 0.025 if gap >= 0.12 else (0.012 if gap >= 0.05 else 0.005)

    return size(_gap(words, p.a - 1)), size(_gap(words, p.b - 1))


def reference_seconds(words: list[Word], p: Patch) -> float:
    """As much of the recording's own speech as the voice clone uses well (4–12 s), from what there is."""
    have = sum(w.end - w.start for i, w in enumerate(words) if i < p.a or i >= p.b)
    return float(min(12.0, max(4.0, have)))


def trim_to_text(take_words: list[Word], expected: list[str], length: float) -> tuple[float, float, list[str]]:
    """Where the expected words are in a take: (start, end, extra words said at its edges).

    A take sometimes says a word of its reference again before its text, or runs on after it; those words
    are cut off so nothing is said twice at the seams."""
    exp = [norm(t) for t in expected if norm(t)]
    got = [norm(w.text) for w in take_words]
    if not exp or not take_words:
        return 0.0, length, []
    blocks = [m for m in difflib.SequenceMatcher(a=got, b=exp, autojunk=False).get_matching_blocks() if m.size]
    if not blocks:
        return 0.0, length, []
    # only what is clearly beyond the text is cut: the take's first / last expected word must be there
    first = blocks[0].a if blocks[0].b == 0 else 0
    last = blocks[-1].a + blocks[-1].size - 1 if blocks[-1].b + blocks[-1].size == len(exp) else len(take_words) - 1
    extra = [w.text for w in take_words[:first]] + [w.text for w in take_words[last + 1 :]]
    if not extra:
        return 0.0, length, []
    start = 0.0
    if first:  # in the gap before the first word, never into the extra word
        prev_end, w_start = take_words[first - 1].end, take_words[first].start
        start = max(prev_end, w_start - 0.06) if w_start - prev_end >= 0.12 else (prev_end + w_start) / 2
    end = length
    if last < len(take_words) - 1:
        w_end, next_start = take_words[last].end, take_words[last + 1].start
        end = min(next_start, w_end + 0.10) if next_start - w_end >= 0.12 else (w_end + next_start) / 2
    return max(0.0, start), max(start, min(end, length)), extra


def splice(audio: np.ndarray, sr: int, start: float, end: float, new: np.ndarray, fade: float = 0.03,
           fade_out: float | None = None) -> np.ndarray:
    """``new`` in place of [start, end); ``fade`` at the left cut, ``fade_out`` at the right one (same if None)."""
    a, b = int(start * sr), int(end * sr)
    n1 = int(fade * sr)
    n2 = int((fade if fade_out is None else fade_out) * sr)
    left, right = audio[:a].copy(), audio[b:].copy()
    new = new.astype(np.float32).copy()
    if len(new) > n1 + n2:
        if n1:
            new[:n1] *= np.linspace(0, 1, n1, dtype=np.float32)
        if n2:
            new[-n2:] *= np.linspace(1, 0, n2, dtype=np.float32)
    if n1 and len(left) > n1:
        left[-n1:] *= np.linspace(1, 0.4, n1, dtype=np.float32)
    if n2 and len(right) > n2:
        right[:n2] *= np.linspace(0.4, 1, n2, dtype=np.float32)
    return np.concatenate([left, new, right])


def match_loudness(new: np.ndarray, around: np.ndarray) -> np.ndarray:
    """The new speech as loud as the speech around it (RMS of the loud part), peaks under −1 dB."""

    def speech_rms(x):
        if len(x) < 400:
            return rms(x)
        w = 240
        frames = x[: len(x) // w * w].reshape(-1, w)
        e = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        loud = e[e >= e.max() * 0.1] if e.max() > 0 else e
        return float(np.sqrt(np.mean(loud**2))) if len(loud) else 0.0

    target, have = speech_rms(around), speech_rms(new)
    if target <= 0 or have <= 0:
        return new
    out = new * (target / have)
    peak = float(np.max(np.abs(out)))
    if peak > 0.89:
        out *= 0.89 / peak
    return out.astype(np.float32)


def load_mono(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sr)



# ---------------------------------------------------------------- seams checked by ear
# where a cut may go relative to the word timings, most likely first: timings are often short at a word's end,
# and a short word next to a long one can swallow part of it
LEFT_TRIES = (0.0, -0.05, 0.04, -0.10, 0.10, -0.18, 0.18)
RIGHT_TRIES = (0.0, 0.05, -0.04, 0.10, -0.10, 0.15, -0.20)


def seam_errors(heard: list[str], edited: list[str], p: Patch) -> int:
    """Words wrong around a patch in what was heard: the patch's words and its two neighbours must be there,
    once, in order (words far from the seams are not counted: the window cuts them anyway)."""
    lo = max(0, p.ja - 6)
    exp = [norm(t) for t in edited[lo : p.jb + 6]]
    got = [norm(t) for t in heard if norm(t)]
    c0, c1 = max(0, p.ja - 1 - lo), min(len(exp), p.jb + 1 - lo)  # neighbour before .. neighbour after
    cost = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=got, b=exp, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if j1 == j2:  # said but not expected: counts when it sits at a seam
            if c0 <= j1 <= c1:
                cost += i2 - i1
            continue
        overlap = min(j2, c1) - max(j1, c0)
        if overlap > 0:
            cost += max(overlap, i2 - i1 if tag == "replace" else 0)
    return cost


def best_seams(audio: np.ndarray, sr: int, words: list[Word], p: Patch, new: np.ndarray, edited: list[str],
               hear, fades: tuple[float, float], margin: float = 1.2, log=None) -> tuple[float, float, int]:
    """The cut points whose splice sounds right: a few positions around the word timings are tried and a short
    window around the seams is heard (``hear(audio) -> words``); the one with the fewest wrong words wins
    (ties: the closest to the timings). Returns (start, end, errors)."""
    total = len(audio) / sr
    base_s, base_e = p.start, p.end
    lo_s = words[p.a - 1].start if p.a > 0 else 0.0
    hi_e = words[p.b].end if p.b < len(words) else total

    # the window always holds both neighbours whole (a word clipped by the window is lost to the ear)
    prev_start = words[p.a - 1].start if p.a > 0 else 0.0
    next_end = words[p.b].end if p.b < len(words) else total

    def judge(s, e):
        fixed = splice(audio, sr, s, e, new, *fades)
        w0 = max(0.0, min(s - margin, prev_start - 0.15))
        w1 = min(len(fixed) / sr, s + len(new) / sr + max(margin, next_end - e + 0.15))
        heard = hear(fixed[int(w0 * sr) : int(w1 * sr)])
        err = seam_errors(heard, edited, p)
        if log:
            log(f"[fixer] cut {s:.2f}–{e:.2f} s · {err} wrong · heard: {' '.join(heard)}")
        return err

    def snap(t, lo, hi):
        return snap_to_valley(audio, sr, t, max(lo, t - 0.02), min(hi, t + 0.02))

    best = (judge(base_s, base_e), 0.0, base_s, base_e)
    if best[0] == 0:
        return base_s, base_e, 0
    # the right cut first (timings are most often short at a word's end), then the left one
    for off in RIGHT_TRIES[1:]:
        if p.b >= len(words) or p.b == p.a and not p.new:
            break
        e = snap(min(hi_e, max(base_s + 0.03, base_e + off)), base_s + 0.03, hi_e)
        err = judge(best[2], e)
        if (err, abs(off)) < (best[0], best[1]):
            best = (err, abs(off), best[2], e)
        if err == 0:
            break
    s_best = best
    for off in LEFT_TRIES[1:]:
        if best[0] == 0 or p.a == 0:
            break
        s = snap(max(lo_s, min(best[3] - 0.03, base_s + off)), lo_s, best[3] - 0.03)
        err = judge(s, s_best[3])
        if (err, abs(off)) < (best[0], best[1]):
            best = (err, abs(off), s, s_best[3])
    return best[2], best[3], best[0]
