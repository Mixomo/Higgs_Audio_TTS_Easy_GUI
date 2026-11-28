"""Montage: a multi-track timeline of voice, music and effect clips.

The timeline lives in the page (``MONTAGE_JS`` in a ``gr.HTML``); its value is
the project as JSON, read by Python to export the mix and written by Python
when another tab sends its blocks here. Clips point at files on disk and play
in the browser through Gradio's file route.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf

from .paths import OUTPUTS_DIR, ROOT

MONTAGE_DIR = ROOT / "projects" / "montage"
TRACK_KINDS = ("voice", "music", "fx")


def file_url(path: str) -> str:
    return "/gradio_api/file=" + quote(str(Path(path).resolve()).replace("\\", "/"), safe="/:")


def _id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def empty_project() -> dict:
    return {
        "version": 1,
        "rev": 0,
        "zoom": 60.0,
        "tracks": [
            {"id": "voice1", "name": "Voice", "kind": "voice", "gain": 0.0, "mute": False, "solo": False},
            {"id": "music1", "name": "Music", "kind": "music", "gain": -8.0, "mute": False, "solo": False},
            {"id": "fx1", "name": "FX", "kind": "fx", "gain": 0.0, "mute": False, "solo": False},
        ],
        "clips": [],
    }


def load_value(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        data = json.loads(value or "")
        if isinstance(data, dict) and "tracks" in data:
            return data
    except (TypeError, json.JSONDecodeError):
        pass
    return empty_project()


def dump(project: dict) -> str:
    return json.dumps(project, ensure_ascii=False)


def initial_value() -> str:
    """The app opens with an empty timeline: a montage comes back from 📂 Load or from a GUI preset."""
    return dump(empty_project())


def duration_of(path: str) -> float:
    try:
        return float(sf.info(path).duration)
    except Exception:
        import librosa

        return float(librosa.get_duration(path=path))


def project_end(project: dict, tracks: set[str] | None = None) -> float:
    ends = [c["start"] + c["dur"] for c in project["clips"] if tracks is None or c["track"] in tracks]
    return max(ends, default=0.0)


def _voice_track(project: dict, speaker: str) -> str:
    """The voice track of a speaker: one lane per voice, so a dialogue lays out like a real session."""
    name = speaker or "Voice"
    for t in project["tracks"]:
        if t["kind"] == "voice" and t["name"] == name:
            return t["id"]
    empty_default = next((t for t in project["tracks"] if t["kind"] == "voice" and t["name"] == "Voice"
                          and not any(c["track"] == t["id"] for c in project["clips"])), None)
    if empty_default is not None:
        empty_default["name"] = name
        return empty_default["id"]
    track = {"id": _id("voice"), "name": name, "kind": "voice", "gain": 0.0, "mute": False, "solo": False}
    voices = [i for i, t in enumerate(project["tracks"]) if t["kind"] == "voice"]
    project["tracks"].insert((voices[-1] + 1) if voices else 0, track)
    return track["id"]


def add_blocks(value, blocks: list[dict], mode: str, gap: float = 0.6) -> tuple[str, str]:
    """``blocks`` from a finished render (file, text, speaker, start, seconds). ``mode``: append or replace."""
    project = empty_project() if mode == "replace" else load_value(value)
    if mode == "replace":
        prev = load_value(value)
        project["zoom"] = prev.get("zoom", project["zoom"])
        project["rev"] = int(prev.get("rev", 0))
    blocks = [b for b in (blocks or []) if b.get("file") and Path(b["file"]).is_file()]
    if not blocks:
        return dump(project), "Nothing to send: generate something first."
    offset = project_end(project) + (gap if project["clips"] else 0.0)
    for b in blocks:
        dur = float(b.get("seconds") or 0) or duration_of(b["file"])
        project["clips"].append({
            "id": _id("c"),
            "track": _voice_track(project, b.get("speaker") or ""),
            "path": str(Path(b["file"]).resolve()),
            "url": file_url(b["file"]),
            "label": (b.get("text") or Path(b["file"]).stem)[:120],
            "start": round(offset + float(b.get("start") or 0.0), 4),
            "offset": 0.0,
            "dur": round(dur, 4),
            "src_dur": round(dur, 4),
            "gain": 0.0,
            "fade_in": 0.0,
            "fade_out": 0.0,
            "source": b.get("source", ""),
            "text": b.get("text") or "",
            "voice": b.get("voice") or "",
            "lora": b.get("lora") or "",
            "speaker": b.get("speaker") or "",
        })
    project["rev"] = int(project.get("rev", 0)) + 1
    what = "Replaced the timeline with" if mode == "replace" else "Appended"
    return dump(project), f"{what} {len(blocks)} clip(s) from {blocks[0].get('source') or 'the render'}."


# ---------------------------------------------------------------- server functions (called from the page)
def montage_probe(*args):
    """Duration and URL of a file the page uploaded or picked."""
    from .live import _server_args

    (path,) = _server_args(args, 1)
    try:
        return {"ok": True, "dur": duration_of(path), "url": file_url(path), "path": str(Path(path).resolve())}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}



# ---------------------------------------------------------------- save / load / export
def list_saved() -> list[str]:
    return sorted(p.stem for p in MONTAGE_DIR.glob("*.json")) if MONTAGE_DIR.is_dir() else []


def save_project(name: str, value) -> str:
    from .audio_utils import safe_name

    name = safe_name(name or "montage", "montage")
    MONTAGE_DIR.mkdir(parents=True, exist_ok=True)
    (MONTAGE_DIR / f"{name}.json").write_text(dump(load_value(value)), encoding="utf-8")
    return name


def load_project(name: str) -> str:
    path = MONTAGE_DIR / f"{name}.json"
    project = load_value(path.read_text(encoding="utf-8"))
    project["rev"] = int(project.get("rev", 0)) + 1000
    missing = [c["path"] for c in project["clips"] if not Path(c["path"]).is_file()]
    if missing:
        print(f"[montage] {len(missing)} clip file(s) of {name} are missing: {missing[:3]}", flush=True)
    for c in project["clips"]:
        c["url"] = file_url(c["path"])
    return dump(project)


def _read(path: str, sr: int, channels: int, cache: dict) -> np.ndarray:
    key = (path, sr, channels)
    if key not in cache:
        import librosa

        data, src_sr = sf.read(path, dtype="float32", always_2d=True)
        data = data.T  # (ch, n)
        if src_sr != sr:
            data = librosa.resample(data, orig_sr=src_sr, target_sr=sr, axis=-1)
        if channels == 1:
            data = data.mean(axis=0, keepdims=True)
        elif data.shape[0] == 1:
            data = np.repeat(data, 2, axis=0)
        else:
            data = data[:2]
        cache[key] = data.astype(np.float32)
    return cache[key]


def _pan(x: np.ndarray, pan: float) -> np.ndarray:
    """Stereo pan, the same law as the browser's StereoPannerNode on a stereo input (centre = untouched)."""
    pan = max(-1.0, min(1.0, float(pan or 0.0)))
    if abs(pan) < 1e-4 or x.shape[0] != 2:
        return x
    left, right = x[0], x[1]
    if pan < 0:
        ang = (pan + 1.0) * np.pi / 2
        return np.stack([left + right * np.cos(ang), right * np.sin(ang)]).astype(np.float32)
    ang = pan * np.pi / 2
    return np.stack([left * np.cos(ang), right + left * np.sin(ang)]).astype(np.float32)


def mix(value, sr: int = 44100, channels: int = 2, normalize_peak: float = -1.0) -> tuple[np.ndarray, float]:
    project = load_value(value)
    tracks = {t["id"]: t for t in project["tracks"]}
    solo = any(t.get("solo") for t in tracks.values())
    end = project_end(project)
    if end <= 0:
        raise ValueError("The timeline is empty.")
    out = np.zeros((channels, int(end * sr) + sr // 10), dtype=np.float32)
    cache: dict = {}
    for c in project["clips"]:
        t = tracks.get(c["track"])
        if t is None or t.get("mute") or (solo and not t.get("solo")):
            continue
        if not Path(c["path"]).is_file():
            raise FileNotFoundError(f"A clip's file is missing: {c['path']}")
        src = _read(c["path"], sr, channels, cache)
        a = int(float(c.get("offset", 0)) * sr)
        n = int(float(c["dur"]) * sr)
        piece = src[:, a : a + n].copy()
        if piece.shape[1] == 0:
            continue
        gain = 10 ** ((float(c.get("gain", 0)) + float(t.get("gain", 0))) / 20)
        piece *= gain
        fi = min(int(float(c.get("fade_in", 0)) * sr), piece.shape[1])
        fo = min(int(float(c.get("fade_out", 0)) * sr), piece.shape[1])
        if fi:
            piece[:, :fi] *= np.linspace(0, 1, fi, dtype=np.float32)
        if fo:
            piece[:, -fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
        piece = _pan(piece, float(t.get("pan", 0)) / 100.0)  # the track's pan (linear: per clip = per bus)
        s = int(float(c["start"]) * sr)
        e = min(s + piece.shape[1], out.shape[1])
        out[:, s:e] += piece[:, : e - s]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    ceiling = 10 ** (normalize_peak / 20)
    if peak > ceiling:
        out *= ceiling / peak
    return out, end


def export(value, name: str = "montage", fmt: str = "44.1 kHz stereo WAV") -> str:
    from .audio_utils import safe_name

    sr, ch = (24000, 1) if fmt.startswith("24") else (48000, 2) if fmt.startswith("48") else (44100, 2)
    audio, _end = mix(value, sr, ch)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"montage_{safe_name(name or 'montage', 'montage')}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    from . import loudness

    audio = loudness.normalize(audio.T, sr).T  # one gain for the whole mix: the app's loudness, the balance stays
    sf.write(str(path), audio.T, sr, subtype="PCM_24")
    return str(path)


# ---------------------------------------------------------------- the page
MONTAGE_HTML = '<div class="tl-root" tabindex="0"><div class="tl-loading">Loading the timeline…</div></div>'

MONTAGE_CSS = r"""
.tl-root { --tl-bg:#101016; --tl-lane:#14141b; --tl-lane2:#17171f; --tl-line:#24242f; --tl-text:#e4e4e7; --tl-muted:#a1a1aa;
  --tl-accent:#8b5cf6; --tl-sel:#ffd166; --tl-voice:#7c5cff; --tl-music:#22a06b; --tl-fx:#d9822b; --tl-head:150px;
  position:relative; background:var(--tl-bg); color:var(--tl-text); border:1px solid var(--tl-line); border-radius:10px;
  font:12px/1.3 system-ui,sans-serif; user-select:none; outline:none; overflow:hidden; }
.tl-root:focus { border-color:var(--tl-accent); }
.tl-bar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; padding:6px 8px; border-bottom:1px solid var(--tl-line); }
.tl-bar button, .tl-bar label.tl-btn { background:#1d1d27; color:var(--tl-text); border:1px solid #2a2a36; border-radius:6px;
  padding:3px 8px; cursor:pointer; font-size:12px; }
.tl-bar button:hover, .tl-bar label.tl-btn:hover { border-color:var(--tl-accent); }
.tl-bar button.on { background:#2e2550; border-color:var(--tl-accent); }
.tl-time { font-variant-numeric:tabular-nums; min-width:150px; padding:0 6px; color:#fff; font-size:13px; }
.tl-sep { width:1px; height:18px; background:var(--tl-line); }
.tl-help { margin-left:auto; color:var(--tl-muted); font-size:11px; }
.tl-body { display:flex; position:relative; }
.tl-heads { width:var(--tl-head); flex:none; border-right:1px solid var(--tl-line); }
.tl-ruler-pad { height:24px; border-bottom:1px solid var(--tl-line); }
.tl-head { height:64px; border-bottom:1px solid var(--tl-line); padding:4px 6px; box-sizing:border-box; display:flex;
  flex-direction:column; gap:3px; background:#101016; }
.tl-head .tl-name { background:transparent; border:none; color:var(--tl-text); font-weight:600; width:100%; font-size:12px; padding:0; }
.tl-head .tl-row { display:flex; gap:3px; align-items:center; }
.tl-head .tl-row button { background:#1d1d27; border:1px solid #2a2a36; color:var(--tl-muted); border-radius:4px; font-size:10px;
  padding:1px 5px; cursor:pointer; }
.tl-head .tl-row button.on { color:#111; background:var(--tl-sel); }
.tl-head .tl-row input[type=range] { width:62px; background:transparent; accent-color:var(--tl-accent); }
.tl-head .tl-kind { width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:4px; }
.tl-scroll { flex:1; overflow-x:auto; overflow-y:hidden; position:relative; }
.tl-canvas { position:relative; }
.tl-ruler { height:24px; border-bottom:1px solid var(--tl-line); position:relative; cursor:pointer; background:#101016; }
.tl-tick { position:absolute; top:0; bottom:0; border-left:1px solid #3a404c; color:var(--tl-muted); font-size:10px; padding-left:3px; }
.tl-tick.minor { top:16px; border-color:#2a2f38; }
.tl-lane { height:64px; border-bottom:1px solid var(--tl-line); position:relative; background:var(--tl-lane); }
.tl-lane:nth-child(even) { background:var(--tl-lane2); }
.tl-lane.drop { box-shadow: inset 0 0 0 2px var(--tl-accent); }
.tl-clip { position:absolute; top:4px; height:56px; border-radius:6px; overflow:hidden; box-sizing:border-box;
  border:1px solid rgba(255,255,255,.25); cursor:grab; }
.tl-clip.voice { background:rgba(124,92,255,.32); }
.tl-clip.music { background:rgba(34,160,107,.35); }
.tl-clip.fx { background:rgba(217,130,43,.38); }
.tl-clip.sel { border:2px solid var(--tl-sel); }
.tl-clip.missing { background:repeating-linear-gradient(45deg,#5a2020,#5a2020 6px,#3a1515 6px,#3a1515 12px); }
.tl-clip canvas { position:absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; }
.tl-clip .tl-label { position:absolute; left:6px; top:2px; right:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  font-size:11px; color:#fff; text-shadow:0 1px 2px #000; pointer-events:none; }
.tl-clip .tl-edge { position:absolute; top:0; bottom:0; width:7px; cursor:ew-resize; }
.tl-clip .tl-edge.l { left:0; } .tl-clip .tl-edge.r { right:0; }
.tl-clip .tl-edge:hover { background:rgba(255,255,255,.25); }
.tl-play { position:absolute; top:0; bottom:0; width:2px; background:#ff5a5f; pointer-events:none; z-index:5; }
.tl-play::before { content:""; position:absolute; top:0; left:-5px; border:6px solid transparent; border-top-color:#ff5a5f; }
.tl-marquee { position:absolute; border:1px dashed var(--tl-sel); background:rgba(255,209,102,.12); pointer-events:none; z-index:6; }
.tl-menu { position:fixed; z-index:10000; background:#1d1d27; border:1px solid #2a2a36; border-radius:8px; padding:4px; min-width:190px;
  box-shadow:0 8px 24px rgba(0,0,0,.5); }
.tl-menu div { padding:5px 10px; border-radius:5px; cursor:pointer; color:#e4e4e7; font-size:12px; }
.tl-menu div:hover { background:#2e2550; }
.tl-menu hr { border:none; border-top:1px solid #2a2a36; margin:3px 0; }
.tl-empty { position:absolute; left:16px; top:40px; color:var(--tl-muted); pointer-events:none; font-size:13px; }
.tl-modal-bg { position:fixed; inset:0; background:rgba(0,0,0,.55); z-index:9000; display:flex; align-items:center; justify-content:center; }
.tl-modal { width:min(720px, calc(100vw - 32px)); max-height:calc(100vh - 40px); overflow:auto; background:#14141b; color:#e4e4e7;
  border:1px solid #2a2a36; border-radius:12px; padding:14px 16px; box-shadow:0 18px 48px rgba(0,0,0,.6); font:13px/1.35 system-ui,sans-serif; }
.tl-modal h3 { margin:0 0 10px; font-size:15px; }
.tl-modal .tl-mrow { display:flex; gap:8px; margin-bottom:8px; flex-wrap:wrap; align-items:flex-end; }
.tl-modal label { display:flex; flex-direction:column; gap:3px; font-size:11px; color:#a1a1aa; flex:1; min-width:180px; }
.tl-modal select, .tl-modal input[type=text], .tl-modal textarea { background:var(--input-background-fill, #0f0f15); color:var(--body-text-color, #e4e4e7);
  border:1px solid var(--border-color-primary, #2a2a36); border-radius:10px; padding:7px 9px; font:13px system-ui,sans-serif; }
.tl-modal textarea { width:100%; min-height:110px; resize:vertical; box-sizing:border-box; }
.tl-modal button { background:var(--button-secondary-background-fill, #1d1d27); color:var(--body-text-color, #e4e4e7);
  border:1px solid var(--border-color-primary, #2a2a36); border-radius:10px; padding:7px 14px; cursor:pointer; transition:border-color .12s, background .12s; }
.tl-modal button:hover { border-color:var(--color-accent, #8b5cf6); }
.tl-modal button.hv-x { border:0; background:transparent; padding:0; }
.tl-modal select:focus, .tl-modal input:focus, .tl-modal textarea:focus { outline:none; border-color:var(--color-accent, #8b5cf6); box-shadow:0 0 0 3px rgba(139,92,246,.18); }
.tl-modal details.tl-ref { margin-bottom:8px; font-size:12px; color:#d4d4d8; }
.tl-modal details.tl-ref summary { cursor:pointer; margin-bottom:6px; }
.tl-modal .tl-refbox { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:6px; }
.tl-modal .tl-refbox audio { height:30px; max-width:300px; }
.tl-modal .tl-up { background:#1d1d27; border:1px solid #2a2a36; border-radius:6px; padding:5px 12px; cursor:pointer; color:#e4e4e7;
  flex:0 0 auto; min-width:0; font-size:12px; display:inline-block; }
.tl-modal .tl-sepnote { font-size:11px; color:#a1a1aa; }
.tl-modal button.primary { background:var(--button-primary-background-fill, #8b5cf6); border-color:transparent; color:#fff; font-weight:600; }
.tl-modal button:disabled { opacity:.5; cursor:wait; }
.tl-modal .tl-llm { border:1px solid var(--border-color-primary, #24242f); border-radius:12px; padding:8px; margin:8px 0; }
.tl-modal .tl-llm-title { font-size:11px; color:#a1a1aa; margin-bottom:6px; }
.tl-modal .tl-mstatus { min-height:18px; color:#ffd166; font-size:12px; margin:6px 0; }
.tl-modal .tl-mbtns { display:flex; gap:8px; justify-content:flex-end; align-items:center; }
.tl-modal .tl-check { flex-direction:row; align-items:center; gap:6px; font-size:12px; color:#d4d4d8; margin-right:auto; }
.tl-clip.busy { outline:2px dashed #ffd166; }
.tl-mixer { display:flex; gap:6px; padding:8px; border-top:1px solid var(--tl-line); overflow-x:auto; background:#171a20; }
.tl-strip { flex:0 0 96px; display:flex; flex-direction:column; align-items:center; gap:5px; padding:6px 4px; border:1px solid #24242f;
  border-radius:6px; background:#14141b; color:var(--tl-text); font-size:11px; }
.tl-strip.master { flex-basis:110px; border-color:#2a2a36; }
.tl-sname { width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-align:center; display:flex; gap:4px; align-items:center; justify-content:center; }
.tl-fwrap { display:flex; gap:6px; align-items:stretch; height:120px; }
.tl-meter { width:10px; height:120px; border-radius:2px; background:#262633; }
.tl-fader { writing-mode:vertical-lr; direction:rtl; width:22px; height:120px; accent-color:#8b5cf6; cursor:ns-resize; }
.tl-db { width:52px; background:#0f0f15; color:var(--tl-text); border:1px solid #2a2a36; border-radius:4px; font-size:11px; padding:1px 3px; text-align:right; }
.tl-panrow { display:flex; gap:4px; align-items:center; width:100%; }
.tl-panrow input { width:62px; accent-color:#d9822b; }
.tl-panl { width:26px; text-align:center; color:var(--tl-muted); font-size:10px; }
.tl-strip .tl-row button { background:#1d1d27; border:1px solid #2a2a36; color:var(--tl-muted); border-radius:4px; font-size:10px; padding:1px 7px; cursor:pointer; }
.tl-strip .tl-row button.on { background:#8b5cf6; color:#111; }
.tl-mnote { font-size:10px; color:var(--tl-muted); text-align:center; line-height:1.25; }
.tl-status { padding:3px 8px; color:var(--tl-muted); font-size:11px; border-top:1px solid var(--tl-line); min-height:16px; }
"""

MONTAGE_JS = r"""
const root = element.querySelector('.tl-root');
const KIND_COLORS = { voice: '#7c5cff', music: '#22a06b', fx: '#d9822b' };
let P = null;            // the project
let selected = new Set();
let anchorId = null;     // for shift-click ranges
let history = [], future = [];
let playhead = 0, playing = false, playStartCtx = 0, playStartPos = 0, nodes = [];
let ctx = null, snap = true;
const buffers = new Map();   // url -> AudioBuffer | Promise | 'error'
const peaksCache = new Map();
let lastSentRev = null, saveTimer = null;
let mixerOpen = false, meterRaf = 0, master = null, preMix = null;
const buses = new Map();      // track id -> {g, pan, an}: volume -> pan -> meter -> master
const meterHold = new Map();

function parse(v) { try { const p = JSON.parse(v || ''); if (p && p.tracks) return p; } catch (e) {} return null; }
function clone(o) { return JSON.parse(JSON.stringify(o)); }
function uid(p) { return p + Math.random().toString(16).slice(2, 10); }
function fmt(t) { t = Math.max(0, t); const m = Math.floor(t / 60), s = t - m * 60; return `${String(m).padStart(2,'0')}:${s.toFixed(2).padStart(5,'0')}`; }
function trackOf(id) { return P.tracks.find((t) => t.id === id); }
function clipOf(id) { return P.clips.find((c) => c.id === id); }
function end() { return P.clips.reduce((m, c) => Math.max(m, c.start + c.dur), 0); }
function z() { return P.zoom || 60; }

function commit(label) {
  applyMix();
  P.rev = (P.rev || 0) + 1;
  lastSentRev = P.rev;
  props.value = JSON.stringify(P);
  status(label || '');
}
function snapshot() { history.push(JSON.stringify(P)); if (history.length > 200) history.shift(); future = []; }
function undo() { if (!history.length) return; future.push(JSON.stringify(P)); P = JSON.parse(history.pop()); selected.clear(); render(); commit('Undo'); }
function redo() { if (!future.length) return; history.push(JSON.stringify(P)); P = JSON.parse(future.pop()); selected.clear(); render(); commit('Redo'); }
function status(msg) { const s = root.querySelector('.tl-status'); if (s) s.textContent = msg; }

// ---------------------------------------------------------------- audio
function audioCtx() { ctx = ctx || new AudioContext(); if (ctx.state === 'suspended') ctx.resume(); return ctx; }
// ---------------------------------------------------------------- mixer: one bus per track
function trackK(t) { const solo = P.tracks.some((x) => x.solo); return (t.mute || (solo && !t.solo)) ? 0 : Math.pow(10, (t.gain || 0) / 20); }
function masterNode() {
  const ac = audioCtx();
  if (!master) { const an = ac.createAnalyser(); an.fftSize = 1024; an.connect(ac.destination); master = { an }; }
  return master.an;
}
function busOf(t) {
  const ac = audioCtx();
  let b = buses.get(t.id);
  if (!b) {
    const g = ac.createGain(); g.channelCount = 2; g.channelCountMode = 'explicit'; g.channelInterpretation = 'speakers';
    const pan = ac.createStereoPanner(); const an = ac.createAnalyser(); an.fftSize = 1024;
    g.connect(pan); pan.connect(an); an.connect(masterNode());
    b = { g, pan, an }; buses.set(t.id, b);
  }
  b.g.gain.setTargetAtTime(trackK(t), ac.currentTime, 0.015);
  b.pan.pan.setTargetAtTime(Math.max(-1, Math.min(1, (t.pan || 0) / 100)), ac.currentTime, 0.015);
  return b;
}
function applyMix() { if (!ctx || !P) return; for (const t of P.tracks) if (buses.has(t.id)) busOf(t); }
function panLabel(p) { p = Math.round(p || 0); return p === 0 ? 'C' : (p < 0 ? 'L' + (-p) : 'R' + p); }
function mixerHtml() {
  return `<div class="tl-mixer">${P.tracks.map((t) => `<div class="tl-strip" data-track="${t.id}">
      <div class="tl-sname" title="${escA(t.name)}"><span class="tl-kind" style="background:${KIND_COLORS[t.kind] || '#888'}"></span>${escA(t.name)}</div>
      <div class="tl-fwrap"><canvas class="tl-meter" width="10" height="120"></canvas>
        <input type="range" class="tl-fader" min="-40" max="12" step="0.5" value="${t.gain || 0}" data-mx="gain" title="Volume · double-click: 0 dB"></div>
      <div><input type="number" class="tl-db" step="0.5" min="-40" max="12" value="${t.gain || 0}" data-mx="gaindb"> dB</div>
      <div class="tl-panrow"><input type="range" min="-100" max="100" step="1" value="${t.pan || 0}" data-mx="pan" title="Pan · double-click: centre"><span class="tl-panl">${panLabel(t.pan)}</span></div>
      <div class="tl-row"><button data-mx="mute" class="${t.mute ? 'on' : ''}" title="Mute">M</button><button data-mx="solo" class="${t.solo ? 'on' : ''}" title="Solo">S</button></div>
    </div>`).join('')}
    <div class="tl-strip master"><div class="tl-sname">Master</div><div class="tl-fwrap"><canvas class="tl-meter" data-master width="10" height="120"></canvas></div>
      <div class="tl-mnote">The export keeps this balance at the app's loudness.</div></div></div>`;
}
const meterBuf = new Float32Array(1024);
function drawMeter(cv, an, key) {
  const g = cv.getContext('2d'), h = cv.height, w = cv.width;
  let pk = 0;
  if (an && playing) { an.getFloatTimeDomainData(meterBuf); for (let i = 0; i < meterBuf.length; i++) { const v = Math.abs(meterBuf[i]); if (v > pk) pk = v; } }
  const db = pk > 1e-5 ? 20 * Math.log10(pk) : -90;
  const shown = Math.max(db, (meterHold.get(key) ?? -90) - 1.2);
  meterHold.set(key, shown);
  const frac = Math.max(0, Math.min(1, (shown + 60) / 60));
  g.fillStyle = '#262633'; g.fillRect(0, 0, w, h);
  g.fillStyle = shown > -3 ? '#e5484d' : shown > -12 ? '#e5b84d' : '#3fb950';
  g.fillRect(0, h - frac * h, w, frac * h);
}
function meterLoop() {
  meterRaf = 0;
  if (!mixerOpen) return;
  root.querySelectorAll('.tl-strip').forEach((s) => {
    const cv = s.querySelector('.tl-meter'); if (!cv) return;
    if (s.classList.contains('master')) drawMeter(cv, master && master.an, 'master'); else drawMeter(cv, buses.get(s.dataset.track)?.an, s.dataset.track);
  });
  if (playing || [...meterHold.values()].some((v) => v > -89)) meterRaf = requestAnimationFrame(meterLoop);
}
function startMeters() { if (mixerOpen && !meterRaf) meterRaf = requestAnimationFrame(meterLoop); }

function load(url) {
  if (buffers.has(url)) return buffers.get(url);
  const p = fetch(url).then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .then((b) => audioCtx().decodeAudioData(b))
    .then((buf) => { buffers.set(url, buf); drawAll(); return buf; })
    .catch((e) => { buffers.set(url, 'error'); render(); return 'error'; });
  buffers.set(url, p);
  return p;
}
function peaks(buf, from, len, cols) {
  const key = `${buf.length}|${from.toFixed(3)}|${len.toFixed(3)}|${cols}`;
  if (peaksCache.has(key)) return peaksCache.get(key);
  const data = buf.getChannelData(0), sr = buf.sampleRate;
  const a = Math.floor(from * sr), n = Math.max(1, Math.floor(len * sr));
  const out = new Float32Array(cols);
  const step = n / cols;
  for (let i = 0; i < cols; i++) {
    let m = 0; const s = a + Math.floor(i * step), e = Math.min(data.length, a + Math.floor((i + 1) * step));
    for (let k = s; k < e; k += Math.max(1, Math.floor((e - s) / 64))) { const v = Math.abs(data[k]); if (v > m) m = v; }
    out[i] = m;
  }
  if (peaksCache.size > 400) peaksCache.clear();
  peaksCache.set(key, out);
  return out;
}
function drawClip(el, c) {
  const cv = el.querySelector('canvas'); const buf = buffers.get(c.url);
  if (!cv || !(buf instanceof AudioBuffer)) return;
  const w = Math.max(1, Math.floor(c.dur * z())), h = 56;
  cv.width = Math.min(w, 4000); cv.height = h;
  const g = cv.getContext('2d'); g.clearRect(0, 0, cv.width, h);
  const pk = peaks(buf, c.offset, c.dur, cv.width);
  g.fillStyle = 'rgba(255,255,255,.75)';
  for (let i = 0; i < pk.length; i++) { const v = Math.min(1, pk[i] * 1.6) * (h / 2 - 4); g.fillRect(i, h / 2 - v, 1, Math.max(1, v * 2)); }
}
function drawAll() { root.querySelectorAll('.tl-clip').forEach((el) => { const c = clipOf(el.dataset.id); if (c) drawClip(el, c); }); }

function stop() {
  for (const n of nodes) { try { n.stop(); } catch (e) {} }
  nodes = [];
  if (playing) { playhead = playStartPos + (ctx.currentTime - playStartCtx); }
  playing = false;
  updatePlayButton();
}
async function play() {
  const ac = audioCtx();
  await Promise.all(P.clips.map((c) => load(c.url)));
  const t0 = ac.currentTime + 0.08;
  for (const c of P.clips) {
    const tr = trackOf(c.track); const buf = buffers.get(c.url);
    if (!tr || !(buf instanceof AudioBuffer)) continue;
    const cEnd = c.start + c.dur; if (cEnd <= playhead) continue;
    const skip = Math.max(0, playhead - c.start);
    const when = t0 + Math.max(0, c.start - playhead);
    const src = ac.createBufferSource(); src.buffer = buf;
    const gain = ac.createGain();
    const k = Math.pow(10, (c.gain || 0) / 20);  // the track's volume and pan are on its bus
    gain.gain.setValueAtTime(k, when);
    if (c.fade_in > 0 && skip < c.fade_in) { gain.gain.setValueAtTime(k * skip / c.fade_in, when); gain.gain.linearRampToValueAtTime(k, when + (c.fade_in - skip)); }
    if (c.fade_out > 0) { const fs = when + (c.dur - skip) - c.fade_out; if (fs > when) { gain.gain.setValueAtTime(k, fs); gain.gain.linearRampToValueAtTime(0, fs + c.fade_out); } }
    src.connect(gain).connect(busOf(tr).g);
    src.start(when, c.offset + skip, c.dur - skip);
    nodes.push(src);
  }
  playStartCtx = t0; playStartPos = playhead; playing = true; updatePlayButton();
  requestAnimationFrame(tick);
  startMeters();
}
function toggle() { if (playing) stop(); else play(); }
function tick() {
  if (!playing) return;
  const t = playStartPos + (ctx.currentTime - playStartCtx);
  if (t >= end() + 0.2) { stop(); playhead = 0; drawPlayhead(); return; }
  setPlayhead(t, true);
  requestAnimationFrame(tick);
}
function setPlayhead(t, follow) {
  playhead = Math.max(0, t); drawPlayhead();
  if (follow) { const sc = root.querySelector('.tl-scroll'); const x = playhead * z(); if (x > sc.scrollLeft + sc.clientWidth - 40) sc.scrollLeft = x - 80; }
}
function seek(t) { const was = playing; if (was) stop(); setPlayhead(t); if (was) play(); }
function drawPlayhead() {
  const ph = root.querySelector('.tl-play'); if (ph) ph.style.left = (playhead * z()) + 'px';
  const tm = root.querySelector('.tl-time'); if (tm) tm.textContent = `${fmt(playhead)} / ${fmt(end())}`;
}
function updatePlayButton() { const b = root.querySelector('[data-act=play]'); if (b) b.textContent = playing ? '⏸ Pause' : '▶ Play'; }

// ---------------------------------------------------------------- render
function render() {
  if (!P) return;
  const scrollLeft = root.querySelector('.tl-scroll')?.scrollLeft || 0;
  const width = Math.max(end() + 30, 60) * z();
  root.innerHTML = `
    <div class="tl-bar">
      <button data-act="play">▶ Play</button><button data-act="stop">⏹</button>
      <span class="tl-time"></span><span class="tl-sep"></span>
      <button data-act="zout" title="Zoom out (wheel)">－</button><button data-act="zin" title="Zoom in (wheel)">＋</button>
      <button data-act="fit" title="Fit the whole timeline">⤢ Fit</button>
      <button data-act="snap" class="${snap ? 'on' : ''}" title="Snap to clip edges and the playhead">🧲 Snap</button>
      <span class="tl-sep"></span>
      <button data-act="split" title="Split at the playhead (S)">✂ Split</button>
      <button data-act="dup" title="Duplicate (Ctrl+D)">⧉ Duplicate</button>
      <button data-act="del" title="Delete (Del)">🗑 Delete</button>
      <button data-act="undo" title="Ctrl+Z">↶</button><button data-act="redo" title="Ctrl+Y">↷</button>
      <span class="tl-sep"></span>
      <label class="tl-btn" title="Import voice, music or effect audio">📂 Import audio<input type="file" accept="audio/*" multiple hidden></label>
      <button data-act="addtrack" title="Add a track">＋ Track</button>
      <button data-act="addline" title="Voice a new dialogue line at the playhead">🎙 ＋ Line at playhead</button>
      <button data-act="mixer" class="${mixerOpen ? 'on' : ''}" title="Volume, pan, mute / solo and level of every track">🎚 Mixer</button>
      <span class="tl-help">Space play · wheel zoom · Shift+wheel scroll · drag to move, edges to trim · Ctrl/Shift+click, drag a box to select · Del delete · S split · right-click for more</span>
    </div>
    <div class="tl-body">
      <div class="tl-heads"><div class="tl-ruler-pad"></div>
        ${P.tracks.map((t) => `<div class="tl-head" data-track="${t.id}">
          <input class="tl-name" value="${(t.name || '').replace(/"/g, '&quot;')}" title="Track name">
          <div class="tl-row"><span class="tl-kind" style="background:${KIND_COLORS[t.kind] || '#888'}"></span>
            <button data-tact="mute" class="${t.mute ? 'on' : ''}" title="Mute">M</button>
            <button data-tact="solo" class="${t.solo ? 'on' : ''}" title="Solo">S</button>
            <input type="range" min="-30" max="12" step="0.5" value="${t.gain || 0}" data-tact="gain" title="Track gain: ${t.gain || 0} dB">
            <button data-tact="deltrack" title="Delete the track and its clips">✕</button></div></div>`).join('')}
      </div>
      <div class="tl-scroll"><div class="tl-canvas" style="width:${width}px">
        <div class="tl-ruler"></div>
        ${P.tracks.map((t) => `<div class="tl-lane" data-track="${t.id}"></div>`).join('')}
        <div class="tl-play"></div>
      </div></div>
    </div>
    ${mixerOpen ? mixerHtml() : ''}
    <div class="tl-status"></div>`;
  drawRuler(width);
  for (const c of P.clips) {
    const lane = root.querySelector(`.tl-lane[data-track="${c.track}"]`); if (!lane) continue;
    const tr = trackOf(c.track);
    const el = document.createElement('div');
    const buf = buffers.get(c.url);
    el.className = `tl-clip ${tr.kind}${selected.has(c.id) ? ' sel' : ''}${buf === 'error' ? ' missing' : ''}`;
    el.dataset.id = c.id;
    el.style.left = (c.start * z()) + 'px'; el.style.width = Math.max(4, c.dur * z()) + 'px';
    el.title = `${c.label}\n${fmt(c.start)} → ${fmt(c.start + c.dur)} (${c.dur.toFixed(2)} s)${c.gain ? ` · ${c.gain} dB` : ''}`;
    el.innerHTML = `<canvas></canvas><div class="tl-label">${(c.label || '').replace(/</g, '&lt;')}</div><div class="tl-edge l"></div><div class="tl-edge r"></div>`;
    lane.appendChild(el);
    if (buf instanceof AudioBuffer) drawClip(el, c); else load(c.url);
  }
  if (!P.clips.length) root.querySelector('.tl-scroll').insertAdjacentHTML('beforeend', '<div class="tl-empty">Empty timeline: send a render here from Inference, Dialogue, Podcast or Fixer, or import audio.</div>');
  root.querySelector('.tl-scroll').scrollLeft = scrollLeft;
  drawPlayhead(); updatePlayButton(); bind();
}
function drawRuler(width) {
  const r = root.querySelector('.tl-ruler'); const zoom = z();
  const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
  const major = steps.find((s) => s * zoom >= 70) || 600; const minor = major / 5;
  let html = '';
  for (let t = 0; t * zoom < width; t += minor) {
    const isMajor = Math.abs(t / major - Math.round(t / major)) < 1e-6;
    html += `<div class="tl-tick${isMajor ? '' : ' minor'}" style="left:${t * zoom}px">${isMajor ? fmt(t).replace(/\.00$/, '') : ''}</div>`;
  }
  r.innerHTML = html;
}

// ---------------------------------------------------------------- editing
function edges(except) {
  const out = [playhead, 0];
  for (const c of P.clips) if (!except.has(c.id)) { out.push(c.start, c.start + c.dur); }
  return out;
}
function snapTime(t, except) {
  if (!snap) return t;
  const lim = 8 / z(); let best = t, bd = lim;
  for (const e of edges(except)) { const d = Math.abs(e - t); if (d < bd) { bd = d; best = e; } }
  return best;
}
function selectClip(id, ev) {
  if (ev.ctrlKey || ev.metaKey) { if (selected.has(id)) selected.delete(id); else selected.add(id); anchorId = id; }
  else if (ev.shiftKey && anchorId && clipOf(anchorId)) {
    const a = clipOf(anchorId), b = clipOf(id);
    const lo = Math.min(a.start, b.start), hi = Math.max(a.start, b.start);
    selected = new Set(P.clips.filter((c) => c.start >= lo - 1e-6 && c.start <= hi + 1e-6).map((c) => c.id));
  } else if (!selected.has(id)) { selected = new Set([id]); anchorId = id; }
  else { anchorId = id; }
  root.querySelectorAll('.tl-clip').forEach((el) => el.classList.toggle('sel', selected.has(el.dataset.id)));
}
function laneAt(y) {
  for (const lane of root.querySelectorAll('.tl-lane')) { const r = lane.getBoundingClientRect(); if (y >= r.top && y < r.bottom) return lane.dataset.track; }
  return null;
}
function del() {
  if (!selected.size) return; snapshot();
  P.clips = P.clips.filter((c) => !selected.has(c.id)); const n = selected.size; selected.clear(); render(); commit(`Deleted ${n} clip(s)`);
}
function dup() {
  if (!selected.size) return; snapshot();
  const sel = P.clips.filter((c) => selected.has(c.id));
  const span = Math.max(...sel.map((c) => c.start + c.dur)) - Math.min(...sel.map((c) => c.start));
  const copies = sel.map((c) => ({ ...clone(c), id: uid('c'), start: c.start + span }));
  P.clips.push(...copies); selected = new Set(copies.map((c) => c.id)); render(); commit(`Duplicated ${copies.length} clip(s)`);
}
function split() {
  const hit = P.clips.filter((c) => playhead > c.start + 0.01 && playhead < c.start + c.dur - 0.01 && (!selected.size || selected.has(c.id)));
  if (!hit.length) { status('Put the playhead over a clip to split it'); return; }
  snapshot();
  for (const c of hit) {
    const cut = playhead - c.start;
    const right = { ...clone(c), id: uid('c'), start: playhead, offset: c.offset + cut, dur: c.dur - cut, fade_in: 0 };
    c.dur = cut; c.fade_out = 0; P.clips.push(right);
  }
  render(); commit(`Split ${hit.length} clip(s)`);
}
function addTrack(kind) {
  snapshot();
  const n = P.tracks.filter((t) => t.kind === kind).length + 1;
  P.tracks.push({ id: uid(kind), name: `${kind[0].toUpperCase() + kind.slice(1)} ${n}`, kind, gain: 0, mute: false, solo: false });
  render(); commit('Track added');
}
function fit() { const sc = root.querySelector('.tl-scroll'); P.zoom = Math.max(2, Math.min(600, (sc.clientWidth - 40) / Math.max(end(), 5))); render(); commit(); }
function zoomAt(factor, clientX) {
  const sc = root.querySelector('.tl-scroll'); const rect = sc.getBoundingClientRect();
  const x = (clientX ?? rect.left + rect.width / 2) - rect.left; const t = (sc.scrollLeft + x) / z();
  P.zoom = Math.max(2, Math.min(800, z() * factor)); render();
  root.querySelector('.tl-scroll').scrollLeft = t * z() - x;
  clearTimeout(saveTimer); saveTimer = setTimeout(() => commit(), 600);
}
async function importFiles(files) {
  const voice = P.tracks.find((t) => t.kind === 'music') || P.tracks[0];
  let at = playhead; const added = [];
  status(`Importing ${files.length} file(s)…`);
  for (const f of files) {
    try {
      const up = await upload(f);
      const info = await server.montage_probe(up.path);
      if (!info.ok) { status(`${f.name}: ${info.error}`); continue; }
      const kind = /music|song|bgm|track/i.test(f.name) ? 'music' : (/fx|sfx|effect/i.test(f.name) ? 'fx' : (info.dur > 20 ? 'music' : 'fx'));
      const tr = P.tracks.find((t) => t.kind === kind) || voice;
      added.push({ id: uid('c'), track: tr.id, path: info.path, url: info.url, label: f.name, start: at, offset: 0, dur: info.dur, src_dur: info.dur, gain: 0, fade_in: 0, fade_out: 0, source: 'import' });
    } catch (e) { status(`${f.name}: ${e}`); }
  }
  if (added.length) { snapshot(); P.clips.push(...added); selected = new Set(added.map((c) => c.id)); render(); commit(`Imported ${added.length} file(s) at ${fmt(at)}`); }
}
function menu(x, y, items) {
  root.querySelectorAll('.tl-menu').forEach((m) => m.remove());
  const m = document.createElement('div'); m.className = 'tl-menu';
  m.style.left = x + 'px'; m.style.top = y + 'px';
  for (const it of items) {
    if (it === '-') { m.appendChild(document.createElement('hr')); continue; }
    const d = document.createElement('div'); d.textContent = it[0]; d.onclick = () => { m.remove(); it[1](); }; m.appendChild(d);
  }
  // inside the component (its scoped styles apply; fixed: still where the pointer is), its clicks kept from the timeline
  ['pointerdown', 'mousedown', 'click', 'dblclick', 'contextmenu', 'wheel'].forEach((t) => m.addEventListener(t, (e) => e.stopPropagation()));
  root.appendChild(m);
  const r = m.getBoundingClientRect();  // kept on screen
  if (r.right > innerWidth - 8) m.style.left = Math.max(8, innerWidth - r.width - 8) + 'px';
  if (r.bottom > innerHeight - 8) m.style.top = Math.max(8, y - r.height - 14) + 'px';
  setTimeout(() => document.addEventListener('pointerdown', function off(e) { if (!m.contains(e.target)) { m.remove(); document.removeEventListener('pointerdown', off); } }), 0);
}
function adjust(fn, label) { if (!selected.size) return; snapshot(); for (const c of P.clips) if (selected.has(c.id)) fn(c); render(); commit(label); }

// ---------------------------------------------------------------- input
function bind() {
  root.querySelector('.tl-bar').onclick = (e) => {
    const act = e.target.closest('button')?.dataset.act; if (!act) return;
    ({ play: toggle, stop: () => { stop(); setPlayhead(0); }, zin: () => zoomAt(1.4), zout: () => zoomAt(1 / 1.4), fit,
       snap: () => { snap = !snap; render(); }, split, dup, del, undo, redo,
       addline: () => openLine({ mode: 'insert', at: playhead }),
       mixer: () => { mixerOpen = !mixerOpen; render(); startMeters(); },
       addtrack: () => menu(e.clientX, e.clientY + 10, [['Voice track', () => addTrack('voice')], ['Music track', () => addTrack('music')], ['FX track', () => addTrack('fx')]]) })[act]?.();
  };
  root.querySelector('input[type=file]').onchange = (e) => { importFiles([...e.target.files]); e.target.value = ''; };
  const mx = root.querySelector('.tl-mixer');
  if (mx) {
    const stripTrack = (el) => { const s = el.closest('.tl-strip'); return s ? trackOf(s.dataset.track) : null; };
    mx.addEventListener('pointerdown', (e) => { if (e.target.dataset.mx) preMix = JSON.stringify(P); });
    mx.addEventListener('input', (e) => {
      const t = stripTrack(e.target); const k = e.target.dataset.mx; if (!t) return;
      const s = e.target.closest('.tl-strip');
      if (k === 'gain') { t.gain = parseFloat(e.target.value); s.querySelector('.tl-db').value = t.gain; }
      else if (k === 'pan') { t.pan = parseInt(e.target.value, 10); s.querySelector('.tl-panl').textContent = panLabel(t.pan); }
      else return;
      applyMix();
    });
    const done = (label) => { history.push(preMix || JSON.stringify(P)); if (history.length > 200) history.shift(); future = []; preMix = null; render(); commit(label); };
    mx.addEventListener('change', (e) => {
      const t = stripTrack(e.target); const k = e.target.dataset.mx; if (!t) return;
      if (k === 'gaindb') { preMix = preMix || JSON.stringify(P); t.gain = Math.max(-40, Math.min(12, parseFloat(e.target.value) || 0)); }
      if (k === 'gain' || k === 'gaindb') done(`${t.name}: ${t.gain} dB`);
      else if (k === 'pan') done(`${t.name}: pan ${panLabel(t.pan)}`);
    });
    mx.addEventListener('dblclick', (e) => {
      const t = stripTrack(e.target); const k = e.target.dataset.mx; if (!t) return;
      preMix = JSON.stringify(P);
      if (k === 'gain') { t.gain = 0; done(`${t.name}: 0 dB`); }
      else if (k === 'pan') { t.pan = 0; done(`${t.name}: pan centre`); }
    });
    mx.addEventListener('click', (e) => {
      const t = stripTrack(e.target); const k = e.target.dataset.mx; if (!t || (k !== 'mute' && k !== 'solo')) return;
      preMix = JSON.stringify(P);
      if (k === 'mute') t.mute = !t.mute; else t.solo = !t.solo;
      done(`${t.name}: ${k} ${t[k] ? 'on' : 'off'}`);
    });
    startMeters();
  }
  root.querySelectorAll('.tl-head').forEach((h) => {
    const t = trackOf(h.dataset.track);
    h.querySelector('.tl-name').onchange = (e) => { snapshot(); t.name = e.target.value; commit('Track renamed'); };
    h.onclick = (e) => {
      const act = e.target.dataset.tact; if (!act || act === 'gain') return;
      snapshot();
      if (act === 'mute') t.mute = !t.mute;
      if (act === 'solo') t.solo = !t.solo;
      if (act === 'deltrack') { P.clips = P.clips.filter((c) => c.track !== t.id); P.tracks = P.tracks.filter((x) => x.id !== t.id); }
      render(); commit();
    };
    h.querySelector('input[type=range]').onchange = (e) => { snapshot(); t.gain = parseFloat(e.target.value); render(); commit(`Track gain ${t.gain} dB`); };
  });
  const sc = root.querySelector('.tl-scroll');
  sc.onwheel = (e) => {
    if (e.shiftKey) { sc.scrollLeft += (e.deltaY || e.deltaX); e.preventDefault(); return; }
    e.preventDefault(); zoomAt(e.deltaY < 0 ? 1.18 : 1 / 1.18, e.clientX);
  };
  root.querySelector('.tl-ruler').onpointerdown = (e) => {
    const r = root.querySelector('.tl-canvas').getBoundingClientRect();
    const move = (ev) => seek((ev.clientX - r.left) / z());
    move(e);
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  };
  const canvas = root.querySelector('.tl-canvas');
  canvas.onpointerdown = (e) => {
    if (e.button !== 0 || e.target.closest('.tl-ruler')) return;
    root.focus({ preventScroll: true });
    const clipEl = e.target.closest('.tl-clip');
    const r = canvas.getBoundingClientRect();
    const t0 = (e.clientX - r.left) / z();
    if (!clipEl) {
      // marquee selection on an empty spot; a plain click moves the playhead
      const box = document.createElement('div'); box.className = 'tl-marquee'; canvas.appendChild(box);
      const x0 = e.clientX - r.left, y0 = e.clientY - r.top; let moved = false;
      const additive = e.ctrlKey || e.metaKey || e.shiftKey; const base = additive ? new Set(selected) : new Set();
      const move = (ev) => {
        const x1 = ev.clientX - r.left, y1 = ev.clientY - r.top; moved = moved || Math.abs(x1 - x0) + Math.abs(y1 - y0) > 4;
        Object.assign(box.style, { left: Math.min(x0, x1) + 'px', top: Math.min(y0, y1) + 'px', width: Math.abs(x1 - x0) + 'px', height: Math.abs(y1 - y0) + 'px' });
        if (!moved) return;
        const sel = new Set(base); const br = box.getBoundingClientRect();
        root.querySelectorAll('.tl-clip').forEach((el) => { const cr = el.getBoundingClientRect(); if (cr.right > br.left && cr.left < br.right && cr.bottom > br.top && cr.top < br.bottom) sel.add(el.dataset.id); });
        selected = sel; root.querySelectorAll('.tl-clip').forEach((el) => el.classList.toggle('sel', selected.has(el.dataset.id)));
      };
      const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); box.remove();
        if (!moved) { if (!additive) { selected.clear(); root.querySelectorAll('.tl-clip.sel').forEach((el) => el.classList.remove('sel')); } seek(t0); } };
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
      return;
    }
    const id = clipEl.dataset.id; const c = clipOf(id);
    const edge = e.target.classList.contains('tl-edge') ? (e.target.classList.contains('l') ? 'l' : 'r') : null;
    selectClip(id, e);
    if (e.ctrlKey || e.metaKey || e.shiftKey) return;
    const moving = P.clips.filter((x) => selected.has(x.id));
    const orig = new Map(moving.map((x) => [x.id, { start: x.start, dur: x.dur, offset: x.offset, track: x.track }]));
    const trackIdx = (tid) => P.tracks.findIndex((t) => t.id === tid);
    let changed = false; const before = JSON.stringify(P);
    const move = (ev) => {
      const dt = (ev.clientX - e.clientX) / z();
      if (Math.abs(ev.clientX - e.clientX) + Math.abs(ev.clientY - e.clientY) > 2) changed = true;
      if (!changed) return;
      if (edge === 'l') {
        const o = orig.get(id); let ns = snapTime(o.start + dt, new Set([id]));
        ns = Math.max(o.start - o.offset, Math.min(ns, o.start + o.dur - 0.05));
        c.offset = o.offset + (ns - o.start); c.dur = o.dur - (ns - o.start); c.start = ns;
      } else if (edge === 'r') {
        const o = orig.get(id); let ne = snapTime(o.start + o.dur + dt, new Set([id]));
        ne = Math.min(ne, o.start + (c.src_dur - o.offset)); c.dur = Math.max(0.05, ne - o.start);
      } else {
        const o = orig.get(id); let ns = Math.max(0, o.start + dt);
        const snapped = snapTime(ns, selected); const snappedEnd = snapTime(ns + o.dur, selected) - o.dur;
        ns = Math.abs(snapped - ns) <= Math.abs(snappedEnd - ns) ? snapped : snappedEnd;
        const shift = Math.max(ns - o.start, -Math.min(...moving.map((x) => orig.get(x.id).start)));
        const lane = laneAt(ev.clientY); const dTrack = lane ? trackIdx(lane) - trackIdx(o.track) : 0;
        for (const x of moving) {
          const ox = orig.get(x.id); x.start = Math.max(0, ox.start + shift);
          const ti = Math.max(0, Math.min(P.tracks.length - 1, trackIdx(ox.track) + dTrack)); x.track = P.tracks[ti].id;
        }
      }
      for (const x of moving) {
        const el = root.querySelector(`.tl-clip[data-id="${x.id}"]`); if (!el) continue;
        el.style.left = (x.start * z()) + 'px'; el.style.width = Math.max(4, x.dur * z()) + 'px';
        const lane = root.querySelector(`.tl-lane[data-track="${x.track}"]`); if (lane && el.parentElement !== lane) lane.appendChild(el);
        if (edge) drawClip(el, x);
      }
      drawPlayhead();
    };
    const up = () => {
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up);
      if (changed) { history.push(before); future = []; render(); commit(edge ? 'Trimmed' : `Moved ${moving.length} clip(s)`); }
    };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  };
  canvas.ondblclick = (e) => {
    const clipEl = e.target.closest('.tl-clip'); if (!clipEl) return;
    e.preventDefault(); openLine({ mode: 'replace', clip: clipOf(clipEl.dataset.id) });
  };
  canvas.oncontextmenu = (e) => {
    e.preventDefault();
    const clipEl = e.target.closest('.tl-clip');
    const r = canvas.getBoundingClientRect(); const t = (e.clientX - r.left) / z();
    if (clipEl) {
      if (!selected.has(clipEl.dataset.id)) selectClip(clipEl.dataset.id, {});
      menu(e.clientX, e.clientY, [
        ['✎ Edit text, voice and LoRA · voice it again', () => openLine({ mode: 'replace', clip: clipOf(clipEl.dataset.id) })],
        ['🎙 ＋ New line after this clip', () => { const c0 = clipOf(clipEl.dataset.id); openLine({ mode: 'insert', at: c0.start + c0.dur, track: c0.track, like: c0 }); }],
        ['▶ Play from here', () => { seek(clipOf(clipEl.dataset.id).start); if (!playing) play(); }],
        ['✂ Split at the playhead', split], ['⧉ Duplicate', dup], ['🗑 Delete', del], '-',
        ['Fade in 0.3 s', () => adjust((c) => { c.fade_in = c.fade_in ? 0 : Math.min(0.3, c.dur / 2); }, 'Fade in')],
        ['Fade out 0.3 s', () => adjust((c) => { c.fade_out = c.fade_out ? 0 : Math.min(0.3, c.dur / 2); }, 'Fade out')],
        ['Gain +3 dB', () => adjust((c) => { c.gain = (c.gain || 0) + 3; }, 'Gain +3 dB')],
        ['Gain −3 dB', () => adjust((c) => { c.gain = (c.gain || 0) - 3; }, 'Gain −3 dB')],
        ['Reset trim and gain', () => adjust((c) => { c.start -= c.offset; c.offset = 0; c.dur = c.src_dur; c.gain = 0; c.fade_in = c.fade_out = 0; }, 'Reset')], '-',
        ['Close the gaps after this clip', () => { snapshot(); const c0 = clipOf(clipEl.dataset.id); let at = c0.start + c0.dur;
          P.clips.filter((c) => c.track === c0.track && c.start > c0.start).sort((a, b) => a.start - b.start).forEach((c) => { c.start = at; at += c.dur; }); render(); commit('Gaps closed'); }],
        ...P.tracks.filter((tr) => tr.id !== clipOf(clipEl.dataset.id).track).map((tr) => [`Move to “${tr.name}”`, () => adjust((c) => { c.track = tr.id; }, `Moved to ${tr.name}`)]),
      ]);
    } else {
      const lane = laneAt(e.clientY);
      menu(e.clientX, e.clientY, [['🎙 ＋ Add a dialogue line here', () => openLine({ mode: 'insert', at: snapTime(t, new Set()), track: lane })],
        ['Move the playhead here', () => seek(t)], ['Select all (Ctrl+A)', () => { selected = new Set(P.clips.map((c) => c.id)); render(); }],
        ['Add a voice track', () => addTrack('voice')], ['Add a music track', () => addTrack('music')], ['Add an FX track', () => addTrack('fx')]]);
    }
  };
}
root.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase(); const mod = e.ctrlKey || e.metaKey;
  if (k === ' ') { e.preventDefault(); toggle(); }
  else if (k === 'delete' || k === 'backspace') { e.preventDefault(); del(); }
  else if (mod && k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
  else if (mod && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); redo(); }
  else if (mod && k === 'a') { e.preventDefault(); selected = new Set(P.clips.map((c) => c.id)); render(); }
  else if (mod && k === 'd') { e.preventDefault(); dup(); }
  else if (k === 's' && !mod) { e.preventDefault(); split(); }
  else if (k === 'escape') { selected.clear(); render(); }
  else if (k === 'home') { seek(0); }
  else if (k === 'end') { seek(end()); }
  else if (k === 'arrowleft' || k === 'arrowright') {
    e.preventDefault(); const dir = k === 'arrowleft' ? -1 : 1;
    if (selected.size && e.altKey) { adjust((c) => { c.start = Math.max(0, c.start + dir * (e.shiftKey ? 1 : 0.1)); }, 'Nudged'); }
    else { seek(playhead + dir * (e.shiftKey ? 5 : 1)); }
  }
  else if (k === '+' || k === '=') zoomAt(1.4);
  else if (k === '-') zoomAt(1 / 1.4);
});
root.addEventListener('dragover', (e) => { e.preventDefault(); });
root.addEventListener('drop', (e) => { e.preventDefault(); const files = [...(e.dataTransfer?.files || [])].filter((f) => f.type.startsWith('audio')); if (files.length) importFiles(files); });

// ---------------------------------------------------------------- the line editor: voice a new line or a clip again
let options = null, modal = null, pendingId = null;
async function getOptions() {
  if (!options) { try { options = await server.montage_options(); } catch (e) { options = { voices: ['None'], loras: [['Same as Single voice', '__inherit__']], ops: [] }; } }
  return options;
}
function escA(s) { return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }
function optHtml(list, cur) { return list.map((o) => { const [l, v] = Array.isArray(o) ? o : [o, o]; return `<option value="${escA(v)}"${v === cur ? ' selected' : ''}>${escA(l)}</option>`; }).join(''); }
function request(req) {
  req.id = uid('r'); pendingId = req.id;
  P.request = req; P.rev = (P.rev || 0) + 1; lastSentRev = P.rev;
  props.value = JSON.stringify(P);
  setTimeout(() => trigger('submit'), 30);
  return req.id;
}
function closeLine() { if (modal) { modal.remove(); modal = null; } pendingId = null; root.querySelectorAll('.tl-clip.busy').forEach((el) => el.classList.remove('busy')); }
async function openLine(o) {
  closeLine();
  options = null;
  const opt = await getOptions();
  const clip = o.clip || null;
  const like = clip || o.like || [...P.clips].filter((c) => c.text && c.start <= (o.at ?? 0)).sort((a, b) => b.start - a.start)[0] || null;
  const voice = (like && like.voice) || (opt.voices.find((v) => v !== 'None') || 'None');
  const lora = (like && like.lora) || '__inherit__';
  const voices = opt.voices.includes(voice) ? opt.voices : [...opt.voices, voice];
  const loras = opt.loras.some((l) => l[1] === lora) ? opt.loras : [...opt.loras, [lora + ' (missing)', lora]];
  // the line's own reference audio (wins over the library voice); a clip voiced from a library voice has none
  const own = like && like.ref_audio && (!like.voice || like.voice === 'None') ? like.ref_audio : '';
  const refUrl = (path) => '/gradio_api/file=' + encodeURI(String(path).replace(/\\/g, '/'));
  const ref = { audio: own, url: own ? refUrl(own) : '', name: own ? String(own).split(/[\\/]/).pop() : '', text: own ? (like.ref_text || '') : '' };
  let refOrig = null;
  const engines = window.__hvSepEngines || [];
  const bg = document.createElement('div'); bg.className = 'tl-modal-bg hv-modal hv-js';
  bg.innerHTML = `<div class="tl-modal hv-card" role="dialog" aria-modal="true">
    <div class="hv-head"><div class="hv-title">${clip ? '✎ Edit the line and voice it again' : `🎙 New dialogue line at ${fmt(o.at || 0)}`}</div>
      <button class="hv-x" data-act="cancel" title="Close (Esc)">✕</button></div>
    <div class="tl-mrow">
      <label>Voice<select data-f="voice">${optHtml(voices, voice)}</select></label>
      <label>LoRA<select data-f="lora">${optHtml(loras, lora)}</select></label>
    </div>
    <details class="tl-ref"${own ? ' open' : ''}><summary>🎙 Own reference audio (optional, wins over the library voice · 5–30 s works best)</summary>
      <div class="tl-refbox" data-ref></div>
      <div class="tl-mrow"><label style="flex:3">Reference transcript<input type="text" data-f="ref_text" placeholder="What the reference says (empty: no transcript)"></label></div>
      <div class="tl-mrow"><label style="flex:3">🎧 Separate the voice from music, noise or other voices<select data-f="sep">${optHtml(engines, 'voc_fv7+gabox_denoise+becruily_karaoke')}</select></label>
        <button data-act="sep">🎧 Separate</button><button data-act="unsep">↩ Original</button></div>
      <div class="tl-sepnote" data-sepnote></div>
    </details>
    <label style="min-width:100%">Text (type @ for tags, {voice:…} / {lora:…} switch the speaker)<textarea class="tag-ac" data-f="text" placeholder="What this voice says…">${escA(clip ? (clip.text || clip.label || '') : '')}</textarea></label>
    <div class="tl-llm"><div class="tl-llm-title">🧠 LLM director on this line</div>
      <div class="tl-mrow"><label>Operation<select data-f="op">${optHtml(opt.ops || [], (opt.ops?.[0] || [])[1])}</select></label>
        <label style="flex:2">Instruction (optional)<input type="text" data-f="instr" placeholder="Review: what to fix · Continue: what comes next"></label>
        <button data-act="llm">🧠 Run</button></div></div>
    <div class="tl-mstatus"></div>
    <div class="tl-mbtns">
      ${clip ? '<label class="tl-check"><input type="checkbox" data-f="ripple" checked> Shift the later clips of this track by the change in length</label>' : '<span style="margin-right:auto"></span>'}
      <button data-act="cancel">Cancel</button>
      <button data-act="voice" class="primary">${clip ? '🎙 Voice it again' : '🎙 Voice and insert'}</button>
    </div></div>`;
  root.appendChild(bg); modal = bg;  // inside the component: its scoped styles apply (fixed: still over the page)
  // the timeline's own mouse handlers must not see what happens in the dialog
  ['pointerdown', 'mousedown', 'click', 'dblclick', 'contextmenu', 'wheel'].forEach((t) => bg.addEventListener(t, (e) => e.stopPropagation()));
  const q = (f) => bg.querySelector(`[data-f="${f}"]`);
  const busy = (on, msg) => { bg.querySelectorAll('button').forEach((b) => { if (b.dataset.act !== 'cancel') b.disabled = on; }); bg.querySelector('.tl-mstatus').textContent = msg || ''; };
  bg.busy = busy;
  const drawRef = () => {
    bg.querySelector('[data-ref]').innerHTML = (ref.audio ? `<audio controls preload="none" src="${escA(ref.url)}"></audio><span>${escA(ref.name)}</span>
      <button data-act="refclear" title="Use the library voice again">✕ Remove</button>` : '<span>None: the library voice above is used.</span>')
      + `<label class="tl-up">📎 ${ref.audio ? 'Replace' : 'Upload'} audio<input type="file" accept="audio/*" hidden data-f="refup"></label>`;
  };
  q('ref_text').value = ref.text;
  drawRef();
  const sepNote = (t) => { bg.querySelector('[data-sepnote]').textContent = t || ''; };
  bg.addEventListener('change', async (e) => {
    if (e.target.dataset.f !== 'refup' || !e.target.files?.length) return;
    const f = e.target.files[0];
    try { const up = await upload(f); Object.assign(ref, { audio: up.path, url: up.url, name: f.name }); refOrig = null; sepNote(''); }
    catch (err) { sepNote('❌ Upload failed: ' + err); }
    drawRef();
  });
  bg.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Escape' && !e.defaultPrevented) closeLine(); });
  bg.addEventListener('pointerdown', (e) => { if (e.target === bg && !pendingId) closeLine(); });
  bg.addEventListener('click', (e) => {
    const act = e.target.dataset.act; if (!act) return;
    if (act === 'cancel') { closeLine(); return; }
    if (act === 'refclear') { Object.assign(ref, { audio: '', url: '', name: '' }); refOrig = null; sepNote(''); drawRef(); return; }
    if (act === 'unsep') { if (refOrig) { Object.assign(ref, refOrig); refOrig = null; sepNote('↩ Original audio back.'); drawRef(); } return; }
    if (act === 'sep') {
      if (!ref.audio) { sepNote('Upload a reference audio first.'); return; }
      e.target.disabled = true; sepNote('Separating the voice… (models download once, 0.2–0.9 GB)');
      server.separate_ref_server(ref.audio, q('sep').value).then((r) => {
        if (r && r.ok) { if (!refOrig) refOrig = { ...ref }; Object.assign(ref, { audio: r.path, url: r.url, name: r.name }); sepNote(r.note); drawRef(); }
        else sepNote('❌ ' + ((r && r.error) || 'Separation failed'));
      }).catch((err) => sepNote('❌ ' + err)).finally(() => { e.target.disabled = false; });
      return;
    }
    const text = q('text').value.trim();
    if (act === 'llm') {
      if (!text && q('op').value !== 'continue') { busy(false, 'Write the line first.'); return; }
      const before = P.clips.filter((c) => c.text && c.id !== clip?.id && c.start < (clip ? clip.start : (o.at || 0)) + 1e-3).sort((a, b) => a.start - b.start)
        .slice(-30).map((c) => ({ speaker: c.voice || c.speaker || trackOf(c.track)?.name || '', text: c.text }));
      busy(true, '🧠 The LLM is working… (progress under the timeline)');
      request({ kind: 'llm', op: q('op').value, text, instruction: q('instr').value, voice: q('voice').value, context: before });
      return;
    }
    if (act === 'voice') {
      if (!text) { busy(false, 'Write what the voice says.'); return; }
      busy(true, '🎙 Voicing… (progress under the timeline)');
      if (clip) root.querySelector(`.tl-clip[data-id="${clip.id}"]`)?.classList.add('busy');
      request({ kind: 'voice', mode: clip ? 'replace' : 'insert', clip_id: clip?.id || null, at: clip ? clip.start : (o.at || 0),
                track: clip ? clip.track : (o.track && trackOf(o.track)?.kind === 'voice' ? o.track : null),
                text, voice: q('voice').value, lora: q('lora').value, ripple: !!q('ripple')?.checked,
                ref_audio: ref.audio || '', ref_text: ref.audio ? q('ref_text').value : '' });
    }
  });
  setTimeout(() => q('text').focus(), 30);
}
function handleReply(r) {
  if (!r || !modal || r.id !== pendingId) return;
  if (r.error) { modal.busy(false, '❌ ' + r.error); pendingId = null; return; }
  if (typeof r.text === 'string') { const ta = modal.querySelector('[data-f="text"]'); ta.value = r.text; }
  if (r.done) { closeLine(); status(r.note || 'Done'); return; }
  modal.busy(false, r.note || ''); pendingId = null;
}

function adopt() {
  const p = parse(props.value);
  if (!p) { P = P || { version: 1, rev: 0, zoom: 60, tracks: [], clips: [] }; render(); return; }
  if (P && p.rev === lastSentRev) return;   // our own edit coming back
  const reply = p.reply; delete p.reply; delete p.request;
  if (reply) { handleReply(reply); if (P && JSON.stringify(P.clips) === JSON.stringify(p.clips) && JSON.stringify(P.tracks) === JSON.stringify(p.tracks)) { delete P.request; P.rev = p.rev; lastSentRev = p.rev; return; } }
  if (P && p.rev !== P.rev) { history.push(JSON.stringify(P)); }
  stop(); P = p; lastSentRev = p.rev; selected.clear(); render();
}
watch('value', adopt);
adopt();
"""
