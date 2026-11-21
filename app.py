from __future__ import annotations

import json
import atexit
import gc
import logging
import math
import os
import re
import shutil
import sys
import time
import warnings
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


def _quiet_asyncio_connection_reset():
    """Hide harmless browser disconnect noise from Gradio/uvicorn on Windows."""
    import asyncio

    if getattr(asyncio.BaseEventLoop.default_exception_handler, "_higgs_quiet", False):
        return
    original = asyncio.BaseEventLoop.default_exception_handler

    def quiet(self, context):
        exc = context.get("exception")
        if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
            return
        return original(self, context)

    quiet._higgs_quiet = True
    asyncio.BaseEventLoop.default_exception_handler = quiet


if sys.platform.startswith("win"):
    _quiet_asyncio_connection_reset()


def _configure_windows_inference_scheduling() -> None:
    """Keep local TTS decoding at full speed when the console is not focused."""
    if not sys.platform.startswith("win"):
        return
    try:
        from higgs_local.win_scheduling import configure_process

        print(configure_process(), flush=True)
    except Exception as exc:
        print(f"[runtime] Windows scheduling policy was not applied: {exc}", flush=True)


_configure_windows_inference_scheduling()

APP_ROOT = Path(__file__).resolve().parent
APP_CACHE = APP_ROOT / "models" / ".cache"
_CPU_THREADS = str(max(2, min(8, os.cpu_count() or 4)))
os.environ.setdefault("HIGGS_CPU_THREADS", _CPU_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", os.environ["HIGGS_CPU_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS", os.environ["HIGGS_CPU_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["HIGGS_CPU_THREADS"])
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(APP_CACHE / "huggingface"))
os.environ.setdefault("HF_XET_CACHE", str(APP_CACHE / "xet"))
os.environ["HF_MODULES_CACHE"] = str(APP_CACHE / "hf_modules")
warnings.filterwarnings(
    "ignore",
    message=".*HTTP_422_UNPROCESSABLE_ENTITY.*",
    category=Warning,
)
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_CONTENT.*", category=Warning)
warnings.filterwarnings("ignore", category=Warning, module=r"gradio\.routes")
warnings.filterwarnings(
    "ignore",
    message="Trying to convert audio automatically from float32 to 16-bit int format.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Kwargs passed to `processor.__call__`.*",
)


class _QuietKnownNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        blocked = [
            "Kwargs passed to `processor.__call__` have to be in `processor_kwargs`",
            "HTTP_422_UNPROCESSABLE_ENTITY",
            "Trying to convert audio automatically from float32 to 16-bit int format",
        ]
        return not any(item in msg for item in blocked)


logging.getLogger("transformers").addFilter(_QuietKnownNoise())
logging.getLogger("gradio").addFilter(_QuietKnownNoise())
logging.getLogger("uvicorn").addFilter(_QuietKnownNoise())

import gradio as gr

from higgs_local.ui_theme import CSS as UI_CSS
from higgs_local.ui_theme import TRAINING_WARNING_HTML as UI_WARNING_HTML
from higgs_local.ui_theme import build_theme, header_html, section
from higgs_local.training_dashboard import audio_evolution_html, monitor_html
from higgs_local.projects import (
    dataset_dirs,
    list_projects,
    load_project,
    open_project_folder,
    project_card_html,
    safe_project_name,
    save_project_section,
)

from higgs_local.audio_utils import SAMPLE_RATE, safe_name, save_wav
from higgs_local.adapters import (
    NONE_ADAPTER,
    current_run_checkpoints,
    adapter_project,
    list_v2_lora_adapters,
    list_v3_lora_adapters,
    lora_checkpoints,
    lora_projects,
    normalize_adapter_choice,
)
from higgs_local.asr import ASRManager, WHISPER_LANGS
from higgs_local.engines import ModelManager, text_chunks_for_ui
from higgs_local.model_registry import (
    DEFAULT_ASR_MODEL,
    ASR_MODELS,
    HIGGS_ASR_MODELS,
    PROCESSOR_MODELS,
    TOKENIZER_MODELS,
    TTS_MODELS,
    ensure_tts_model,
    ensure_asr_model,
    ensure_model,
    model_status,
)
from higgs_local.paths import OUTPUTS_DIR, ROOT, SAMPLES_DIR, TEMP_DIR, V2_DEFAULT_MODEL, V2_DEFAULT_TOKENIZER, ensure_local_dirs
from higgs_local.training_utils import (
    DEFAULT_TRAINING_24GB,
    analyze_training_dataset,
    append_eval_sample,
    build_higgs_dataset,
    build_training_command,
    build_v3_training_command,
    dataset_task_type,
    list_train_datasets,
    slugify,
    validate_higgs_dataset,
    reset_project_for_fresh_run,
)
from higgs_local.training_runner import TrainingProcessManager
from higgs_local.voice_library import list_voice_names, normalize_voice_library, resolve_voice, save_voice_sample


APP_TITLE = "Higgs Audio Local"
MODEL_CHOICES = [
    ("Higgs V3 TTS - 6 - 12+ GB VRAM - better stability", "Higgs V3 TTS"),
    ("Higgs V2 TTS - 16 - 24+ GB VRAM, less language support & stability", "Higgs V2 TTS"),
]
TRAINING_MODEL_CHOICES = [
    ("Higgs V2 TTS", "Higgs V2 TTS"),
    ("Higgs V3 TTS", "Higgs V3 TTS"),
]
SEAMLESS_MODE = "Seamless (auto)"
CHUNK_CHOICES = [SEAMLESS_MODE, "None", "Paragraph/Sentence Auto", "Periods", "Paragraphs", "Lines", "Speaker turns"]
DIALOGUE_MAX_SEGMENTS = 20
WHISPER_VRAM = {
    "CrisperWhisper 2.0 large": "best · ~5 GB VRAM",
    "CrisperWhisper 2.0 turbo": "fast · ~2.5 GB VRAM",
    "CrisperWhisper 2.0 medium": "~2.5 GB VRAM",
    "CrisperWhisper 2.0 small": "light · ~1 GB VRAM",
}
ASR_CHOICES = [(f"{name} - {WHISPER_VRAM.get(name, 'VRAM varies')}", name) for name in ASR_MODELS]
LOG_LINES: list[str] = []
MANAGER = ModelManager()
ASR = ASRManager()
TRAINING = TrainingProcessManager()
LAST_TRAINING_DONE_KEY: tuple[str, int] | None = None
LAST_INFERENCE_SEED: int | None = None
SETTINGS_FILE = ROOT / "config" / "ui_settings.json"
APP_JS = """
() => {
  const scrollConsole = () => {
    const body = document.getElementById('higgs-console-body');
    if (!body || window._hcPaused) return;
    body.scrollTop = body.scrollHeight;
  };
  const tick = () => { scrollConsole(); };
  new MutationObserver(tick).observe(document.body, {
    childList: true, subtree: true, attributes: true
  });
  setInterval(tick, 500);
  tick();
}
"""

class _CmdMirror:
    def __init__(self, stream):
        self.stream = stream
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, data):
        if data:
            _mirror_write(str(data))
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


CMD_MIRROR_LINES = deque(maxlen=1200)
CMD_MIRROR_LOCK = threading.Lock()
CMD_MIRROR_CURRENT = ""
CMD_MIRROR_LIVE_GEN = ""
CMD_MIRROR_OVERWRITE = False


def _clean_cmd_line(line: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).rstrip()


def _is_gen_line(line: str) -> bool:
    return "[gen]" in line and "[v3-gen-diagnostics]" not in line


def _mirror_commit(line: str) -> None:
    global CMD_MIRROR_LIVE_GEN
    line = _clean_cmd_line(line)
    if not line.strip():
        return
    if "[v3-gen-diagnostics]" in line:
        return
    if _is_gen_line(line):
        if " OK:" in line:
            if CMD_MIRROR_LIVE_GEN:
                CMD_MIRROR_LINES.append(CMD_MIRROR_LIVE_GEN)
            CMD_MIRROR_LIVE_GEN = ""
            CMD_MIRROR_LINES.append(line)
            return
        CMD_MIRROR_LIVE_GEN = line
        return
    CMD_MIRROR_LINES.append(line)


def _mirror_write(data: str) -> None:
    global CMD_MIRROR_CURRENT, CMD_MIRROR_OVERWRITE, CMD_MIRROR_LIVE_GEN
    for ch in data:
        if ch == "\r":
            if _is_gen_line(_clean_cmd_line(CMD_MIRROR_CURRENT)):
                CMD_MIRROR_LIVE_GEN = _clean_cmd_line(CMD_MIRROR_CURRENT)
            CMD_MIRROR_CURRENT = ""
            CMD_MIRROR_OVERWRITE = True
            continue
        if ch == "\n":
            _mirror_commit(CMD_MIRROR_CURRENT)
            CMD_MIRROR_CURRENT = ""
            CMD_MIRROR_OVERWRITE = False
            continue
        if CMD_MIRROR_OVERWRITE:
            CMD_MIRROR_CURRENT = ""
            CMD_MIRROR_OVERWRITE = False
        CMD_MIRROR_CURRENT += ch


def _install_cmd_mirror():
    if getattr(sys.stdout, "_higgs_mirror", False):
        return

    class LockedMirror(_CmdMirror):
        _higgs_mirror = True

        def write(self, data):
            with CMD_MIRROR_LOCK:
                return super().write(data)

    sys.stdout = LockedMirror(sys.stdout)
    sys.stderr = LockedMirror(sys.stderr)


def cmd_mirror_text() -> str:
    """Return raw lines (kept for any legacy call sites)."""
    with CMD_MIRROR_LOCK:
        lines = list(CMD_MIRROR_LINES)
        current = _clean_cmd_line(CMD_MIRROR_CURRENT)
        live_gen = CMD_MIRROR_LIVE_GEN
    if current.strip():
        if "[v3-gen-diagnostics]" in current:
            pass
        elif _is_gen_line(current):
            live_gen = current
        else:
            lines.append(current)
    if live_gen:
        lines.append(live_gen)
    return "\n".join(lines[-80:]) or "Idle."


_HTML_ESC = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})
_ATTR_ESC = str.maketrans({"&": "&amp;", '"': "&quot;", "<": "&lt;", ">": "&gt;"})


def _line_color(line: str) -> str:
    l = line.lower()
    if "error" in l or "✗" in l or "traceback" in l or "exception" in l:
        return "#f87171"   # red
    if "warn" in l or "⚠" in l:
        return "#fbbf24"   # amber
    if "[v3-gen-diagnostics]" in l:
        return "#a78bfa"   # purple — final diagnostic
    if "[gen]" in l:
        return "#34d399"   # teal — live progress
    if "[ui]" in l or "[training" in l or "[dataset" in l:
        return "#60a5fa"   # blue — ui/system
    if "saved:" in l or "ready:" in l or "done" in l:
        return "#4ade80"   # green
    return "#cccccc"       # default


def cmd_mirror_html() -> str:
    """Return a self-contained iframe console with smart auto-scroll."""
    with CMD_MIRROR_LOCK:
        lines = list(CMD_MIRROR_LINES)
        current = _clean_cmd_line(CMD_MIRROR_CURRENT)
        live_gen = CMD_MIRROR_LIVE_GEN
    if current.strip():
        if "[v3-gen-diagnostics]" in current:
            pass
        elif _is_gen_line(current):
            live_gen = current
        else:
            lines.append(current)
    if live_gen:
        lines.append(live_gen)
    display = [line for line in lines if "[v3-gen-diagnostics]" not in line][-120:] if lines else []
    if not display:
        display = ["Idle."]

    html_rows = []
    for line in display:
        safe = line.translate(_HTML_ESC)
        color = _line_color(line)
        html_rows.append(f'<div style="color:{color};white-space:pre;line-height:1.55">{safe}</div>')
    content = "\n".join(html_rows)

    srcdoc = f"""<!doctype html>
<html><head><style>
html,body{{margin:0;background:#111;color:#ccc;font-family:Consolas,ui-monospace,monospace;font-size:12px;}}
#wrap{{height:333px;border-radius:8px;border:1px solid #333;overflow:hidden;box-sizing:border-box;}}
#body{{height:333px;overflow:auto;padding:8px 20px 8px 12px;box-sizing:border-box;scrollbar-width:thin;scrollbar-color:#555 transparent;}}
#body::-webkit-scrollbar{{width:5px;height:5px}}#body::-webkit-scrollbar-thumb{{background:#555;border-radius:3px}}
</style></head><body>
<div id="wrap"><div id="body">{content}<div id="anchor"></div></div></div>
<script>
const b=document.getElementById('body');
b.onscroll=()=>{{window._paused=!(b.scrollTop+b.clientHeight>=b.scrollHeight-40);}};
if(!window._paused) b.scrollTop=b.scrollHeight;
setTimeout(()=>{{if(!window._paused)b.scrollTop=b.scrollHeight;}},50);
</script></body></html>"""
    return f'<iframe class="cmd-mirror" scrolling="no" style="display:block;width:100%;height:333px;border:0;border-radius:8px;overflow:hidden;" srcdoc="{srcdoc.translate(_ATTR_ESC)}"></iframe>'


def _load_ui_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_ui_settings(**updates) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = _load_ui_settings()
    data.update(updates)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def v3_compile_default() -> bool:
    return bool(_load_ui_settings().get("v3_torch_compile", False))


def log(message: str) -> str:
    LOG_LINES.append(message)
    del LOG_LINES[:-250]
    return "\n".join(LOG_LINES[-80:])


def mini_log(message: str) -> str:
    lines = [line for line in str(message or "").splitlines() if line.strip()]
    return "\n".join(lines[-3:]) or "Idle."


def cmd_gen_preview(model_version: str, max_new_tokens) -> str:
    name = "Higgs V3" if str(model_version).startswith("Higgs V3") else "Higgs V2"
    return f"[gen] {name}:   0%| starting | 0/{int(max_new_tokens)} frames, ~0.0s audio (live frame/s in CMD)"


def set_button_busy(label: str):
    return gr.update(value=label, interactive=False, variant="secondary")


def restore_button(label: str, variant: str = "primary"):
    return gr.update(value=label, interactive=True, variant=variant)


def _resolve_reference(uploaded_audio, voice_name):
    preset_audio, preset_text = resolve_voice(voice_name)
    return uploaded_audio or preset_audio, preset_text


def _clean_tmp_dir() -> None:
    if not TEMP_DIR.exists():
        return
    for path in TEMP_DIR.iterdir():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError:
            pass


atexit.register(_clean_tmp_dir)


def _save_output(prefix: str, sr: int, wav, text: str = "", reference_name: str = "", persist: bool = True) -> str:
    root = OUTPUTS_DIR if persist else TEMP_DIR / "generated"
    match = re.search(r".*?(?:\(\.\.\.\)|\.\.\.|[.!?…])", (text or "").strip(), flags=re.S)
    excerpt = (match.group(0) if match else (text or "").strip())[:110]
    parts = [
        safe_name(prefix, "higgs")[:36],
        safe_name(reference_name, "no_reference")[:48],
        safe_name(excerpt, "speech")[:110],
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"{'_'.join(parts)}_{stamp}.wav"
    save_wav(path, sr, wav)
    return str(path)


def fresh_audio_value(path: str):
    # Gradio 6 keeps the player's last `playback_position` on the client and
    # seeks the freshly loaded waveform back to it; force the playhead to 0.
    return gr.update(value=str(path), visible=True, playback_position=0)


def _player_audio(sr: int, wav):
    if getattr(wav, "ndim", 1) == 1:
        wav = wav.reshape(-1, 1).repeat(2, axis=1)
    elif getattr(wav, "ndim", 1) == 2 and wav.shape[1] == 1:
        wav = wav.repeat(2, axis=1)
    return int(sr), wav


def keep_inference_audio_output():
    return gr.update()


def notify_done() -> None:
    """Play the completion chime."""
    sound = ROOT / "assets" / "inference_training_done.wav"
    try:
        if sys.platform.startswith("win"):
            import winsound

            if sound.exists():
                winsound.PlaySound(str(sound), winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                winsound.MessageBeep()
    except Exception:
        pass


def _fmt_clock(seconds) -> str:
    seconds = max(0, int(round(float(seconds or 0))))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes}:{sec:02d}"


def progress_bar_html(percent, title: str, detail: str = "", kind: str = "run") -> str:
    """Inference progress bar. ``percent=None`` draws an indeterminate bar."""
    import html as _html

    if percent is None:
        fill = '<div class="pb-fill pb-indeterminate"></div>'
        pct_label = ""
    else:
        pct = max(0.0, min(100.0, float(percent)))
        fill = f'<div class="pb-fill" style="width:{pct:.1f}%"></div>'
        pct_label = f"{pct:.0f}%"
    return (
        f'<div class="pb pb-{kind}">'
        f'<div class="pb-head"><span class="pb-title">{_html.escape(title)}</span><span class="pb-pct">{pct_label}</span></div>'
        f'<div class="pb-track">{fill}</div>'
        f'<div class="pb-detail">{_html.escape(detail)}</div>'
        "</div>"
    )


IDLE_PROGRESS_HTML = (
    '<div class="pb pb-idle"><div class="pb-head"><span class="pb-title">Ready</span><span class="pb-pct"></span></div>'
    '<div class="pb-track"><div class="pb-fill" style="width:0%"></div></div>'
    '<div class="pb-detail">Progress, speed and ETA appear here while generating.</div></div>'
)


def build_segments(text: str, chunk_mode: str, gap_seconds: float = 0.5) -> list[dict]:
    """Text -> ordered segments ({"text", "gap"}) rendered as one continuous take.

    Higgs V3 stays accurate up to roughly 20-25 s of speech per pass (about 300
    characters); longer single passes start skipping text. "Seamless (auto)"
    packs whole sentences up to that size and joins them with short natural
    pauses, so the split is not audible.
    """
    import re as _re

    from higgs_local.audio_utils import paragraph_sentence_split

    text = (text or "").strip()
    if not text:
        return []
    if chunk_mode != SEAMLESS_MODE:
        chunks = text_chunks_for_ui(text, chunk_mode)
        return [{"text": c, "gap": float(gap_seconds) if i else 0.0} for i, c in enumerate(chunks)]
    max_chars = 300
    segments: list[dict] = []
    for p_idx, paragraph in enumerate(p for p in _re.split(r"\n\s*\n+", text) if p.strip()):
        pieces: list[str] = []
        for chunk in paragraph_sentence_split(paragraph, max_chars=max_chars):
            if len(chunk) <= int(max_chars * 1.3):
                pieces.append(chunk)
                continue
            # A single very long sentence: break at clause punctuation.
            current = ""
            for part in _re.split(r"(?<=[,;:])\s+", chunk):
                candidate = f"{current} {part}".strip()
                if current and len(candidate) > max_chars:
                    pieces.append(current)
                    current = part
                else:
                    current = candidate
            if current:
                pieces.append(current)
        for i, piece in enumerate(pieces):
            gap = 0.0 if not segments else (0.15 if i else 0.45)
            segments.append({"text": piece, "gap": gap})
    return segments


def _engine_kwargs(
    text,
    ref_audio,
    ref_text,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    ras_win_len,
    ras_win_max_num_repeat,
    seed,
    chunk_mode,
):
    return {
        "text": text,
        "ref_audio": ref_audio,
        "ref_text": ref_text or "",
        "temperature": max(float(temperature), 0.3),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
        "ras_win_len": int(ras_win_len),
        "ras_win_max_num_repeat": int(ras_win_max_num_repeat),
        "seed": int(seed),
        "chunk_mode": chunk_mode,
    }


def _effective_seed(seed) -> int:
    value = int(seed)
    if value < 0:
        import random

        value = random.randint(0, 2_147_483_647)
    return value


def _remember_inference_seed(seed: int) -> None:
    global LAST_INFERENCE_SEED
    if int(seed) >= 0:
        LAST_INFERENCE_SEED = int(seed)


def use_last_inference_seed(current_seed):
    if LAST_INFERENCE_SEED is None:
        return gr.update(value=current_seed), "No previous generation seed is available yet."
    return gr.update(value=LAST_INFERENCE_SEED), f"Loaded last generation seed: {LAST_INFERENCE_SEED}"


def restore_random_seed():
    return gr.update(value=-1), "Seed restored to random mode (-1)."


def _dialogue_rows_from_values(values):
    rows = []
    for idx in range(0, len(values), 4):
        text, speaker, audio, ref_text = values[idx : idx + 4]
        rows.append({"text": text, "speaker": speaker, "audio": audio, "ref_text": ref_text})
    return rows


def _finish_sentence(text: str) -> str:
    text = (text or "").strip()
    if text and not text.endswith((".", "!", "?", ",", ";", ":", '"', "'", "</SE_e>", "</SE>", ">")):
        return text + "."
    return text


def _dialogue_updates(rows, count, note):
    rows = rows[:DIALOGUE_MAX_SEGMENTS]
    while len(rows) < DIALOGUE_MAX_SEGMENTS:
        rows.append({"text": "", "speaker": "None", "audio": None, "ref_text": ""})
    count = max(1, min(int(count), DIALOGUE_MAX_SEGMENTS))
    updates = [gr.update(value=row["text"]) for row in rows]
    updates += [gr.update(value=row["speaker"]) for row in rows]
    updates += [gr.update(value=row["audio"]) for row in rows]
    updates += [gr.update(value=row["ref_text"]) for row in rows]
    updates += [gr.update(visible=i < count) for i in range(DIALOGUE_MAX_SEGMENTS)]
    return [count, *updates, note]


def add_dialogue_row(index, count, *values):
    rows = _dialogue_rows_from_values(values)
    idx = int(index)
    count = int(count)
    if count < DIALOGUE_MAX_SEGMENTS:
        rows.insert(idx + 1, {"text": "", "speaker": rows[idx]["speaker"], "audio": rows[idx]["audio"], "ref_text": rows[idx]["ref_text"]})
        count += 1
    return _dialogue_updates(rows, count, f"Added dialogue row after {idx + 1}.")


def copy_dialogue_row(index, count, *values):
    rows = _dialogue_rows_from_values(values)
    idx = int(index)
    count = int(count)
    if count < DIALOGUE_MAX_SEGMENTS:
        rows.insert(idx + 1, rows[idx].copy())
        count += 1
    return _dialogue_updates(rows, count, f"Copied dialogue row {idx + 1}.")


def delete_dialogue_row(index, count, *values):
    rows = _dialogue_rows_from_values(values)
    idx = int(index)
    count = int(count)
    if count > 1:
        rows.pop(idx)
        rows.append({"text": "", "speaker": "None", "audio": None, "ref_text": ""})
        count -= 1
    return _dialogue_updates(rows, count, f"Deleted dialogue row {idx + 1}.")


def reset_dialogue_rows(default_speaker):
    rows = [
        {"text": "", "speaker": default_speaker or "None", "audio": None, "ref_text": ""},
        {"text": "", "speaker": default_speaker or "None", "audio": None, "ref_text": ""},
    ]
    return _dialogue_updates(rows, 2, "Dialogue rows reset to default.")


def clear_dialogue_rows(count, *values):
    rows = _dialogue_rows_from_values(values)
    for row in rows[: int(count)]:
        row["text"] = ""
    return _dialogue_updates(rows, count, "Dialogue row text cleared.")


def remove_empty_dialogue_rows(count, *values):
    rows = [row for row in _dialogue_rows_from_values(values) if (row.get("text") or "").strip()]
    if not rows:
        rows = [{"text": "", "speaker": "None", "audio": None, "ref_text": ""}]
    return _dialogue_updates(rows, len(rows), f"Removed empty rows. Active rows: {len(rows) if (rows[0].get('text') or '').strip() else 0}.")


def _progress_adapter(progress, start: float, end: float):
    def update(fraction, desc: str):
        if progress is None:
            return
        if fraction is None:
            progress(start, desc=desc)
            return
        value = start + (end - start) * min(max(float(fraction), 0.0), 1.0)
        progress(value, desc=desc)

    return update


def ensure_tts_for_flow(model_version: str, progress=None) -> str:
    missing_before = model_status(TTS_MODELS[model_version]) != "installed"
    if model_version == "Higgs V2 TTS":
        missing_before = missing_before or model_status(TOKENIZER_MODELS["Higgs V2 tokenizer"]) != "installed"
    path = ensure_tts_model(model_version, progress_cb=_progress_adapter(progress, 0.05, 0.2))
    if missing_before:
        notify_done()
    return f"{model_version} ready: {path}"


def ensure_asr_for_flow(asr_model: str, progress=None) -> str:
    missing_before = model_status(ASR_MODELS[asr_model]) != "installed"
    if asr_model in HIGGS_ASR_MODELS:
        missing_before = missing_before or model_status(PROCESSOR_MODELS["Whisper large-v3 processor"]) != "installed"
    ensure_asr_model(asr_model, progress_cb=_progress_adapter(progress, 0.1, 0.35))
    if asr_model in HIGGS_ASR_MODELS:
        ensure_model(PROCESSOR_MODELS["Whisper large-v3 processor"], progress_cb=_progress_adapter(progress, 0.35, 0.45))
    if missing_before:
        notify_done()
    return f"{asr_model} ready."


def run_inference(
    text,
    model_version,
    lora_adapter,
    voice_preset,
    ref_audio,
    ref_text,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    ras_win_len,
    ras_win_max_num_repeat,
    seed,
    reuse_chunk_seed,
    save_to_outputs,
    chunk_mode,
    gap_seconds=0.5,
    progress=gr.Progress(track_tqdm=False),
):
    def out(audio=None, file=None, console=None, busy=True, bar=None):
        """One UI update for the 7 inference outputs."""
        return (
            gr.update() if audio is None else audio,
            gr.update() if file is None else file,
            text,
            gr.update() if console is None else console,
            set_button_busy("Generating…") if busy else restore_button("⚡ Generate Speech"),
            gr.update(interactive=busy),
            gr.update() if bar is None else bar,
        )

    started = time.perf_counter()
    state = {
        "phase": "Preparing",
        "label": "",
        "seg_idx": 1,
        "seg_total": 1,
        "seg_est": 50,
        "seg_start": started,
        "frames": 0,
        "done_est": 0,
        "total_est": 1,
    }

    def bar(final: str | None = None, kind: str = "run"):
        now = time.perf_counter()
        elapsed = now - started
        if final is not None:
            return progress_bar_html(100 if kind == "done" else 0, final, f"Elapsed {_fmt_clock(elapsed)}", kind=kind)
        if state["phase"] != "Generating":
            return progress_bar_html(None, state["phase"], f"Elapsed {_fmt_clock(elapsed)}")
        frames = int(state["frames"])
        seg_elapsed = max(now - state["seg_start"], 1e-3)
        fps = frames / seg_elapsed
        seg_frac = min(frames / max(state["seg_est"], frames + 1), 0.99)
        done = state["done_est"] + seg_frac * state["seg_est"]
        pct = 100.0 * done / max(state["total_est"], 1)
        remaining = max(state["total_est"] - done, 0)
        eta = remaining / fps if fps > 1 else None
        title = "Generating" + (f" · segment {state['seg_idx']}/{state['seg_total']}" if state["seg_total"] > 1 else "")
        detail = (
            f"{frames} frames · {fps:.0f} fps ({fps / 25.0:.2f}x realtime) · ~{frames / 25.0:.1f}s audio"
            f" · elapsed {_fmt_clock(elapsed)}" + (f" · ETA {_fmt_clock(eta)}" if eta is not None else "")
        )
        return progress_bar_html(min(pct, 99.0), title, detail)

    def start_segment(idx: int, total: int, seg_text: str):
        if idx > 1:
            state["done_est"] += state["seg_est"]
        state.update(
            phase="Generating",
            seg_idx=int(idx),
            seg_total=int(total),
            seg_est=max(50, int(len(seg_text or "") * 1.8)),
            seg_start=time.perf_counter(),
            frames=0,
        )

    def frame_progress(frame, total, desc):
        state["frames"] = int(frame)
        if int(frame) % 64 < 16:
            print(f"[ui-progress] {desc}", flush=True)

    def run_job(job):
        """Run ``job`` on a worker thread and refresh the bar while it works."""
        from higgs_local.win_scheduling import configure_current_thread

        result: dict = {}

        def target():
            configure_current_thread()
            try:
                result["value"] = job()
            except BaseException as exc:  # re-raised in the UI thread
                result["error"] = exc

        worker = threading.Thread(target=target, name="higgs-tts-job", daemon=True)
        worker.start()
        try:
            while worker.is_alive():
                worker.join(0.3)
                yield out(bar=bar())
        finally:
            if worker.is_alive():  # UI cancelled: stop the GPU loop too
                MANAGER.v3.cancel()
                worker.join(timeout=30)
        if "error" in result:
            raise result["error"]
        job_result.append(result.get("value"))

    job_result: list = []
    try:
        if not (text or "").strip():
            message = "Error: text is empty."
            log(message)
            yield out(console=mini_log(message), busy=False, bar=bar("Nothing to generate: the script is empty.", "error"))
            return
        yield out(console="[ui] Preparing inference model", bar=bar())
        print("[ui] Preparing inference model", flush=True)
        ASR.unload()
        MANAGER.set_v2_paths(V2_DEFAULT_MODEL, V2_DEFAULT_TOKENIZER)
        MANAGER.set_lora_adapter(lora_adapter)
        state["phase"] = "Checking model files"
        ready = ensure_tts_for_flow(model_version, progress)
        preset_audio, preset_text = _resolve_reference(ref_audio, voice_preset)
        final_ref_text = ref_text or preset_text
        reference_note = ""
        if preset_audio and lora_adapter and lora_adapter != NONE_ADAPTER:
            reference_note = "LoRA + reference audio active; the same reference is reused for every segment."
        segments = build_segments(text, chunk_mode, float(gap_seconds))
        user_seed = int(seed)
        use_shared_seed = bool(reuse_chunk_seed) or user_seed >= 0 or len(segments) <= 1
        effective_seed = _effective_seed(seed) if use_shared_seed else -1
        kwargs = _engine_kwargs(
            text,
            preset_audio,
            final_ref_text,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            ras_win_len,
            ras_win_max_num_repeat,
            effective_seed,
            chunk_mode,
        )
        kwargs["progress_cb"] = frame_progress
        state["total_est"] = sum(max(50, int(len(s["text"]) * 1.8)) for s in segments) or 1
        if chunk_mode == "None" and len(text) > 400:
            print("[ui] Warning: 'None' mode with long text; Higgs V3 may skip words after ~25 s per pass.", flush=True)

        state["phase"] = "Loading model, codec and LoRA"
        fast_engine = False
        if str(model_version).startswith("Higgs V3"):
            print("[ui] Loading runtime, tokenizer, codec and LoRA", flush=True)

            def load_v3():
                MANAGER.unload_other(model_version)
                MANAGER.v3.load()

            yield from run_job(load_v3)
            fast_engine = MANAGER.v3.fast_active

        print(f"[ui] Synthesizing {len(segments)} segment(s) on the {'fast' if fast_engine else 'compatibility'} engine", flush=True)
        if fast_engine:

            def fast_job():
                wav_out = None
                for event in MANAGER.v3.stream(
                    text,
                    ref_audio=kwargs.get("ref_audio"),
                    ref_text=kwargs.get("ref_text") or "",
                    temperature=kwargs.get("temperature", 1.0),
                    top_p=kwargs.get("top_p", 0.95),
                    top_k=kwargs.get("top_k", 50),
                    max_new_tokens=kwargs.get("max_new_tokens", 2048),
                    seed=effective_seed,
                    progress_cb=frame_progress,
                    segments=segments,
                    emit_audio=False,
                ):
                    if event["type"] == "segment":
                        start_segment(event["index"], event["total"], event.get("text", ""))
                    elif event["type"] == "done":
                        wav_out = event["audio"]
                return wav_out

            yield from run_job(fast_job)
            wav = job_result[-1]
            if wav is None:
                message = "Generation cancelled."
                log(message)
                yield out(console=mini_log(message), busy=False, bar=bar("Cancelled", "error"))
                return
            sr = SAMPLE_RATE
            rtf = MANAGER.v3.last_rtf or 0.0
            engine_log = (
                f"Fast engine: {len(segments)} segment(s), {len(wav) / SAMPLE_RATE:.1f}s audio, "
                f"{(1.0 / rtf) if rtf else 0.0:.2f}x realtime."
                + (f" Adapter: {MANAGER.v3.loaded_adapter_path}." if MANAGER.v3.loaded_adapter_path else "")
            )
        else:
            payloads = [{"text": s["text"], "ref_audio": preset_audio, "ref_text": final_ref_text} for s in segments]
            base_kwargs = {k: v for k, v in kwargs.items() if k not in {"text", "ref_audio", "ref_text"}}
            base_kwargs["chunk_mode"] = "None"
            if use_shared_seed:
                base_kwargs["seed"] = effective_seed

            def chunk_progress(idx, total, chunk_text, chunk=None):
                print(f"[ui] segment {idx}/{total}: {len(chunk_text or '')} chars", flush=True)
                start_segment(idx, total, chunk_text)

            def legacy_job():
                return MANAGER.generate_many(model_version, payloads, float(gap_seconds), chunk_progress=chunk_progress, **base_kwargs)

            yield from run_job(legacy_job)
            sr, wav, engine_log = job_result[-1]
            engine_log = f"{len(segments)} segment(s).\n{engine_log}"

        state["phase"] = "Saving WAV"
        if effective_seed >= 0:
            seed_note = f"Seed used: {effective_seed}."
        else:
            seed_note = "Seed used: random per segment."
        engine_log = f"{seed_note}\n{engine_log}"
        if reference_note:
            engine_log = f"{reference_note}\n{engine_log}"
        reference_name = voice_preset if voice_preset and voice_preset != "None" else Path(ref_audio).stem if ref_audio else "no_reference"
        out_path = _save_output(model_version, sr, wav, text=text, reference_name=reference_name, persist=bool(save_to_outputs))
        _remember_inference_seed(effective_seed)
        save_label = "Saved" if save_to_outputs else "Preview temp file"
        message = f"{ready}\n{engine_log}\n{save_label}: {out_path}"
        log(message)
        notify_done()
        seconds = len(wav) / float(sr) if sr else 0.0
        elapsed = time.perf_counter() - started
        done_label = f"Done · {seconds:.1f}s of audio in {_fmt_clock(elapsed)} ({seconds / max(elapsed, 1e-6):.2f}x realtime)"
        yield out(audio=fresh_audio_value(out_path), file=out_path, console=message, busy=False, bar=bar(done_label, "done"))
    except Exception as exc:
        message = f"Error: {exc}"
        log(message)
        yield out(console=message, busy=False, bar=bar(f"Error: {exc}", "error"))


def run_longform(
    text,
    model_version,
    lora_adapter,
    voice_preset,
    ref_audio,
    ref_text,
    split_mode,
    gap_seconds,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    ras_win_len,
    ras_win_max_num_repeat,
    seed,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        chunks = text_chunks_for_ui(text, split_mode)
        if not chunks:
            return None, None, log("Error: no text chunks found.")
        ASR.unload()
        MANAGER.set_v2_paths(V2_DEFAULT_MODEL, V2_DEFAULT_TOKENIZER)
        MANAGER.set_lora_adapter(lora_adapter)
        ready = ensure_tts_for_flow(model_version, progress)
        preset_audio, preset_text = _resolve_reference(ref_audio, voice_preset)
        final_ref_text = ref_text or preset_text
        payloads = []
        for idx, chunk in enumerate(chunks, 1):
            if progress:
                progress((idx - 1, len(chunks)), desc=f"Generating chunk {idx}/{len(chunks)}")
            payloads.append({"text": chunk, "ref_audio": preset_audio, "ref_text": final_ref_text})
        base_kwargs = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_new_tokens": int(max_new_tokens),
            "ras_win_len": int(ras_win_len),
            "ras_win_max_num_repeat": int(ras_win_max_num_repeat),
            "seed": int(seed),
            "chunk_mode": "None",
        }

        def chunk_progress(idx, total, chunk_text):
            print(f"[ui] long-form chunk {idx}/{total}: {len(chunk_text or '')} chars", flush=True)
            progress(0.2 + 0.72 * ((idx - 1) / max(total, 1)), desc=f"Synthesizing chunk {idx}/{total}")

        sr, wav, engine_log = MANAGER.generate_many(
            model_version, payloads, float(gap_seconds), chunk_progress=chunk_progress, **base_kwargs
        )
        reference_name = voice_preset if voice_preset and voice_preset != "None" else Path(ref_audio).stem if ref_audio else "no_reference"
        out_path = _save_output(model_version, sr, wav, text=text, reference_name=reference_name)
        if progress:
            progress((len(chunks), len(chunks)), desc="Done")
        notify_done()
        return fresh_audio_value(out_path), out_path, log(f"{ready}\nLong-form chunks: {len(chunks)}\n{engine_log}\nSaved: {out_path}")
    except Exception as exc:
        return None, None, log(f"Error: {exc}")


def run_dialogue(
    model_version,
    lora_adapter,
    gap_seconds,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    ras_win_len,
    ras_win_max_num_repeat,
    seed,
    reuse_chunk_seed,
    save_to_outputs,
    chunk_mode,
    row_count,
    *speaker_values,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        progress(0.03, desc="Preparing dialogue model")
        ASR.unload()
        MANAGER.set_v2_paths(V2_DEFAULT_MODEL, V2_DEFAULT_TOKENIZER)
        MANAGER.set_lora_adapter(lora_adapter)
        print("[ui] Preparing dialogue model", flush=True)
        ready = ensure_tts_for_flow(model_version, progress)
        turns = []
        reference_names = []
        visible_values = speaker_values[: max(1, int(row_count)) * 4]
        active_rows = [
            visible_values[idx : idx + 4]
            for idx in range(0, len(visible_values), 4)
            if (visible_values[idx] or "").strip()
        ]
        skipped_empty = (len(visible_values) // 4) - len(active_rows)
        if skipped_empty:
            print(f"[ui] Skipping {skipped_empty} empty dialogue rows.", flush=True)
        active_row_total = len(active_rows)
        for row_idx, (text, preset, audio, ref_text) in enumerate(active_rows, 1):
            preset_audio, preset_text = _resolve_reference(audio, preset)
            reference_names.append(preset if preset and preset != "None" else Path(audio).stem if audio else "no_reference")
            row_chunks = text_chunks_for_ui(text, chunk_mode)
            for chunk_idx, chunk in enumerate(row_chunks, 1):
                turns.append({
                    "text": _finish_sentence(chunk),
                    "ref_audio": preset_audio,
                    "ref_text": ref_text or preset_text,
                    "_row_idx": row_idx,
                    "_row_total": active_row_total,
                    "_chunk_idx": chunk_idx,
                    "_chunk_total": len(row_chunks),
                })
        if not turns:
            message = "Error: dialogue has no speaker text."
            return None, None, mini_log(log(message)), restore_button("⚡ Generate Dialogue"), gr.update(interactive=False)
        payloads = []
        segment_total = len(turns)
        for idx, turn in enumerate(turns, 1):
            print(
                f"[ui] Preparing dialogue row {turn['_row_idx']}/{turn['_row_total']} "
                f"chunk {turn['_chunk_idx']}/{turn['_chunk_total']} "
                f"(segment {idx}/{segment_total})",
                flush=True,
            )
            payloads.append(turn)
        base_kwargs = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_new_tokens": int(max_new_tokens),
            "ras_win_len": int(ras_win_len),
            "ras_win_max_num_repeat": int(ras_win_max_num_repeat),
            "seed": int(seed),
            "chunk_mode": "None",
        }
        progress_state = {
            "start": time.perf_counter(),
            "total_start": time.perf_counter(),
            "segment": 1,
            "row_idx": 1,
            "row_total": active_row_total,
            "chunk_idx": 1,
            "chunk_total": 1,
        }

        def frame_progress(frame, total, desc):
            frame = int(frame)
            total = int(total)
            elapsed = max(time.perf_counter() - progress_state["start"], 1e-3)
            fps = frame / elapsed
            frames_per_sec = getattr(getattr(MANAGER, "v3", None), "frames_per_sec", None) if str(model_version).startswith("Higgs V3") else 25.0
            frames_per_sec = frames_per_sec or 25.0
            audio_sec = frame / float(frames_per_sec)
            segment = int(progress_state["segment"])
            done_frames = (segment - 1) * total + frame
            total_frames = max(segment_total * total, 1)
            total_elapsed = max(time.perf_counter() - progress_state["total_start"], 1e-3)
            eta = max(total_frames - done_frames, 0) / max(done_frames / total_elapsed, 1e-3)
            progress_desc = (
                f"row {progress_state['row_idx']}/{progress_state['row_total']} "
                f"chunk {progress_state['chunk_idx']}/{progress_state['chunk_total']} "
                f"(segment {segment}/{segment_total}): {frame}/{total} frames · "
                f"{fps:.2f} frame/s · ~{audio_sec:.1f}s audio · ETA total {eta / 60:.0f}m {eta % 60:02.0f}s"
            )
            if frame == 1 or frame % 32 == 0:
                print(f"[ui-progress] {progress_desc}", flush=True)
            progress((done_frames, total_frames), desc=progress_desc)

        base_kwargs["progress_cb"] = frame_progress

        def chunk_progress(idx, total, chunk_text, chunk=None):
            chunk = chunk or {}
            progress_state["segment"] = int(idx)
            progress_state["row_idx"] = int(chunk.get("_row_idx", idx))
            progress_state["row_total"] = int(chunk.get("_row_total", active_row_total))
            progress_state["chunk_idx"] = int(chunk.get("_chunk_idx", 1))
            progress_state["chunk_total"] = int(chunk.get("_chunk_total", 1))
            progress_state["start"] = time.perf_counter()
            print(
                f"[ui] dialogue row {progress_state['row_idx']}/{progress_state['row_total']} "
                f"chunk {progress_state['chunk_idx']}/{progress_state['chunk_total']} "
                f"(segment {idx}/{segment_total}): {len(chunk_text or '')} chars",
                flush=True,
            )
            progress(
                ((idx - 1) * int(max_new_tokens), max(segment_total * int(max_new_tokens), 1)),
                desc=(
                    f"Loading/generating dialogue row {progress_state['row_idx']}/{progress_state['row_total']} "
                    f"chunk {progress_state['chunk_idx']}/{progress_state['chunk_total']} "
                    f"(segment {idx}/{segment_total})"
                ),
            )

        sr, wav, engine_log = MANAGER.generate_many(
            model_version, payloads, float(gap_seconds), chunk_progress=chunk_progress, **base_kwargs
        )
        refs = "_".join(dict.fromkeys(reference_names)) or "no_reference"
        out_path = _save_output(
            model_version,
            sr,
            wav,
            text=" ".join(turn.get("text", "") for turn in turns),
            reference_name=refs,
            persist=bool(save_to_outputs),
        )
        progress(0.97, desc="Completion chime")
        notify_done()
        progress(1.0, desc="Done")
        save_label = "Saved" if save_to_outputs else "Preview temp file"
        skipped_note = f"\nSkipped empty rows: {skipped_empty}" if skipped_empty else ""
        message = f"{ready}\nDialogue rows: {active_row_total}{skipped_note}\nGenerated segments: {len(turns)}\nSplit mode: {chunk_mode}\n{engine_log}\n{save_label}: {out_path}"
        log(message)
        return fresh_audio_value(out_path), out_path, message, restore_button("⚡ Generate Dialogue"), gr.update(interactive=False)
    except Exception as exc:
        message = f"Error: {exc}"
        log(message)
        return None, None, message, restore_button("⚡ Generate Dialogue"), gr.update(interactive=False)


def refresh_voices():
    choices = list_voice_names()
    return [gr.update(choices=choices, value="None") for _ in range(6)] + [
        log(f"Voice library refreshed: {len(choices) - 1} voices.")
    ]


def save_sample(audio_path, sample_name, transcript):
    status = save_voice_sample(audio_path, sample_name, transcript)
    notify_done()
    choices = list_voice_names()
    return [gr.update(choices=choices, value="None") for _ in range(6)] + [log(status)]


def stop_generation():
    MANAGER.v3.cancel()
    return (
        log("Stop requested. V3 checks this internally when supported; Gradio also cancels queued events."),
        restore_button("⚡ Generate Speech"),
        gr.update(interactive=False),
    )


def stop_reference_transcription(label: str):
    return mini_log(log("Stop requested. Gradio will cancel queued/running ASR when supported.")), restore_button(label, "secondary"), gr.update(interactive=False)


def stop_dialogue_generation():
    MANAGER.v3.cancel()
    return (
        log("Stop requested. Dialogue generation will stop when the active backend supports cancellation."),
        restore_button("⚡ Generate Dialogue"),
        gr.update(interactive=False),
    )


def stop_dataset_build():
    return (
        "Dataset build stop requested. Gradio will cancel queued/running work when supported.",
        "### Dataset Build\nStop requested.",
        gr.update(value="Build Higgs Dataset", interactive=True, variant="primary"),
        gr.update(interactive=False),
    )


def release_all_models(reason: str = "") -> str:
    """Unload TTS (V2/V3 + fast engine graphs), in-process ASR and free VRAM.

    Called before every heavy flow (dataset build, training, transcription) so
    no model from a previous flow stays resident. Only the ~0.4 GB CUDA context
    of this process remains until the app is closed.
    """
    MANAGER.unload_all()
    ASR.unload()
    gc.collect()
    vram = ""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            vram = f" (VRAM held by the app: {torch.cuda.memory_reserved() / 1024**3:.2f} GB)"
    except Exception:
        pass
    message = f"[vram] All models unloaded{' for ' + reason if reason else ''}{vram}."
    print(message, flush=True)
    return message


def unload_models():
    release_all_models("manual unload")
    return log("Models unloaded and VRAM cleanup requested.")


def _delete_files_in(folder: Path, suffixes: set[str]) -> str:
    deleted = 0
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            path.unlink()
            deleted += 1
    return f"Deleted {deleted} files from {folder}."


def delete_output_audios():
    return log(_delete_files_in(OUTPUTS_DIR, {".wav", ".mp3", ".flac", ".ogg", ".m4a"}))


def delete_reference_samples():
    return log(_delete_files_in(SAMPLES_DIR, {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".txt", ".json"}))


def transcribe_audio(audio_path, asr_model, language, progress=gr.Progress(track_tqdm=False)):
    try:
        progress(0.05, desc="Unloading models")
        release_all_models("transcription")
        ensure_asr_for_flow(asr_model, progress)
        progress(0.45, desc=f"Transcribing with {asr_model}")
        text, status = ASR.transcribe(audio_path, asr_model, language)
        if asr_model in HIGGS_ASR_MODELS and not (text or "").strip():
            status = (
                f"{asr_model} ran but returned an empty transcript. "
                "This Higgs STT checkpoint targets transformers 4.51.0, while the local V3 TTS runtime needs "
                "transformers >=5.5. Use CrisperWhisper for reference transcription in this build."
            )
        notify_done()
        progress(1.0, desc="Done")
        return text, log(status)
    except Exception as exc:
        return "", log(f"ASR error: {exc}")


def transcribe_sample_ui(audio_path, asr_model, language, progress=gr.Progress(track_tqdm=False)):
    text, status = transcribe_audio(audio_path, asr_model, language, progress)
    return text, mini_log(status), restore_button("🔍 Transcribe", "secondary"), gr.update(interactive=False)


def transcribe_reference_ui(audio_path, asr_model, language, progress=gr.Progress(track_tqdm=False)):
    text, status = transcribe_audio(audio_path, asr_model, language, progress)
    return text, mini_log(status), restore_button("🎙️ Transcribe", "secondary"), gr.update(interactive=False)


def save_sample_ui(audio_path, sample_name, transcript):
    result = save_sample(audio_path, sample_name, transcript)
    status = result[-1]
    return [*result[:-1], mini_log(status), restore_button("💾 Save Sample")]


def set_v3_runtime(precision, attention_backend, compile_enabled):
    _save_ui_settings(v3_torch_compile=bool(compile_enabled))
    MANAGER.set_v3_runtime(precision, attention_backend, compile_enabled)
    state = "enabled" if compile_enabled else "disabled"
    return log(
        f"V3 runtime updated: precision={precision}, attention={attention_backend}, torch.compile={state}. "
        "Settings apply on the next V3 model load/generation."
    )


def _lora_kind(model_version: str | None) -> str:
    return "v2" if model_version == "Higgs V2 TTS" else "v3"


def lora_choices_for_model(model_version: str | None) -> list[str]:
    return list_v2_lora_adapters() if _lora_kind(model_version) == "v2" else list_v3_lora_adapters()


def default_lora_for_model(model_version: str | None) -> str:
    # LoRA voices are an explicit choice: start on the base model.
    return NONE_ADAPTER


def lora_project_choices(model_version: str | None) -> list[str]:
    return [NONE_ADAPTER, *lora_projects(_lora_kind(model_version))]


def lora_ui_updates(model_version, adapter_value):
    """(project update, checkpoint update) showing ``adapter_value`` if it still exists."""
    kind = _lora_kind(model_version)
    adapter_value = normalize_adapter_choice(adapter_value)
    projects = lora_project_choices(model_version)
    project = adapter_project(adapter_value)
    if project not in projects:
        project = NONE_ADAPTER
    checkpoints = lora_checkpoints(kind, project)
    values = [value for _, value in checkpoints]
    if adapter_value not in values:
        adapter_value = values[0] if values else None
    return (
        gr.update(choices=projects, value=project),
        gr.update(choices=checkpoints, value=adapter_value, interactive=bool(checkpoints)),
    )


def on_lora_project_change(model_version, project):
    checkpoints = lora_checkpoints(_lora_kind(model_version), project)
    return gr.update(choices=checkpoints, value=checkpoints[0][1] if checkpoints else None, interactive=bool(checkpoints))


def refresh_lora_adapters(model_version, current_adapter):
    return lora_ui_updates(model_version, current_adapter)


def sync_lora_for_model(model_version, current_adapter, v2_lora, v3_lora):
    current_adapter = normalize_adapter_choice(current_adapter)
    if _lora_kind(model_version) == "v2":
        v3_lora = current_adapter if current_adapter.endswith("/qwen3_lora") else v3_lora
        remembered = v2_lora
    else:
        v2_lora = current_adapter if current_adapter.endswith("/lora_adapter") else v2_lora
        remembered = v3_lora
    project_update, adapter_update = lora_ui_updates(model_version, remembered)
    return project_update, adapter_update, v2_lora, v3_lora


def set_lora_adapter(model_version, adapter_choice, v2_lora, v3_lora):
    adapter_choice = normalize_adapter_choice(adapter_choice)
    MANAGER.set_lora_adapter(adapter_choice)
    if model_version == "Higgs V2 TTS":
        v2_lora = adapter_choice or NONE_ADAPTER
    else:
        v3_lora = adapter_choice or NONE_ADAPTER
    return log(f"LoRA adapter selected: {adapter_choice or NONE_ADAPTER}"), v2_lora, v3_lora


def list_training_projects() -> list[str]:
    # One project list for Dataset Preparation and Training (data/ + exp/ + projects/).
    return list_projects()


def _running_project() -> str | None:
    return TRAINING.run_name if TRAINING.is_running() else None


def project_card(project_name) -> str:
    return project_card_html(project_name, _running_project())


_DATASET_SETTING_KEYS = (
    "source_folder", "asr_model", "language", "val_split", "recursive",
    "isolate_voice", "segment_voice", "min_segment_sec", "max_segment_sec",
)


def on_dataset_project_change(project_name):
    """Show the project card and restore the dataset settings saved for it."""
    saved = load_project(project_name).get("dataset", {})
    updates = [gr.update(value=saved[key]) if key in saved and saved[key] is not None else gr.update() for key in _DATASET_SETTING_KEYS]
    return (project_card(project_name), *updates)


def _checkpoint_number(path: Path) -> int:
    name = path.name
    if not name.startswith("checkpoint-"):
        return -1
    try:
        return int(name.rsplit("-", 1)[1])
    except ValueError:
        return -1


def best_resume_checkpoint_for_project(project_name: str | None) -> str:
    safe_name = slugify(project_name or "", "")
    if not safe_name:
        return ""
    run_dir = ROOT / "exp" / safe_name
    if not run_dir.exists():
        return ""
    if (run_dir / "trainer_state.pt").exists():
        return run_dir.relative_to(ROOT).as_posix()
    current = current_run_checkpoints(run_dir)  # only the latest run's snapshots
    checkpoints = [
        path
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and path.name in current and (path / "trainer_state.pt").exists()
    ]
    if checkpoints:
        best = max(checkpoints, key=_checkpoint_number)
        return best.relative_to(ROOT).as_posix()
    return ""


def list_project_resume_choices(project_name: str | None) -> list[str]:
    choices = ["None"]
    safe_name = slugify(project_name or "", "")
    if not safe_name:
        return choices
    run_dir = ROOT / "exp" / safe_name
    if not run_dir.exists():
        return choices
    if (run_dir / "trainer_state.pt").exists():
        choices.append(run_dir.relative_to(ROOT).as_posix())
    current = current_run_checkpoints(run_dir)  # only the latest run's snapshots
    checkpoints = [
        path
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and path.name in current and (path / "trainer_state.pt").exists()
    ]
    for path in sorted(checkpoints, key=_checkpoint_number, reverse=True):
        choices.append(path.relative_to(ROOT).as_posix())
    return choices


def resume_choice_to_state(choice):
    return "" if not choice or choice == "None" else choice


def _rel_to_root(value) -> str:
    if not value:
        return ""
    path = Path(str(value))
    try:
        if path.is_absolute():
            return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(value)
    return path.as_posix()


def _load_trainer_args(project_name: str | None) -> tuple[dict, str]:
    resume = best_resume_checkpoint_for_project(project_name)
    if resume:
        state_path = ROOT / resume / "trainer_state.pt"
        try:
            import torch

            state = torch.load(state_path, map_location="cpu", weights_only=False)
            return dict(state.get("args") or {}), resume
        except Exception as exc:
            print(f"[training-project] Could not load {state_path}: {exc}", flush=True)

    safe_name = slugify(project_name or "", "")
    config_path = ROOT / "exp" / safe_name / "project_config.json" if safe_name else None
    if config_path and config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            args = dict(payload.get("args") or {})
            if payload.get("model"):
                args["_project_model"] = payload.get("model")
            return args, resume
        except Exception as exc:
            print(f"[training-project] Could not load {config_path}: {exc}", flush=True)
    return {}, resume


def save_project_config(project_name: str, model_name: str, args: dict, resume_checkpoint: str = "") -> None:
    safe_name = slugify(project_name or "", "")
    if not safe_name:
        return
    project_dir = ROOT / "exp" / safe_name
    project_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "project": safe_name,
        "model": model_name,
        "resume_checkpoint": resume_checkpoint or "",
        "args": args,
    }
    (project_dir / "project_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def refresh_training_projects():
    projects = list_training_projects()
    return gr.update(choices=projects)


def default_training_values():
    return (
        gr.update(value="Higgs V2 TTS"),
        gr.update(),
        gr.update(),
        gr.update(value="LoRA adapter"),
        gr.update(value=3),
        gr.update(value=1000),
        gr.update(value=DEFAULT_TRAINING_24GB["batch"]),
        gr.update(value=DEFAULT_TRAINING_24GB["grad_accum"]),
        gr.update(value=DEFAULT_TRAINING_24GB["lr"]),
        gr.update(value=DEFAULT_TRAINING_24GB["lora_rank"]),
        gr.update(value=True),
        gr.update(value=DEFAULT_TRAINING_24GB["bf16"]),
        gr.update(value=DEFAULT_TRAINING_24GB["logging_steps"]),
        gr.update(value=DEFAULT_TRAINING_24GB["save_steps"]),
        gr.update(value=DEFAULT_TRAINING_24GB["eval_steps"]),
        gr.update(value=False),
        gr.update(value=""),
        gr.update(value=1000),
    )


VRAM_PRESETS = [
    "24 GB VRAM",
    "32 GB VRAM",
    "48 GB VRAM",
    "80 GB VRAM",
    "96 GB VRAM",
]


def _autotune_no_update(message: str, report: str = ""):
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        log(message),
        report or f"### Dataset Auto-Tune\n\n{message}",
    )


def _choose_autotune_batch(vram_preset: str, avg_duration: float, sample_count: int) -> tuple[int, float, int]:
    if vram_preset == "96 GB VRAM":
        if avg_duration <= 4.0 and sample_count >= 256:
            return 10, 96.0, 8
        if avg_duration <= 8.0 and sample_count >= 192:
            return 8, 96.0, 8
        if avg_duration <= 15.0 and sample_count >= 128:
            return 6, 96.0, 8
        if avg_duration <= 25.0 and sample_count >= 64:
            return 3, 96.0, 8
        return 1, 96.0, 8
    if vram_preset == "80 GB VRAM":
        if avg_duration <= 4.0 and sample_count >= 256:
            return 8, 80.0, 8
        if avg_duration <= 8.0 and sample_count >= 192:
            return 6, 80.0, 8
        if avg_duration <= 15.0 and sample_count >= 128:
            return 4, 80.0, 8
        if avg_duration <= 25.0 and sample_count >= 64:
            return 2, 80.0, 8
        return 1, 80.0, 8
    if vram_preset == "48 GB VRAM":
        if avg_duration <= 4.0 and sample_count >= 192:
            return 6, 48.0, 8
        if avg_duration <= 6.0 and sample_count >= 128:
            return 4, 48.0, 8
        if avg_duration <= 10.0 and sample_count >= 96:
            return 3, 48.0, 8
        if avg_duration <= 15.0 and sample_count >= 64:
            return 2, 48.0, 8
        return 1, 48.0, 8
    if vram_preset == "32 GB VRAM":
        if avg_duration <= 3.0 and sample_count >= 128:
            return 3, 36.0, 8
        if avg_duration <= 8.0 and sample_count >= 64:
            return 2, 36.0, 8
        return 1, 36.0, 8
    if avg_duration <= 3.5 and sample_count >= 64:
        return 2, 24.0, 8
    return 1, 24.0, 8


def _format_seconds(value) -> str:
    return "unknown" if value is None else f"{float(value):.2f}s"


def _format_percent(value) -> str:
    return "unknown" if value is None else f"{float(value) * 100:.0f}%"


def _format_autotune_report(analysis, vram_preset: str, settings: dict | None, warnings_list: list[str]) -> str:
    lines = [
        "### Dataset Auto-Tune",
        "",
        f"**Source:** {analysis.source}",
        f"**Trainable:** {'yes' if analysis.trainable else 'no'}",
        f"**Samples:** {analysis.sample_count}",
        f"**Duration:** {'unknown' if analysis.total_duration is None else f'{analysis.total_duration / 60.0:.1f} min'}",
        f"**Average clip:** {_format_seconds(analysis.avg_duration)}",
        f"**Median clip:** {_format_seconds(analysis.median_duration)}",
        f"**Clip range:** {_format_seconds(analysis.min_duration)} - {_format_seconds(analysis.max_duration)}",
        f"**Transcript coverage:** {_format_percent(analysis.transcript_coverage)}",
        f"**Languages:** {', '.join(analysis.languages) if analysis.languages else 'unknown'}",
        f"**Speakers:** {', '.join(analysis.speakers) if analysis.speakers else 'unknown'}",
    ]
    if settings:
        lines += [
            "",
            "### Recommended Training Settings",
            "",
            f"**VRAM preset:** {vram_preset}",
            f"**Batch:** {settings['batch']}",
            f"**Gradient accumulation:** {settings['grad_accum']}",
            f"**Effective batch:** {settings['effective_batch']} samples/update",
            f"**Approx audio/update:** {settings['approx_audio_per_update']:.1f}s",
            f"**Steps/epoch:** {settings['steps_per_epoch']}",
            f"**Target passes:** {settings['target_passes']}",
            f"**Epochs field:** {settings['epochs']} approximate passes needed for the selected max steps",
            f"**Max steps:** {settings['max_steps']}",
            f"**Learning rate:** {settings['lr']:g}",
            f"**LoRA rank:** {settings['lora_rank']}",
            f"**Save/Eval every:** {settings['save_eval']} steps",
        ]
    if warnings_list:
        lines += ["", "### Warnings", ""]
        lines += [f"- {item}" for item in warnings_list]
    return "\n".join(lines)


def _v3_recipe(minutes: float, sample_count: int, avg_duration: float, vram_preset: str) -> dict:
    """Higgs V3 LoRA recipe for the padded-batch trainer (warmup + cosine).

    Measured on a 20 min single-speaker set (193 clips): LR 1e-4, rank 32,
    effective batch 8, cosine -> best held-out loss 4.021 at ~5 epochs (3 min),
    vs 4.036 at best (step 732, ~85 min) for 1.5e-5 constant, which then
    overfit past the base model by step ~2000. Keep the Best eval checkpoint.
    """
    vram = {"24 GB VRAM": 24, "32 GB VRAM": 32, "48 GB VRAM": 48, "80 GB VRAM": 80, "96 GB VRAM": 96}.get(vram_preset, 24)
    # ~25 s of audio per micro-batch fits 24 GB without gradient checkpointing.
    audio_budget = 25.0 * vram / 24.0
    batch = int(max(1, min(16, audio_budget // max(avg_duration, 1.0))))
    effective_target = 8 if minutes < 30 else 16
    grad_accum = int(max(1, math.ceil(effective_target / batch)))
    if minutes < 8:
        epochs, lr, rank = 12, 1e-4, 16
    elif minutes < 30:
        epochs, lr, rank = 8, 1e-4, 32
    elif minutes < 90:
        epochs, lr, rank = 5, 8e-5, 32
    else:
        epochs, lr, rank = 4, 8e-5, 64
    steps_per_epoch = max(1, math.ceil(sample_count / max(batch * grad_accum, 1)))
    eval_every = int(max(10, min(250, steps_per_epoch)))
    return {
        "batch": batch,
        "grad_accum": grad_accum,
        "effective_batch": batch * grad_accum,
        "approx_audio_per_update": batch * grad_accum * avg_duration,
        "steps_per_epoch": steps_per_epoch,
        "target_passes": epochs,
        "epochs": epochs,
        "max_steps": -1,
        "lr": lr,
        "lora_rank": rank,
        "save_eval": eval_every,
    }


def auto_tune_training_ui(train_data_dir, eval_data_dir="", vram_preset="24 GB VRAM", model_name="Higgs V3 TTS", optimizer="adamw"):
    """V3: full dataset-driven plan (recipe + anti-overfitting). V2: legacy auto-tune."""
    keep = [gr.update()] * 13
    if not str(model_name).startswith("Higgs V3") or not train_data_dir:
        return (*auto_tune_training_from_dataset(train_data_dir, eval_data_dir, vram_preset, model_name), *keep)
    try:
        from higgs_local.autotune import plan_markdown, plan_training, profile_dataset

        train_dir = ROOT / train_data_dir if not os.path.isabs(str(train_data_dir)) else Path(train_data_dir)
        eval_dir = (ROOT / eval_data_dir if not os.path.isabs(str(eval_data_dir)) else Path(eval_data_dir)) if eval_data_dir else None
        profile = profile_dataset(train_dir, eval_dir)
        vram_gb = int(str(vram_preset or "24").split()[0])
        plan = plan_training(profile, vram_gb, optimizer or "adamw")
        v = plan.values
        report = plan_markdown(profile, plan, str(train_data_dir))
        message = f"V3 auto-tune: {profile.clips} clips, {profile.minutes:.1f} min -> {v['epochs']} epochs max, eval every {v['eval_steps'] or '-'} steps."
        return (
            gr.update(value=v["epochs"]),
            gr.update(value=-1),
            gr.update(value=v["batch"]),
            gr.update(value=v["grad_accum"]),
            gr.update(value=v["lr"]),
            gr.update(value=v["lora_rank"]),
            gr.update(value=True),
            gr.update(value=True),
            gr.update(value=v["logging_steps"]),
            gr.update(value=v["save_steps"]),
            gr.update(value=v["eval_steps"]),
            log(message),
            report,
            gr.update(value=v["lora_alpha"]),
            gr.update(value=v["lr_scheduler"]),
            gr.update(value=v["warmup_ratio"]),
            gr.update(value=v["weight_decay"]),
            gr.update(value=v["train_audio_modules"]),
            gr.update(value=v["grad_ckpt"]),
            gr.update(value=v["max_clip"]),
            gr.update(value=v["prodigy_d_coef"]),
            gr.update(value=v["patience"]),
            gr.update(value=v["min_delta"]),
            gr.update(value=v["restore_best"]),
            gr.update(value=v["max_batch_seconds"]),
            gr.update(value=v["eval_audio"]),
        )
    except Exception as exc:
        return (*_autotune_no_update(f"Auto-tune failed: {exc}"), *keep)


def auto_tune_training_from_dataset(train_data_dir, eval_data_dir="", vram_preset="24 GB VRAM", model_name="Higgs V3 TTS"):
    if not train_data_dir:
        return _autotune_no_update(
            "Auto-tune needs a selected train dataset.",
            "### Dataset Auto-Tune\n\nSelect a train dataset, choose a VRAM preset, then click **Analyze & Auto-tune**.",
        )
    try:
        analysis = analyze_training_dataset(train_data_dir)
        warnings_list = list(analysis.warnings)
        if not analysis.trainable:
            warnings_list.append("This folder is not trainable yet because Higgs V2/V3 trainers require metadata.json.")
            report = _format_autotune_report(analysis, vram_preset, None, warnings_list)
            return _autotune_no_update("Dataset analyzed, but hyperparameters were not changed.", report)

        sample_count = int(analysis.sample_count)
        total_duration = float(analysis.total_duration or 0.0)
        minutes = total_duration / 60.0
        avg_duration = analysis.avg_duration
        if avg_duration is None:
            avg_duration = 5.0
            warnings_list.append("Duration unavailable; assuming 5.0s average clip length for planning.")

        if str(model_name).startswith("Higgs V3"):
            settings = _v3_recipe(minutes, sample_count, float(avg_duration), vram_preset or "24 GB VRAM")
            if sample_count < 20:
                warnings_list.append("Very small dataset; high overfit risk. Watch eval loss and use the Best eval checkpoint.")
            if analysis.max_duration is not None and analysis.max_duration > 30:
                warnings_list.append("Clips over 30 s are skipped by the V3 trainer; rebuild the dataset with segmentation on.")
            if not eval_data_dir:
                warnings_list.append("No validation dataset selected: no eval loss and no Best eval checkpoint.")
            report = _format_autotune_report(analysis, vram_preset or "24 GB VRAM", settings, warnings_list)
            message = f"V3 auto-tune applied from {analysis.source}: {sample_count} samples, {minutes:.1f} min, {vram_preset}."
            return (
                gr.update(value=settings["epochs"]),
                gr.update(value=-1),
                gr.update(value=settings["batch"]),
                gr.update(value=settings["grad_accum"]),
                gr.update(value=settings["lr"]),
                gr.update(value=settings["lora_rank"]),
                gr.update(value=True),
                gr.update(value=True),
                gr.update(value=10),
                gr.update(value=settings["save_eval"]),
                gr.update(value=settings["save_eval"]),
                log(message),
                report,
            )

        batch_value, target_audio_seconds_per_update, max_grad_accum = _choose_autotune_batch(
            vram_preset or "24 GB VRAM",
            float(avg_duration),
            sample_count,
        )
        grad_accum_value = max(
            1,
            min(max_grad_accum, int(math.ceil(target_audio_seconds_per_update / max(float(avg_duration) * batch_value, 1.0)))),
        )
        effective_batch = batch_value * grad_accum_value
        approx_audio_per_update = effective_batch * float(avg_duration)
        steps_per_epoch = max(1, math.ceil(sample_count / max(effective_batch, 1)))

        if minutes < 8:
            target_passes = 18
            lr_value = 1e-5
            rank_value = 8
        elif minutes < 30:
            target_passes = 15
            lr_value = 1.5e-5
            rank_value = 16
        elif minutes < 90:
            target_passes = 10
            lr_value = 2e-5
            rank_value = 16
        else:
            target_passes = 6
            lr_value = 2e-5
            rank_value = 16

        max_steps_value = int(max(600, min(2500, steps_per_epoch * target_passes)))
        if sample_count < 80:
            max_steps_value = int(min(max_steps_value, 1200))
        epochs_value = int(max(1, math.ceil(max_steps_value / max(steps_per_epoch, 1))))
        save_eval = int(max(100, min(250, max_steps_value // 4)))

        if sample_count < 20:
            warnings_list.append("Very small dataset; high overfit risk.")
        if minutes < 5:
            warnings_list.append("Very short dataset; LoRA rank 8 is recommended.")
        if avg_duration > 15:
            warnings_list.append("Long clips detected; consider splitting into shorter segments.")
        if analysis.max_duration is not None and analysis.max_duration > 29:
            warnings_list.append("Some clips exceed ~29s; Higgs generation/training may behave worse with long clips.")
        if analysis.transcript_coverage is not None and analysis.transcript_coverage < 0.9:
            warnings_list.append("Transcript coverage below 90%; check missing or empty TXT files.")
        if not eval_data_dir:
            warnings_list.append("No validation dataset selected; eval loss/audio preview may be limited.")
        if vram_preset in {"48 GB VRAM", "80 GB VRAM", "96 GB VRAM"} and avg_duration > 20:
            warnings_list.append("High-VRAM preset selected, but clips are very long. Batch was kept conservative; consider splitting long clips.")
        if vram_preset in {"80 GB VRAM", "96 GB VRAM"} and sample_count < 128:
            warnings_list.append("Large VRAM preset selected with a small dataset. Higher batch may reduce update frequency; monitor overfitting.")

        settings = {
            "batch": batch_value,
            "grad_accum": grad_accum_value,
            "effective_batch": effective_batch,
            "approx_audio_per_update": approx_audio_per_update,
            "steps_per_epoch": steps_per_epoch,
            "target_passes": target_passes,
            "epochs": epochs_value,
            "max_steps": max_steps_value,
            "lr": lr_value,
            "lora_rank": rank_value,
            "save_eval": save_eval,
        }
        report = _format_autotune_report(analysis, vram_preset or "24 GB VRAM", settings, warnings_list)
        message = f"Auto-tune applied from {analysis.source}: {sample_count} samples, {minutes:.1f} min, {vram_preset}."
        return (
            gr.update(value=epochs_value),
            gr.update(value=max_steps_value),
            gr.update(value=batch_value),
            gr.update(value=grad_accum_value),
            gr.update(value=lr_value),
            gr.update(value=rank_value),
            gr.update(value=True),
            gr.update(value=True),
            gr.update(value=10),
            gr.update(value=save_eval),
            gr.update(value=save_eval),
            log(message),
            report,
        )
    except Exception as exc:
        return _autotune_no_update(f"Auto-tune failed: {exc}")


V3_FIELD_COUNT = 14


def _v3_field_updates(args: dict) -> tuple:
    """Restore the V3 recipe widgets from a run's stored arguments (missing = keep)."""

    def pick(key, cast):
        if key not in args or args[key] is None:
            return gr.update()
        try:
            return gr.update(value=cast(args[key]))
        except (TypeError, ValueError):
            return gr.update()

    alpha = gr.update()
    if args.get("lora_alpha") is not None:
        try:
            alpha_value = int(args["lora_alpha"])
            rank_value = int(args.get("lora_rank") or 0)
            alpha = gr.update(value=0 if rank_value and alpha_value == 2 * rank_value else alpha_value)
        except (TypeError, ValueError):
            pass
    train_audio = gr.update(value=not bool(args["freeze_audio_head"])) if "freeze_audio_head" in args else gr.update()
    optimizer = args.get("optimizer")
    return (
        alpha,
        pick("lr_scheduler", str),
        pick("warmup_ratio", float),
        pick("weight_decay", float),
        train_audio,
        pick("audio_lr_scale", float),
        pick("gradient_checkpointing", bool),
        pick("max_audio_seconds", float),
        gr.update(value=optimizer) if optimizer in ("adamw", "prodigy", "prodigy_sf") else gr.update(),
        pick("prodigy_d_coef", float),
        pick("early_stop_patience", int),
        pick("early_stop_min_delta", float),
        pick("restore_best", bool),
        pick("max_batch_seconds", float),
    )


def select_training_project(project_name, current_model):
    project_name = project_name or ""
    args, resume = _load_trainer_args(project_name)
    if not args:
        datasets = list_train_datasets()
        status = (
            ""
            if project_name
            else "Select or type a project name."
        )
        train_default, eval_default = dataset_dirs(project_name) if project_name else ("", "")
        return (
            gr.update(value=current_model or "Higgs V3 TTS"),
            gr.update(choices=datasets, value=train_default or None),
            gr.update(choices=[""] + datasets, value=eval_default),
            gr.update(value="LoRA adapter"),
            gr.update(value=3),
            gr.update(value=1000),
            gr.update(value=DEFAULT_TRAINING_24GB["batch"]),
            gr.update(value=DEFAULT_TRAINING_24GB["grad_accum"]),
            gr.update(value=DEFAULT_TRAINING_24GB["lr"]),
            gr.update(value=DEFAULT_TRAINING_24GB["lora_rank"]),
            gr.update(value=True),
            gr.update(value=DEFAULT_TRAINING_24GB["bf16"]),
            gr.update(value=DEFAULT_TRAINING_24GB["logging_steps"]),
            gr.update(value=DEFAULT_TRAINING_24GB["save_steps"]),
            gr.update(value=DEFAULT_TRAINING_24GB["eval_steps"]),
            gr.update(value=False),
            gr.update(),
            gr.update(value=1000),
            gr.update(choices=list_project_resume_choices(project_name), value="None"),
            "",
            status,
            "",
            *(gr.update() for _ in range(V3_FIELD_COUNT)),
        )

    model_name = args.get("_project_model") or current_model or "Higgs V2 TTS"
    if (ROOT / "exp" / project_name / "qwen3_lora").exists() or list((ROOT / "exp" / project_name).glob("checkpoint-*/qwen3_lora")):
        model_name = "Higgs V3 TTS"
    elif (ROOT / "exp" / project_name / "lora_adapter").exists() or list((ROOT / "exp" / project_name).glob("checkpoint-*/lora_adapter")):
        model_name = "Higgs V2 TTS"
    train_dir = _rel_to_root(args.get("train_data_dir"))
    eval_dir = _rel_to_root(args.get("eval_data_dir"))
    if not train_dir or not (ROOT / train_dir).exists():
        # Stored dataset moved/deleted: fall back to the project's own dataset.
        train_dir, eval_dir = dataset_dirs(project_name)
    datasets = list_train_datasets()
    use_lora_value = bool(args.get("use_lora", True))
    status = ""
    log_tail = (
        f"Project loaded: {project_name}\n"
        f"Model: {model_name}\n"
        f"Train: {train_dir or 'not stored'}\n"
        f"Eval: {eval_dir or 'none'}\n"
        f"Resume: {resume or 'none'}"
    )
    return (
        gr.update(value=model_name),
        gr.update(choices=datasets, value=train_dir) if train_dir else gr.update(choices=datasets),
        gr.update(choices=[""] + datasets, value=eval_dir) if eval_dir else gr.update(choices=[""] + datasets),
        gr.update(value="LoRA adapter" if use_lora_value else "Full fine-tune"),
        gr.update(value=int(args.get("num_train_epochs", 3))),
        gr.update(value=int(args.get("max_steps", -1))),
        gr.update(value=int(args.get("batch_size", DEFAULT_TRAINING_24GB["batch"]))),
        gr.update(value=int(args.get("gradient_accumulation_steps", DEFAULT_TRAINING_24GB["grad_accum"]))),
        gr.update(value=float(args.get("learning_rate", DEFAULT_TRAINING_24GB["lr"]))),
        gr.update(value=int(args.get("lora_rank", DEFAULT_TRAINING_24GB["lora_rank"]))),
        gr.update(value=use_lora_value),
        gr.update(value=bool(args.get("bf16", DEFAULT_TRAINING_24GB["bf16"]))),
        gr.update(value=int(args.get("logging_steps", DEFAULT_TRAINING_24GB["logging_steps"]))),
        gr.update(value=int(args.get("save_steps", DEFAULT_TRAINING_24GB["save_steps"]))),
        gr.update(value=int(args.get("eval_steps", DEFAULT_TRAINING_24GB["eval_steps"]))),
        gr.update(value=bool(args.get("enable_eval_audio", False))),
        gr.update(value=str(args.get("eval_text") or "")),
        gr.update(value=int(args.get("eval_audio_max_new_tokens", 1000))),
        gr.update(choices=list_project_resume_choices(project_name), value=resume or "None"),
        resume,
        status,
        log_tail,
        *_v3_field_updates(args),
    )


def reload_training_project(project_name, current_model):
    """🔄 button: re-read everything about the project from disk."""
    return (
        gr.update(choices=list_training_projects()),
        *select_training_project(project_name, current_model),
        project_card(project_name),
        *training_monitor_updates(project_name, force=True),
    )


def delete_training_project(project_name):
    safe_name = slugify(project_name or "", "")
    if not safe_name:
        message = "No project selected."
        return (
            gr.update(choices=list_training_projects(), value=None),
            *default_training_values(),
            gr.update(choices=["None"], value="None"),
            "",
            "⚠️ No project selected.",
            message,
        )
    project_dir = (ROOT / "exp" / safe_name).resolve()
    exp_root = (ROOT / "exp").resolve()
    try:
        project_dir.relative_to(exp_root)
    except ValueError:
        message = f"Refusing to delete outside exp: {project_dir}"
        return (
            gr.update(choices=list_training_projects(), value=project_name),
            *default_training_values(),
            gr.update(choices=["None"], value="None"),
            "",
            "⚠️ Delete refused.",
            message,
        )
    if project_dir.exists():
        shutil.rmtree(project_dir)
        message = f"Deleted project: {project_dir}"
    else:
        message = f"Project did not exist: {project_dir}"
    return (
        gr.update(choices=list_training_projects(), value=None),
        *default_training_values(),
        gr.update(choices=["None"], value="None"),
        "",
        "Project deleted. Fields reset.",
        message,
    )


def load_sample_for_ui(name, use_ref_text=True):
    audio, text = resolve_voice(name)
    return audio, text if use_ref_text else ""


def refresh_sample_dropdown():
    return gr.update(choices=list_voice_names())


def update_clips_count(text, chunk_mode):
    text = (text or "").strip()
    if not text:
        return gr.update(value="")
    if chunk_mode == SEAMLESS_MODE:
        count = max(len(build_segments(text, chunk_mode)), 1)
    elif chunk_mode == "None":
        count = 1
    else:
        count = max(len(text_chunks_for_ui(text, chunk_mode)), 1)
    words = len(text.split())
    est = len(text) / 14.0  # ~14 characters per second of speech
    est_label = f"{est:.0f} s" if est < 90 else f"{est / 60:.1f} min"
    parts = [f"{len(text):,} characters", f"{words:,} words", f"{count} segment{'s' if count != 1 else ''}", f"≈ {est_label} of audio"]
    warn = " · ⚠️ single pass over ~25 s may skip words" if chunk_mode == "None" and len(text) > 400 else ""
    return gr.update(value=" · ".join(parts) + warn)


def refresh_dataset_manifests():
    datasets = list_train_datasets()
    train_default = next((item for item in datasets if item.endswith("/train")), None)
    eval_default = next((item for item in datasets if item.endswith("/eval")), None)
    return gr.update(choices=datasets, value=train_default), gr.update(choices=[""] + datasets, value=eval_default or "")


def prepare_higgs_dataset(
    source_folder,
    dataset_name,
    asr_model,
    asr_language,
    asr_batch_size,
    val_split,
    recursive,
    isolate_voice=True,
    segment_voice=True,
    min_segment_sec=2.0,
    max_segment_sec=15.0,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        from higgs_local.dataset_audio import SegmentOptions

        release_all_models("dataset build")
        ensure_asr_for_flow(asr_model, progress)
        min_sec = float(min_segment_sec)
        max_sec = max(float(max_segment_sec), min_sec + 1.0)
        audio_options = SegmentOptions(isolate=bool(isolate_voice), segment=bool(segment_voice), min_sec=min_sec, max_sec=max_sec)

        def asr_transcribe(audio_path: str) -> str:
            text, status = ASR.transcribe(audio_path, asr_model, asr_language)
            print(f"[dataset-asr] {Path(audio_path).name}: {status}", flush=True)
            return text

        def asr_transcribe_many(audio_paths: list[str]) -> dict[str, str]:
            return ASR.transcribe_many_whisper(
                audio_paths,
                asr_model,
                asr_language,
                batch_size=int(asr_batch_size),
                progress_cb=lambda done, total, desc: progress((done, total), desc=desc),
            )

        summary = build_higgs_dataset(
            source_folder=source_folder,
            dataset_name=dataset_name,
            asr_transcribe=asr_transcribe,
            asr_transcribe_many=asr_transcribe_many,
            language=asr_language,
            dataset_task_type="single_speaker_smart_voice",
            ref_audio_path=None,
            ref_transcript="",
            val_split=float(val_split),
            recursive=bool(recursive),
            progress=lambda fraction, desc: progress(fraction, desc=desc),
            audio_options=audio_options,
        )
        release_all_models("dataset build finished")
        project = summary.dataset_dir.name
        save_project_section(
            project,
            "dataset",
            {
                "source_folder": source_folder,
                "asr_model": asr_model,
                "language": asr_language,
                "val_split": float(val_split),
                "recursive": bool(recursive),
                "isolate_voice": bool(isolate_voice),
                "segment_voice": bool(segment_voice),
                "min_segment_sec": float(min_segment_sec),
                "max_segment_sec": float(max_segment_sec),
                "train_clips": summary.train_samples,
                "eval_clips": summary.eval_samples,
                "minutes": round(summary.total_duration / 60.0, 2),
                "built_at": time.strftime("%Y-%m-%d %H:%M"),
            },
        )
        ok_train, train_report = validate_higgs_dataset(summary.train_dir)
        eval_report = ""
        if summary.eval_dir:
            _, eval_report = validate_higgs_dataset(summary.eval_dir)
        datasets = list_train_datasets()
        notify_done()
        message = (
            f"Higgs dataset ready: {summary.dataset_dir}\n"
            "Training format: single-speaker WAV/TXT pairs\n"
            f"Train: {summary.train_samples} samples\n"
            f"Eval: {summary.eval_samples} samples\n"
            f"Duration: {summary.total_duration / 3600:.3f} h, avg {summary.avg_duration:.2f}s\n"
            f"Languages: {', '.join(summary.languages)}\n"
            f"Speakers: {', '.join(summary.speakers)}\n\n"
            f"{train_report}"
        )
        if eval_report:
            message += f"\n\n{eval_report}"
        if not ok_train:
            message = "Dataset created but train validation found issues.\n" + message
        return (
            gr.update(choices=datasets, value=summary.train_dir.relative_to(ROOT).as_posix()),
            gr.update(choices=[""] + datasets, value=summary.eval_dir.relative_to(ROOT).as_posix() if summary.eval_dir else ""),
            message,
            f"### Dataset Build\nDone: {summary.train_samples} train / {summary.eval_samples} eval samples. "
            f"Project **{project}** is selected in Training with this dataset.",
            gr.update(value="Build Higgs Dataset", interactive=True, variant="primary"),
            gr.update(interactive=False),
            gr.update(choices=list_projects(), value=project),
            project_card(project),
            gr.update(choices=list_projects(), value=project),
        )
    except Exception as exc:
        import traceback

        message = f"Dataset error: {exc}\n\n{traceback.format_exc()}"
        print(message, flush=True)
        release_all_models("dataset build failed")
        return (
            gr.update(),
            gr.update(),
            message,
            "### Dataset Build\nFailed.",
            gr.update(value="Build Higgs Dataset", interactive=True, variant="primary"),
            gr.update(interactive=False),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def browse_source_folder(current_value=""):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=current_value or str(ROOT))
        root.destroy()
        if selected:
            return selected, f"Selected source folder: {selected}"
        return current_value, "Folder selection cancelled."
    except Exception as exc:
        return current_value, f"Folder picker error: {exc}"


def add_eval_sample_to_dataset(train_data_dir, eval_data_dir, eval_text, eval_audio, eval_ref_text):
    if not train_data_dir:
        return gr.update(), log("Eval sample error: select a train dataset first.")
    try:
        eval_dir, report = append_eval_sample(
            train_dataset_dir=train_data_dir,
            eval_dataset_dir=eval_data_dir or "",
            target_text=eval_text,
            target_audio_path=eval_audio,
            reference_transcript=eval_ref_text or "",
        )
        datasets = list_train_datasets()
        notify_done()
        message = f"Eval sample added: {eval_dir}\n\n{report}"
        return gr.update(choices=[""] + datasets, value=eval_dir.relative_to(ROOT).as_posix()), log(message)
    except Exception as exc:
        return gr.update(), log(f"Eval sample error: {exc}")


def start_higgs_training(
    train_model_select,
    train_data_dir,
    eval_data_dir,
    output_name,
    training_mode,
    epochs,
    max_steps,
    batch_size,
    grad_accum,
    learning_rate,
    lora_rank,
    use_lora,
    bf16,
    logging_steps,
    save_steps,
    eval_steps,
    resume_checkpoint,
    enable_eval_audio,
    eval_text,
    eval_audio_max_new_tokens,
    v3_lora_alpha=0,
    v3_lr_scheduler="cosine",
    v3_warmup_ratio=0.05,
    v3_weight_decay=0.01,
    v3_train_audio_modules=False,
    v3_audio_lr_scale=0.1,
    v3_grad_ckpt=False,
    v3_max_clip=30,
    v3_optimizer="adamw",
    v3_prodigy_dcoef=1.0,
    v3_patience=3,
    v3_min_delta=0.002,
    v3_restore_best=True,
    v3_max_batch_sec=40,
):
    global LAST_TRAINING_DONE_KEY
    LAST_TRAINING_DONE_KEY = None
    if not train_data_dir:
        message = "Training error: select a Higgs train dataset."
        return (
            log(message),
            "⚠️ Select a train dataset first.",
            message,
            restore_button("🚀 Start Training"),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    try:
        if TRAINING.is_running():
            raise RuntimeError(f"Training is already running: {TRAINING.run_name}")
        release_all_models("training")
        if not resume_checkpoint or resume_checkpoint == "None":
            # Training from scratch: wipe the previous run of this project so the
            # resume / LoRA dropdowns only ever show this run's checkpoints.
            default_project = "higgs_v3_lora" if train_model_select == "Higgs V3 TTS" else "higgs_v2_lora"
            reset_project_for_fresh_run(output_name or default_project, default_project)
        save_project_section(
            output_name or ("higgs_v3_lora" if train_model_select == "Higgs V3 TTS" else "higgs_v2_lora"),
            "training",
            {
                "model": train_model_select,
                "train_data_dir": train_data_dir,
                "eval_data_dir": eval_data_dir or "",
                "resume": resume_checkpoint or "",
                "optimizer": v3_optimizer or "adamw",
                "last_started": time.strftime("%Y-%m-%d %H:%M"),
            },
        )
        if train_model_select == "Higgs V3 TTS":
            print("[training] Ensuring Higgs V3 model and tokenizer are available...", flush=True)
            ensure_tts_model("Higgs V3 TTS")
            ensure_model(TOKENIZER_MODELS["Higgs V2 tokenizer"])
            project_args = {
                "train_data_dir": str(ROOT / train_data_dir if not os.path.isabs(train_data_dir) else train_data_dir),
                "eval_data_dir": str(ROOT / eval_data_dir if eval_data_dir and not os.path.isabs(eval_data_dir) else eval_data_dir or ""),
                "output_dir": str(ROOT / "exp" / slugify(output_name or "higgs_v3_lora", "higgs_v3_lora")),
                "num_train_epochs": int(epochs),
                "max_steps": int(max_steps),
                "batch_size": int(batch_size),
                "gradient_accumulation_steps": int(grad_accum),
                "learning_rate": float(learning_rate),
                "lora_rank": int(lora_rank),
                "use_lora": bool(use_lora),
                "bf16": bool(bf16),
                "logging_steps": int(logging_steps),
                "save_steps": int(save_steps),
                "eval_steps": int(eval_steps),
                "enable_eval_audio": bool(enable_eval_audio),
                "eval_text": eval_text or "",
                "eval_audio_max_new_tokens": int(eval_audio_max_new_tokens),
                "lora_alpha": int(v3_lora_alpha or 0) or 2 * int(lora_rank),
                "lr_scheduler": str(v3_lr_scheduler or "cosine"),
                "warmup_ratio": float(v3_warmup_ratio),
                "weight_decay": float(v3_weight_decay),
                "freeze_audio_head": not bool(v3_train_audio_modules),
                "audio_lr_scale": float(v3_audio_lr_scale),
                "gradient_checkpointing": bool(v3_grad_ckpt),
                "max_audio_seconds": float(v3_max_clip),
                "optimizer": str(v3_optimizer or "adamw"),
                "prodigy_d_coef": float(v3_prodigy_dcoef),
                "early_stop_patience": int(v3_patience),
                "early_stop_min_delta": float(v3_min_delta),
                "restore_best": bool(v3_restore_best),
                "max_batch_seconds": float(v3_max_batch_sec),
            }
            save_project_config(output_name or "higgs_v3_lora", "Higgs V3 TTS", project_args, resume_checkpoint or "")
            cmd, pretty = build_v3_training_command(
                train_data_dir=train_data_dir,
                eval_data_dir=eval_data_dir or "",
                output_name=output_name or "higgs_v3_lora",
                task_type="single_speaker_smart_voice",
                epochs=int(epochs),
                max_steps=int(max_steps),
                batch_size=int(batch_size),
                grad_accum=int(grad_accum),
                learning_rate=float(learning_rate),
                lora_rank=int(lora_rank),
                use_lora=bool(use_lora),
                bf16=bool(bf16),
                logging_steps=int(logging_steps),
                save_steps=int(save_steps),
                eval_steps=int(eval_steps),
                freeze_audio_head=not bool(v3_train_audio_modules),
                resume_checkpoint=resume_checkpoint or "",
                enable_eval_audio=bool(enable_eval_audio),
                eval_text=eval_text or "",
                eval_audio_max_new_tokens=int(eval_audio_max_new_tokens),
                lora_alpha=int(v3_lora_alpha or 0),
                lr_scheduler=str(v3_lr_scheduler or "cosine"),
                warmup_ratio=float(v3_warmup_ratio),
                weight_decay=float(v3_weight_decay),
                audio_lr_scale=float(v3_audio_lr_scale),
                gradient_checkpointing=bool(v3_grad_ckpt),
                max_audio_seconds=float(v3_max_clip),
                optimizer=str(v3_optimizer or "adamw"),
                prodigy_d_coef=float(v3_prodigy_dcoef or 1.0),
                early_stop_patience=int(v3_patience or 0),
                early_stop_min_delta=float(v3_min_delta or 0.0),
                restore_best=bool(v3_restore_best),
                max_batch_seconds=float(v3_max_batch_sec or 40),
            )
            status, tail = TRAINING.start(cmd, output_name or "higgs_v3_lora")
            message = (
                f"Custom Higgs V3 training started for {training_mode}.\n\n"
                f"{pretty}\n\n"
                "This uses the local teacher-forced audio-code CE trainer over delayed V3 audio codebooks."
            )
            return (
                log(message),
                status,
                tail,
                gr.update(value="Training running... check logs", interactive=False, variant="secondary"),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )
        print("[training] Ensuring Higgs V2 model and tokenizer are available...", flush=True)
        ensure_tts_model("Higgs V2 TTS")
        ensure_model(TOKENIZER_MODELS["Higgs V2 tokenizer"])
        project_args = {
            "train_data_dir": str(ROOT / train_data_dir if not os.path.isabs(train_data_dir) else train_data_dir),
            "eval_data_dir": str(ROOT / eval_data_dir if eval_data_dir and not os.path.isabs(eval_data_dir) else eval_data_dir or ""),
            "output_dir": str(ROOT / "exp" / slugify(output_name or "higgs_v2_lora", "higgs_v2_lora")),
            "num_train_epochs": int(epochs),
            "max_steps": int(max_steps),
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(grad_accum),
            "learning_rate": float(learning_rate),
            "lora_rank": int(lora_rank),
            "use_lora": bool(use_lora),
            "bf16": bool(bf16),
            "logging_steps": int(logging_steps),
            "save_steps": int(save_steps),
            "eval_steps": int(eval_steps),
            "enable_eval_audio": bool(enable_eval_audio),
            "eval_text": eval_text or "",
            "eval_audio_max_new_tokens": int(eval_audio_max_new_tokens),
        }
        save_project_config(output_name or "higgs_v2_lora", "Higgs V2 TTS", project_args, resume_checkpoint or "")
        cmd, pretty = build_training_command(
            train_data_dir=train_data_dir,
            eval_data_dir=eval_data_dir or "",
            output_name=output_name or "higgs_v2_lora",
            epochs=int(epochs),
            max_steps=int(max_steps),
            batch_size=int(batch_size),
            grad_accum=int(grad_accum),
            learning_rate=float(learning_rate),
            lora_rank=int(lora_rank),
            use_lora=bool(use_lora),
            bf16=bool(bf16),
            logging_steps=int(logging_steps),
            save_steps=int(save_steps),
            eval_steps=int(eval_steps),
            resume_checkpoint=resume_checkpoint or "",
            enable_eval_audio=bool(enable_eval_audio),
            eval_text=eval_text or "",
            eval_audio_max_new_tokens=int(eval_audio_max_new_tokens),
        )
        status, tail = TRAINING.start(cmd, output_name or "higgs_v2_lora")
        message = (
            f"Training started for {training_mode}.\n\n"
            f"{pretty}\n\n"
            "The live trainer output is mirrored to CMD and saved under logs/training."
        )
        return (
            log(message),
            status,
            tail,
            gr.update(value="Training running... check logs", interactive=False, variant="secondary"),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    except Exception as exc:
        message = f"Training command error: {exc}"
        return (
            log(message),
            "⚠️ Training could not start: see the console.",
            message,
            restore_button("🚀 Start Training"),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


def stop_higgs_training():
    status, tail = TRAINING.stop()
    notify_done()
    return (
        log("Training stop requested."),
        status,
        tail,
        restore_button("🚀 Start Training"),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


TRAINING_IDLE_HTML = (
    '<div class="pb pb-idle"><div class="pb-head"><span class="pb-title">No training running</span><span class="pb-pct"></span></div>'
    '<div class="pb-track"><div class="pb-fill" style="width:0%"></div></div>'
    '<div class="pb-detail">Steps, loss, best eval and ETA appear here while training.</div></div>'
)


def training_progress_html() -> str:
    """Training progress bar (same look as the inference bar)."""
    info = TRAINING.progress_info()
    state = info.get("state")
    if state == "idle":
        return TRAINING_IDLE_HTML
    run = info.get("run") or "training"
    step, total = info.get("step"), info.get("total")
    parts = []
    if step is not None and total:
        parts.append(f"step {step}/{total}")
    if info.get("loss") is not None:
        parts.append(f"train loss {info['loss']:.3f}")
    if info.get("last_eval") is not None:
        eval_part = f"eval {info['last_eval']:.3f} @ {info['last_eval_step']}"
        if info.get("best_step") != info.get("last_eval_step"):
            eval_part += f" (best {info['best_eval']:.3f} @ {info['best_step']}, saved in checkpoint-best)"
        parts.append(eval_part)
    if info.get("lr") is not None:
        parts.append(f"lr {info['lr']:.1e}")
    if step and info.get("train_elapsed"):
        parts.append(f"{info['train_elapsed'] / max(step, 1):.1f} s/step")
    parts.append(f"elapsed {_fmt_clock(info.get('elapsed', 0))}")
    if info.get("overfitting_since") is not None:
        parts.append(f"⚠️ overfitting: eval rising since step {info['overfitting_since']}, more steps won't help")
    if state == "running":
        if info.get("eta") is not None:
            parts.append(f"ETA {_fmt_clock(info['eta'])}")
        if step is None or not total:
            return progress_bar_html(None, f"Training {run} · loading model and caching audio codes", " · ".join(parts))
        return progress_bar_html(min(100.0 * step / total, 99.5), f"Training {run}", " · ".join(parts))
    if state == "done":
        label = "Stopped and saved" if info.get("stopped") else "Training finished"
        pct = 100.0 if not info.get("stopped") or not total else 100.0 * (step or 0) / total
        return progress_bar_html(pct, f"{label} · {run}", " · ".join(parts), kind="done")
    detail = info.get("error_line") or f"Process exited with code {info.get('code')}"
    return progress_bar_html(100.0 * (step or 0) / total if total else 0, f"Training failed · {run}", detail, kind="error")


_AUDIO_PANEL_CACHE: dict[str, str] = {}


def training_monitor_updates(project_name, force: bool = False):
    """(charts html, audio html update). The audio panel is only re-sent when a
    new preview appears, so a playing <audio> is never reset by the timer."""
    key = safe_project_name(project_name) or ""
    charts = monitor_html(key, running=TRAINING.is_running() and TRAINING.run_name == key)
    audio = audio_evolution_html(key)
    if force or _AUDIO_PANEL_CACHE.get(key) != audio:
        _AUDIO_PANEL_CACHE[key] = audio
        return charts, audio
    return charts, gr.update()


def refresh_training_status(project_name=None):
    global LAST_TRAINING_DONE_KEY
    status, tail = TRAINING.refresh()
    running = TRAINING.is_running()
    finished_process = TRAINING.process is not None and not running
    resume_choice_update, resume_state_update = gr.update(), gr.update()
    if finished_process:
        code = TRAINING.process.poll()
        done_key = (TRAINING.run_name, int(code) if code is not None else 0)
        if done_key != LAST_TRAINING_DONE_KEY:
            LAST_TRAINING_DONE_KEY = done_key
            notify_done()
            # The run just ended: show its checkpoints right away for resuming.
            latest = best_resume_checkpoint_for_project(project_name)
            resume_choice_update = gr.update(choices=list_project_resume_choices(project_name), value=latest or "None")
            resume_state_update = latest
    return (
        status,
        tail,
        restore_button("🚀 Start Training") if finished_process else gr.update(),
        gr.update(interactive=running),
        gr.update(interactive=running or finished_process),
        gr.update(interactive=finished_process or running),
        training_progress_html(),
        project_card(project_name),
        *training_monitor_updates(project_name),
        resume_choice_update,
        resume_state_update,
    )


def launch_tensorboard(output_name):
    try:
        status, tail = TRAINING.launch_tensorboard(slugify(output_name or "higgs_v2_lora"))
        return log(status), status, tail
    except Exception as exc:
        message = f"TensorBoard error: {exc}"
        return log(message), "⚠️ TensorBoard launch failed.", message


def insert_tag(text, tag):
    text = text or ""
    tag = tag or ""
    return f"{text}{tag}" if not text or text.endswith((" ", "\n")) else f"{text} {tag}"


TAG_CATALOG = {
    "emotion": [
        "affection", "amusement", "anger", "arousal", "awe", "bitterness", "confusion",
        "contemplation", "contentment", "determination", "disgust", "elation", "enthusiasm",
        "fear", "helplessness", "longing", "pride", "relief", "sadness", "shame", "surprise",
    ],
    "prosody": [
        "speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast", "pitch_low",
        "pitch_high", "expressive_high", "expressive_low", "pause", "long_pause",
    ],
    "env": ["music", "noise"],
    "style": ["singing", "shouting", "whispering"],
    "sfx": ["cough", "laughter", "crying", "screaming", "burping", "humming", "sigh", "sniff", "sneeze"],
}


def tag_reference_text() -> str:
    return """## Higgs V3 Tags

V3 tags use `<|category:value|>` syntax and can be inserted in the target text.

Only the 43 tags listed here are recognized. Anything else can degrade output or be read literally.

Placement:

- Sentence-level: emotion, style, and prosody `speed_*`, `pitch_*`, `expressive_*`. Put these at the start of the sentence.
- Inline: sound effects and prosody `pause` / `long_pause`. Put these exactly where the effect should happen.
- SFX gotcha: use `<|sfx:tag|>onomatopoeia` with no space between the tag and the sound word.

### Global Copy/Paste Example

```text
<|emotion:enthusiasm|><|prosody:speed_fast|>Welcome back. <|prosody:pause|><|style:whispering|>Here is a secret. <|sfx:laughter|>Haha, that was unexpected.
```

### Emotion Variants

```text
<|emotion:elation|>
<|emotion:amusement|>
<|emotion:enthusiasm|>
<|emotion:determination|>
<|emotion:pride|>
<|emotion:contentment|>
<|emotion:affection|>
<|emotion:relief|>
<|emotion:contemplation|>
<|emotion:confusion|>
<|emotion:surprise|>
<|emotion:awe|>
<|emotion:longing|>
<|emotion:arousal|>
<|emotion:anger|>
<|emotion:fear|>
<|emotion:disgust|>
<|emotion:bitterness|>
<|emotion:sadness|>
<|emotion:shame|>
<|emotion:helplessness|>
```

### Style Variants

```text
<|style:singing|>
<|style:shouting|>
<|style:whispering|>
```

### Sound Effect Variants

Pair each token with matching onomatopoeia immediately after it.

```text
<|sfx:cough|>Ahem
<|sfx:laughter|>Haha
<|sfx:laughter|>Hehe
<|sfx:crying|>Boohoo
<|sfx:crying|>Sob
<|sfx:screaming|>Ahh
<|sfx:screaming|>Aaah
<|sfx:burping|>Burp
<|sfx:humming|>Hmm
<|sfx:humming|>Mmm
<|sfx:sigh|>Uh
<|sfx:sigh|>Ahh
<|sfx:sniff|>Sff
<|sfx:sneeze|>Achoo
```

### Prosody Variants

```text
<|prosody:speed_very_slow|>
<|prosody:speed_slow|>
<|prosody:speed_fast|>
<|prosody:speed_very_fast|>
<|prosody:pitch_low|>
<|prosody:pitch_high|>
<|prosody:pause|>
<|prosody:long_pause|>
<|prosody:expressive_high|>
<|prosody:expressive_low|>
```

### Stacking Examples

Use sentence-level tags before the sentence. Stack emotion, style, and prosody with no separator.
Put inline tags such as `pause`, `long_pause`, and `sfx` exactly where they should happen.

```text
<|emotion:contentment|><|prosody:expressive_low|>This is a calm and grounded sentence.

<|emotion:enthusiasm|><|prosody:speed_fast|>This line should feel energetic, quick, and upbeat.

<|style:shouting|><|emotion:determination|>We are not giving up today.

<|style:whispering|><|prosody:pitch_low|>Keep your voice low. This part is confidential.

<|emotion:amusement|>That was unexpected. <|sfx:laughter|>Haha, I did not see that coming.

<|emotion:contemplation|><|prosody:speed_slow|>Let me think about that for a moment. <|prosody:long_pause|>Now I understand.

<|emotion:surprise|><|prosody:pitch_high|>Wait, did that really just happen?
```

## Higgs V2 Event Tags

V2's official generation script expects easy markers like `[laugh]` and converts them internally to `<SE>...</SE>`. Use the bracket syntax in text fields.

### Instant Events

```text
[laugh]
[music]
[applause]
[cheering]
[cough]
```

### Start / End Events

```text
[humming start]
[humming end]
[music start]
[music end]
[sing start]
[sing end]
```

### Global Copy/Paste Example

```text
Welcome everyone [applause]. [music start] The intro fades in [music end]. I tried to stay serious [laugh], but then I had to [cough] and keep going.
```
"""


def language_reference_text() -> str:
    return """## Higgs V3 Languages

Higgs V3 does not expose an explicit `language` argument in `generate_speech`. Language is inferred from target text and, for cloning, helped by the reference audio/transcript.

The model documentation reports single-digit WER/CER on 102 languages, split into two tiers.

### WER/CER under 5 - polished, production-quality

Afrikaans · Arabic · Armenian · Assamese · Asturian · Azerbaijani · Bashkir · Basque · Belarusian · Bengali · Bosnian · Bulgarian · Catalan · Cebuano · Central Kurdish · Chinese · Croatian · Czech · Danish · Dutch · Eastern Mari · English · Esperanto · Estonian · Finnish · French · Galician · Georgian · German · Greek · Gujarati · Haitian Creole · Hausa · Hebrew · Hindi · Hungarian · Indonesian · Italian · Japanese · Javanese · Kannada · Kazakh · Korean · Kinyarwanda · Kyrgyz · Latvian · Lingala · Lithuanian · Luo · Macedonian · Malay · Malayalam · Maltese · Maori · Marathi · Mongolian · Nepali · Norwegian · Occitan · Persian · Polish · Portuguese · Romanian · Russian · Sepedi · Serbian · Shona · Slovak · Slovene · Spanish · Swahili · Swedish · Tagalog · Tajik · Tamil · Telugu · Thai · Turkish · Ukrainian · Urdu · Uyghur · Uzbek · Vietnamese · Xhosa · Zulu

### WER/CER between 5 and 10 - usable, less polished

Albanian · Chichewa/Nyanja · Eastern Punjabi · Ganda · Icelandic · Irish · Kabyle · Kabuverdianu · Kamba · Latin · Luxembourgish · Oromo · Pashto · Sindhi · Somali · Umbundu · Welsh

## Higgs V2 Languages

V2 documentation lists these languages:

```text
English
Mandarin
Korean
German
Spanish
```
"""


def chunk_mode_reference_text() -> str:
    return """### Chunk Mode

- **None**: sends the full text as one generation request. V3 can stop early when the model emits its internal end signal, often around 30 seconds even with a high frame limit.
- **Paragraph/Sentence Auto**: keeps short paragraphs as one generation; long paragraphs are split by sentence boundaries.
- **Periods**: splits at each period.
- **Paragraphs**: splits on paragraph breaks (double enter) and joins the generated audio.
- **Lines**: splits on line breaks (single enter) and joins the generated audio.
- **Speaker turns**: keeps `[SPEAKER0]`, `[SPEAKER1]` style turns separated when possible.

Changing chunk mode changes how many clips are generated and joined.
"""


def build_ui():
    ensure_local_dirs()
    normalize_voice_library()
    sample_choices = list_voice_names()
    default_sample_name = next((name for name in sample_choices if name != "None"), "None")
    default_audio, default_text = load_sample_for_ui(default_sample_name)
    empty_sample_name = "None"
    inference_default_sample = "None"
    saved_v3_compile = v3_compile_default()
    MANAGER.set_v3_runtime("auto", "sdpa", saved_v3_compile)

    with gr.Blocks(title=APP_TITLE, fill_width=True) as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.HTML(header_html("Higgs Audio Easy GUI"))
        with gr.Row(elem_classes="header-actions"):
            top_unload_btn = gr.Button("⏏ Unload models", size="sm", variant="secondary", scale=0)
            delete_outputs_btn = gr.Button("🗑 Delete output audios", size="sm", variant="secondary", scale=0, elem_classes=["red-btn"])
            delete_samples_btn = gr.Button("🗑 Delete reference samples", size="sm", variant="secondary", scale=0, elem_classes=["red-btn"])
            gr.Button("📖 Higgs Audio docs", size="sm", variant="secondary", scale=0, link="https://github.com/boson-ai/higgs-audio")

        with gr.Tabs(elem_classes="main-tabs", selected="tab_studio"):
            # ================================================================ STUDIO
            with gr.Tab("🎙️ Inference", id="tab_studio"):
                with gr.Tabs():
                    with gr.Tab("Single voice"):
                        with gr.Row(equal_height=False, elem_classes="studio-grid"):
                            # ---------------- left: script + output
                            with gr.Column(scale=7, min_width=420, elem_classes="studio-main"):
                                with gr.Column(elem_classes="card"):
                                    gr.HTML(section("Script"))
                                    infer_text = gr.Textbox(
                                        show_label=False,
                                        placeholder=(
                                            "Type or paste what the voice should say. Long texts are fine: they are split "
                                            "into natural segments and joined into one continuous take.\n\n"
                                            "Tip: Higgs V3 control tokens like <|emotion:elation|> go inline."
                                        ),
                                        lines=10,
                                        max_lines=30,
                                        value="",
                                        elem_classes=["script-box"],
                                    )
                                    infer_clips_count = gr.Markdown(value="", elem_classes="text-stats")
                                    with gr.Row(equal_height=True):
                                        infer_gen_btn = gr.Button(
                                            "⚡ Generate Speech",
                                            variant="primary",
                                            size="lg",
                                            scale=5,
                                            elem_classes="button-primary",
                                        )
                                        infer_stop_btn = gr.Button(
                                            "⏹ Stop",
                                            variant="stop",
                                            size="lg",
                                            scale=1,
                                            min_width=96,
                                            elem_classes="button-stop",
                                            interactive=False,
                                        )
                                with gr.Column(elem_classes=["card", "output-card"]):
                                    gr.HTML(section("Output"))
                                    infer_progress = gr.HTML(IDLE_PROGRESS_HTML, elem_classes=["progress-card"])
                                    infer_audio_out = gr.Audio(
                                        label="Result",
                                        type="filepath",
                                        visible=True,
                                        elem_id="infer-generated-audio",
                                        elem_classes=["audio-safe-space", "output-clean"],
                                    )
                                    with gr.Row(equal_height=True):
                                        infer_file_out = gr.Textbox(
                                            label="Saved WAV",
                                            interactive=False,
                                            scale=5,
                                            elem_classes=["output-path"],
                                        )
                                        infer_save_outputs = gr.Checkbox(
                                            label="Save to outputs",
                                            value=True,
                                            scale=1,
                                            min_width=140,
                                        )
                                with gr.Accordion("✨ Control tokens & languages", open=False):
                                    with gr.Row():
                                        with gr.Column():
                                            gr.Markdown(tag_reference_text())
                                        with gr.Column():
                                            gr.Markdown(language_reference_text())

                            # ---------------- voice / settings (grid places them)
                            with gr.Column(min_width=320, elem_classes=["card", "studio-voice"]):
                                gr.HTML(section("Voice"))
                                infer_model_select = gr.Dropdown(
                                    label="Model",
                                    choices=MODEL_CHOICES,
                                    value="Higgs V3 TTS",
                                )
                                with gr.Row(equal_height=True):
                                    lora_project = gr.Dropdown(
                                        choices=lora_project_choices("Higgs V3 TTS"),
                                        value=NONE_ADAPTER,
                                        label="LoRA voice",
                                        scale=4,
                                    )
                                    lora_adapter = gr.Dropdown(
                                        choices=[],
                                        value=None,
                                        label="Checkpoint",
                                        interactive=False,
                                        scale=4,
                                    )
                                    refresh_lora_adapter_btn = gr.Button("🔄", scale=1, min_width=48, variant="secondary")
                                gr.Markdown(
                                    "Pick a trained voice, then a checkpoint. *Final* is the exported adapter; "
                                    "steps are snapshots saved during training. Switching is instant on the fast engine.",
                                    elem_classes="hint",
                                )
                                with gr.Row(equal_height=True):
                                    infer_sample_select = gr.Dropdown(
                                        choices=list_voice_names(),
                                        value=inference_default_sample,
                                        label="Reference voice (library)",
                                        scale=6,
                                    )
                                    refresh_infer_sample_btn = gr.Button("🔄", scale=1, min_width=48, variant="secondary")
                                infer_ref_audio = gr.Audio(
                                    label="Reference audio · 5–30 s",
                                    type="filepath",
                                    value=None,
                                    sources=["upload", "microphone"],
                                    elem_classes=["audio-safe-space"],
                                )
                                infer_ref_text = gr.Textbox(
                                    label="Reference transcript",
                                    placeholder="Loaded from the sample library, editable. Transcribe only when needed.",
                                    lines=2,
                                    value="",
                                    interactive=True,
                                )
                                with gr.Row(equal_height=True):
                                    infer_use_ref_text = gr.Checkbox(
                                        label="Use transcript",
                                        value=True,
                                        scale=3,
                                    )
                                    infer_transcribe_ref_btn = gr.Button("🎙️ Transcribe", scale=2, min_width=110, variant="secondary")
                                    stop_infer_transcribe_btn = gr.Button(
                                        "⏹",
                                        variant="stop",
                                        scale=1,
                                        min_width=48,
                                        elem_classes="button-stop",
                                        interactive=False,
                                    )
                                gr.Markdown(
                                    "Turning the transcript off can sound more expressive when the reference already carries strong emotion.",
                                    elem_classes="hint",
                                )

                            with gr.Column(min_width=320, elem_classes=["card", "studio-settings"]):
                                gr.HTML(section("Settings"))
                                with gr.Accordion("📏 Long text & seed", open=True):
                                    infer_chunk_mode = gr.Dropdown(
                                        CHUNK_CHOICES,
                                        value=SEAMLESS_MODE,
                                        label="Long text mode",
                                        info="Seamless splits at sentences (~300 chars, the size Higgs V3 reads reliably) and joins them into one continuous take.",
                                    )
                                    with gr.Row(equal_height=True):
                                        infer_seed = gr.Number(
                                            value=-1,
                                            precision=0,
                                            label="Seed (-1 = random)",
                                            scale=2,
                                        )
                                        use_last_seed_btn = gr.Button("Use last", variant="secondary", scale=1, min_width=90)
                                        restore_random_seed_btn = gr.Button("Random", variant="secondary", scale=1, min_width=90)
                                    infer_reuse_chunk_seed = gr.Checkbox(
                                        label="Reuse one seed across segments",
                                        value=True,
                                        info="Keeps the voice steadier when the text is split.",
                                    )
                                    with gr.Accordion("How the split modes work", open=False):
                                        gr.Markdown(chunk_mode_reference_text())
                                with gr.Accordion("🎛️ Sampling", open=False):
                                    with gr.Row():
                                        infer_temperature = gr.Slider(
                                            0.3,
                                            2.0,
                                            value=1.0,
                                            step=0.05,
                                            label="Temperature",
                                            info="Lower = steadier, higher = more expressive (and more artifacts).",
                                        )
                                        infer_top_p = gr.Slider(
                                            0.1,
                                            1.0,
                                            value=0.95,
                                            step=0.01,
                                            label="Top-p",
                                            info="Nucleus cutoff. Lower constrains style and pronunciation.",
                                        )
                                    with gr.Row():
                                        infer_top_k = gr.Slider(
                                            0,
                                            200,
                                            value=50,
                                            step=1,
                                            label="Top-k",
                                            info="0 disables. 30–80 is a practical range.",
                                        )
                                        infer_max_new = gr.Slider(
                                            128,
                                            8192,
                                            value=4096,
                                            step=128,
                                            label="Max frames per pass",
                                            info="Safety cap per segment (25 frames = 1 s).",
                                        )
                                    with gr.Row():
                                        infer_ras_win_len = gr.Slider(
                                            0,
                                            32,
                                            value=7,
                                            step=1,
                                            label="V2 RAS window",
                                            info="V2 only. 0 disables repetition-aware sampling.",
                                        )
                                        infer_ras_max_repeat = gr.Slider(
                                            1,
                                            8,
                                            value=2,
                                            step=1,
                                            label="V2 RAS max repeats",
                                            info="V2 only. 1 = strongest anti-loop.",
                                        )
                                with gr.Accordion("⚗️ Runtime", open=False):
                                    v3_precision = gr.Dropdown(
                                        [
                                            ("Auto (bf16 · fast engine)", "auto"),
                                            ("bf16 · 12+ GB VRAM · fast engine", "bf16"),
                                            ("8-bit · 8+ GB VRAM · compatibility engine", "8bit"),
                                            ("4-bit · 6+ GB VRAM · compatibility engine", "4bit"),
                                        ],
                                        value="auto",
                                        label="Higgs V3 precision",
                                        info="bf16 runs on the fast CUDA-graph engine. 8/4-bit use the slower compatibility loop.",
                                    )
                                    v3_attention_backend = gr.Dropdown(
                                        [
                                            ("SDPA - stable PyTorch attention", "sdpa"),
                                            ("Eager - compatibility fallback", "eager"),
                                        ],
                                        value="sdpa",
                                        label="Attention backend (compatibility engine)",
                                    )
                                    v3_torch_compile = gr.Checkbox(
                                        label="torch.compile (compatibility engine only)",
                                        value=saved_v3_compile,
                                        info="Not needed on the fast engine. First run can take minutes to compile.",
                                    )
                                with gr.Accordion("🛰️ Transcription (ASR)", open=False):
                                    infer_asr_model = gr.Dropdown(
                                        label="ASR model",
                                        choices=ASR_CHOICES,
                                        value=DEFAULT_ASR_MODEL,
                                    )
                                    infer_asr_lang = gr.Dropdown(
                                        label="ASR language",
                                        choices=list(WHISPER_LANGS),
                                        value="Auto-detect",
                                    )

                    with gr.Tab("Dialogue"):
                        with gr.Column(elem_classes="card"):
                            gr.HTML(section("Multi-speaker dialogue"))
                            speaker_inputs = []
                            dialogue_rows = []
                            dialogue_add_btns = []
                            dialogue_copy_btns = []
                            dialogue_delete_btns = []
                            dialogue_row_count = gr.State(2)
                            with gr.Row():
                                dialogue_reset_btn = gr.Button("Reset rows", size="sm", variant="secondary")
                                dialogue_clear_btn = gr.Button("Clear rows", size="sm", variant="secondary")
                                dialogue_compact_btn = gr.Button("Remove empty rows", size="sm", variant="secondary")
                            for i in range(DIALOGUE_MAX_SEGMENTS):
                                with gr.Row(visible=i < 2, equal_height=True) as row:
                                    with gr.Column(scale=3, min_width=180):
                                        s = gr.Dropdown(
                                            choices=list_voice_names(),
                                            label=f"Speaker {i + 1}",
                                            value=default_sample_name if i < 2 else "None",
                                        )
                                    t = gr.Textbox(
                                        placeholder=f"What speaker {i + 1} says...",
                                        label=f"Line {i + 1}",
                                        scale=7,
                                        lines=3,
                                    )
                                    with gr.Column(scale=1, min_width=64):
                                        add_btn = gr.Button("＋", variant="secondary", size="sm", elem_classes=["green-btn"])
                                        copy_btn = gr.Button("⧉", variant="secondary", size="sm")
                                        delete_btn = gr.Button("✕", variant="secondary", size="sm", elem_classes=["red-btn"])
                                    a = gr.Audio(label=f"Reference {i + 1}", type="filepath", visible=False, elem_classes=["audio-safe-space"])
                                    rt = gr.Textbox(label=f"Reference Text {i + 1}", visible=False)
                                    speaker_inputs.extend([t, s, a, rt])
                                    dialogue_rows.append(row)
                                    dialogue_add_btns.append(add_btn)
                                    dialogue_copy_btns.append(copy_btn)
                                    dialogue_delete_btns.append(delete_btn)
                            dialogue_silence_slider = gr.Slider(0, 5, value=0.5, step=0.1, label="Silence between speakers (s)")
                            with gr.Row(equal_height=True):
                                dialogue_gen_btn = gr.Button(
                                    "⚡ Generate Dialogue",
                                    variant="primary",
                                    size="lg",
                                    scale=5,
                                    elem_classes="button-primary",
                                )
                                dialogue_stop_btn = gr.Button(
                                    "⏹ Stop",
                                    variant="stop",
                                    size="lg",
                                    scale=1,
                                    min_width=96,
                                    elem_classes="button-stop",
                                    interactive=False,
                                )
                        with gr.Column(elem_classes=["card", "output-card"]):
                            gr.HTML(section("Output"))
                            dialogue_audio_out = gr.Audio(
                                label="Generated dialogue",
                                type="filepath",
                                elem_id="dialogue-generated-audio",
                                elem_classes=["audio-safe-space", "output-clean"],
                            )
                            dialogue_file_out = gr.Textbox(label="Saved WAV", interactive=False, elem_classes=["output-path"])
                        gr.Markdown(
                            "Uses the model, LoRA and sampling settings from **Single voice**.",
                            elem_classes="hint",
                        )

                infer_console = gr.Textbox(label="Internal Status", lines=1, interactive=False, value="Idle.", visible=False)

            # ========================================================= VOICE LIBRARY
            with gr.Tab("📚 Voice Library", id="tab_prep_samples"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=280, elem_classes="card"):
                        gr.HTML(section("Your samples"))
                        sample_dropdown = gr.Dropdown(
                            choices=sample_choices,
                            value=empty_sample_name,
                            label="Select sample",
                            interactive=True,
                        )
                        refresh_samples_btn = gr.Button("🔄 Refresh list", size="sm", variant="secondary")
                        gr.Markdown(
                            "Upload or record a clean 5–30 s clip, transcribe it, then save it. "
                            "Saved samples appear as reference voices in the Inference tab.",
                            elem_classes="hint",
                        )

                    with gr.Column(scale=2, elem_classes="card"):
                        gr.HTML(section("Editor"))
                        prep_audio_player = gr.Audio(
                            label="Audio · 5–30 s recommended (use the trim icon to edit)",
                            type="filepath",
                            interactive=True,
                            value=None,
                            sources=["upload", "microphone"],
                            elem_classes=["audio-safe-space"],
                        )
                        prep_transcription = gr.Textbox(
                            label="Transcript",
                            placeholder="Transcription will appear here, or type it manually...",
                            lines=4,
                            interactive=True,
                            value="",
                        )
                        with gr.Row(equal_height=True):
                            prep_asr_model = gr.Dropdown(
                                label="ASR model",
                                choices=ASR_CHOICES,
                                value=DEFAULT_ASR_MODEL,
                                scale=3,
                            )
                            prep_asr_lang = gr.Dropdown(
                                label="Language",
                                choices=list(WHISPER_LANGS),
                                value="Auto-detect",
                                scale=2,
                            )
                            transcribe_prep_btn = gr.Button("🔍 Transcribe", variant="secondary", scale=1, min_width=120)
                            stop_prep_transcribe_btn = gr.Button(
                                "⏹",
                                variant="stop",
                                scale=1,
                                min_width=48,
                                elem_classes="button-stop",
                                interactive=False,
                            )
                        with gr.Row(equal_height=True):
                            save_sample_name = gr.Textbox(
                                label="Sample ID",
                                placeholder="e.g. news_anchor_1",
                                scale=4,
                                value="",
                            )
                            save_sample_btn = gr.Button("💾 Save sample", variant="primary", scale=1, min_width=140)
                        prep_sample_console = gr.Textbox(label="Console", lines=1, interactive=False, value="Idle.", visible=False)

            with gr.Tab("📂 Dataset Preparation", id="tab_dataset"):
                gr.HTML(UI_WARNING_HTML)
                gr.Markdown("### 🛠️ Higgs Dataset Builder")
                gr.Markdown(
                    "Point it at a folder of recordings of one speaker (any length, any format). The builder isolates the voice, "
                    "cuts natural speech segments, normalises loudness and transcribes them verbatim. Files that already have a "
                    "sidecar `.txt` / `.json` transcript are cleaned but kept whole.",
                    elem_classes="hint",
                )
                with gr.Row(equal_height=True):
                    dataset_name_input = gr.Dropdown(
                        choices=list_projects(),
                        value=None,
                        allow_custom_value=True,
                        label="Project",
                        info="Pick a project or type a new name. Dataset Preparation and Training share the same project.",
                        scale=6,
                    )
                    dataset_open_btn = gr.Button("📂 Open folder", scale=0, min_width=130, variant="secondary")
                dataset_project_card = gr.HTML(project_card_html(None))
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="form-section"):
                        gr.Markdown("#### 📁 Folder Selection")
                        with gr.Row():
                            src_folder = gr.Textbox(label="Source Audio Folder", placeholder="J:\\path\\to\\audio_folder", scale=5)
                            browse_src_folder_btn = gr.Button("Search Folder", scale=1, min_width=130)
                        recursive_dataset = gr.Checkbox(label="Scan subfolders", value=True)
                    with gr.Column(scale=1, elem_classes="form-section"):
                        gr.Markdown("#### 🛰️ ASR & Split")
                        dataset_asr_model = gr.Dropdown(ASR_CHOICES, value=DEFAULT_ASR_MODEL, label="ASR Model")
                        dataset_asr_lang = gr.Dropdown(list(WHISPER_LANGS), value="Auto-detect", label="Language")
                        dataset_asr_batch = gr.Slider(1, 32, value=8, step=1, label="ASR batch size", visible=False)
                        val_split_slider = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Validation Split")
                    with gr.Column(scale=1, elem_classes="form-section"):
                        gr.Markdown("#### 🎧 Clean-up & Segmentation")
                        dataset_isolate = gr.Checkbox(
                            label="Isolate voice (Inworld Voice Isolator)",
                            value=True,
                            info="Removes background noise and music, keeps the foreground voice.",
                        )
                        dataset_segment = gr.Checkbox(
                            label="Split into natural segments (Inworld SAVA)",
                            value=True,
                            info="Voice-activity + end-of-turn model; long turns are cut at the deepest pause.",
                        )
                        with gr.Row():
                            dataset_min_sec = gr.Slider(1.0, 8.0, value=2.0, step=0.5, label="Min segment (s)")
                            dataset_max_sec = gr.Slider(6.0, 25.0, value=15.0, step=0.5, label="Max segment (s)")
                        gr.Markdown(
                            "3–15 s clips train Higgs V3 best: it reads up to ~25 s reliably per pass.",
                            elem_classes="hint",
                        )
                        with gr.Row():
                            prep_btn = gr.Button("Build Higgs Dataset", variant="primary", elem_classes="button-primary")
                            stop_prep_btn = gr.Button(
                                "⏹️ Stop",
                                variant="stop",
                                elem_classes="button-stop",
                                interactive=False,
                            )
                dataset_status_box = gr.Markdown("### Dataset Build\nIdle.")
                dataset_log_box = gr.Textbox(label="Dataset Preparation Log", lines=1, interactive=False, visible=False)

            with gr.Tab("🚀 Training", id="tab_train"):
                gr.HTML(UI_WARNING_HTML)
                training_project_card = gr.HTML(project_card_html(None))
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="form-section"):
                        gr.Markdown("#### 📁 Model & Dataset Selection")
                        with gr.Row():
                            output_name = gr.Dropdown(
                                choices=list_training_projects(),
                                value=None,
                                label="Select Project",
                                allow_custom_value=True,
                                scale=4,
                            )
                            refresh_project_btn = gr.Button("🔄", scale=1, min_width=50)
                            delete_project_btn = gr.Button("🗑️", variant="stop", scale=1, min_width=50)
                            train_open_btn = gr.Button("📂", scale=1, min_width=50)
                        train_model_select = gr.Dropdown(TRAINING_MODEL_CHOICES, value="Higgs V3 TTS", label="Base Model")
                        initial_datasets = list_train_datasets()
                        train_manifest = gr.Dropdown(choices=initial_datasets, label="Train Dataset Directory")
                        val_manifest = gr.Dropdown(choices=[""] + initial_datasets, label="Validation Dataset Directory")
                        with gr.Row():
                            refresh_train_btn = gr.Button("🔄 Refresh Datasets", size="sm")
                            auto_tune_btn = gr.Button("🧠 Analyze & Auto-tune", size="sm")
                        train_vram_preset = gr.Radio(
                            VRAM_PRESETS,
                            value="24 GB VRAM",
                            label="Auto-tune preset",
                        )
                        training_autotune_report = gr.Markdown(
                            "Select a train dataset, choose a VRAM preset, then click **Analyze & Auto-tune**.\n\n"
                            "<small>ℹ️ Auto-tune is only a starting point and does not guarantee a successful training. "
                            "If the model doesn't sound like the dataset, adjust the values by ear.</small>"
                        )
                    with gr.Column(scale=1, elem_classes="form-section"):
                        gr.Markdown("#### ⚙️ Core Hyperparameters")
                        training_mode = gr.Dropdown(
                            ["LoRA adapter", "Full fine-tune"],
                            value="LoRA adapter",
                            label="Training Mode",
                        )
                        epochs = gr.Number(value=3, precision=0, label="Epochs / approximate passes")
                        steps = gr.Number(value=1000, precision=0, label="Max Steps (priority; -1 = use epochs)")
                        train_batch_size = gr.Number(value=DEFAULT_TRAINING_24GB["batch"], precision=0, label="Batch size")
                        train_grad_accum = gr.Number(value=DEFAULT_TRAINING_24GB["grad_accum"], precision=0, label="Gradient accumulation")
                        lr = gr.Number(value=DEFAULT_TRAINING_24GB["lr"], label="Learning Rate")
                        lora_rank = gr.Number(value=DEFAULT_TRAINING_24GB["lora_rank"], precision=0, label="LoRA Rank")
                        use_lora = gr.Checkbox(label="Use LoRA", value=True)
                        bf16_train = gr.Checkbox(label="bf16 mixed precision", value=DEFAULT_TRAINING_24GB["bf16"])
                        with gr.Row():
                            train_logging_steps = gr.Number(value=DEFAULT_TRAINING_24GB["logging_steps"], precision=0, label="Logging steps")
                            train_save_steps = gr.Number(value=DEFAULT_TRAINING_24GB["save_steps"], precision=0, label="Save steps")
                            train_eval_steps = gr.Number(value=DEFAULT_TRAINING_24GB["eval_steps"], precision=0, label="Eval steps")
                        resume_checkpoint_choice = gr.Dropdown(
                            choices=["None"],
                            value="None",
                            label="Resume checkpoint",
                        )
                        gr.Markdown(
                            "Existing projects auto-select their latest restorable checkpoint from `exp/<project>`. "
                            "Choose `None` to retrain from scratch."
                        )
                with gr.Accordion("🧪 V3 recipe (advanced)", open=False, elem_classes="accordion"):
                    gr.Markdown(
                        "Applies to **Higgs V3** training. Defaults follow the tuned recipe: warmup + cosine decay, "
                        "LoRA alpha = 2×rank, frozen codec embedding/head, best-eval checkpoint saved as `checkpoint-best`.",
                        elem_classes="hint",
                    )
                    with gr.Row():
                        v3_optimizer = gr.Dropdown(
                            [
                                ("AdamW · measured recipe (uses Learning Rate)", "adamw"),
                                ("Prodigy · finds the LR on its own", "prodigy"),
                                ("Prodigy Schedule-Free · auto LR, no schedule", "prodigy_sf"),
                            ],
                            value="adamw",
                            label="Optimizer",
                            info="Prodigy ignores the Learning Rate field and estimates it while training.",
                            scale=3,
                        )
                        v3_prodigy_dcoef = gr.Number(
                            value=1.0,
                            label="Prodigy d_coef",
                            info="Scales Prodigy's LR estimate: 0.5 = half, 2 = double.",
                            scale=1,
                        )
                    with gr.Row():
                        v3_lora_alpha = gr.Number(value=0, precision=0, label="LoRA alpha (0 = 2×rank)")
                        v3_lr_scheduler = gr.Dropdown(["cosine", "linear", "constant"], value="cosine", label="LR schedule")
                        v3_warmup_ratio = gr.Slider(0.0, 0.2, value=0.05, step=0.01, label="Warmup ratio")
                        v3_weight_decay = gr.Number(value=0.01, label="Weight decay")
                    with gr.Row():
                        v3_train_audio_modules = gr.Checkbox(
                            label="Also train codec embedding/head",
                            value=False,
                            info="Off is safer: the 8-codebook table is shared by every voice.",
                        )
                        v3_audio_lr_scale = gr.Number(value=0.1, label="Codec embedding LR × (when trained)")
                        v3_grad_ckpt = gr.Checkbox(label="Gradient checkpointing", value=False, info="Less VRAM, ~30% slower.")
                        v3_max_clip = gr.Slider(8, 40, value=30, step=1, label="Skip clips longer than (s)")
                    gr.Markdown("**Anti-overfitting**", elem_classes="hint")
                    with gr.Row():
                        v3_patience = gr.Number(
                            value=3, precision=0, label="Early-stop patience (evals)",
                            info="Stop after this many evals without improvement. 0 = never stop early.",
                        )
                        v3_min_delta = gr.Number(
                            value=0.002, label="Min improvement (relative)",
                            info="0.002 = an eval must beat the best by 0.2 % to count as progress.",
                        )
                        v3_restore_best = gr.Checkbox(
                            label="Final adapter = best eval", value=True,
                            info="When training ends or stops, the exported LoRA is the best-eval checkpoint.",
                        )
                        v3_max_batch_sec = gr.Number(
                            value=40, precision=0, label="Max audio per micro-batch (s)",
                            info="Caps padded audio per batch to avoid out-of-memory with long clips.",
                        )
                with gr.Accordion("🎧 Eval Zone (Optional Audio Preview)", open=False, elem_classes="accordion"):
                    gr.Markdown(
                        "Enable this to force periodic audio previews during training. "
                        "`Eval steps` are optimizer steps, not raw micro-batches. "
                        "When enabled, the trainer logs eval loss when a validation dataset exists, "
                        "saves WAV previews under `exp/<run>/eval_audio`, writes them to TensorBoard, "
                        "and always writes one final preview even if the run ends before the next eval interval."
                    )
                    enable_eval_audio = gr.Checkbox(label="Generate eval audio previews", value=True)
                    eval_text = gr.Textbox(
                        label="Eval preview text",
                        lines=3,
                        placeholder="Leave empty to use an unseen sentence from the eval set (its real recording is shown as the reference).",
                        value="",
                    )
                    eval_audio_max_new = gr.Number(value=1000, precision=0, label="Eval audio max new tokens")
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary", elem_classes="button-primary")
                    stop_train_btn = gr.Button("⏹️ Stop Training", variant="stop", elem_classes="button-stop", interactive=False)
                    refresh_training_btn = gr.Button("🔄 Refresh Status", variant="secondary", interactive=False)
                    tb_btn = gr.Button("📊 TensorBoard", variant="secondary", interactive=False)
                with gr.Column(elem_classes=["card", "output-card"]):
                    training_progress = gr.HTML(TRAINING_IDLE_HTML, elem_classes=["progress-card"])
                with gr.Accordion("📈 Training monitor · loss curves, overfitting guard and eval audio", open=True):
                    training_monitor = gr.HTML(monitor_html(None))
                    gr.Markdown("**Eval audio evolution**", elem_classes="hint")
                    training_audio = gr.HTML(audio_evolution_html(None))
                training_status_box = gr.Markdown("", elem_classes="hint")
                training_log_box = gr.Textbox(
                    label="Training Process Log",
                    lines=1,
                    interactive=False,
                    visible=False,
                )
                training_timer = gr.Timer(2.0, active=True) if hasattr(gr, "Timer") else None

        with gr.Accordion("🖥️ Console", open=True, elem_classes=["console-accordion"]):
            cmd_console = gr.HTML(value=cmd_mirror_html())
        cmd_console_timer = gr.Timer(1.0, active=True) if hasattr(gr, "Timer") else None

        logs_box = gr.Textbox(label="Logs", lines=1, visible=False)
        lora_v2_state = gr.State(default_lora_for_model("Higgs V2 TTS"))
        lora_v3_state = gr.State(default_lora_for_model("Higgs V3 TTS"))
        resume_checkpoint = gr.State("")

        common_params = [
            infer_temperature,
            infer_top_p,
            infer_top_k,
            infer_max_new,
            infer_ras_win_len,
            infer_ras_max_repeat,
            infer_seed,
            infer_reuse_chunk_seed,
        ]
        voice_dropdowns = [sample_dropdown, infer_sample_select, *speaker_inputs[1::4]]
        sample_dropdown.change(
            lambda name: (*load_sample_for_ui(name), name),
            sample_dropdown,
            [prep_audio_player, prep_transcription, save_sample_name],
            queue=False,
        )
        refresh_samples_btn.click(refresh_sample_dropdown, None, sample_dropdown, queue=False)
        prep_transcribe_start = transcribe_prep_btn.click(
            lambda: (set_button_busy("Transcribing... check the console"), gr.update(interactive=True), "Transcribing sample..."),
            None,
            [transcribe_prep_btn, stop_prep_transcribe_btn, prep_sample_console],
            queue=False,
            show_progress="hidden",
        )
        ev_prep_transcribe = prep_transcribe_start.then(
            transcribe_sample_ui,
            [prep_audio_player, prep_asr_model, prep_asr_lang],
            [prep_transcription, prep_sample_console, transcribe_prep_btn, stop_prep_transcribe_btn],
        )
        stop_prep_transcribe_btn.click(
            lambda: stop_reference_transcription("🔍 Transcribe"),
            None,
            [prep_sample_console, transcribe_prep_btn, stop_prep_transcribe_btn],
            queue=True,
            show_progress="hidden",
            cancels=[ev_prep_transcribe],
        )
        save_sample_start = save_sample_btn.click(
            lambda: (set_button_busy("Saving sample..."), "Saving voice sample..."),
            None,
            [save_sample_btn, prep_sample_console],
            queue=False,
            show_progress="hidden",
        )
        save_sample_start.then(
            save_sample_ui,
            [prep_audio_player, save_sample_name, prep_transcription],
            [*voice_dropdowns, prep_sample_console, save_sample_btn],
        )

        infer_sample_select.change(
            load_sample_for_ui,
            [infer_sample_select, infer_use_ref_text],
            [infer_ref_audio, infer_ref_text],
            queue=False,
        )
        refresh_infer_sample_btn.click(refresh_sample_dropdown, None, infer_sample_select, queue=False)
        infer_ref_transcribe_start = infer_transcribe_ref_btn.click(
            lambda: (set_button_busy("Transcribing... check the console"), gr.update(interactive=True), "Transcribing reference audio..."),
            None,
            [infer_transcribe_ref_btn, stop_infer_transcribe_btn, infer_console],
            queue=False,
            show_progress="hidden",
        )
        ev_infer_ref_transcribe = infer_ref_transcribe_start.then(
            transcribe_reference_ui,
            [infer_ref_audio, infer_asr_model, infer_asr_lang],
            [infer_ref_text, infer_console, infer_transcribe_ref_btn, stop_infer_transcribe_btn],
        )
        stop_infer_transcribe_btn.click(
            lambda: stop_reference_transcription("🎙️ Transcribe"),
            None,
            [infer_console, infer_transcribe_ref_btn, stop_infer_transcribe_btn],
            queue=True,
            show_progress="hidden",
            cancels=[ev_infer_ref_transcribe],
        )
        infer_text.change(update_clips_count, [infer_text, infer_chunk_mode], infer_clips_count, queue=False)
        infer_chunk_mode.change(update_clips_count, [infer_text, infer_chunk_mode], infer_clips_count, queue=False)
        use_last_seed_btn.click(
            use_last_inference_seed,
            infer_seed,
            [infer_seed, infer_console],
            queue=False,
            show_progress="hidden",
        )
        restore_random_seed_btn.click(
            restore_random_seed,
            None,
            [infer_seed, infer_console],
            queue=False,
            show_progress="hidden",
        )
        infer_start = infer_gen_btn.click(
            lambda: (set_button_busy("Generating…"), gr.update(interactive=True), "Preparing speech generation...", gr.update(), None),
            None,
            [infer_gen_btn, infer_stop_btn, infer_console, infer_audio_out, infer_file_out],
            queue=False,
            show_progress="hidden",
        )
        ev_inf = infer_start.then(
            run_inference,
            [
                infer_text,
                infer_model_select,
                lora_adapter,
                infer_sample_select,
                infer_ref_audio,
                infer_ref_text,
                *common_params,
                infer_save_outputs,
                infer_chunk_mode,
            ],
            [infer_audio_out, infer_file_out, infer_text, infer_console, infer_gen_btn, infer_stop_btn, infer_progress],
            show_progress="hidden",
        )
        infer_stop_btn.click(
            stop_generation,
            None,
            [infer_console, infer_gen_btn, infer_stop_btn],
            queue=True,
            show_progress="hidden",
            cancels=[ev_inf],
        )

        dialogue_start = dialogue_gen_btn.click(
            lambda: (set_button_busy("Generating dialogue…"), gr.update(interactive=True), "Preparing dialogue generation...", None, None),
            None,
            [dialogue_gen_btn, dialogue_stop_btn, infer_console, dialogue_audio_out, dialogue_file_out],
            queue=False,
            show_progress="hidden",
        )
        ev_dialogue = dialogue_start.then(
            run_dialogue,
            [
                infer_model_select,
                lora_adapter,
                dialogue_silence_slider,
                *common_params,
                infer_save_outputs,
                infer_chunk_mode,
                dialogue_row_count,
                *speaker_inputs,
            ],
            [dialogue_audio_out, dialogue_file_out, infer_console, dialogue_gen_btn, dialogue_stop_btn],
            show_progress_on=[dialogue_audio_out],
        )
        dialogue_stop_btn.click(
            stop_dialogue_generation,
            None,
            [infer_console, dialogue_gen_btn, dialogue_stop_btn],
            queue=True,
            show_progress="hidden",
            cancels=[ev_dialogue],
        )
        dialogue_text_inputs = speaker_inputs[0::4]
        dialogue_speaker_inputs = speaker_inputs[1::4]
        dialogue_audio_inputs = speaker_inputs[2::4]
        dialogue_ref_text_inputs = speaker_inputs[3::4]
        dialogue_edit_outputs = [
            dialogue_row_count,
            *dialogue_text_inputs,
            *dialogue_speaker_inputs,
            *dialogue_audio_inputs,
            *dialogue_ref_text_inputs,
            *dialogue_rows,
            infer_console,
        ]
        dialogue_reset_btn.click(
            reset_dialogue_rows,
            gr.State(default_sample_name),
            dialogue_edit_outputs,
            queue=False,
            show_progress="hidden",
        )
        dialogue_clear_btn.click(
            clear_dialogue_rows,
            [dialogue_row_count, *speaker_inputs],
            dialogue_edit_outputs,
            queue=False,
            show_progress="hidden",
        )
        dialogue_compact_btn.click(
            remove_empty_dialogue_rows,
            [dialogue_row_count, *speaker_inputs],
            dialogue_edit_outputs,
            queue=False,
            show_progress="hidden",
        )
        for i in range(DIALOGUE_MAX_SEGMENTS):
            dialogue_add_btns[i].click(
                add_dialogue_row,
                [gr.State(i), dialogue_row_count, *speaker_inputs],
                dialogue_edit_outputs,
                queue=False,
                show_progress="hidden",
            )
            dialogue_copy_btns[i].click(
                copy_dialogue_row,
                [gr.State(i), dialogue_row_count, *speaker_inputs],
                dialogue_edit_outputs,
                queue=False,
                show_progress="hidden",
            )
            dialogue_delete_btns[i].click(
                delete_dialogue_row,
                [gr.State(i), dialogue_row_count, *speaker_inputs],
                dialogue_edit_outputs,
                queue=False,
                show_progress="hidden",
            )

        browse_src_folder_btn.click(
            browse_source_folder,
            src_folder,
            [src_folder, dataset_log_box],
            queue=False,
        )
        prep_start = prep_btn.click(
            lambda: (
                gr.update(value="Building...", interactive=False, variant="secondary"),
                gr.update(interactive=True),
                "### Dataset Build\nBuilding...",
                "Starting dataset build...",
            ),
            None,
            [prep_btn, stop_prep_btn, dataset_status_box, dataset_log_box],
            queue=False,
        )
        ev_prep_build = prep_start.then(
            prepare_higgs_dataset,
            [
                src_folder,
                dataset_name_input,
                dataset_asr_model,
                dataset_asr_lang,
                dataset_asr_batch,
                val_split_slider,
                recursive_dataset,
                dataset_isolate,
                dataset_segment,
                dataset_min_sec,
                dataset_max_sec,
            ],
            [train_manifest, val_manifest, dataset_log_box, dataset_status_box, prep_btn, stop_prep_btn, output_name, dataset_project_card, dataset_name_input],
        )
        stop_prep_btn.click(
            stop_dataset_build,
            None,
            [dataset_log_box, dataset_status_box, prep_btn, stop_prep_btn],
            queue=True,
            show_progress="hidden",
            cancels=[ev_prep_build],
        )
        refresh_train_btn.click(refresh_dataset_manifests, None, [train_manifest, val_manifest], queue=False)
        auto_tune_btn.click(
            auto_tune_training_ui,
            [train_manifest, val_manifest, train_vram_preset, train_model_select, v3_optimizer],
            [
                epochs,
                steps,
                train_batch_size,
                train_grad_accum,
                lr,
                lora_rank,
                use_lora,
                bf16_train,
                train_logging_steps,
                train_save_steps,
                train_eval_steps,
                logs_box,
                training_autotune_report,
                v3_lora_alpha,
                v3_lr_scheduler,
                v3_warmup_ratio,
                v3_weight_decay,
                v3_train_audio_modules,
                v3_grad_ckpt,
                v3_max_clip,
                v3_prodigy_dcoef,
                v3_patience,
                v3_min_delta,
                v3_restore_best,
                v3_max_batch_sec,
                enable_eval_audio,
            ],
            queue=False,
        )
        refresh_project_btn.click(
            reload_training_project,
            [output_name, train_model_select],
            [
                output_name,
                train_model_select,
                train_manifest,
                val_manifest,
                training_mode,
                epochs,
                steps,
                train_batch_size,
                train_grad_accum,
                lr,
                lora_rank,
                use_lora,
                bf16_train,
                train_logging_steps,
                train_save_steps,
                train_eval_steps,
                enable_eval_audio,
                eval_text,
                eval_audio_max_new,
                resume_checkpoint_choice,
                resume_checkpoint,
                training_status_box,
                training_log_box,
                v3_lora_alpha,
                v3_lr_scheduler,
                v3_warmup_ratio,
                v3_weight_decay,
                v3_train_audio_modules,
                v3_audio_lr_scale,
                v3_grad_ckpt,
                v3_max_clip,
                v3_optimizer,
                v3_prodigy_dcoef,
                v3_patience,
                v3_min_delta,
                v3_restore_best,
                v3_max_batch_sec,
                training_project_card,
                training_monitor,
                training_audio,
            ],
            queue=False,
        )
        delete_project_btn.click(
            delete_training_project,
            output_name,
            [
                output_name,
                train_model_select,
                train_manifest,
                val_manifest,
                training_mode,
                epochs,
                steps,
                train_batch_size,
                train_grad_accum,
                lr,
                lora_rank,
                use_lora,
                bf16_train,
                train_logging_steps,
                train_save_steps,
                train_eval_steps,
                enable_eval_audio,
                eval_text,
                eval_audio_max_new,
                resume_checkpoint_choice,
                resume_checkpoint,
                training_status_box,
                training_log_box,
            ],
            queue=False,
        )
        dataset_name_input.change(
            on_dataset_project_change,
            dataset_name_input,
            [
                dataset_project_card,
                src_folder,
                dataset_asr_model,
                dataset_asr_lang,
                val_split_slider,
                recursive_dataset,
                dataset_isolate,
                dataset_segment,
                dataset_min_sec,
                dataset_max_sec,
            ],
            queue=False,
            show_progress="hidden",
        )
        dataset_open_btn.click(lambda name: log(open_project_folder(name)), dataset_name_input, logs_box, queue=False)
        train_open_btn.click(lambda name: log(open_project_folder(name)), output_name, logs_box, queue=False)
        output_name.change(project_card, output_name, training_project_card, queue=False, show_progress="hidden")
        output_name.change(
            lambda name: training_monitor_updates(name, force=True),
            output_name,
            [training_monitor, training_audio],
            queue=False,
            show_progress="hidden",
        )
        output_name.change(
            select_training_project,
            [output_name, train_model_select],
            [
                train_model_select,
                train_manifest,
                val_manifest,
                training_mode,
                epochs,
                steps,
                train_batch_size,
                train_grad_accum,
                lr,
                lora_rank,
                use_lora,
                bf16_train,
                train_logging_steps,
                train_save_steps,
                train_eval_steps,
                enable_eval_audio,
                eval_text,
                eval_audio_max_new,
                resume_checkpoint_choice,
                resume_checkpoint,
                training_status_box,
                training_log_box,
                v3_lora_alpha,
                v3_lr_scheduler,
                v3_warmup_ratio,
                v3_weight_decay,
                v3_train_audio_modules,
                v3_audio_lr_scale,
                v3_grad_ckpt,
                v3_max_clip,
                v3_optimizer,
                v3_prodigy_dcoef,
                v3_patience,
                v3_min_delta,
                v3_restore_best,
                v3_max_batch_sec,
            ],
            queue=False,
        )
        resume_checkpoint_choice.change(resume_choice_to_state, resume_checkpoint_choice, resume_checkpoint, queue=False)
        training_start = start_btn.click(
            lambda: (set_button_busy("Starting training... check logs"), "Starting training command..."),
            None,
            [start_btn, training_log_box],
            queue=False,
            show_progress="hidden",
        )
        training_start.then(
            start_higgs_training,
            [
                train_model_select,
                train_manifest,
                val_manifest,
                output_name,
                training_mode,
                epochs,
                steps,
                train_batch_size,
                train_grad_accum,
                lr,
                lora_rank,
                use_lora,
                bf16_train,
                train_logging_steps,
                train_save_steps,
                train_eval_steps,
                resume_checkpoint,
                enable_eval_audio,
                eval_text,
                eval_audio_max_new,
                v3_lora_alpha,
                v3_lr_scheduler,
                v3_warmup_ratio,
                v3_weight_decay,
                v3_train_audio_modules,
                v3_audio_lr_scale,
                v3_grad_ckpt,
                v3_max_clip,
                v3_optimizer,
                v3_prodigy_dcoef,
                v3_patience,
                v3_min_delta,
                v3_restore_best,
                v3_max_batch_sec,
            ],
            [logs_box, training_status_box, training_log_box, start_btn, stop_train_btn, refresh_training_btn, tb_btn],
        )
        stop_train_btn.click(
            stop_higgs_training,
            None,
            [logs_box, training_status_box, training_log_box, start_btn, stop_train_btn, refresh_training_btn, tb_btn],
            queue=False,
        )
        refresh_training_btn.click(
            refresh_training_status,
            output_name,
            [training_status_box, training_log_box, start_btn, stop_train_btn, refresh_training_btn, tb_btn, training_progress, training_project_card, training_monitor, training_audio, resume_checkpoint_choice, resume_checkpoint],
            queue=False,
        )
        if training_timer is not None:
            training_timer.tick(
                refresh_training_status,
                output_name,
                [training_status_box, training_log_box, start_btn, stop_train_btn, refresh_training_btn, tb_btn, training_progress, training_project_card, training_monitor, training_audio, resume_checkpoint_choice, resume_checkpoint],
                queue=False,
                show_progress="hidden",
            )
        if cmd_console_timer is not None:
            cmd_console_timer.tick(cmd_mirror_html, None, cmd_console, queue=False, show_progress="hidden")
        demo.load(fn=None, js=APP_JS, queue=False)
        tb_btn.click(launch_tensorboard, output_name, [logs_box, training_status_box, training_log_box], queue=False)

        v3_precision.change(set_v3_runtime, [v3_precision, v3_attention_backend, v3_torch_compile], logs_box, queue=False)
        v3_attention_backend.change(set_v3_runtime, [v3_precision, v3_attention_backend, v3_torch_compile], logs_box, queue=False)
        v3_torch_compile.change(set_v3_runtime, [v3_precision, v3_attention_backend, v3_torch_compile], logs_box, queue=False)
        lora_project.input(on_lora_project_change, [infer_model_select, lora_project], lora_adapter, queue=False, show_progress="hidden")
        infer_model_select.change(
            sync_lora_for_model,
            [infer_model_select, lora_adapter, lora_v2_state, lora_v3_state],
            [lora_project, lora_adapter, lora_v2_state, lora_v3_state],
            queue=False,
            show_progress="hidden",
        )
        lora_adapter.change(
            set_lora_adapter,
            [infer_model_select, lora_adapter, lora_v2_state, lora_v3_state],
            [logs_box, lora_v2_state, lora_v3_state],
            queue=False,
        )
        refresh_lora_adapter_btn.click(
            refresh_lora_adapters,
            [infer_model_select, lora_adapter],
            [lora_project, lora_adapter],
            queue=False,
        )
        top_unload_btn.click(
            unload_models,
            None,
            logs_box,
            queue=True,
            cancels=[ev_inf, ev_dialogue],
        )
        delete_outputs_btn.click(
            delete_output_audios,
            None,
            logs_box,
            queue=True,
        )
        delete_samples_btn.click(
            delete_reference_samples,
            None,
            logs_box,
            queue=True,
        )

    return demo


if __name__ == "__main__":
    _install_cmd_mirror()
    build_ui().queue(default_concurrency_limit=1).launch(inbrowser=True, css=UI_CSS, js=APP_JS, theme=build_theme(), allowed_paths=[str(ROOT / "exp")])
