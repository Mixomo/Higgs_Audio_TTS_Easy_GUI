from __future__ import annotations

import json
import numpy as np
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
from higgs_local.dataset_audio import CLEAN_ENGINES
from higgs_local.engines import ModelManager, text_chunks_for_ui
from higgs_local import takecheck as tc
from higgs_local import fixer as fx
from higgs_local import llm, loudness, montage
from higgs_local.block_editors import DIALOGUE_EDITOR_CSS, DIALOGUE_EDITOR_HTML, DIALOGUE_EDITOR_JS
from higgs_local.live import LIVE_PLAYER_CSS, LIVE_PLAYER_HTML, LIVE_PLAYER_JS, live_claim, live_poll
from higgs_local.render import (
    LOUDNESS_TARGETS,
    READ_AS_WORDS,
    Advanced,
    Block,
    Cancelled,
    Renderer,
    Sampling,
    clear_take_cache,
    results_report,
)
from higgs_local.model_registry import (
    DEFAULT_ASR_MODEL,
    ASR_MODELS,
    HIGGS_ASR_MODELS,
    PROCESSOR_MODELS,
    TOKENIZER_MODELS,
    TTS_MODELS,
    WHISPER_MODELS,
    ensure_tts_model,
    ensure_asr_model,
    ensure_model,
    model_status,
)
from higgs_local.paths import CACHE_DIR, OUTPUTS_DIR, ROOT, SAMPLES_DIR, TEMP_DIR, V2_DEFAULT_MODEL, V2_DEFAULT_TOKENIZER, ensure_local_dirs
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

  // ---------------- modals: ✕, a click outside the card and Esc press the modal's Cancel button
  const modalCancel = (m) => m.querySelector('button.hv-cancel, .hv-cancel button');
  const shownModals = () => [...document.querySelectorAll('.hv-modal:not(.hv-js)')].filter((m) => m.getClientRects().length);
  document.addEventListener('click', (e) => {
    const m = e.target.closest && e.target.closest('.hv-modal'); if (!m || m.classList.contains('hv-js')) return;
    if (e.target === m || e.target.closest('.hv-x')) { const c = modalCancel(m); if (c) c.click(); }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || e.defaultPrevented) return;
    const m = shownModals().pop(); if (!m) return;
    const c = modalCancel(m); if (c) { e.preventDefault(); c.click(); }
  });
  // a modal that opens takes the focus (Enter = its main button)
  new MutationObserver(() => {
    for (const m of shownModals()) {
      if (m.dataset.hvOpen) continue;
      m.dataset.hvOpen = '1';
      // Enter = the main button, but never a destructive one (those start on Cancel)
      const main = m.querySelector('.hv-actions button:last-child');
      const pick = main && main.classList.contains('stop') ? modalCancel(m) : main;
      if (pick) setTimeout(() => pick.focus(), 60);
    }
    document.querySelectorAll('.hv-modal[data-hv-open]').forEach((m) => { if (!m.getClientRects().length) delete m.dataset.hvOpen; });
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style'] });

  // ---------------- a tab left open across an app restart that did not notice it (Gradio reloads the ones
  // that do). Same code: it keeps working. Updated code: it would call the new server's handlers by the old
  // numbers (wrong inputs, endless errors), so it stops talking to the server and offers a reload, without
  // forcing one: what is on screen stays to copy from.
  const cfg0 = window.gradio_config || {};
  const base = String(cfg0.root || location.origin).replace(/[/]$/, '');
  const depSig = (c) => JSON.stringify((c.dependencies || []).map((d) => [d.id, d.targets, d.inputs, d.outputs]));
  let bootId = String(cfg0.app_id || ''), stale = false;
  const mySig = depSig(cfg0);
  const realFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const url = String((input && input.url) || input || '');
    if (stale && url.includes('/gradio_api/') && !url.includes('/file=')) {
      return Promise.reject(new Error('This tab is from before the app was updated: reload it.'));
    }
    return realFetch(input, init);
  };
  if (bootId) {
    setInterval(async () => {
      if (stale) return;
      try {
        const r = await realFetch(base + '/gradio_api/app_id', { cache: 'no-store' });
        if (!r.ok) return;
        const now = String((await r.json()).app_id || '');
        if (!now || now === bootId) return;
        const c = await (await realFetch(base + '/config', { cache: 'no-store' })).json();
        if (depSig(c) === mySig) { bootId = now; return; }  // restarted with the same code: carry on
        stale = true;
        const bar = document.createElement('div');
        bar.className = 'hv-stale';
        bar.innerHTML = '⚠️ The app was restarted with updated code. This tab is paused so it does not call the wrong '
          + 'functions; copy anything you need, then <button type="button">⟳ Reload</button>';
        bar.querySelector('button').onclick = () => location.reload();
        document.body.appendChild(bar);
      } catch (e) { /* server down or restarting: try again later */ }
    }, 3000);
  }

  // ---------------- the speech settings only where speech is made (not in Voice Library, Dataset, Training)
  const NO_SPEECH = ['Voice Library', 'Dataset Preparation', 'Training'];
  const syncAdvanced = () => {
    const tab = document.querySelector('.main-tabs [role=tab][aria-selected=true]');
    const adv = document.querySelector('.advanced-speech');
    if (!tab || !adv) return;
    const hide = NO_SPEECH.some((n) => tab.textContent.includes(n));
    if (adv.classList.contains('hv-off') !== hide) adv.classList.toggle('hv-off', hide);
  };
  new MutationObserver(syncAdvanced).observe(document.body, { subtree: true, attributes: true, attributeFilter: ['aria-selected'] });
  setTimeout(syncAdvanced, 500);

  // ---------------- @ tag autocomplete (Inference script, dialogue lines, eval / compare texts)
  const CATALOG = __TAG_CATALOG__;
  window.__hvSepEngines = __SEP_ENGINES__;
  // library voices and LoRAs, asked from the server (they change while the app runs)
  const dyn = { t: 0, model: '', items: [] };
  const refreshDyn = async (ta, model) => {
    if (!window.__hvTagOptions || (Date.now() - dyn.t < 5000 && dyn.model === model)) return;
    dyn.t = Date.now(); dyn.model = model;
    try {
      const r = await window.__hvTagOptions(model);
      dyn.items = [
        ...(r.voices || []).map((v) => ({ cat: 'Voice from here on (library)', label: `{voice:${v}}`, hint: 'speaker', insert: `{voice:${v}} ` })),
        ...(r.loras || []).map((l) => ({ cat: 'LoRA from here on', label: `{lora:${l}}`, hint: 'final adapter', insert: `{lora:${l}} ` })),
        { cat: 'LoRA from here on', label: '{lora:none}', hint: 'base model', insert: '{lora:none} ' },
        ...[250, 500, 800, 1200, 2000].map((ms) => ({ cat: 'Pause before what follows', label: `{pause:${ms}}`, hint: `${ms} ms`, insert: `{pause:${ms}} ` })),
      ];
      if (st && st.ta === ta) update(ta);
    } catch (e) { /* the server is busy: try again on the next key */ dyn.t = 0; }
  };
  const modelFor = (ta) => {
    if (ta.closest('.tag-ac-v3')) return 'Higgs V3 TTS';
    const sel = document.querySelector('#infer-model-select input');
    return sel && /V2/.test(sel.value || '') ? 'Higgs V2 TTS' : 'Higgs V3 TTS';
  };
  const pop = document.createElement('div');
  pop.id = 'tag-ac-pop';
  pop.setAttribute('role', 'listbox');
  document.body.appendChild(pop);
  let st = null;  // {ta, start, items, idx}

  const caretXY = (ta, pos) => {
    const cs = getComputedStyle(ta);
    const m = document.createElement('div');
    for (const p of ['boxSizing','width','paddingTop','paddingRight','paddingBottom','paddingLeft',
                     'borderTopWidth','borderRightWidth','borderBottomWidth','borderLeftWidth',
                     'fontFamily','fontSize','fontWeight','fontStyle','letterSpacing','lineHeight',
                     'textTransform','wordSpacing','tabSize']) m.style[p] = cs[p];
    m.style.position = 'absolute'; m.style.visibility = 'hidden'; m.style.top = '0'; m.style.left = '-9999px';
    m.style.whiteSpace = 'pre-wrap'; m.style.wordWrap = 'break-word'; m.style.overflow = 'hidden';
    m.textContent = ta.value.slice(0, pos);
    const mark = document.createElement('span');
    mark.textContent = ta.value.slice(pos) || '.';
    m.appendChild(mark);
    document.body.appendChild(m);
    const r = ta.getBoundingClientRect();
    const lh = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.3;
    const xy = { x: r.left + mark.offsetLeft - ta.scrollLeft, y: r.top + mark.offsetTop - ta.scrollTop + lh };
    m.remove();
    return xy;
  };
  const close = () => { st = null; pop.style.display = 'none'; };
  const render = () => {
    pop.innerHTML = '';
    if (!st.items.length) {
      const e = document.createElement('div'); e.className = 'tag-ac-empty';
      e.textContent = 'No tag matches · Esc to close'; pop.appendChild(e); return;
    }
    let lastCat = null;
    st.items.forEach((it, i) => {
      if (it.cat !== lastCat) {
        const h = document.createElement('div'); h.className = 'tag-ac-cat'; h.textContent = it.cat; pop.appendChild(h);
        lastCat = it.cat;
      }
      const row = document.createElement('div');
      row.className = 'tag-ac-item' + (i === st.idx ? ' active' : '');
      row.setAttribute('role', 'option');
      const a = document.createElement('span'); a.className = 'tag-ac-label'; a.textContent = it.label;
      const b = document.createElement('span'); b.className = 'tag-ac-hint'; b.textContent = it.hint;
      row.append(a, b);
      row.addEventListener('mousedown', (ev) => { ev.preventDefault(); st.idx = i; choose(); });
      pop.appendChild(row);
    });
    const act = pop.querySelector('.tag-ac-item.active');
    if (act) act.scrollIntoView({ block: 'nearest' });
  };
  const update = (ta) => {
    const pos = ta.selectionStart, before = ta.value.slice(0, pos);
    const at = before.lastIndexOf('@');
    if (at < 0 || ta.selectionEnd !== pos) return close();
    const q = before.slice(at + 1);
    if (/\\s/.test(q) || q.length > 40) return close();
    refreshDyn(ta, modelFor(ta));
    const list = [...dyn.items, ...(CATALOG[modelFor(ta)] || [])];
    const words = q.toLowerCase().split(/[:|_<>{}\\[\\]-]+/).filter(Boolean);
    const items = list.filter((it) => {
      const hay = (it.cat + ' ' + it.label).toLowerCase();
      return words.every((w) => hay.includes(w));
    });
    const keep = st && st.ta === ta && st.start === at ? Math.min(st.idx, Math.max(items.length - 1, 0)) : 0;
    st = { ta, start: at, items, idx: keep };
    render();
    const xy = caretXY(ta, at);
    pop.style.display = 'block';
    const w = pop.offsetWidth, h = pop.offsetHeight;
    let x = Math.min(xy.x, window.innerWidth - w - 8), y = xy.y + 4;
    if (y + h > window.innerHeight - 8) y = Math.max(8, xy.y - h - 28);
    pop.style.left = Math.max(8, x) + 'px'; pop.style.top = y + 'px';
  };
  const choose = () => {
    if (!st || !st.items.length) return close();
    const { ta, start } = st, it = st.items[st.idx];
    const end = ta.selectionStart;
    const v = ta.value;
    const next = v.slice(end);
    const ins = it.insert + (it.hint === 'inline' && next && !/^\\s/.test(next) ? ' ' : '');
    ta.value = v.slice(0, start) + ins + next;
    const caret = start + ins.length;
    ta.setSelectionRange(caret, caret);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    close();
    ta.focus();
  };
  const isTarget = (el) => el && el.tagName === 'TEXTAREA' && el.closest('.tag-ac, .tag-ac-v3');
  document.addEventListener('input', (e) => { if (isTarget(e.target)) update(e.target); }, true);
  document.addEventListener('keydown', (e) => {
    if (!st || e.target !== st.ta) return;
    const n = st.items.length;
    if (e.key === 'ArrowDown' && n) { st.idx = (st.idx + 1) % n; render(); e.preventDefault(); }
    else if (e.key === 'ArrowUp' && n) { st.idx = (st.idx - 1 + n) % n; render(); e.preventDefault(); }
    else if ((e.key === 'Enter' || e.key === 'Tab') && n) { e.preventDefault(); e.stopPropagation(); choose(); }
    else if (e.key === 'Escape') { e.preventDefault(); close(); }
  }, true);
  document.addEventListener('click', (e) => { if (st && e.target !== st.ta && !pop.contains(e.target)) close(); }, true);
  document.addEventListener('keyup', (e) => {
    if (st && e.target === st.ta && (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'Home' || e.key === 'End')) update(e.target);
  }, true);
  window.addEventListener('scroll', (e) => { if (st && !pop.contains(e.target) && e.target !== pop) update(st.ta); }, true);
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


def _init_loudness() -> None:
    s = _load_ui_settings()
    loudness.configure(s.get("loudness_on", True), s.get("loudness_target", -20.0))


_init_loudness()


def v3_compile_default() -> bool:
    return bool(_load_ui_settings().get("v3_torch_compile", False))


def ref_max_default() -> float:
    try:
        return max(0.0, float(_load_ui_settings().get("v3_ref_max_sec", 0)))
    except (TypeError, ValueError):
        return 0.0


def set_ref_max(seconds):
    seconds = max(0.0, float(seconds or 0))
    _save_ui_settings(v3_ref_max_sec=seconds)
    MANAGER.set_ref_max_seconds(seconds)
    return log("Reference length: " + (f"the first {seconds:g} s are used." if seconds else "no limit, the whole reference is used."))


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
atexit.register(lambda: llm.SERVER.stop())  # the local LLM server never outlives the app


def _save_output(prefix: str, sr: int, wav, text: str = "", reference_name: str = "", persist: bool = True, adv=None) -> str:
    if adv is not None and getattr(adv, "enhance", False):
        from higgs_local.enhance import ENHANCER

        t0 = time.perf_counter()
        wav, sr = ENHANCER.enhance(wav, sr, denoise=bool(getattr(adv, "enhance_denoise", False)))
        print(f"[enhance] LavaSR: highs restored, {sr} Hz, in {time.perf_counter() - t0:.1f} s", flush=True)
        if not getattr(adv, "keep_check_models", True):
            ENHANCER.unload()
    root = OUTPUTS_DIR if persist else TEMP_DIR / "generated"
    text = VOICE_TAG_RE.sub(" ", text or "")
    match = re.search(r".*?(?:\(\.\.\.\)|\.\.\.|[.!?…])", (text or "").strip(), flags=re.S)
    excerpt = (match.group(0) if match else (text or "").strip())[:110]
    parts = [
        safe_name(prefix, "higgs")[:36],
        safe_name(reference_name, "no_reference")[:48],
        safe_name(excerpt, "speech")[:110],
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"{'_'.join(parts)}_{stamp}.wav"
    # the same loudness for every output (after enhancing, which adds highs), unless a delivery target was asked
    explicit = adv is not None and LOUDNESS_TARGETS.get(getattr(adv, "final_loudness", None))
    if explicit:
        from higgs_local.render import final_loudness as _final_loudness

        wav = _final_loudness(np.asarray(wav, dtype=np.float32), sr, adv.final_loudness)
    save_wav(path, sr, wav, level=not explicit)
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


# ---------------------------------------------------------------- Utils tab (storage)
STORAGE_HEADERS = ["Name", "Type", "Size", "Details", "Path"]


def _storage_items(kind, run_key=None):
    from higgs_local import storage

    if kind == "models":
        return [i for i in storage.scan_models() if i.size >= 1024]
    if kind == "loras":
        return storage.scan_loras()
    if kind == "datasets":
        return storage.scan_datasets()
    if kind == "parts":
        return storage.scan_run_parts(run_key) if run_key else []
    return []


def storage_view(kind, run_key=None):
    """(table, total markdown, delete-selector update) for one section."""
    from higgs_local import storage

    items = _storage_items(kind, run_key)
    label = {"models": "models and caches", "loras": "LoRA runs", "datasets": "datasets and downloads",
             "parts": "this run"}[kind]
    summary = f"**{len(items)} items · total {storage.human(storage.total(items))}** in {label}"
    if kind == "models":
        caches = sum(i.size for i in items if i.kind == "Cache")
        summary += f" (caches {storage.human(caches)})"
    choices = [(f"{i.name} · {storage.human(i.size)} · {i.key}", i.key) for i in items]
    return storage.table(items), summary, gr.update(choices=choices, value=[])


def storage_ask(keys, kind):
    """First click: show the confirmation with what will be deleted."""
    from higgs_local import storage

    keys = [k for k in (keys or []) if k]
    if not keys:
        return [], gr.update(visible=False), "Select at least one item to delete."
    sizes = {i.key: i for i in _storage_items(kind) if i.key in keys} if kind != "parts" else {}
    lines = []
    total = 0
    for key in keys:
        size = sizes[key].size if key in sizes else storage.dir_size(ROOT / key)
        total += size
        lines.append(f"- `{key}` · {storage.human(size)}")
    warning = {
        "models": "Deleted models download again the next time a feature needs them.",
        "loras": "The LoRA voices, their checkpoints and training history are removed.",
        "datasets": "The prepared clips are removed; projects keep their settings.",
        "parts": "Deleted checkpoints can no longer be used for inference or to resume training.",
    }[kind]
    text = (
        f"**{len(keys)} item(s), {storage.human(total)}.** {warning} **This cannot be undone.**\n\n"
        + "\n".join(lines[:25]) + ("\n- …" if len(lines) > 25 else "")
    )
    return keys, gr.update(visible=True), text


def storage_ask_keep_best(run_key):
    from higgs_local import storage

    if not run_key:
        return [], gr.update(visible=False), "Pick a LoRA run first."
    if not (ROOT / run_key / "checkpoint-best").exists():
        return [], gr.update(visible=False), "This run has no `checkpoint-best`; nothing was changed."
    targets = storage.keep_only_best_targets(run_key)
    if not targets:
        return [], gr.update(visible=False), "Only `checkpoint-best` is left already."
    keys, group, text = storage_ask([t.key for t in targets], "parts")
    text = text.replace("### ⚠️ Delete", "### ⭐ Keep only the best: delete", 1) + (
        "\n\n`checkpoint-best` and the final adapter stay. Without numbered checkpoints this run can no longer be resumed."
    )
    return keys, group, text


def _storage_guard(keys):
    """Refuse to delete what the running training uses; unload models so Windows can delete their files."""
    if TRAINING.is_running():
        run = TRAINING.run_name or ""
        for key in keys:
            if key.startswith(("models/", "data/")) or key == f"exp/{run}" or key.startswith(f"exp/{run}/"):
                raise RuntimeError(f"A training run ({run}) is active: stop it before deleting `{key}`.")
    release_all_models("storage cleanup")


def storage_delete(keys, kind, run_key=None):
    from higgs_local import storage

    keys = list(keys or [])
    try:
        _storage_guard(keys)
        freed, errors = storage.delete(keys)
        message = f"Deleted {len(keys) - len(errors)} item(s), freed {storage.human(freed)}."
        if errors:
            message += "\n\nCould not delete:\n" + "\n".join(f"- {e}" for e in errors)
    except Exception as exc:
        message = f"Nothing deleted: {exc}"
    table, summary, selector = storage_view(kind, run_key)
    return [], gr.update(visible=False), message, table, summary, selector


def compare_checkpoint_choices(project):
    choices = lora_checkpoints("v3", project) if project else []
    return gr.update(choices=choices, value=[c[1] for c in choices[:3]])


def compare_select_all(project):
    return gr.update(value=[value for _label, value in lora_checkpoints("v3", project)] if project else [])


def _compare_html(rows, text, seed, voice):
    """Plain audio players (same look as the training eval previews)."""
    import html as _html

    from higgs_local.training_dashboard import _file_url

    if not rows:
        return '<div class="tm"><div class="tm-muted">Results appear here, one player per checkpoint.</div></div>'
    head = (
        f'<div class="tm-text"><b>Text:</b> “{_html.escape(text)}” · seed {seed}'
        f' · voice {_html.escape(voice or "None")}</div>'
    )
    cards = []
    for row in rows:
        best = " best" if row["label"].startswith("Best eval") else ""
        if row.get("path"):
            body = f'<audio controls preload="none" src="{_file_url(row["path"])}"></audio>'
            meta = f'{row["seconds"]:.1f} s'
        else:
            body, meta = "", _html.escape(row.get("error") or "generating…")
        cards.append(
            f'<div class="tm-audio{best}"><div class="tm-audio-tag">{_html.escape(row["label"])}</div>'
            f'{body}<div class="tm-audio-meta">{meta}</div></div>'
        )
    return f'<div class="tm">{head}<div class="tm-audios">{"".join(cards)}</div></div>'


def compare_checkpoints(project, adapters, text, voice, seed, temperature, progress=gr.Progress(track_tqdm=False)):
    """Same text, same seed, same voice on each chosen checkpoint: the only difference is the LoRA.

    WAVs are kept in exp/<run>/checkpoint_compare/<time>/ (served by the app like the eval previews)."""
    import soundfile as sf

    adapters = list(adapters or [])
    text = (text or "").strip()
    seed = int(seed) if seed is not None and int(seed) >= 0 else 1234
    if not project or not adapters:
        yield "Pick a LoRA run and at least one checkpoint.", _compare_html([], text, seed, voice)
        return
    if not text:
        yield "Type the text to compare.", _compare_html([], text, seed, voice)
        return
    if TRAINING.is_running():
        yield "A training run is using the GPU: wait for it to finish.", _compare_html([], text, seed, voice)
        return
    labels = dict((value, label) for label, value in lora_checkpoints("v3", project))
    order = {value: i for i, (_l, value) in enumerate(lora_checkpoints("v3", project))}
    adapters.sort(key=lambda a: order.get(a, 999))
    out_dir = ROOT / "exp" / slugify(project, "lora") / "checkpoint_compare" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(f"{text}\nseed={seed}\nvoice={voice}\n", encoding="utf-8")
    rows = [{"label": labels.get(a, a)} for a in adapters]
    for n, adapter in enumerate(adapters):
        label = rows[n]["label"]
        progress((n, len(adapters)), desc=f"Generating with {label}")
        yield f"Generating {n + 1}/{len(adapters)}: **{label}** …", _compare_html(rows, text, seed, voice)
        last_file, last_console = None, ""
        for update in run_inference(
            text, "Higgs V3 TTS", adapter, voice or "None", None, "",
            float(temperature), 0.95, 50, 4096, 7, 2, seed, True, False, SEAMLESS_MODE,
        ):
            if isinstance(update[1], str):
                last_file = update[1]
            if isinstance(update[2], str):
                last_console = update[2]
        if last_file and Path(last_file).exists():
            dst = out_dir / f"{n:02d}_{safe_name(label, 'checkpoint')[:60]}.wav"
            shutil.copyfile(last_file, dst)
            rows[n].update(path=str(dst), seconds=sf.info(str(dst)).duration)
        else:
            rows[n]["error"] = last_console.strip().splitlines()[-1] if last_console.strip() else "failed"
    ok = sum(1 for r in rows if r.get("path"))
    yield (
        f"Done: {ok}/{len(adapters)} checkpoint(s), same text, voice and seed {seed}. Saved in `{out_dir.relative_to(ROOT).as_posix()}`.",
        _compare_html(rows, text, seed, voice),
    )


def storage_run_choices():
    from higgs_local import storage

    return [(f"{i.name} · {storage.human(i.size)}", i.key) for i in storage.scan_loras()]


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


def _fmt_duration(seconds) -> str:
    """Audio length for people: 9.4 s, 2:05 min, 1:02:30 h."""
    seconds = max(0.0, float(seconds or 0))
    if seconds < 60:
        return f"{seconds:.1f} s"
    clock = _fmt_clock(int(seconds + 0.5))  # half up, like the player's duration label
    return f"{clock} h" if seconds >= 3600 else f"{clock} min"


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


def build_segments(text: str, chunk_mode: str, gap_seconds: float = 0.5, carry_tags: bool = True) -> list[dict]:
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
        segments = [{"text": c, "gap": float(gap_seconds) if i else 0.0} for i, c in enumerate(chunks)]
        # Speaker turns are different voices: each turn keeps only its own tags.
        return segments if chunk_mode == "Speaker turns" or not carry_tags else carry_sentence_tags(segments)
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
    return carry_sentence_tags(segments) if carry_tags else segments


# Sentence-level Higgs V3 tags (they set the delivery of what follows). Inline ones
# (sfx, prosody pause / long_pause) only mark a moment and are never carried.
_SENTENCE_TAG = r"<\|(?:emotion|style|env|prosody:(?:speed|pitch|expressive)_[a-z_]+)(?::[a-z_]+)?\|>"


def _tag_slot(tag: str) -> str:
    """Category a sentence tag belongs to: a newer tag of the same slot replaces the old one."""
    body = tag[2:-2]
    if body.startswith("prosody:"):
        return "prosody:" + body.split(":", 1)[1].split("_", 1)[0]  # speed / pitch / expressive
    return body.split(":", 1)[0]  # emotion / style / env


def _leading_sentence_tags(text: str) -> list[str]:
    import re as _re

    match = _re.match(rf"\s*((?:{_SENTENCE_TAG}\s*)+)", text or "")
    return _re.findall(_SENTENCE_TAG, match.group(1)) if match else []


def carry_sentence_tags(segments: list[dict]) -> list[dict]:
    """Keep the intonation steady across segments.

    Each pass of the model only sees its own text, so a segment would fall back to a
    neutral delivery. The sentence-level tags written so far (the first segment's, plus
    any added later) are prepended to each following segment. Tags add up per category:
    a new emotion replaces the previous emotion, a new speed the previous speed, and so
    on, while the other categories stay. Tags typed at the start of a segment win.
    """
    import re as _re

    active: dict[str, str] = {}
    for segment in segments:
        text = segment["text"]
        own = {_tag_slot(t) for t in _leading_sentence_tags(text)}
        carried = "".join(tag for slot, tag in active.items() if slot not in own)
        if carried:
            segment["text"] = carried + text.lstrip()
            segment["carried_tags"] = carried
        for tag in _re.findall(_SENTENCE_TAG, text):
            active.pop(_tag_slot(tag), None)
            active[_tag_slot(tag)] = tag
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


def _finish_sentence(text: str) -> str:
    text = (text or "").strip()
    if text and not text.endswith((".", "!", "?", ",", ";", ":", '"', "'", "</SE_e>", "</SE>", ">")):
        return text + "."
    return text


# ---------------------------------------------------------------- dialogue builder (any number of lines)
def _new_row_id() -> str:
    import uuid

    return uuid.uuid4().hex[:8]


def dialogue_row(text="", voice="None", lora="__inherit__", audio="", ref_text="") -> dict:
    from higgs_local.montage import file_url

    audio = audio if audio and Path(str(audio)).is_file() else ""
    return {"id": _new_row_id(), "text": text or "", "voice": voice or "None", "lora": lora or "__inherit__",
            "audio": audio, "audio_url": file_url(audio) if audio else "", "audio_name": Path(audio).name if audio else "",
            "ref_text": ref_text or ""}


def _dialogue_data(value) -> dict:
    try:
        data = json.loads(value) if isinstance(value, str) else (value or {})
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("rows", [])
    return data


def dialogue_rows_of(value) -> list[dict]:
    return [r for r in _dialogue_data(value)["rows"] if isinstance(r, dict)]


def dialogue_value(value=None, rows: list[dict] | None = None, model_version: str | None = "Higgs V3 TTS",
                   default_voice: str | None = None) -> str:
    """The editor's JSON with fresh voice and LoRA lists (rows kept unless given)."""
    data = _dialogue_data(value)
    if rows is not None:
        data["rows"] = rows
    data["voices"] = list_voice_names()
    data["loras"] = [[label, val] for label, val in dialogue_lora_choices(model_version)]
    if default_voice is not None:
        data["default_voice"] = default_voice
    data["rev"] = int(data.get("rev", 0)) + 1
    return json.dumps(data, ensure_ascii=False)


def dialogue_sync(value, model_version):
    return dialogue_value(value, None, model_version)


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


#                      ---- advanced settings shared by every tab (order of ADV_KEYS = order of the UI inputs)
ADV_KEYS = [
    "asr_check", "asr_model", "language", "similarity", "similarity_min", "strictness", "takes", "retries",
    "sound_checks", "pause_same_ms", "pause_other_ms", "tidy_pauses", "level_pieces", "room_tone",
    "final_loudness", "words", "live", "keep_check_models", "enhance", "enhance_denoise",
]


def advanced_from_values(values) -> Advanced:
    data = dict(zip(ADV_KEYS, values))
    data["strictness"] = str(data.get("strictness") or "Normal")
    data["takes"] = max(1, int(data.get("takes") or 1))
    data["retries"] = max(0, int(data.get("retries") or 0))
    data["pause_same_ms"] = int(data.get("pause_same_ms") or 0)
    data["pause_other_ms"] = int(data.get("pause_other_ms") or 0)
    data["sound_checks"] = list(data.get("sound_checks") or [])
    data["words"] = list(data.get("words") or [])
    AUTOPLAY["on"] = not bool(data.get("live"))
    return Advanced(**data)


def plain_advanced() -> Advanced:
    """No checks, no cache, no live playback: what the checkpoint compare uses."""
    return Advanced(sound_checks=[], live=False, only_changed=False)


def _segmenter(gap_seconds: float = 0.5):
    def segment(text, chunk_mode, carry_tags):
        return build_segments(text, chunk_mode, float(gap_seconds), carry_tags=bool(carry_tags))

    return segment


def _render_progress(state: dict, started: float):
    now = time.perf_counter()
    elapsed = now - started
    title = state.get("title") or "Preparing"
    if state.get("phase") and state["phase"] != title:
        title = f"{title} · {state['phase']}"
    frames = int(state.get("frames", 0))
    total = max(int(state.get("total", 1)), 1)
    idx = int(state.get("index", 0))
    seg_elapsed = max(now - state.get("block_start", now), 1e-3)
    fps = frames / seg_elapsed if frames else 0.0
    est = max(50, int(state.get("block_chars", 50) * 1.8))
    block_frac = min(frames / max(est, frames + 1), 0.99) if frames else 0.0
    pct = 100.0 * (max(idx - 1, 0) + block_frac) / total if idx else None
    detail = f"Elapsed {_fmt_clock(elapsed)}"
    if frames:
        detail = f"{frames} frames · {fps:.0f} fps ({fps / 25.0:.2f}x realtime) · " + detail
    if state.get("last_take"):
        detail += f" · last take: {state['last_take']}"
    return progress_bar_html(None if pct is None else min(pct, 99.0), title, detail)


def render_job(blocks, sampling: Sampling, adv: Advanced, gap_seconds: float, label: str, renderer_box: list | None = None):
    """Generator: renders ``blocks`` on a worker thread.

    Yields ``("progress", html)`` while it works and finally ``("done", wav, results, elapsed)``.
    """
    from higgs_local.win_scheduling import configure_current_thread

    started = time.perf_counter()
    expanded, problems = expand_voice_tags(list(blocks), sampling.model_version)
    blocks[:] = expanded  # the caller's list too: its results line up with these blocks
    if problems:
        gr.Warning(f"Not found, the line keeps its own voice: {', '.join(problems)}")
        print(f"[tags] not found: {', '.join(problems)}", flush=True)
    state: dict = {"title": "Loading model, codec and LoRA", "total": len(blocks)}

    def events(ev):
        kind = ev.get("type")
        if kind == "block":
            state.update(title=f"Block {ev['index']}/{ev['total']}" + (" · reused" if ev.get("cached") else ""),
                         index=ev["index"], frames=0, block_start=time.perf_counter(),
                         block_chars=len(ev.get("text", "")), phase="Generating")
        elif kind == "frames":
            state["frames"] = ev["frames"]
        elif kind == "phase":
            state["phase"] = ev["phase"]
            state["frames"] = 0
            state["block_start"] = time.perf_counter()
        elif kind == "take":
            state["last_take"] = ev["report"]

    asr = ASR if (adv.asr_check or adv.sound_checks) else None
    renderer = Renderer(MANAGER, sampling, adv, _segmenter(gap_seconds), asr=asr, events=events)
    if renderer_box is not None:
        renderer_box.append(renderer)
    result: dict = {}

    def target():
        configure_current_thread()
        try:
            result["value"] = renderer.render(blocks, live_label=label)
        except BaseException as exc:
            result["error"] = exc

    worker = threading.Thread(target=target, name="higgs-render", daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.3)
            yield ("progress", _render_progress(state, started))
    finally:
        if worker.is_alive():  # the UI cancelled: stop the GPU loop too
            renderer.cancel()
            worker.join(timeout=30)
        if not adv.keep_check_models:
            tc.EMBEDDER.unload()
    if "error" in result:
        raise result["error"]
    wav, results = result["value"]
    yield ("done", wav, results, time.perf_counter() - started)


VOICE_TAG_RE = re.compile(r"\{\s*(voice|lora|pause)\s*:\s*([^{}]*?)\s*\}", re.IGNORECASE)
NO_VOICE_WORDS = {"", "none", "off", "base", "default"}
TRAILING_TAGS_RE = re.compile(r"(?:\s*(?:<\|[^|>]*\|>|\[[^\]]*\]|</?SE[^>]*>))+\s*$")


def voice_tag_options(*args):
    """Server function for the @ autocomplete: the library voices and the LoRA projects of the model."""
    from higgs_local.live import _server_args

    (model,) = _server_args(args, 1)
    kind = _lora_kind(model or "Higgs V3 TTS")
    return {"voices": [v for v in list_voice_names() if v != "None"],
            "loras": [p for p in lora_projects(kind) if p and p != NONE_ADAPTER]}


def _lora_from_tag(name: str, model_version: str | None):
    """A LoRA named in a tag: a project (its final adapter), a checkpoint label or value; None when unknown."""
    n = (name or "").strip().lower()
    if n in NO_VOICE_WORDS:
        return NONE_ADAPTER
    kind = _lora_kind(model_version)
    for project in lora_projects(kind):
        if str(project).lower() == n:
            checkpoints = lora_checkpoints(kind, project)
            return checkpoints[0][1] if checkpoints else None
    for label, val in dialogue_lora_choices(model_version):
        if n in (str(label).lower(), str(val).lower()) and val != DIALOGUE_INHERIT:
            return val
    return None


def pause_seconds(value) -> float | None:
    """``800``, ``800ms``, ``1.5s`` → seconds (0–30); None when it is not a duration."""
    s = str(value if value is not None else "").strip().lower().replace(",", ".")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s)?", s)
    if not m:
        return None
    n = float(m.group(1))
    return max(0.0, min(30.0, n if m.group(2) == "s" else n / 1000.0))


def split_voice_runs(text: str) -> list[tuple[str, str | None, str | None, float | None]]:
    """``(text, voice, lora, pause)`` runs of a text with ``{voice:name}``, ``{lora:name}`` and ``{pause:ms}`` tags.

    A voice or LoRA tag holds from where it stands to the end of the text (or the next tag); None = the block's
    own. A pause tag is the silence before the words that follow it (None = the usual pause). Expression tags
    written just before a tag go with the words after it."""
    runs: list[tuple[str, str | None, str | None, float | None]] = []
    voice = lora = pause = None
    pos = 0
    carry = ""
    for m in VOICE_TAG_RE.finditer(text or ""):
        chunk = carry + text[pos:m.start()]
        if tc.visible(chunk).strip():
            trailing = TRAILING_TAGS_RE.search(chunk)  # expression tags written before the tag go with its words
            carry = trailing.group(0) if trailing else ""
            runs.append((chunk[: len(chunk) - len(carry)], voice, lora, pause))
            pause = None
        else:
            carry = chunk
        kind = m.group(1).lower()
        if kind == "voice":
            voice = m.group(2)
        elif kind == "lora":
            lora = m.group(2)
        else:
            secs = pause_seconds(m.group(2))
            pause = secs if secs is not None else pause
        pos = m.end()
    tail = carry + (text or "")[pos:]
    if tail.strip() or not runs:
        runs.append((tail, voice, lora, pause))
    return runs


def _tagged_block(base: Block, text: str, voice, lora, model_version, problems: list[str]) -> Block:
    b = Block(**{**base.__dict__, "text": text})
    if voice is not None:
        names = {v.lower(): v for v in list_voice_names() if v != "None"}
        if voice.strip().lower() in NO_VOICE_WORDS:
            b.ref_audio, b.ref_text, b.voice, b.speaker = None, "", "", "Model voice"
        elif voice.strip().lower() in names:
            name = names[voice.strip().lower()]
            audio, ref_text = _resolve_reference(None, name)
            b.ref_audio, b.ref_text, b.voice, b.speaker = audio or None, ref_text or "", name, name
        else:
            problems.append(f"voice “{voice}”")
    if lora is not None:
        value = _lora_from_tag(lora, model_version)
        if value is None:
            problems.append(f"LoRA “{lora}”")
        else:
            b.lora = value
            if voice is None or voice.strip().lower() in NO_VOICE_WORDS:
                b.speaker = adapter_project(value) if value != NONE_ADAPTER else (b.speaker or "Model voice")
    return b


def expand_voice_tags(blocks: list, model_version: str | None) -> tuple[list, list[str]]:
    """Split every block whose text switches voice or LoRA with a tag into one block per voice."""
    out: list = []
    problems: list[str] = []
    for block in blocks:
        if not VOICE_TAG_RE.search(block.text or ""):
            out.append(block)
            continue
        for k, (text, voice, lora, pause) in enumerate(split_voice_runs(block.text)):
            b = _tagged_block(block, text.strip(), voice, lora, model_version, problems)
            if pause is not None:
                b.pause = pause
            elif k:
                b.pause = None
            out.append(b)
    return out, sorted(set(problems))


def blocks_payload(results, wav_path: str, source: str, blocks: list | None = None) -> list[dict]:
    """What a finished render hands to the montage timeline: one clip per block, with what made it."""
    out = []
    for r in results:
        b = blocks[r.index - 1] if blocks and 0 < r.index <= len(blocks) else None
        out.append({"file": r.file, "text": r.text, "speaker": r.speaker, "start": round(r.start, 3),
                    "seconds": round(r.seconds, 3), "source": source,
                    "voice": getattr(b, "voice", "") or "", "lora": getattr(b, "lora", "") or "",
                    "ref_audio": getattr(b, "ref_audio", "") or "", "ref_text": getattr(b, "ref_text", "") or ""})
    return out or [{"file": wav_path, "text": "", "speaker": "", "start": 0.0, "seconds": 0.0, "source": source}]


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
    carry_tags=True,
    gap_seconds=0.5,
    adv: Advanced | None = None,
):
    adv = adv or plain_advanced()
    started = time.perf_counter()

    def out(audio=None, file=None, console=None, busy=True, bar=None, blocks=None):
        """One UI update for the 7 inference outputs (the script is not one: it stays editable while generating)."""
        return (
            gr.update() if audio is None else audio,
            gr.update() if file is None else file,
            gr.update() if console is None else console,
            set_button_busy("Generating…") if busy else restore_button("⚡ Generate Speech"),
            gr.update(interactive=busy),
            gr.update() if bar is None else bar,
            gr.update() if blocks is None else blocks,
        )

    try:
        if not (text or "").strip():
            message = "Error: text is empty."
            log(message)
            yield out(console=mini_log(message), busy=False, bar=progress_bar_html(0, "Nothing to generate: the script is empty.", "", kind="error"))
            return
        yield out(console="[ui] Preparing inference model", bar=progress_bar_html(None, "Checking model files"))
        if not adv.asr_check:
            ASR.unload()
        MANAGER.set_v2_paths(V2_DEFAULT_MODEL, V2_DEFAULT_TOKENIZER)
        MANAGER.set_lora_adapter(lora_adapter)
        ready = ensure_tts_for_flow(model_version)
        preset_audio, preset_text = _resolve_reference(ref_audio, voice_preset)
        final_ref_text = ref_text or preset_text
        voice_name = voice_preset if voice_preset and voice_preset != "None" else (Path(ref_audio).stem if ref_audio else "")
        # {voice:name} / {lora:name} tags switch the speaker from there on: one text, several voices
        blocks = []
        problems: list[str] = []
        for run_text, run_voice, run_lora, run_pause in split_voice_runs(text):
            base = Block(text="", ref_audio=preset_audio or None, ref_text=final_ref_text or "", lora=lora_adapter,
                         speaker=voice_name, voice=voice_preset if voice_preset and voice_preset != "None" else "")
            base = _tagged_block(base, "", run_voice, run_lora, model_version, problems)
            # each segment is a block: with "voice only what changed" an edited sentence is the only one voiced again
            for k, s in enumerate(build_segments(run_text, chunk_mode, float(gap_seconds), carry_tags=bool(carry_tags))):
                pause = None if (k == 0 and blocks) else float(s.get("gap", 0.0))
                if k == 0 and run_pause is not None:
                    pause = run_pause
                blocks.append(Block(**{**base.__dict__, "text": s["text"], "pause": pause}))
        if problems:
            gr.Warning(f"Not found, the text keeps the voice before it: {', '.join(sorted(set(problems)))}")
        if not blocks:
            raise ValueError("Nothing to voice: the text only has tags.")
        sampling = Sampling(
            model_version=model_version, temperature=max(float(temperature), 0.3), top_p=float(top_p), top_k=int(top_k),
            max_frames=int(max_new_tokens), seed=int(seed), reuse_seed=bool(reuse_chunk_seed), chunk_mode="None",
            carry_tags=False, ras_win_len=int(ras_win_len), ras_max_repeat=int(ras_win_max_num_repeat),
        )
        print(f"[ui] Voicing {len(blocks)} segment(s)", flush=True)
        wav = results = None
        elapsed = 0.0
        for event in render_job(blocks, sampling, adv, float(gap_seconds), "single voice"):
            if event[0] == "progress":
                yield out(bar=event[1])
            else:
                _kind, wav, results, elapsed = event
        sr = SAMPLE_RATE
        report = results_report(results)
        reference_name = voice_name or "no_reference"
        out_path = _save_output(model_version, sr, wav, text=text, reference_name=reference_name, persist=bool(save_to_outputs), adv=adv)
        if int(seed) >= 0:
            _remember_inference_seed(int(seed))
        save_label = "Saved" if save_to_outputs else "Preview temp file"
        message = f"{ready}\n{report}\n{save_label}: {out_path}"
        log(message)
        notify_done()
        seconds = len(wav) / float(sr)
        done_label = f"Done · {_fmt_duration(seconds)} of audio in {_fmt_clock(elapsed)} ({seconds / max(elapsed, 1e-6):.2f}x realtime)"
        failed = sum(1 for r in results if not r.ok)
        if failed:
            done_label += f" · {failed} block(s) kept with issues"
        yield out(audio=result_audio_value(out_path), file=out_path, console=message, busy=False,
                  bar=progress_bar_html(100, done_label, report.splitlines()[-1] if report else "", kind="done"),
                  blocks=blocks_payload(results, out_path, "Single voice", blocks))
    except Cancelled:
        message = "Generation cancelled."
        log(message)
        yield out(console=mini_log(message), busy=False, bar=progress_bar_html(0, "Cancelled", "", kind="error"))
    except Exception as exc:
        message = f"Error: {exc}"
        log(message)
        yield out(console=message, busy=False, bar=progress_bar_html(0, f"Error: {exc}", "", kind="error"))


def run_inference_ui(*values):
    """The Generate button: the 17 inference inputs, then the advanced settings (ADV_KEYS order)."""
    yield from run_inference(*values[:17], adv=advanced_from_values(values[17:]))


DIALOGUE_INHERIT = "__inherit__"


def dialogue_lora_choices(model_version: str | None) -> list[tuple[str, str]]:
    """One flat LoRA list for a dialogue turn: Single voice's, none, then every checkpoint of every run."""
    kind = _lora_kind(model_version)
    out = [("Same as Single voice", DIALOGUE_INHERIT), ("None (base model)", NONE_ADAPTER)]
    for project in lora_projects(kind):
        for label, value in lora_checkpoints(kind, project):
            out.append((f"{project} · {label.split(' · ')[0]}", value))
    return out


def run_dialogue(model_version, lora_adapter, temperature, top_p, top_k, max_new_tokens, ras_win_len,
                 ras_win_max_num_repeat, seed, reuse_chunk_seed, save_to_outputs, chunk_mode, carry_tags, dialogue_json,
                 only_changed, *values):
    """Every turn with its own voice: library voice or reference audio, and its own LoRA."""
    adv = advanced_from_values(values)
    adv.only_changed = bool(only_changed)
    started = time.perf_counter()

    def out(audio=None, file=None, console=None, busy=True, bar=None, blocks=None):
        return (
            gr.update() if audio is None else audio,
            gr.update() if file is None else file,
            gr.update() if console is None else console,
            set_button_busy("Generating dialogue…") if busy else restore_button("⚡ Generate Dialogue"),
            gr.update(interactive=busy),
            gr.update() if bar is None else bar,
            gr.update() if blocks is None else blocks,
        )

    try:
        if not adv.asr_check:
            ASR.unload()
        MANAGER.set_v2_paths(V2_DEFAULT_MODEL, V2_DEFAULT_TOKENIZER)
        yield out(console="Preparing dialogue model", bar=progress_bar_html(None, "Checking model files"))
        ready = ensure_tts_for_flow(model_version)
        rows = dialogue_rows_of(dialogue_json)
        blocks: list[Block] = []
        missing = []
        for i, row in enumerate(rows, 1):
            line = (row.get("text") or "").strip()
            if not line:
                continue
            if row.get("audio") and not Path(row["audio"]).is_file():
                missing.append(i)
                row["audio"] = ""
            if row.get("voice") not in list_voice_names():
                missing.append(i)
                row["voice"] = "None"
            audio, preset_text = _resolve_reference(row.get("audio") or None, row.get("voice"))
            lora = row.get("lora") or DIALOGUE_INHERIT
            lora = lora_adapter if lora == DIALOGUE_INHERIT else lora
            who = row.get("voice") if row.get("voice") and row.get("voice") != "None" else ""
            if not who and row.get("audio"):
                who = Path(row["audio"]).stem
            if not who and lora and lora != NONE_ADAPTER:
                who = adapter_project(lora)
            blocks.append(Block(text=_finish_sentence(line), ref_audio=audio or None,
                                ref_text=(row.get("ref_text") or preset_text or ""), lora=lora,
                                speaker=who or f"Speaker {i}", voice=row.get("voice") or "",
                                pause=pause_seconds(row.get("pause"))))
        if missing:
            gr.Warning(f"Line(s) {', '.join(map(str, sorted(set(missing))))}: a reference audio or library voice is gone; "
                       "they use the model's own voice (or the first line's) instead.")
        if not blocks:
            message = "Error: dialogue has no speaker text."
            yield out(console=mini_log(log(message)), busy=False, bar=progress_bar_html(0, message, "", kind="error"))
            return
        sampling = Sampling(
            model_version=model_version, temperature=max(float(temperature), 0.3), top_p=float(top_p), top_k=int(top_k),
            max_frames=int(max_new_tokens), seed=int(seed), reuse_seed=bool(reuse_chunk_seed), chunk_mode=chunk_mode,
            carry_tags=bool(carry_tags), ras_win_len=int(ras_win_len), ras_max_repeat=int(ras_win_max_num_repeat),
        )
        wav = results = None
        elapsed = 0.0
        for event in render_job(blocks, sampling, adv, 0.15, "dialogue"):
            if event[0] == "progress":
                yield out(bar=event[1])
            else:
                _kind, wav, results, elapsed = event
        report = results_report(results)
        refs = "_".join(dict.fromkeys(b.speaker for b in blocks)) or "no_reference"
        out_path = _save_output(model_version, SAMPLE_RATE, wav, text=" ".join(b.text for b in blocks), reference_name=refs,
                                persist=bool(save_to_outputs), adv=adv)
        notify_done()
        save_label = "Saved" if save_to_outputs else "Preview temp file"
        message = f"{ready}\nDialogue turns: {len(blocks)}\n{report}\n{save_label}: {out_path}"
        log(message)
        seconds = len(wav) / float(SAMPLE_RATE)
        done_label = f"Done · {_fmt_duration(seconds)} of dialogue in {_fmt_clock(elapsed)}"
        yield out(audio=result_audio_value(out_path), file=out_path, console=message, busy=False,
                  bar=progress_bar_html(100, done_label, report.splitlines()[-1] if report else "", kind="done"),
                  blocks=blocks_payload(results, out_path, "Dialogue", blocks))
    except Cancelled:
        yield out(console=mini_log(log("Dialogue cancelled.")), busy=False, bar=progress_bar_html(0, "Cancelled", "", kind="error"))
    except Exception as exc:
        message = f"Error: {exc}"
        log(message)
        yield out(console=message, busy=False, bar=progress_bar_html(0, f"Error: {exc}", "", kind="error"))


def refresh_voices():
    choices = list_voice_names()
    return [gr.update(choices=choices, value="None") for _ in range(2)] + [
        log(f"Voice library refreshed: {len(choices) - 1} voices.")
    ]


def save_sample(audio_path, sample_name, transcript):
    status = save_voice_sample(audio_path, sample_name, transcript)
    if status.startswith("Saved voice sample: "):
        dest = SAMPLES_DIR / status.split(": ", 1)[1].strip()
        if loudness.normalize_file(dest):
            status += f" · loudness {loudness.SETTINGS['target']:g} LUFS"
    notify_done()
    choices = list_voice_names()
    return [gr.update(choices=choices, value="None") for _ in range(2)] + [log(status)]


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


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _files_in(folder: Path, suffixes: set[str]) -> tuple[int, int]:
    n = size = 0
    if folder.exists():
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                n += 1
                size += path.stat().st_size
    return n, size


def danger_ask(kind: str, name: str | None = None):
    """The confirmation of a destructive button: (pending kind, show the modal, what will happen)."""
    if kind == "outputs":
        n, size = _files_in(OUTPUTS_DIR, AUDIO_SUFFIXES)
        text = (f"**{n} audio file(s), {size / 1e6:,.1f} MB** in `outputs/` are deleted: every generated audio, the "
                "timeline's voiced lines and the separated voices. Timeline clips that use them stop playing. "
                "**This cannot be undone.**")
    elif kind == "samples":
        n, _size = _files_in(SAMPLES_DIR, AUDIO_SUFFIXES)
        text = (f"**{n} voice(s)** of the voice library are deleted with their transcripts. "
                "**This cannot be undone.**")
    elif kind == "preset":
        if not name or not (GUI_PRESETS_DIR / f"{name}.json").is_file():
            gr.Warning("Pick a saved preset first.")
            return "", gr.update(visible=False), gr.update()
        text = f"The GUI preset **{name}** is deleted. The settings on the page stay as they are."
    else:
        return "", gr.update(visible=False), gr.update()
    return kind, gr.update(visible=True), text


def danger_run(kind: str, preset_name: str | None):
    if kind == "outputs":
        return "", gr.update(visible=False), delete_output_audios(), gr.update()
    if kind == "samples":
        return "", gr.update(visible=False), delete_reference_samples(), gr.update()
    if kind == "preset":
        return "", gr.update(visible=False), gr.update(), delete_gui_preset(preset_name)
    return "", gr.update(visible=False), gr.update(), gr.update()


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


def infer_save_to_library(audio_path, name, transcript):
    """Inference: keep the current reference (audio + transcript) as a library voice and pick it.

    Returns (Voice Library dropdown, Inference voice dropdown, status). An existing voice is never overwritten here."""
    from higgs_local.audio_utils import safe_name

    keep = gr.update(), gr.update()
    if not audio_path or not Path(str(audio_path)).is_file():
        gr.Warning("Load or record a reference audio first.")
        return *keep, "No reference audio to save."
    if not (name or "").strip():
        gr.Warning("Type a name for the voice.")
        return *keep, "Type a name for the voice."
    clean = safe_name(name, "voice")
    taken = {n.lower() for n in list_voice_names()}
    if clean.lower() in taken:
        gr.Warning(f"“{clean}” is already in the library.")
        return *keep, f"❌ **{clean}** is already in the library: pick another name (or overwrite it in 🎙️ Voice Library)."
    status = save_voice_sample(audio_path, clean, transcript)
    if not status.startswith("Saved voice sample: "):
        return *keep, f"❌ {status}"
    dest = SAMPLES_DIR / status.split(": ", 1)[1].strip()
    if loudness.normalize_file(dest):
        status += f" · loudness {loudness.SETTINGS['target']:g} LUFS"
    log(status)
    choices = list_voice_names()
    note = f"✅ Saved as **{dest.stem}** in `{_rel(dest)}`" + ("" if (transcript or "").strip() else " (no transcript)") +         ": it is now the selected library voice."
    return gr.update(choices=choices), gr.update(choices=choices, value=dest.stem), note


CLEANED_REFS_DIR = OUTPUTS_DIR / "cleaned_refs"
REF_CLEAN_ENGINES = [(label, key) for label, key in CLEAN_ENGINES if key != "off"]
# the default chain in Inference, the Voice Library and Dataset Preparation: voice, then noise, then only the lead voice
DEFAULT_SEP_CHAIN = "voc_fv7+gabox_denoise+becruily_karaoke"


def clean_reference_audio(audio_path, engine, original, progress=gr.Progress(track_tqdm=False)):
    """Separate the voice from music, noise or other voices in a reference clip (any engine or chain).

    Returns (cleaned audio or None, original path kept for ↩, status). The clip keeps its sample rate; mono.
    The player is then emptied and loaded with the cleaned file (see the wiring), so its waveform is always redrawn."""
    if not audio_path or not Path(str(audio_path)).is_file():
        gr.Warning("Load a reference audio first.")
        return None, original, "No audio to clean."
    engine = str(engine or "").strip()
    if not engine or engine == "off":
        return None, original, "Pick a separation model."
    import soundfile as sf

    from higgs_local.dataset_audio import DatasetAudioProcessor, _mono

    src = Path(str(audio_path))
    progress(0.1, desc=f"Separating the voice · {_engine_label(engine)}")
    try:
        dst, note = separate_file(src, engine)
    except Exception as exc:
        gr.Warning(f"Separation failed: {exc}")
        print(f"[clean] {src.name}: {exc}", flush=True)
        return None, original, f"❌ {exc}"
    keep = original if original and Path(str(original)).is_file() else str(src)
    return str(dst), keep, f"{note} The player now holds the clean voice, saved as `{_rel(dst)}`. ↩ brings back the original."


def _engine_label(engine: str) -> str:
    return dict((k, l) for l, k in CLEAN_ENGINES).get(engine, engine)


SEPARATE_LOCK = threading.Lock()


def separate_file(src: Path, engine: str) -> tuple[Path, str]:
    """Keep only the voice of one clip (any engine or chain ``a+b``); mono, same sample rate, in outputs/cleaned_refs."""
    import soundfile as sf

    from higgs_local.dataset_audio import DatasetAudioProcessor, _mono

    with SEPARATE_LOCK:  # one separation at a time on the GPU
        proc = DatasetAudioProcessor()
        try:
            t0 = time.time()
            audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
            clean = proc.clean(_mono(audio), int(sr), engine)
            clean = loudness.normalize(clean, int(sr))
            peak = float(np.max(np.abs(clean))) if len(clean) else 0.0
            if peak > 0.98:
                clean = clean * (0.98 / peak)
            CLEANED_REFS_DIR.mkdir(parents=True, exist_ok=True)
            dst = CLEANED_REFS_DIR / f"{src.stem[:50]}_{engine.replace('+', '-')}_{time.strftime('%H%M%S')}.wav"
            sf.write(str(dst), clean, int(sr), subtype="PCM_16")
        finally:
            proc.unload()
    note = f"✅ {_engine_label(engine)} · {len(clean) / sr:.1f} s in {time.time() - t0:.1f} s."
    print(f"[clean] {src.name}: {note} → {dst}", flush=True)
    return dst, note


def _rel(path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def separate_ref_server(*args):
    """Server function for the page editors (dialogue lines, timeline dialog): separate one reference."""
    from higgs_local.live import _server_args

    path, engine = _server_args(args, 2)
    src = Path(str(path or ""))
    if not src.is_file():
        return {"ok": False, "error": "The reference audio is gone."}
    if not engine or engine == "off":
        return {"ok": False, "error": "Pick a separation model."}
    try:
        dst, note = separate_file(src, str(engine))
    except Exception as exc:
        print(f"[clean] {src.name}: {exc}", flush=True)
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "path": str(dst.resolve()), "url": montage.file_url(str(dst)), "name": dst.name, "note": note}


def restore_reference_audio(original):
    if original and Path(str(original)).is_file():
        return original, None, "↩ Original audio back."
    return gr.update(), None, "No original to bring back."


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


DATASET_MODE_RAW = "🎙️ Raw recordings"
DATASET_MODE_READY = "📦 Ready dataset (audio + .txt)"
DATASET_MODE_YOUTUBE = "▶️ YouTube (yt-dlp)"

_DATASET_SETTING_KEYS = (
    "source_folder", "asr_model", "language", "val_split", "recursive",
    "clean_engine", "segment_voice", "min_segment_sec", "max_segment_sec",
    "source_mode", "ready_path", "ready_use_split", "ready_vad_trim", "ready_min_speech", "ready_normalize",
    "align_text", "align_drop", "youtube_urls",
)


def on_dataset_project_change(project_name):
    """Show the project card and restore the dataset settings saved for it."""
    saved = dict(load_project(project_name).get("dataset", {}))
    if "clean_engine" not in saved and "isolate_voice" in saved:
        saved["clean_engine"] = "inworld" if saved["isolate_voice"] else "off"
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


def set_live_patience(value, project_name):
    """Patience edited while this project trains: the trainer picks it up at its next eval."""
    try:
        if not TRAINING.is_running() or not project_name:
            return
        run = slugify(project_name, "")
        if run != TRAINING.run_name:
            return
        path = ROOT / "exp" / run / "live_settings.json"
        data = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        data["early_stop_patience"] = max(int(value or 0), 0)
        path.write_text(json.dumps(data), encoding="utf-8")
        print(f"[training] Patience set to {data['early_stop_patience']} for the running {run} (next eval).", flush=True)
    except Exception as exc:
        print(f"[training] Could not update live patience: {exc}", flush=True)


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
                warnings_list.append(f"Longest clip is {analysis.max_duration:.0f} s: it is trained too, but long clips need "
                                     "more VRAM (batch 1 and gradient checkpointing help).")
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


def import_dataset_choices_update(current=None):
    from higgs_local.clip_import import dataset_choices

    choices = dataset_choices()
    return gr.update(choices=choices, value=current if current else None)


def import_list_clips(folder):
    from higgs_local.clip_import import clip_choices

    choices, status = clip_choices(folder)
    return gr.update(choices=choices, value=None), status


def import_browse_folder(current):
    folder, _msg = browse_source_folder(current or str(ROOT / "data"))
    return gr.update(value=folder) if folder else gr.update()


def import_load_clip(audio_path, folder):
    """(audio, transcript, status, sample-name suggestion) for the chosen dataset clip."""
    from higgs_local.clip_import import servable_copy, text_for_clip

    if not audio_path:
        return gr.update(), gr.update(), gr.update(), gr.update()
    text = text_for_clip(audio_path, folder)
    if not text:
        return gr.update(), gr.update(), f"`{Path(audio_path).name}` has no transcript.", gr.update()
    return servable_copy(audio_path), text, f"Loaded `{Path(audio_path).name}`.", Path(audio_path).stem[:60]


def import_pick_pair(folder):
    from higgs_local.clip_import import pick_pair_dialog, servable_copy

    start = str(ROOT / folder) if folder and not os.path.isabs(str(folder)) else (folder or "")
    try:
        audio, text, status = pick_pair_dialog(start)
    except Exception as exc:
        return gr.update(), gr.update(), f"File picker error: {exc}", gr.update()
    if not audio:
        return gr.update(), gr.update(), status, gr.update()
    return servable_copy(audio), text, status, Path(audio).stem[:60]


def clip_import_block(label="📥 Import audio + transcript from a dataset"):
    """Dataset / folder / file picker that fills an audio + transcript pair. Returns its components."""
    from higgs_local.clip_import import dataset_choices

    with gr.Accordion(label, open=False):
        with gr.Row(equal_height=True):
            ds = gr.Dropdown(
                dataset_choices(),
                value=None,
                allow_custom_value=True,
                label="Higgs dataset or any folder",
                info="data/<project>/train|eval, or paste a folder path with audio + .txt pairs (metadata.jsonl also works).",
                scale=6,
            )
            folder_btn = gr.Button("📁", scale=1, min_width=48, variant="secondary")
            refresh_btn = gr.Button("🔄", scale=1, min_width=48, variant="secondary")
        clip = gr.Dropdown([], value=None, label="Clip (type to search: name, length, text)", filterable=True)
        with gr.Row(equal_height=True):
            file_btn = gr.Button("🎵 Pick an audio file (+ .txt) from anywhere…", variant="secondary", size="sm")
        status = gr.Markdown("Only clips with a transcript are listed.", elem_classes="hint")
    return ds, folder_btn, refresh_btn, clip, file_btn, status


def youtube_clip_block(label="▶️ From YouTube (or any link yt-dlp supports)"):
    """Link + From / To fields that download one part of a video as the reference audio. Returns its components."""
    with gr.Accordion(label, open=False):
        url = gr.Textbox(label="Video link", placeholder="https://www.youtube.com/watch?v=…  (a ?t=90 link starts there)",
                         lines=1)
        with gr.Row(equal_height=True):
            start = gr.Textbox(label="From (optional)", placeholder="1:20 · empty: the link's t= or the start", scale=2,
                               min_width=120)
            end = gr.Textbox(label="To (optional)", placeholder="1:45 · empty: to the end", scale=2, min_width=120)
            btn = gr.Button("⬇ Download", variant="secondary", scale=1, min_width=120)
        status = gr.Markdown("The whole video, or only From → To, as mono WAV in `downloads/references`; nothing is "
                             "cut to a fixed length. Then trim, separate and transcribe it like any other clip.",
                             elem_classes="hint")
    return url, start, end, btn, status


def youtube_clip_ui(url, start, end, progress=gr.Progress(track_tqdm=False)):
    """Download one clip into the reference player. Returns (audio, transcript, original for ↩, status, name)."""
    from higgs_local.training_utils import slugify
    from higgs_local.youtube import download_clip

    if not (url or "").strip():
        gr.Warning("Paste a video link first.")
        return gr.update(), gr.update(), gr.update(), "Paste a video link first.", gr.update()
    try:
        path, title = download_clip(url, start, end, progress=lambda f, d: progress(f, desc=d))
    except Exception as exc:
        msg = str(exc).replace("ERROR: ", "")
        gr.Warning(f"Download failed: {msg[:200]}")
        print(f"[youtube] {url}: {msg}", flush=True)
        return gr.update(), gr.update(), gr.update(), f"❌ {msg}", gr.update()
    import soundfile as sf

    seconds = sf.info(str(path)).duration
    print(f"[youtube] {title}: {seconds:.1f} s → {path}", flush=True)
    note = f"✅ **{title}** · {seconds:.1f} s, saved as `{_rel(path)}`. Transcribe it (the old transcript was cleared)."
    return str(path), "", None, note, slugify(title, fallback="youtube_voice")[:40]


def wire_youtube_clip(parts, audio_out, text_out, orig_state, name_out=None):
    url, start, end, btn, status = parts
    outputs = [audio_out, text_out, orig_state, status] + ([name_out] if name_out is not None else [])

    def run(u, s, e, progress=gr.Progress(track_tqdm=False)):
        result = youtube_clip_ui(u, s, e, progress)
        return result if name_out is not None else result[:4]

    for trigger in (btn.click, url.submit):
        trigger(run, [url, start, end], outputs, concurrency_limit=1, concurrency_id="youtube_clip")


def wire_clip_import(parts, audio_out, text_out, name_out=None):
    ds, folder_btn, refresh_btn, clip, file_btn, status = parts
    outputs = [audio_out, text_out, status] + ([name_out] if name_out is not None else [])

    def trim(fn):
        def run(*args):
            result = fn(*args)
            return result if name_out is not None else result[:3]

        return run

    ds.change(import_list_clips, ds, [clip, status], queue=False, show_progress="hidden")
    folder_btn.click(import_browse_folder, ds, ds, queue=False)
    refresh_btn.click(import_dataset_choices_update, ds, ds, queue=False)
    clip.change(trim(import_load_clip), [clip, ds], outputs, queue=False, show_progress="hidden")
    file_btn.click(trim(import_pick_pair), ds, outputs, queue=False)


def load_sample_for_ui(name, use_ref_text=True):
    audio, text = resolve_voice(name)
    return audio, text if use_ref_text else ""


def refresh_sample_dropdown():
    return gr.update(choices=list_voice_names())


def update_clips_count(text, chunk_mode):
    text = (text or "").strip()
    if not text:
        return gr.update(value="")
    runs = split_voice_runs(text)
    if chunk_mode == SEAMLESS_MODE:
        count = sum(max(len(build_segments(t, chunk_mode)), 1) for t, *_ in runs)
    elif chunk_mode == "None":
        count = len(runs)
    else:
        count = sum(max(len(text_chunks_for_ui(t, chunk_mode)), 1) for t, *_ in runs)
    voices = len(runs)
    text = VOICE_TAG_RE.sub(" ", text)
    words = len(text.split())
    est = len(text) / 14.0  # ~14 characters per second of speech
    est_label = f"{est:.0f} s" if est < 90 else f"{est / 60:.1f} min"
    parts = [f"{len(text):,} characters", f"{words:,} words", f"{count} segment{'s' if count != 1 else ''}", f"≈ {est_label} of audio"]
    if voices > 1:
        parts.insert(3, f"{voices} parts (voice / pause tags)")
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
    clean_engine="inworld",
    segment_voice=True,
    min_segment_sec=2.0,
    max_segment_sec=15.0,
    source_mode=DATASET_MODE_RAW,
    ready_path="",
    ready_use_split=True,
    ready_vad_trim=True,
    ready_min_speech=0.0,
    ready_normalize=True,
    align_text=True,
    align_drop=True,
    youtube_urls="",
    ready_auto_max=True,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        from higgs_local.dataset_audio import CleanOptions, SegmentOptions

        if clean_engine is True or clean_engine is None:  # legacy checkbox value
            clean_engine = "inworld"
        elif clean_engine is False:
            clean_engine = "off"
        clean_engine = str(clean_engine or "off")
        release_all_models("dataset build")
        if source_mode == DATASET_MODE_READY:
            from higgs_local.ready_dataset import import_ready_dataset

            if not (dataset_name or "").strip():
                raise ValueError("Type a project name for the imported dataset.")
            summary, import_report = import_ready_dataset(
                ready_path,
                dataset_name,
                val_split=float(val_split),
                language=asr_language,
                clean_options=CleanOptions(
                    engine=clean_engine,
                    vad_trim=bool(ready_vad_trim),
                    min_speech_ratio=float(ready_min_speech or 0.0),
                    normalize=bool(ready_normalize),
                    target_lufs=float(loudness.SETTINGS["target"]),
                ),
                use_metadata_split=bool(ready_use_split),
                progress=lambda fraction, desc: progress(fraction, desc=desc),
                # on (default): the dataset's own lengths are kept as they are; off: the sliders filter it
                min_sec=0.0 if ready_auto_max else float(min_segment_sec or 0.0),
                max_sec=None if ready_auto_max else float(max_segment_sec),
            )
            summary, align_report = _align_dataset_step(summary, asr_language, align_text, align_drop, progress)
            return _dataset_build_done(
                summary,
                "\n".join(part for part in (import_report, align_report) if part),
                {
                    "align_text": bool(align_text),
                    "align_drop": bool(align_drop),
                    "source_mode": DATASET_MODE_READY,
                    "ready_path": ready_path,
                    "ready_use_split": bool(ready_use_split),
                    "ready_vad_trim": bool(ready_vad_trim),
                    "ready_min_speech": float(ready_min_speech or 0.0),
                    "ready_normalize": bool(ready_normalize),
                    "ready_auto_max": bool(ready_auto_max),
                    "min_sec": float(min_segment_sec or 0.0),
                    "max_sec": float(max_segment_sec),
                    "clean_engine": clean_engine,
                    "language": asr_language,
                    "val_split": float(val_split),
                },
            )
        youtube_report = ""
        if source_mode == DATASET_MODE_YOUTUBE:
            if not (dataset_name or "").strip():
                raise ValueError("Type a project name before downloading from YouTube.")
            result = _download_youtube(dataset_name, youtube_urls, progress)
            youtube_report = _youtube_report(result)
            if not any(result.folder.glob("*.wav")):
                raise RuntimeError("Nothing was downloaded.\n" + youtube_report)
            source_folder, recursive = str(result.folder), False
        ensure_asr_for_flow(asr_model, progress)
        min_sec = float(min_segment_sec)
        max_sec = max(float(max_segment_sec), min_sec + 1.0)
        audio_options = SegmentOptions(
            isolate=clean_engine != "off", engine=clean_engine, segment=bool(segment_voice), min_sec=min_sec, max_sec=max_sec,
            target_lufs=float(loudness.SETTINGS["target"]),
        )

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
        release_all_models("dataset asr finished")
        summary, align_report = _align_dataset_step(summary, asr_language, align_text, align_drop, progress)
        return _dataset_build_done(
            summary,
            "\n".join(part for part in (youtube_report, align_report) if part),
            {
                "align_text": bool(align_text),
                "align_drop": bool(align_drop),
                "source_mode": source_mode if source_mode == DATASET_MODE_YOUTUBE else DATASET_MODE_RAW,
                "youtube_urls": youtube_urls or "",
                "source_folder": source_folder,
                "asr_model": asr_model,
                "language": asr_language,
                "val_split": float(val_split),
                "recursive": bool(recursive),
                "clean_engine": clean_engine,
                "segment_voice": bool(segment_voice),
                "min_segment_sec": float(min_segment_sec),
                "max_segment_sec": float(max_segment_sec),
            },
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
            f"### Dataset Build\nFailed: {exc}",
            gr.update(value="Build Higgs Dataset", interactive=True, variant="primary"),
            gr.update(interactive=False),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def _youtube_folder(dataset_name):
    from higgs_local.training_utils import slugify
    from higgs_local.youtube import DOWNLOADS_DIR

    return DOWNLOADS_DIR / slugify(dataset_name)


def _download_youtube(dataset_name, youtube_urls, progress):
    from higgs_local.youtube import download_audio

    return download_audio(
        youtube_urls,
        _youtube_folder(dataset_name),
        progress=lambda fraction, desc: progress(fraction, desc=desc),
    )


def _youtube_report(result):
    total = len(list(result.folder.glob("*.wav")))
    lines = [f"YouTube: {len(result.files)} new audio file(s), {total} in total in {result.folder}"]
    if result.errors:
        lines.append("Download problems:\n  " + "\n  ".join(result.errors[:10]))
    return "\n".join(lines)


def download_youtube_ui(dataset_name, youtube_urls, progress=gr.Progress(track_tqdm=False)):
    """Download only (the build also downloads whatever is missing)."""
    from higgs_local.youtube import describe_folder

    try:
        if not (dataset_name or "").strip():
            return "Type a project name first: audio is saved in `downloads/<project>`."
        result = _download_youtube(dataset_name, youtube_urls, progress)
        return _youtube_report(result).replace("\n", "  \n") + "\n\n" + describe_folder(result.folder)
    except Exception as exc:
        return f"Download failed: {exc}"


def describe_youtube_ui(dataset_name):
    from higgs_local.youtube import describe_folder

    if not (dataset_name or "").strip():
        return "Paste YouTube links (videos, playlists or channels), one per line. Add `1:20-5:00` after a link to take only that part."
    return describe_folder(_youtube_folder(dataset_name))


def _align_dataset_step(summary, language, align_text, align_drop, progress):
    """Optional Qwen3-ForcedAligner pass over the built clips: tight edges, drop text/audio mismatches."""
    if not align_text:
        return summary, ""
    from dataclasses import replace

    from higgs_local.aligner import AlignOptions, align_dataset, resolve_language

    code, supported = resolve_language(language)
    if not supported:
        return summary, f"Text alignment skipped: Qwen3-ForcedAligner does not support '{language}'."
    split_dirs = [summary.train_dir] + ([summary.eval_dir] if summary.eval_dir else [])
    report, _kept = align_dataset(
        split_dirs,
        language,
        AlignOptions(trim=True, drop_mismatch=bool(align_drop)),
        progress=lambda fraction, desc: progress(fraction, desc=desc),
    )
    from higgs_local.training_utils import write_metadata

    samples, created_from = [], []
    counts = {}
    for split_dir in split_dirs:
        meta = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
        counts[split_dir] = len(meta.get("samples") or [])
        samples.extend(meta.get("samples") or [])
        created_from = created_from or list((meta.get("dataset_info") or {}).get("created_from") or [])
    if not counts.get(summary.train_dir):
        raise RuntimeError("Text alignment dropped every train clip: the transcripts do not match the audio.")
    write_metadata(summary.dataset_dir, samples, created_from)
    total = sum(float(sample.get("duration", 0.0)) for sample in samples)
    eval_dir = summary.eval_dir if summary.eval_dir and counts.get(summary.eval_dir) else None
    summary = replace(
        summary,
        eval_dir=eval_dir,
        train_samples=counts[summary.train_dir],
        eval_samples=counts.get(summary.eval_dir, 0) if eval_dir else 0,
        total_duration=total,
        avg_duration=total / max(len(samples), 1),
    )
    return summary, "\n".join(report)


def _dataset_build_done(summary, extra_report: str, settings: dict):
    """Common finish for building from recordings and importing a ready dataset."""
    release_all_models("dataset build finished")
    project = summary.dataset_dir.name
    save_project_section(
        project,
        "dataset",
        {
            **settings,
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
    if extra_report:
        message = f"{extra_report}\n\n{message}"
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


def browse_ready_dataset(kind, current_value=""):
    """Native picker for a ready dataset: a folder or a .zip archive."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        start = current_value.strip().strip('"') if current_value else ""
        start_dir = start if start and Path(start).is_dir() else (str(Path(start).parent) if start else str(ROOT))
        if kind == "zip":
            selected = filedialog.askopenfilename(
                initialdir=start_dir, title="Choose a dataset .zip", filetypes=[("Zip archive", "*.zip"), ("All files", "*.*")]
            )
        else:
            selected = filedialog.askdirectory(initialdir=start_dir, title="Choose a dataset folder")
        root.destroy()
        if selected:
            selected = str(Path(selected))
            return selected, describe_ready_dataset(selected)
        return current_value, gr.update()
    except Exception as exc:
        return current_value, f"⚠️ Picker error: {exc}"


def describe_ready_dataset(path):
    if not (path or "").strip():
        return "Enter or pick a folder / .zip with audio + `.txt` pairs (an optional `metadata.jsonl` is used when present)."
    from higgs_local.ready_dataset import describe_ready_source

    return describe_ready_source(path)


def on_dataset_mode_change(mode, engine):
    ready = mode == DATASET_MODE_READY
    youtube = mode == DATASET_MODE_YOUTUBE
    # Ready datasets are usually clean already: default their clean-up to off.
    new_engine = engine
    if ready and engine == "inworld":
        new_engine = "off"
    elif not ready and engine == "off":
        new_engine = "inworld"
    return (
        gr.update(visible=not ready and not youtube),
        gr.update(visible=ready),
        gr.update(visible=not ready),
        gr.update(visible=ready),
        gr.update(visible=not ready),
        gr.update(value=new_engine),
        gr.update(visible=youtube),
    )


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
    v3_max_clip=0,
    v3_optimizer="adamw",
    v3_prodigy_dcoef=1.0,
    v3_patience=3,
    v3_min_delta=0.002,
    v3_restore_best=True,
    v3_max_batch_sec=40,
    v3_best_only=False,
    logging_epochs=0.0,
    save_epochs=0.0,
    eval_epochs=0.0,
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
                "best_only": bool(v3_best_only),
                "logging_epochs": float(logging_epochs or 0.0),
                "save_epochs": float(save_epochs or 0.0),
                "eval_epochs": float(eval_epochs or 0.0),
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
                best_only=bool(v3_best_only),
                logging_epochs=float(logging_epochs or 0.0),
                save_epochs=float(save_epochs or 0.0),
                eval_epochs=float(eval_epochs or 0.0),
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


TAG_HINTS = {
    "emotion": "sentence start",
    "prosody": "sentence start",
    "style": "sentence start",
    "env": "sentence start",
    "sfx": "inline · sound word right after",
}
SFX_SOUNDS = {
    "cough": ["Ahem"], "laughter": ["Haha", "Hehe"], "crying": ["Boohoo", "Sob"],
    "screaming": ["Ahh", "Aaah"], "burping": ["Burp"], "humming": ["Hmm", "Mmm"],
    "sigh": ["Uh", "Ahh"], "sniff": ["Sff"], "sneeze": ["Achoo"],
}
V2_EVENT_TAGS = [
    ("[laugh]", "instant"), ("[music]", "instant"), ("[applause]", "instant"), ("[cheering]", "instant"),
    ("[cough]", "instant"), ("[humming start]", "start"), ("[humming end]", "end"), ("[music start]", "start"),
    ("[music end]", "end"), ("[sing start]", "start"), ("[sing end]", "end"),
]


def tag_autocomplete_catalog() -> dict:
    """Tags offered by the @ autocomplete, per model (same lists as the reference accordion)."""
    v3 = []
    for category, values in TAG_CATALOG.items():
        for value in values:
            tag = f"<|{category}:{value}|>"
            hint = TAG_HINTS.get(category, "")
            if category == "prosody" and value in ("pause", "long_pause"):
                hint = "inline"
            if category == "sfx":
                for sound in SFX_SOUNDS.get(value, [""]):
                    v3.append({"label": tag + sound, "insert": tag + sound, "cat": category, "hint": hint})
            else:
                v3.append({"label": tag, "insert": tag, "cat": category, "hint": hint})
    v2 = [{"label": tag, "insert": tag, "cat": "event", "hint": kind} for tag, kind in V2_EVENT_TAGS]
    return {"Higgs V3 TTS": v3, "Higgs V2 TTS": v2}


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

**Tags across segments:** each segment is generated on its own, so sentence-level V3 tags (emotion, style, prosody speed / pitch / expressive) are carried to the next segments to keep the same delivery (turn it off with **Carry tags across segments**). New tags add up: a new emotion replaces the previous emotion (same for style, speed, pitch and expressiveness) while the other tags stay. Sound effects and pauses are never repeated. Speaker turns keep only their own tags.
"""


# ================================================================ LLM, director, podcast, fixer, montage
LLM_KEYS = ["provider", "local_model", "url", "api_key", "remote_model", "ctx", "temperature", "free_tts"]
LLM_STATE = {"free_tts": True}


def llm_config_from(values) -> llm.LLMConfig:
    data = dict(zip(LLM_KEYS, values))
    LLM_STATE["free_tts"] = bool(data.get("free_tts", True))
    return llm.LLMConfig(
        provider=data.get("provider") or "Local (llama.cpp)",
        local_model=data.get("local_model") or llm.default_local_choice(),
        url=data.get("url") or "",
        api_key=data.get("api_key") or "",
        remote_model=data.get("remote_model") or "",
        ctx=int(data.get("ctx") or 16384),
        temperature=float(data.get("temperature") or 0.7),
        free_tts=bool(data.get("free_tts", True)),
    )


def _free_tts_for_llm():
    """A local LLM is about to load: the speech and check models leave the GPU first when asked."""
    if LLM_STATE.get("free_tts"):
        print("[llm] freeing the speech and check models before the local LLM loads", flush=True)
        MANAGER.unload_all()
        ASR.unload()
        tc.EMBEDDER.unload()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _stop_llm_for_tts():
    if llm.SERVER.running and LLM_STATE.get("free_tts"):
        print("[llm] stopping the local LLM before voicing (frees its VRAM)", flush=True)
        llm.SERVER.stop()


def llm_save_settings(*values):
    data = dict(zip(LLM_KEYS, values))
    data.pop("api_key", None)  # the key stays in memory only
    try:
        llm.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        llm.SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    LLM_STATE["free_tts"] = bool(data.get("free_tts", True))


def llm_load_settings() -> dict:
    try:
        return json.loads(llm.SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def llm_check(provider, url, api_key, local_model):
    if str(provider).startswith("Local"):
        installed = llm.llama_installed()
        m = llm.LOCAL_BY_LABEL.get(local_model)
        have = (m is None and Path(str(local_model)).is_file()) or (m is not None and llm.model_installed(m))
        state = "running" if llm.SERVER.running else "not loaded"
        return (
            f"Local llama.cpp: engine {'installed' if installed else 'not installed (downloaded on first use, 590 MB)'} · "
            f"model {'on disk' if have else 'not downloaded yet'} · server {state}.",
            gr.update(),
        )
    ok, message, models = llm.check_remote(url, api_key)
    return ("✅ " if ok else "❌ ") + message, gr.update(choices=models, value=models[0] if models else None)


def llm_download(local_model, progress=gr.Progress(track_tqdm=False)):
    try:
        def prog(frac, desc):
            progress(frac if frac is not None else 0.0, desc=desc)

        llm.ensure_llama(prog)
        path = llm.ensure_local_model(local_model, prog)
        notify_done()
        return f"✅ Ready: {path.name}", gr.update(choices=llm.local_choices(), value=local_model)
    except Exception as exc:
        return f"❌ {exc}", gr.update()


def llm_unload():
    return llm.SERVER.stop()


def tag_rules_v3() -> tuple[str, set[str]]:
    tags = []
    for category, values in TAG_CATALOG.items():
        for value in values:
            tag = f"<|{category}:{value}|>"
            if category == "sfx":
                tags.append(f"{tag}{SFX_SOUNDS.get(value, [''])[0]} (inline, sound word after)")
            elif category == "prosody" and value in ("pause", "long_pause"):
                tags.append(f"{tag} (inline)")
            else:
                tags.append(tag)
    known = {f"<|{c}:{v}|>" for c, vs in TAG_CATALOG.items() for v in vs}
    return ", ".join(tags), known


DIRECTOR_OPS = {
    "review": "Review the script",
    "tags": "Add tags (director)",
    "punctuate": "Punctuate for pauses (director)",
    "continue": "Continue the script (director)",
    "split": "Split direct speech",
    "accents": "Accents and ё",
}


def _llm_lang(language: str) -> str:
    from higgs_local.asr import WHISPER_LANGS as _W

    return _W.get(language) or (language if language in PODCAST_LANGS else "auto")


def director_lines(op: str, lines: list[dict], instruction: str, language: str, cfg: llm.LLMConfig, progress=None):
    """Apply a text operation to lines [{speaker, text}]: the new texts and how many lines kept their text."""
    lang = _llm_lang(language)
    payload = [{"id": str(i), "speaker": ln.get("speaker", ""), "text": ln["text"]} for i, ln in enumerate(lines)]
    if op == "tags":
        rules, known = tag_rules_v3()
        return llm.add_tags(cfg, payload, lang, rules, known, progress, _free_tts_for_llm)
    if op == "punctuate":
        return llm.punctuate(cfg, payload, lang, progress, _free_tts_for_llm)
    if op == "accents":
        return llm.accents(cfg, payload, lang, progress, _free_tts_for_llm)
    if op == "review":
        return llm.review(cfg, payload, lang, instruction, progress, _free_tts_for_llm)
    raise ValueError(op)


def _director_note(op, changed, refused, total):
    name = DIRECTOR_OPS.get(op, op)
    note = f"🧠 {name}: {changed} of {total} line(s) changed."
    if refused:
        note += f" {refused} answer(s) changed words and were not taken (those lines kept their text)."
    return note


def director_inference(op, text, instruction, language, *llm_values, progress=gr.Progress(track_tqdm=False)):
    """Inference script: paragraphs are the lines; 'continue' appends, 'split' fills the Dialogue rows."""
    cfg = llm_config_from(llm_values)
    text = text or ""
    if not text.strip() and op != "continue":
        return gr.update(), "The script is empty.", [], gr.update()
    prog = lambda f, d: progress(f or 0.0, desc=d)  # noqa: E731
    try:
        if op == "continue":
            more = llm.continue_script(cfg, [{"name": "Narrator"}], [("Narrator", p) for p in re.split(r"\n\s*\n", text) if p.strip()] or [("Narrator", "")],
                                       _llm_lang(language), 1.0, instruction, tag_rules_v3()[0], prog, _free_tts_for_llm)
            added = "\n\n".join(ln["text"] for ln in more)
            return (text.rstrip() + "\n\n" + added).strip(), f"🧠 Continued with {len(more)} paragraph(s).", [], gr.update()
        if op == "split":
            lines = llm.split_direct_speech(cfg, text, "Narrator", _llm_lang(language), prog, _free_tts_for_llm)
            return gr.update(), f"🧠 Split into {len(lines)} lines: sent to the Dialogue tab.", lines, gr.update(selected="tab_dialogue")
        paragraphs = [p for p in re.split(r"(\n\s*\n)", text)]
        idx = [i for i, p in enumerate(paragraphs) if p.strip() and not re.fullmatch(r"\n\s*\n", p)]
        new, refused = director_lines(op, [{"text": paragraphs[i]} for i in idx], instruction, language, cfg, prog)
        changed = sum(1 for i, t in zip(idx, new) if t != paragraphs[i])
        for i, t in zip(idx, new):
            paragraphs[i] = t
        return "".join(paragraphs), _director_note(op, changed, refused, len(idx)), [], gr.update()
    except Exception as exc:
        return gr.update(), f"❌ {exc}", [], gr.update()


def dialogue_from_lines(lines, value, model_version):
    """Lines [{speaker, text}] (a split of the script) added to the dialogue builder."""
    if not lines:
        return gr.update(), gr.update()
    rows = [r for r in dialogue_rows_of(value) if (r.get("text") or "").strip()]
    voices = list_voice_names()
    for ln in lines:
        name = ln.get("speaker") or "None"
        template = next((r for r in rows if r.get("_speaker") == name), None)
        voice = name if name in voices else (template or {}).get("voice", "None")
        row = dialogue_row(ln["text"], voice, (template or {}).get("lora", DIALOGUE_INHERIT), (template or {}).get("audio", ""),
                           (template or {}).get("ref_text", ""))
        row["_speaker"] = name
        rows.append(row)
    return dialogue_value(value, rows, model_version), f"{len(lines)} line(s) added to the dialogue."


def director_dialogue(op, instruction, language, value, model_version, *llm_values, progress=gr.Progress(track_tqdm=False)):
    cfg = llm_config_from(llm_values)
    rows = dialogue_rows_of(value)
    active = [i for i, r in enumerate(rows) if (r.get("text") or "").strip()]
    prog = lambda f, d: progress(f or 0.0, desc=d)  # noqa: E731
    try:
        names = [_row_name(r, i) for i, r in enumerate(rows)]
        if op == "continue":
            speakers = [{"name": n} for n in dict.fromkeys(names[i] for i in active)] or [{"name": "Speaker 1"}]
            more = llm.continue_script(cfg, speakers, [(names[i], rows[i]["text"]) for i in active], _llm_lang(language), 1.0,
                                       instruction, tag_rules_v3()[0], prog, _free_tts_for_llm)
            out = [rows[i] for i in active]
            for ln in more:
                template = next((rows[i] for i in reversed(active) if names[i] == ln["speaker"]), out[-1] if out else dialogue_row())
                out.append({**template, "id": _new_row_id(), "text": ln["text"]})
            return dialogue_value(value, out, model_version), f"🧠 Continued with {len(more)} line(s)."
        if op == "split":
            return gr.update(), "Split direct speech works on the Single voice script."
        new, refused = director_lines(op, [{"speaker": names[i], "text": rows[i]["text"]} for i in active], instruction, language, cfg, prog)
        changed = 0
        for i, t in zip(active, new):
            changed += t != rows[i]["text"]
            rows[i]["text"] = t
        return dialogue_value(value, rows, model_version), _director_note(op, changed, refused, len(active))
    except Exception as exc:
        return gr.update(), f"❌ {exc}"


def _row_name(row, i):
    if row.get("_speaker"):
        return row["_speaker"]
    if row.get("voice") and row["voice"] != "None":
        return row["voice"]
    if row.get("audio"):
        return Path(row["audio"]).stem
    if row.get("lora") and row["lora"] not in (NONE_ADAPTER, DIALOGUE_INHERIT):
        return adapter_project(row["lora"])
    return f"Speaker {i + 1}"


# ---------------------------------------------------------------- podcast
PODCAST_MAX_SPEAKERS = 6
PODCAST_LANGS = ["auto", "es", "en", "pt", "fr", "de", "it", "ru", "zh", "ja", "ko", "pl", "nl", "tr", "uk"]


PODCAST_SPK_FIELDS = 7  # name, manner, voice, LoRA, own reference audio, its transcript, pause before (ms)


def podcast_speakers(count, values) -> list[dict]:
    out = []
    n = PODCAST_SPK_FIELDS
    for i in range(max(1, min(int(count), PODCAST_MAX_SPEAKERS))):
        name, manner, voice, lora, audio, ref_text, pause = (list(values[i * n : i * n + n]) + [None] * n)[:n]
        audio = str(audio) if audio and Path(str(audio)).is_file() else ""
        out.append({"name": (name or f"Speaker {i + 1}").strip(), "manner": (manner or "").strip(), "voice": voice or "None",
                    "lora": lora or DIALOGUE_INHERIT, "audio": audio, "ref_text": (ref_text or "").strip(),
                    "pause": pause_seconds(pause)})
    return out


def podcast_value(value, speakers: list[dict] | None = None, blocks: list[dict] | None = None, model_version="Higgs V3 TTS") -> str:
    data = _podcast_data(value)
    if speakers is not None:
        data["speakers"] = [{"name": s["name"]} for s in speakers]
    if blocks is not None:
        data["blocks"] = blocks
    data["voices"] = list_voice_names()
    data["loras"] = [[label, val] for label, val in dialogue_lora_choices(model_version)]
    data["rev"] = int(data.get("rev", 0)) + 1
    return json.dumps(data, ensure_ascii=False)


def _podcast_data(value) -> dict:
    try:
        data = json.loads(value) if isinstance(value, str) else (value or {})
    except json.JSONDecodeError:
        data = {}
    data.setdefault("speakers", [])
    data.setdefault("blocks", [])
    return data


def _new_block(speaker, text, direction=""):
    import uuid

    return {"id": uuid.uuid4().hex[:8], "speaker": speaker, "text": text, "direction": direction, "voice": "", "lora": ""}


def podcast_sync(value, model_version, count, *speaker_values):
    return podcast_value(value, podcast_speakers(count, speaker_values), None, model_version)


def _spoken_words(texts) -> int:
    return sum(len(llm.TAG_LIKE.sub(" ", t or "").split()) for t in texts)


def _podcast_fill(cfg, speakers, lines, language, target_words, direction, rules, progress, close_at_end=True):
    """An LLM writes shorter than asked: the script is written without its ending, continued until the length is
    nearly reached, then closed by a last round (so the goodbyes come once, at the end)."""
    for _round in range(12):
        have = _spoken_words([ln["text"] for ln in lines])
        missing = target_words - have
        if not close_at_end and have >= target_words * 0.92:
            break
        closing = (missing <= max(60, target_words * 0.2)) if close_at_end else None
        progress(min(0.95, have / max(target_words, 1)),
                 desc=f"{'Closing the episode' if closing else 'Writing more'}: {have}/{target_words} words…")
        more = llm.continue_script(cfg, speakers, [(ln["speaker"], ln["text"]) for ln in lines], language,
                                   max(0.25, max(missing, 40) / llm.WORDS_PER_MINUTE), direction, rules, None,
                                   _free_tts_for_llm, closing=closing)
        if not more:
            break
        lines = lines + more
        if (closing and close_at_end) or _spoken_words([ln["text"] for ln in more]) < 10:
            break
    return lines


def podcast_write(value, topic, language, minutes, model_version, count, *values, progress=gr.Progress(track_tqdm=False)):
    sp_values, llm_values = values[: PODCAST_MAX_SPEAKERS * PODCAST_SPK_FIELDS], values[PODCAST_MAX_SPEAKERS * PODCAST_SPK_FIELDS :]
    cfg = llm_config_from(llm_values)
    speakers = podcast_speakers(count, sp_values)
    if not (topic or "").strip():
        return gr.update(), "Write the topic or a prompt for the conversation first."
    try:
        target = int(float(minutes) * llm.WORDS_PER_MINUTE)
        rules = tag_rules_v3()[0]
        lines = llm.write_podcast(cfg, topic.strip(), speakers, language, float(minutes) * 0.85, rules,
                                  lambda f, d: progress(f or 0.0, desc=d), _free_tts_for_llm, closing=False)
        lines = _podcast_fill(cfg, speakers, lines, language, target, "", rules, progress)
        blocks = [_new_block(ln["speaker"], ln["text"], ln.get("direction", "")) for ln in lines]
        words = _spoken_words([b["text"] for b in blocks])
        return (podcast_value(value, speakers, blocks, model_version),
                f"🧠 {len(blocks)} block(s), about {words} words (~{words / llm.WORDS_PER_MINUTE:.1f} min of the {float(minutes):g} asked).")
    except Exception as exc:
        return gr.update(), f"❌ {exc}"


def podcast_continue(value, direction, language, minutes, model_version, count, *values, progress=gr.Progress(track_tqdm=False)):
    sp_values, llm_values = values[: PODCAST_MAX_SPEAKERS * PODCAST_SPK_FIELDS], values[PODCAST_MAX_SPEAKERS * PODCAST_SPK_FIELDS :]
    cfg = llm_config_from(llm_values)
    speakers = podcast_speakers(count, sp_values)
    data = _podcast_data(value)
    written = [(b["speaker"], b["text"]) for b in data["blocks"] if b.get("text", "").strip()]
    if not written:
        return gr.update(), "There is no script to continue: write one first."
    try:
        rules = tag_rules_v3()[0]
        lines = llm.continue_script(cfg, speakers, written, language, float(minutes), direction or "", rules,
                                    lambda f, d: progress(f or 0.0, desc=d), _free_tts_for_llm)
        before = [{"speaker": s, "text": t} for s, t in written]
        target = _spoken_words(t for _s, t in written) + int(float(minutes) * llm.WORDS_PER_MINUTE)
        lines = _podcast_fill(cfg, speakers, before + lines, language, target, direction or "", rules, progress,
                              close_at_end=False)[len(before):]
        blocks = data["blocks"] + [_new_block(ln["speaker"], ln["text"], ln.get("direction", "")) for ln in lines]
        return podcast_value(value, speakers, blocks, model_version), f"🧠 {len(lines)} block(s) added after the last one."
    except Exception as exc:
        return gr.update(), f"❌ {exc}"


def podcast_clear(value, model_version, count, *speaker_values):
    return podcast_value(value, podcast_speakers(count, speaker_values), [], model_version), "Blocks cleared; speakers and settings stay."


def podcast_director(op, instruction, language, value, model_version, count, *values, progress=gr.Progress(track_tqdm=False)):
    sp_values, llm_values = values[: PODCAST_MAX_SPEAKERS * PODCAST_SPK_FIELDS], values[PODCAST_MAX_SPEAKERS * PODCAST_SPK_FIELDS :]
    cfg = llm_config_from(llm_values)
    data = _podcast_data(value)
    blocks = data["blocks"]
    if op == "continue":
        return podcast_continue(value, instruction, _llm_lang(language), 1.0, model_version, count, *values, progress=progress)
    if op == "split":
        return gr.update(), "Split direct speech works on the Single voice script."
    if not blocks:
        return gr.update(), "There are no blocks yet."
    try:
        new, refused = director_lines(op, [{"speaker": b["speaker"], "text": b["text"]} for b in blocks], instruction, language, cfg,
                                      lambda f, d: progress(f or 0.0, desc=d))
        changed = 0
        for b, t in zip(blocks, new):
            changed += t != b["text"]
            b["text"] = t
        return podcast_value(json.dumps(data), podcast_speakers(count, sp_values), blocks, model_version), _director_note(op, changed, refused, len(blocks))
    except Exception as exc:
        return gr.update(), f"❌ {exc}"


def podcast_to_dialogue(value, dlg_value, model_version, count, *values):
    sp_values = values[: PODCAST_MAX_SPEAKERS * PODCAST_SPK_FIELDS]
    speakers = {s["name"]: s for s in podcast_speakers(count, sp_values)}
    data = _podcast_data(value)
    rows = []
    for b in data["blocks"]:
        sp = speakers.get(b["speaker"], {})
        own = not b.get("voice") and sp.get("audio")
        row = dialogue_row(b["text"], b.get("voice") or sp.get("voice", "None"), b.get("lora") or sp.get("lora", DIALOGUE_INHERIT),
                           sp.get("audio", "") if own else "", sp.get("ref_text", "") if own else "")
        if sp.get("pause") is not None:
            row["pause"] = int(round(sp["pause"] * 1000))
        row["_speaker"] = b["speaker"]
        rows.append(row)
    if not rows:
        return gr.update(), "No blocks to send.", gr.update(), gr.update()
    return (dialogue_value(dlg_value, rows, model_version), f"Sent {len(rows)} block(s) to Dialogue.",
            gr.update(selected="tab_studio"), gr.update(selected="tab_dialogue"))


def run_podcast(value, model_version, lora_adapter, temperature, top_p, top_k, max_new_tokens, ras_win_len, ras_max_repeat,
                seed, reuse_seed, save_to_outputs, chunk_mode, carry_tags, pause_same, pause_other, only_changed, count,
                *values):
    sp_values = values[: PODCAST_MAX_SPEAKERS * PODCAST_SPK_FIELDS]
    adv = advanced_from_values(values[PODCAST_MAX_SPEAKERS * PODCAST_SPK_FIELDS :])
    adv.only_changed = bool(only_changed)
    # the podcast's own pauses: between two lines of one speaker, and when the speaker changes
    adv.pause_same_ms = max(0, int(pause_same or 0))
    adv.pause_other_ms = max(0, int(pause_other or 0))
    speakers = {s["name"]: s for s in podcast_speakers(count, sp_values)}
    data = _podcast_data(value)

    def out(audio=None, file=None, msg=None, bar=None, blocks=None, busy=True):
        return (gr.update() if audio is None else audio, gr.update() if file is None else file, gr.update() if msg is None else msg,
                gr.update() if bar is None else bar, gr.update() if blocks is None else blocks,
                set_button_busy("Voicing…") if busy else restore_button("🎙 Voice the podcast"), gr.update(interactive=busy))

    try:
        _stop_llm_for_tts()
        if not adv.asr_check:
            ASR.unload()
        blocks = []
        for b in data["blocks"]:
            text = (b.get("text") or "").strip()
            if not text:
                continue
            sp = speakers.get(b["speaker"], {"voice": "None", "lora": DIALOGUE_INHERIT})
            voice = b.get("voice") or sp.get("voice") or "None"
            lora = b.get("lora") or sp.get("lora") or DIALOGUE_INHERIT
            lora = lora_adapter if lora == DIALOGUE_INHERIT else lora
            audio, ref_text = _resolve_reference(None, voice)
            if not b.get("voice") and sp.get("audio"):  # the speaker's own reference wins over its library voice
                audio, ref_text, voice = sp["audio"], sp.get("ref_text") or "", "None"
            blocks.append(Block(text=_finish_sentence(text), ref_audio=audio or None, ref_text=ref_text or "", lora=lora,
                                speaker=b["speaker"], voice=voice if voice != "None" else "",
                                pause=sp.get("pause")))  # the speaker's own pause before its lines, or the usual
        if not blocks:
            yield out(msg="There are no blocks to voice.", busy=False)
            return
        yield out(msg="Preparing…", bar=progress_bar_html(None, "Checking model files"))
        ready = ensure_tts_for_flow(model_version)
        sampling = Sampling(model_version=model_version, temperature=max(float(temperature), 0.3), top_p=float(top_p),
                            top_k=int(top_k), max_frames=int(max_new_tokens), seed=int(seed), reuse_seed=bool(reuse_seed),
                            chunk_mode=chunk_mode, carry_tags=bool(carry_tags), ras_win_len=int(ras_win_len),
                            ras_max_repeat=int(ras_max_repeat))
        wav = results = None
        elapsed = 0.0
        for event in render_job(blocks, sampling, adv, 0.15, "podcast"):
            if event[0] == "progress":
                yield out(bar=event[1])
            else:
                _k, wav, results, elapsed = event
        report = results_report(results)
        path = _save_output("Podcast", SAMPLE_RATE, wav, text=" ".join(b.text for b in blocks), reference_name="podcast",
                            persist=bool(save_to_outputs), adv=adv)
        notify_done()
        log(f"{ready}\nPodcast: {len(blocks)} blocks\n{report}\nSaved: {path}")
        seconds = len(wav) / SAMPLE_RATE
        yield out(audio=result_audio_value(path), file=path, msg=report, busy=False,
                  bar=progress_bar_html(100, f"Done · {_fmt_duration(seconds)} of podcast in {_fmt_clock(elapsed)}", "", kind="done"),
                  blocks=blocks_payload(results, path, "Podcast", blocks))
    except Cancelled:
        yield out(msg="Cancelled.", busy=False, bar=progress_bar_html(0, "Cancelled", "", kind="error"))
    except Exception as exc:
        log(f"Error: {exc}")
        yield out(msg=f"❌ {exc}", busy=False, bar=progress_bar_html(0, f"Error: {exc}", "", kind="error"))


# ---------------------------------------------------------------- dialogue fixer
def fixer_transcribe(audio_path, use_sava, *adv_values, progress=gr.Progress(track_tqdm=False)):
    adv = advanced_from_values(adv_values)
    if not audio_path:
        return [], "", "", "Upload or record the audio to fix first."
    try:
        _stop_llm_for_tts()
        progress(0.1, desc="Transcribing with word timings")
        model = adv.asr_model if adv.asr_model in WHISPER_MODELS else DEFAULT_ASR_MODEL
        text, words, lang = ASR.transcribe_words(audio_path, model, adv.language)
        ws = fx.words_from_result(type("R", (), {"words": words})())
        if use_sava:
            progress(0.7, desc="Refining word edges with SAVA")
            from higgs_local.dataset_audio import DatasetAudioProcessor

            audio, sr = fx.load_mono(audio_path)
            proc = DatasetAudioProcessor()
            ws = fx.refine_with_sava(ws, audio, sr, proc)
            proc.unload()
        transcript = fx.transcript_text(ws)
        data = [{"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)} for w in ws]
        if data:
            data[0]["lang"] = lang  # the seams are heard later in the same language
        dur = sf_info_duration(audio_path)
        return (data, transcript, transcript, f"Transcribed {len(ws)} words ({lang}, {dur:.1f} s). Edit the text below and press Fix.")
    except Exception as exc:
        return [], "", "", f"❌ {exc}"


def sf_info_duration(path) -> float:
    import soundfile as _sf

    try:
        return float(_sf.info(path).duration)
    except Exception:
        return 0.0


def fixer_plan(words, edited):
    """What will be voiced again, shown while the transcript is edited (the spans are chosen automatically)."""
    if not words:
        return ""
    ws = [fx.Word(w["text"], float(w["start"]), float(w["end"])) for w in words]
    patches = fx.find_patches(ws, edited or "")
    if not patches:
        return "No change yet: edit the words you want said otherwise."
    rows = []
    for p in patches:
        if not p.after:
            rows.append(f"- ✂ **{p.before}** is cut out")
        else:
            rows.append(f"- 🎙 **{p.before or '…'}** → **{p.after}**")
    return f"**{len(patches)} change(s)** · each span is grown to a natural cut by itself:\n" + "\n".join(rows)


def fixer_run(audio_path, words, edited, match_loud, lora, model_version, temperature, top_p, top_k,
              max_new_tokens, ras_win_len, ras_max_repeat, seed, reuse_seed, *adv_values):
    adv = advanced_from_values(adv_values)
    adv.live = False  # the fixer never streams: the fixed recording plays when it is done
    AUTOPLAY["on"] = True
    adv.asr_check = True  # every take is heard: a wrong or repeated word is voiced again
    adv.retries = max(int(adv.retries or 0), 2)
    adv.asr_model = adv.asr_model if adv.asr_model in WHISPER_MODELS else DEFAULT_ASR_MODEL

    def out(audio=None, report=None, bar=None, blocks=None, busy=True):
        return (gr.update() if audio is None else audio, gr.update() if report is None else report,
                gr.update() if bar is None else bar, gr.update() if blocks is None else blocks,
                set_button_busy("Fixing…") if busy else restore_button("🩹 Fix the changed words"), gr.update(interactive=busy))

    try:
        if not audio_path or not words:
            yield out(report="Transcribe the recording first.", busy=False)
            return
        ws = [fx.Word(w["text"], float(w["start"]), float(w["end"])) for w in words]
        patches = fx.find_patches(ws, edited)  # spans chosen by the recording itself
        if not patches:
            yield out(report="Nothing changed in the transcript: edit the words you want said otherwise.", busy=False)
            return
        _stop_llm_for_tts()
        audio, sr = fx.load_mono(audio_path)
        total = len(audio) / sr
        FIX = fx.FIX_DIR
        FIX.mkdir(parents=True, exist_ok=True)
        blocks = []
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for k, p in enumerate(patches):
            p.start, p.end = fx.cut_points(ws, p, total, audio, sr)
            ref, ref_text = fx.reference_excerpt(audio, sr, ws, p, fx.reference_seconds(ws, p))
            ref_path = FIX / f"ref_{stamp}_{k}.wav"
            import soundfile as _sf

            _sf.write(str(ref_path), ref, sr)
            text = p.after.strip()
            blocks.append(Block(text=text or "", ref_audio=str(ref_path), ref_text=ref_text, lora=lora or NONE_ADAPTER,
                                speaker="recording", pause=0.0))
        speak = [b for b in blocks if b.text]
        yield out(report=f"{len(patches)} change(s) found; voicing {len(speak)} phrase(s)…", bar=progress_bar_html(None, "Loading the voice"))
        ensure_tts_for_flow(model_version)
        sampling = Sampling(model_version=model_version, temperature=max(float(temperature), 0.3), top_p=float(top_p),
                            top_k=int(top_k), max_frames=int(max_new_tokens), seed=int(seed), reuse_seed=bool(reuse_seed),
                            chunk_mode="None", carry_tags=False, ras_win_len=int(ras_win_len), ras_max_repeat=int(ras_max_repeat))
        adv.tidy_pauses = True  # a patch has no silence at its edges: the cut keeps the recording's own pauses
        adv.level_pieces = False
        adv.room_tone = False
        adv.final_loudness = "Off (as the model made it)"
        results = []
        if speak:
            for event in render_job(speak, sampling, adv, 0.15, "fixer"):
                if event[0] == "progress":
                    yield out(bar=event[1])
                else:
                    _k, _wav, results, _elapsed = event
        import librosa
        import soundfile as _sf

        by_block = {id(b): r for b, r in zip(speak, results)}
        edited_tokens = fx.WORD_RE.findall(edited or "")
        lang_code = (words[0].get("lang") if words and isinstance(words[0], dict) else None) or ""
        lang_name = next((n for n, c in WHISPER_LANGS.items() if c == lang_code), adv.language)
        seam_wav = FIX / f"seam_{stamp}.wav"

        def hear(clip):
            _sf.write(str(seam_wav), clip, sr)
            text, _status = ASR.transcribe_whisper_direct(str(seam_wav), adv.asr_model, lang_name)  # in process, model kept
            return fx.WORD_RE.findall(text or "")

        fixed = audio
        rows = []
        for p, b in sorted(zip(patches, blocks), key=lambda pb: -pb[0].start):
            if b.text:
                r = by_block[id(b)]
                new, nsr = _sf.read(r.file, dtype="float32")
                if new.ndim > 1:
                    new = new.mean(axis=1)
                trimmed = []
                try:  # words said beyond the text at an edge (a reference word said again) are cut off
                    _t, take_words, _lang = ASR.transcribe_words(r.file, adv.asr_model, lang_name)
                    t0, t1, trimmed = fx.trim_to_text(fx.words_from_result(type("R", (), {"words": take_words})()),
                                                      b.text.split(), len(new) / nsr)
                    log(f"[fixer] take heard: {_t} · trimmed: {trimmed or 'nothing'}")
                    if trimmed:
                        new = new[int(t0 * nsr) : int(t1 * nsr)]
                except Exception as exc:
                    log(f"[fixer] take check: {exc}")
                new = render_tidy(new, nsr)
                if nsr != sr:
                    new = librosa.resample(new, orig_sr=nsr, target_sr=sr).astype(np.float32)
                if match_loud:
                    a0, b0 = int(max(0, p.start - 3) * sr), int(min(total, p.end + 3) * sr)
                    around = np.concatenate([audio[a0 : int(p.start * sr)], audio[int(p.end * sr) : b0]])
                    new = fx.match_loudness(new, around)
                else:  # the take comes at the app's loudness: bring it to the recording's own level
                    new = fx.match_loudness(new, audio)
                report = r.report + (f" · trimmed: {' '.join(trimmed)}" if trimmed else "")
            else:
                new = np.zeros(0, dtype=np.float32)
                report = "deleted"
            fades = fx.seam_fades(ws, p)
            try:  # the cut points are heard: a word clipped or left over at a seam moves the cut
                p.start, p.end, errors = fx.best_seams(fixed, sr, ws, p, new, edited_tokens, hear, fades, log=log)
                if errors:
                    report += f" · seam: {errors} word(s) unsure"
            except Cancelled:
                raise
            except Exception as exc:
                log(f"[fixer] seam check: {exc}")
            fixed = fx.splice(fixed, sr, p.start, p.end, new, *fades)
            rows.append((p, report, len(new) / sr))
        del by_block
        out_path = OUTPUTS_DIR / f"fixed_{Path(audio_path).stem[:60]}_{stamp}.wav"
        fixed = loudness.normalize(fixed, sr)  # one gain for the whole recording: the app's loudness
        _sf.write(str(out_path), fixed, sr)
        notify_done()
        lines = ["| # | Time | Before | After | Take |", "|---|---|---|---|---|"]
        for n, (p, report, secs) in enumerate(sorted(rows, key=lambda r: r[0].start), 1):
            lines.append(f"| {n} | {p.start:.2f}–{p.end:.2f} s → {secs:.2f} s | {p.before or '—'} | {p.after or '— (removed)'} | {report} |")
        summary = "\n".join(lines) + f"\n\nSaved: `{out_path}`"
        yield out(audio=result_audio_value(str(out_path)), report=summary, busy=False,
                  bar=progress_bar_html(100, f"Done · {len(patches)} fix(es)", "", kind="done"),
                  blocks=[{"file": str(out_path), "text": edited[:120], "speaker": "Fixed", "start": 0.0,
                           "seconds": len(fixed) / sr, "source": "Fixer"}])
    except Cancelled:
        yield out(report="Cancelled.", busy=False, bar=progress_bar_html(0, "Cancelled", "", kind="error"))
    except Exception as exc:
        log(f"Error: {exc}")
        yield out(report=f"❌ {exc}", busy=False, bar=progress_bar_html(0, f"Error: {exc}", "", kind="error"))


def render_tidy(x, sr):
    from higgs_local.render import tidy as _tidy

    return _tidy(np.asarray(x, dtype=np.float32), sr)


# ---------------------------------------------------------------- montage
def montage_send(mode, blocks, value):
    new_value, note = montage.add_blocks(value, blocks or [], mode)
    return new_value, gr.update(visible=False), note, gr.update(selected="tab_montage")


def montage_export(value, name, fmt):
    try:
        path = montage.export(value, name, fmt)
        notify_done()
        return fresh_audio_value(path), path, f"Exported: `{path}`"
    except Exception as exc:
        return gr.update(), gr.update(), f"❌ {exc}"


def montage_save(value, name):
    try:
        saved = montage.save_project(name, value)
        return gr.update(choices=montage.list_saved(), value=saved), f"Saved as **{saved}** in projects/montage."
    except Exception as exc:
        return gr.update(), f"❌ {exc}"


def montage_load(name):
    if not name:
        return gr.update(), "Pick a saved montage."
    try:
        return montage.load_project(name), f"Loaded **{name}**."
    except Exception as exc:
        return gr.update(), f"❌ {exc}"


def montage_new(value):
    project = montage.empty_project()
    project["rev"] = int(montage.load_value(value).get("rev", 0)) + 1
    return montage.dump(project), "New empty timeline."


PODCAST_EDITOR_HTML = '<div class="pc-root"></div>'

PODCAST_EDITOR_CSS = r"""
.pc-root { display:flex; flex-direction:column; gap:6px; }
.pc-block { display:grid; grid-template-columns: 22px 170px 1fr 34px; gap:6px; align-items:start; padding:6px;
  border:1px solid var(--border-color-primary); border-radius:8px; background:var(--block-background-fill); }
.pc-block.drag-over { border-color:var(--color-accent); box-shadow:0 -3px 0 var(--color-accent) inset; }
.pc-block.dragging { opacity:.45; }
.pc-grip { cursor:grab; font-size:16px; opacity:.6; padding-top:6px; text-align:center; user-select:none; }
.pc-side { display:flex; flex-direction:column; gap:4px; }
.pc-side select, .pc-side input { width:100%; font-size:12px; padding:3px 4px; border-radius:5px; border:1px solid var(--border-color-primary);
  background:var(--input-background-fill); color:var(--body-text-color); }
.pc-text { width:100%; min-height:54px; resize:vertical; font-size:13px; padding:6px; border-radius:6px; border:1px solid var(--border-color-primary);
  background:var(--input-background-fill); color:var(--body-text-color); font-family:inherit; }
.pc-tools { display:flex; flex-direction:column; gap:3px; }
.pc-tools button { font-size:12px; padding:2px 0; border-radius:5px; border:1px solid var(--border-color-primary); background:transparent;
  color:var(--body-text-color); cursor:pointer; }
.pc-tools button:hover { border-color:var(--color-accent); }
.pc-n { font-size:10px; opacity:.6; text-align:center; }
.pc-empty { opacity:.65; padding:14px; border:1px dashed var(--border-color-primary); border-radius:8px; text-align:center; }
.pc-add { align-self:flex-start; font-size:12px; padding:4px 10px; border-radius:6px; border:1px solid var(--border-color-primary);
  background:transparent; color:var(--body-text-color); cursor:pointer; }
"""

PODCAST_EDITOR_JS = r"""
const root = element.querySelector('.pc-root');
let D = null, sent = null, dragId = null, timer = null;
function parse() { try { return JSON.parse(props.value || '{}'); } catch (e) { return {}; } }
function uid() { return Math.random().toString(16).slice(2, 10); }
function esc(s) { return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }
function push(now) {
  clearTimeout(timer);
  const go = () => { D.rev = (D.rev || 0) + 1; sent = JSON.stringify(D); props.value = sent; };
  if (now) go(); else timer = setTimeout(go, 400);
}
function opts(list, cur, withDefault) {
  let h = withDefault ? `<option value="">${withDefault}</option>` : '';
  for (const o of list) { const [label, val] = Array.isArray(o) ? o : [o, o]; h += `<option value="${esc(val)}"${val === cur ? ' selected' : ''}>${esc(label)}</option>`; }
  return h;
}
function render() {
  D.blocks = D.blocks || [];
  const names = (D.speakers || []).map((s) => s.name);
  if (!D.blocks.length) {
    root.innerHTML = '<div class="pc-empty">No blocks yet: write a script with the LLM, or add a block.</div><button class="pc-add" data-act="first">＋ Add a block</button>';
  } else {
    root.innerHTML = D.blocks.map((b, i) => `
      <div class="pc-block" draggable="true" data-id="${b.id}">
        <div><div class="pc-grip" title="Drag to reorder">⠿</div><div class="pc-n">${i + 1}</div></div>
        <div class="pc-side">
          <select data-f="speaker" title="Speaker">${opts(names.includes(b.speaker) ? names : [...names, b.speaker], b.speaker)}</select>
          <select data-f="voice" title="Voice for this block only">${opts((D.voices || []).filter((v) => v !== 'None'), b.voice, 'Voice: speaker’s')}</select>
          <select data-f="lora" title="LoRA for this block only">${opts((D.loras || []).filter((l) => l[1] !== '__inherit__'), b.lora, 'LoRA: speaker’s')}</select>
          <input data-f="direction" value="${esc(b.direction)}" placeholder="Direction (tone, pace)">
        </div>
        <textarea class="pc-text tag-ac" data-f="text">${esc(b.text)}</textarea>
        <div class="pc-tools">
          <button data-act="up" title="Move up">↑</button><button data-act="down" title="Move down">↓</button>
          <button data-act="add" title="Add a block below">＋</button><button data-act="dup" title="Clone">⧉</button>
          <button data-act="del" title="Delete">✕</button>
        </div>
      </div>`).join('') + '<button class="pc-add" data-act="last">＋ Add a block at the end</button>';
  }
  root.querySelectorAll('.pc-text').forEach((t) => { t.style.height = 'auto'; t.style.height = Math.max(54, t.scrollHeight + 2) + 'px'; });
}
root.addEventListener('input', (e) => {
  const el = e.target; const card = el.closest('.pc-block'); if (!card || !el.dataset.f) return;
  const b = D.blocks.find((x) => x.id === card.dataset.id); if (!b) return;
  b[el.dataset.f] = el.value;
  if (el.tagName === 'TEXTAREA') { el.style.height = 'auto'; el.style.height = Math.max(54, el.scrollHeight + 2) + 'px'; }
  push(el.tagName === 'SELECT');
});
root.addEventListener('click', (e) => {
  const act = e.target.dataset.act; if (!act) return;
  const card = e.target.closest('.pc-block'); const i = card ? D.blocks.findIndex((x) => x.id === card.dataset.id) : -1;
  const speaker = (D.speakers || [])[0]?.name || 'Speaker 1';
  if (act === 'first' || act === 'last') D.blocks.push({ id: uid(), speaker: D.blocks.at(-1)?.speaker || speaker, text: '', direction: '', voice: '', lora: '' });
  if (act === 'add') D.blocks.splice(i + 1, 0, { ...D.blocks[i], id: uid(), text: '', direction: '' });
  if (act === 'dup') D.blocks.splice(i + 1, 0, { ...D.blocks[i], id: uid() });
  if (act === 'del') D.blocks.splice(i, 1);
  if (act === 'up' && i > 0) [D.blocks[i - 1], D.blocks[i]] = [D.blocks[i], D.blocks[i - 1]];
  if (act === 'down' && i < D.blocks.length - 1) [D.blocks[i + 1], D.blocks[i]] = [D.blocks[i], D.blocks[i + 1]];
  render(); push(true);
});
root.addEventListener('dragstart', (e) => {
  const card = e.target.closest?.('.pc-block'); if (!card) return;
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') { e.preventDefault(); return; }
  dragId = card.dataset.id; card.classList.add('dragging'); e.dataTransfer.effectAllowed = 'move';
});
root.addEventListener('dragover', (e) => { const card = e.target.closest('.pc-block'); if (!card || !dragId) return; e.preventDefault();
  root.querySelectorAll('.drag-over').forEach((c) => c.classList.remove('drag-over')); card.classList.add('drag-over'); });
root.addEventListener('drop', (e) => {
  const card = e.target.closest('.pc-block'); if (!card || !dragId) return; e.preventDefault();
  const from = D.blocks.findIndex((x) => x.id === dragId); let to = D.blocks.findIndex((x) => x.id === card.dataset.id);
  const [b] = D.blocks.splice(from, 1); if (from < to) to -= 0; D.blocks.splice(to, 0, b);
  dragId = null; render(); push(true);
});
root.addEventListener('dragend', () => { dragId = null; root.querySelectorAll('.dragging,.drag-over').forEach((c) => c.classList.remove('dragging', 'drag-over')); });
function adopt() {
  if (props.value === sent && D) return;
  D = parse(); sent = props.value; render();
}
watch('value', adopt);
adopt();
"""



AUTOPLAY = {"on": False}


def result_audio_value(path: str):
    """A finished result: plays by itself when live playback was off (it already played live otherwise)."""
    return gr.update(value=str(path), visible=True, playback_position=0, autoplay=bool(AUTOPLAY["on"]))


def stop_render(label: str):
    MANAGER.v3.cancel()
    return restore_button(label), gr.update(interactive=False)


def director_bar(title: str, ops: list[str]):
    with gr.Accordion(title, open=False):
        with gr.Row(equal_height=True):
            op = gr.Dropdown([(DIRECTOR_OPS[k], k) for k in ops], value=ops[0], label="LLM operation", scale=2)
            instruction = gr.Textbox(label="Instruction (optional)", scale=4,
                                     placeholder="Review: what to fix · Continue: what comes next")
            btn = gr.Button("🧠 Run", variant="secondary", scale=1, min_width=90)
        note = gr.Markdown("", elem_classes="hint")
    return op, instruction, btn, note


class Modal:
    """A modern dialog over the page (the page blurred behind): ``with Modal("Title", "text") as m:`` then the
    buttons in ``m.actions()``. Show / hide ``m.box`` with ``gr.update(visible=…)``. The button with
    ``elem_classes=["hv-cancel"]`` is what ✕, a click outside the card and Esc press."""

    def __init__(self, title: str, text: str = ""):
        self.title, self.text = title, text

    def __enter__(self):
        import html as _html

        self.box = gr.Column(visible=False, elem_classes=["hv-modal"])
        self.box.__enter__()
        self.card = gr.Column(elem_classes=["hv-card"])
        self.card.__enter__()
        gr.HTML(f'<div class="hv-head"><div class="hv-title">{_html.escape(self.title)}</div>'
                '<button class="hv-x" type="button" title="Close (Esc)">✕</button></div>')
        self.md = gr.Markdown(self.text, elem_classes=["hv-text"])
        return self

    def actions(self):
        return gr.Row(elem_classes=["hv-actions"], equal_height=True)

    def __exit__(self, *exc):
        self.card.__exit__(*exc)
        self.box.__exit__(*exc)
        return False


def montage_send_bar():
    send = gr.Button("🎬 Send to montage", variant="secondary", size="sm")
    with Modal("🎬 Send to the montage",
               "Add these clips **after the end of the timeline**, or **start a new timeline** with only them "
               "(the current one is cleared; undo with Ctrl+Z in the timeline).") as m:
        with m.actions():
            cancel = gr.Button("Cancel", variant="secondary", elem_classes=["hv-cancel"])
            replace = gr.Button("♻ Clear and add", variant="secondary", elem_classes=["red-btn"])
            append = gr.Button("➕ Append to the timeline", variant="primary")
    note = gr.Markdown("", elem_classes="hint")
    return send, m.box, append, replace, cancel, note


def extras_status() -> str:
    rows = ["| Component | Status |", "|---|---|"]
    rows.append(f"| WeSpeaker ResNet34-LM (voice similarity check) | {'✅ installed' if tc.wespeaker_installed() else '⬇ not downloaded (26 MB)'} |")
    rows.append(f"| llama.cpp server b11900 CUDA (local LLM) | {'✅ installed' if llm.llama_installed() else '⬇ not downloaded (~590 MB)'} |")
    for m in llm.LOCAL_MODELS:
        rows.append(f"| {m.label} | {'✅ on disk' if llm.model_installed(m) else '⬇ not downloaded'} |")
    for p in llm.other_ggufs():
        rows.append(f"| {p.name} (your GGUF) | ✅ on disk |")
    from higgs_local.enhance import lava_installed

    rows.append(f"| LavaSR voice enhancer (high frequencies, 48 kHz) | {'✅ installed' if lava_installed() else '⬇ not downloaded (57 MB)'} |")
    rows.append(f"| Local LLM server | {'🟢 running (uses VRAM)' if llm.SERVER.running else '⚪ stopped'} |")
    return "\n".join(rows)


def _size_of(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def extras_paths() -> dict:
    """The extra models on disk, by a readable name."""
    from higgs_local.enhance import LAVA_DIR

    out: dict = {}
    if tc.WESPEAKER_PATH.is_file():
        out["WeSpeaker voice model"] = tc.WESPEAKER_PATH
    if llm.LLAMA_DIR.is_dir() and any(llm.LLAMA_DIR.iterdir()):
        out["llama.cpp engine (tools/llama)"] = llm.LLAMA_DIR
    for m in llm.LOCAL_MODELS:
        if llm.model_path(m).is_file():
            out[m.label] = llm.model_path(m)
    for g in llm.other_ggufs():
        out[f"{g.name} (your GGUF)"] = g
    if LAVA_DIR.is_dir():
        out["LavaSR voice enhancer"] = LAVA_DIR
    return out


def extras_choices():
    out = []
    for name, path in extras_paths().items():
        size = _size_of(path)
        out.append((f"{name} · {size / 1e9:.2f} GB" if size > 5e8 else f"{name} · {size / 1e6:.0f} MB", name))
    return out


def extras_ask(names):
    if not names:
        gr.Warning("Select the extras to delete first.")
        return gr.update(visible=False), gr.update()
    paths = extras_paths()
    lines = [f"- **{n}** · {_size_of(paths[n]) / 1e6:,.0f} MB" for n in names if n in paths]
    return gr.update(visible=True), ("These are deleted from disk:\n\n" + "\n".join(lines)
                                     + "\n\nThey download again when a feature needs them.")


def extras_delete(names):
    import shutil

    paths = extras_paths()
    done, freed = [], 0
    for name in names or []:
        path = paths.get(name)
        if path is None:
            continue
        if "llama" in name.lower() or path.suffix == ".gguf":
            llm.SERVER.stop()
        if "WeSpeaker" in name:
            tc.EMBEDDER.unload()
        if "LavaSR" in name:
            from higgs_local.enhance import ENHANCER

            ENHANCER.unload()
        size = _size_of(path)
        try:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            done.append(name)
            freed += size
        except OSError as exc:
            gr.Warning(f"{name}: {exc}")
    note = f"\n\n🗑 Deleted {len(done)} item(s), {freed / 1e9:.2f} GB freed: {', '.join(done)}." if done else ""
    return extras_status() + note, gr.update(choices=extras_choices(), value=[]), gr.update(visible=False)


def extras_download(what: str, progress=gr.Progress(track_tqdm=False)):
    prog = lambda f, d: progress(f or 0.0, desc=d)  # noqa: E731
    try:
        if what == "wespeaker":
            tc.ensure_wespeaker(prog)
        elif what == "lavasr":
            from higgs_local.enhance import ensure_lava

            ensure_lava(prog)
        else:
            llm.ensure_llama(prog)
    except Exception as exc:
        return extras_status() + f"\n\n❌ {exc}"
    return extras_status()


# ---------------------------------------------------------------- GUI presets (every setting and selection)
GUI_PRESETS_DIR = ROOT / "config" / "gui_presets"
PRESET_SKIP = re.compile(
    r"(console|status|_out$|_note$|_md$|^logs?_|_log|report|progress|_file$|result|preset|api_key|_html$|_info$|monitor|"
    r"^cmd_|^storage|^extras|_tbl$|_pick$|^montage_|confirm|_state$|_blocks$|split_lines|_words)",
)
PRESET_HTML = {"dialogue_editor", "pc_editor", "montage_tl"}


def collect_preset_components(scope: dict) -> dict:
    """The GUI's settings and selections by variable name (lists of components get name[i]); each component once."""
    kinds = (gr.Textbox, gr.Number, gr.Slider, gr.Checkbox, gr.Dropdown, gr.Radio, gr.CheckboxGroup, gr.Audio)
    out: dict = {}
    seen: set[int] = set()

    def take(name, obj):
        if id(obj) in seen or not isinstance(obj, (*kinds, gr.HTML)):
            return
        seen.add(id(obj))  # a skipped component stays skipped under any other name
        if isinstance(obj, gr.HTML):
            if name in PRESET_HTML:
                out[name] = obj
            return
        if PRESET_SKIP.search(name) or getattr(obj, "interactive", None) is False:
            return
        out[name] = obj

    loop_names = {"comp", "pick", "pn", "pm", "pv", "pl", "pa", "pt", "pp", "pse", "psb", "psu", "psn", "s", "t", "a", "lo", "rt", "row", "grp", "btn"}
    for name, obj in scope.items():
        if not name.startswith("_") and name not in loop_names and not isinstance(obj, list):
            take(name, obj)
    for name, obj in scope.items():
        if not name.startswith("_") and isinstance(obj, list):
            for i, o in enumerate(obj):
                take(f"{name}[{i}]", o)
    return out


def list_gui_presets() -> list[str]:
    return sorted(p.stem for p in GUI_PRESETS_DIR.glob("*.json")) if GUI_PRESETS_DIR.is_dir() else []


def save_gui_preset(names: list[str], name, *values):
    from higgs_local.audio_utils import safe_name

    name = safe_name((name or "").strip() or "preset", "preset")
    data = {"version": 1, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "values": {}}
    for key, value in zip(names, values):
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            data["values"][key] = value
    GUI_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    (GUI_PRESETS_DIR / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    gr.Info(f"Preset “{name}” saved: {len(data['values'])} settings.")
    return gr.update(choices=list_gui_presets(), value=name)


def _choice_values(comp) -> list:
    out = []
    for c in getattr(comp, "choices", None) or []:
        out.append(c[1] if isinstance(c, (list, tuple)) else c)
    return out


def _looks_like_path(v) -> bool:
    return isinstance(v, str) and (("/" in v or "\\" in v) and len(v) > 3)


def load_gui_preset(names: list[str], comps: list, name):
    """Every saved setting back in place; whatever is gone (a voice, a LoRA, a file) is skipped and reported."""
    if not name:
        gr.Warning("Pick a preset to load.")
        return [gr.update() for _ in names]
    path = GUI_PRESETS_DIR / f"{name}.json"
    try:
        values = json.loads(path.read_text(encoding="utf-8")).get("values", {})
    except Exception as exc:
        gr.Warning(f"The preset “{name}” could not be read: {exc}")
        return [gr.update() for _ in names]
    updates, problems = _apply_preset_values(names, comps, values)
    if problems:
        shown = "; ".join(problems[:8]) + (f" … and {len(problems) - 8} more" if len(problems) > 8 else "")
        gr.Warning(f"Preset “{name}” loaded, with things that are gone kept as they were: {shown}")
        print(f"[preset] {name}: " + " | ".join(problems), flush=True)
    else:
        gr.Info(f"Preset “{name}” loaded.")
    return updates


def _apply_preset_values(names: list[str], comps: list, values: dict) -> tuple[list, list[str]]:
    """Updates that put saved values back; whatever is gone (a voice, a LoRA, a file) is skipped and reported."""
    voices = set(list_voice_names())
    problems: list[str] = []
    updates = []
    for key, comp in zip(names, comps):
        if key not in values:
            updates.append(gr.update())
            continue
        v = values[key]
        try:
            if isinstance(comp, gr.HTML):
                v = _preset_editor_value(key, v, problems)
            elif isinstance(comp, gr.Audio):
                if v and not Path(str(v)).is_file():
                    problems.append(f"{key}: reference audio gone ({Path(str(v)).name})")
                    v = None
            elif isinstance(comp, (gr.Dropdown, gr.Radio)):
                multi = getattr(comp, "multiselect", False)
                allowed = _choice_values(comp)
                vals = v if (multi and isinstance(v, list)) else [v]
                kept = []
                for x in vals:
                    ok = (x in allowed or x in (None, "", "None") or getattr(comp, "allow_custom_value", False)
                          or (x in voices) or (_looks_like_path(x) and Path(str(x)).exists()))
                    if _looks_like_path(x) and not Path(str(x)).exists():
                        ok = False
                    if ok:
                        kept.append(x)
                    else:
                        problems.append(f"{key}: “{Path(str(x)).name if _looks_like_path(x) else x}” is no longer available")
                if multi:
                    v = kept
                elif not kept:
                    updates.append(gr.update())
                    continue
                else:
                    v = kept[0]
            elif isinstance(comp, gr.CheckboxGroup):
                allowed = _choice_values(comp)
                v = [x for x in (v or []) if x in allowed]
            updates.append(gr.update(value=v))
        except Exception as exc:  # one bad value never stops the rest
            problems.append(f"{key}: {exc}")
            updates.append(gr.update())
    return updates, problems


# ---------------------------------------------------------------- the page's settings across reloads
# Each browser tab keeps its own id (sessionStorage: it survives a reload of that tab, a new tab gets a new one).
# Its settings are saved whenever one changes; a reload of the same tab (after an app restart, say) puts them back.
# A freshly opened tab starts clean.
SESSION_DIR = ROOT / "config" / "session"
SESSION_KEEP_DAYS = 3


def _session_path(tab) -> Path | None:
    tab = str(tab or "")
    if not re.fullmatch(r"[0-9a-fA-F-]{8,40}", tab):
        return None
    return SESSION_DIR / f"{tab}.json"


def session_autosave(names: list[str], tab, *values):
    path = _session_path(tab)
    if path is None:
        return
    data = {k: v for k, v in zip(names, values) if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
    text = json.dumps({"version": 1, "values": data}, ensure_ascii=False)
    try:
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            return
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception as exc:
        print(f"[session] could not save this tab's settings: {exc}", flush=True)


def session_restore(names: list[str], comps: list, tab):
    path = _session_path(tab)
    if path is None or not path.is_file():
        return [gr.update() for _ in names]
    try:
        values = json.loads(path.read_text(encoding="utf-8")).get("values", {})
    except Exception:
        return [gr.update() for _ in names]
    updates, problems = _apply_preset_values(names, comps, values)
    if problems:
        print("[session] restored, skipping: " + " | ".join(problems), flush=True)
    gr.Info("Back where you were: this tab's settings are restored.")
    return updates


def session_cleanup():
    """Drop the saved settings of tabs not used for a few days."""
    cutoff = time.time() - SESSION_KEEP_DAYS * 86400
    for f in SESSION_DIR.glob("*.json") if SESSION_DIR.is_dir() else []:
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _preset_editor_value(key: str, value, problems: list[str]):
    """A block editor's JSON with missing reference files and voices dropped and fresh lists."""
    try:
        data = json.loads(value) if isinstance(value, str) else dict(value or {})
    except Exception:
        return gr.update()
    if key == "montage_tl":  # clips whose audio is gone are dropped
        data = montage.load_value(json.dumps(data))
        kept = []
        for c in data.get("clips", []):
            if c.get("path") and Path(c["path"]).is_file():
                kept.append(c)
            else:
                problems.append(f"timeline: “{c.get('label') or Path(str(c.get('path') or '')).name}” audio is gone")
        data["clips"] = kept
        data.pop("request", None)
        data.pop("reply", None)
        data["rev"] = int(data.get("rev", 0)) + 1000
        return montage.dump(data)
    voices = set(list_voice_names())
    rows = data.get("rows") if "rows" in data else data.get("blocks", [])
    for i, r in enumerate(rows or [], 1):
        if r.get("audio") and not Path(r["audio"]).is_file():
            problems.append(f"{key} line {i}: reference audio gone ({r.get('audio_name') or Path(r['audio']).name})")
            r["audio"] = r["audio_url"] = r["audio_name"] = ""
        if r.get("voice") and r["voice"] not in voices:
            problems.append(f"{key} line {i}: voice “{r['voice']}” is gone")
            r["voice"] = "None" if "rows" in data else ""
        lora = r.get("lora")
        if lora and lora not in (DIALOGUE_INHERIT, NONE_ADAPTER, "") and _looks_like_path(lora) and not Path(lora).exists():
            problems.append(f"{key} line {i}: LoRA gone ({Path(lora).name})")
            r["lora"] = DIALOGUE_INHERIT if "rows" in data else ""
    data["rev"] = int(data.get("rev", 0)) + 1000
    if "rows" in data:
        return dialogue_value(json.dumps(data))
    data["voices"] = list_voice_names()
    return json.dumps(data, ensure_ascii=False)


def delete_gui_preset(name):
    if name and (GUI_PRESETS_DIR / f"{name}.json").is_file():
        (GUI_PRESETS_DIR / f"{name}.json").unlink()
        gr.Info(f"Preset “{name}” deleted.")
    return gr.update(choices=list_gui_presets(), value=None)


# ---------------------------------------------------------------- montage: voice a line from the timeline
MONTAGE_CLIPS_DIR = OUTPUTS_DIR / "montage_clips"
MONTAGE_MODEL = {"version": "Higgs V3 TTS"}


def montage_options(*_args):
    """Server function for the timeline's line editor: voices, LoRAs and the LLM operations."""
    return {
        "voices": list_voice_names(),
        "loras": [[label, val] for label, val in dialogue_lora_choices(MONTAGE_MODEL["version"])],
        "ops": [[DIRECTOR_OPS[k], k] for k in ("review", "tags", "punctuate", "continue", "accents")],
    }


def montage_request(value, model_version, lora_adapter, temperature, top_p, top_k, max_new_tokens, ras_win_len, ras_max_repeat,
                    seed, reuse_seed, chunk_mode, carry_tags, *values):
    """The timeline asked for something: an LLM edit of a line, or a line voiced (new, or a clip again)."""
    n_adv = len(ADV_KEYS)
    adv = advanced_from_values(values[:n_adv])
    cfg = llm_config_from(values[n_adv:])
    MONTAGE_MODEL["version"] = model_version or "Higgs V3 TTS"
    project = montage.load_value(value)
    req = project.pop("request", None)
    project.pop("reply", None)
    if not isinstance(req, dict):
        yield gr.update(), gr.update()
        return

    def answer(bar, **reply):
        project["reply"] = {"id": req.get("id"), **reply}
        project["rev"] = int(project.get("rev", 0)) + 1
        return montage.dump(project), bar

    try:
        if req.get("kind") == "llm":
            yield gr.update(), progress_bar_html(None, "The LLM is working…")
            op, text = req.get("op"), (req.get("text") or "").strip()
            lang = adv.language
            if op == "continue":
                context = [(c.get("speaker") or "Speaker", c.get("text") or "") for c in req.get("context") or [] if c.get("text")]
                me = req.get("voice") if req.get("voice") not in (None, "", "None") else "Speaker"
                if text:
                    context.append((me, text))
                names = list(dict.fromkeys([me, *(s for s, _t in context)]))
                more = llm.continue_script(cfg, [{"name": n} for n in names], context or [(me, "")], _llm_lang(lang), 0.5,
                                           req.get("instruction") or "", tag_rules_v3()[0], None, _free_tts_for_llm)
                mine = [ln["text"] for ln in more if ln.get("speaker") == me] or [more[0]["text"]]
                new_text = (text + " " + mine[0]).strip() if text else mine[0]
                yield answer(progress_bar_html(100, "LLM done", "", kind="done"), text=new_text, note="🧠 Continued the line.")
                return
            new, refused = director_lines(op, [{"speaker": req.get("voice") or "", "text": text}], req.get("instruction") or "",
                                          lang, cfg, None)
            note = _director_note(op, int(new[0] != text), refused, 1)
            yield answer(progress_bar_html(100, "LLM done", "", kind="done"), text=new[0], note=note)
            return

        # ---- voice a line
        text = (req.get("text") or "").strip()
        voice = req.get("voice") or "None"
        if voice not in list_voice_names():
            voice = "None"
        lora = req.get("lora") or DIALOGUE_INHERIT
        lora_used = lora_adapter if lora == DIALOGUE_INHERIT else lora
        if lora_used and lora_used != NONE_ADAPTER and not Path(str(lora_used)).exists():
            yield answer(gr.update(), error=f"The LoRA is gone: {lora_used}")
            return
        _stop_llm_for_tts()
        audio, ref_text = _resolve_reference(None, voice)
        own = str(req.get("ref_audio") or "")
        if own and not Path(own).is_file():
            yield answer(gr.update(), error="The reference audio is gone: upload it again.")
            return
        if own:  # the line's own reference audio wins over the library voice
            audio, ref_text, voice = own, (req.get("ref_text") or "").strip(), "None"
        old = next((c for c in project["clips"] if c["id"] == req.get("clip_id")), None)
        speaker = voice if voice != "None" else (Path(own).stem if own else (old or {}).get("speaker") or "Voice")
        block = Block(text=_finish_sentence(text), ref_audio=audio or None, ref_text=ref_text or "", lora=lora_used,
                      speaker=speaker, voice=voice if voice != "None" else "")
        yield gr.update(), progress_bar_html(None, "Loading the voice")
        ensure_tts_for_flow(model_version)
        sampling = Sampling(model_version=model_version, temperature=max(float(temperature), 0.3), top_p=float(top_p),
                            top_k=int(top_k), max_frames=int(max_new_tokens), seed=int(seed), reuse_seed=bool(reuse_seed),
                            chunk_mode=chunk_mode, carry_tags=bool(carry_tags), ras_win_len=int(ras_win_len),
                            ras_max_repeat=int(ras_max_repeat))
        adv.only_changed = False  # "voice it again" means a new take
        results = []
        wav = None
        for event in render_job([block], sampling, adv, 0.15, "montage"):
            if event[0] == "progress":
                yield gr.update(), event[1]
            else:
                _k, wav, results, _elapsed = event
        MONTAGE_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        path = MONTAGE_CLIPS_DIR / f"line_{time.strftime('%Y%m%d_%H%M%S')}_{safe_name(text[:40], 'line')}.wav"
        sr = SAMPLE_RATE
        if adv.enhance:
            from higgs_local.enhance import ENHANCER

            wav, sr = ENHANCER.enhance(wav, SAMPLE_RATE, denoise=adv.enhance_denoise)
        wav = loudness.normalize(wav, sr)
        import soundfile as _sf

        _sf.write(str(path), np.asarray(wav, dtype=np.float32), sr)
        dur = round(len(wav) / sr, 4)
        info = {"path": str(path.resolve()), "url": montage.file_url(str(path)), "label": text[:120], "offset": 0.0, "dur": dur,
                "src_dur": dur, "text": text, "voice": block.voice, "lora": lora, "speaker": speaker, "source": "timeline",
                "ref_audio": own, "ref_text": (req.get("ref_text") or "") if own else ""}
        report = results[0].report if results else ""
        if old is not None:
            delta = dur - float(old["dur"])
            old_end = float(old["start"]) + float(old["dur"])
            old.update(info)
            if req.get("ripple") and abs(delta) > 1e-3:
                for c in project["clips"]:
                    if c is not old and c["track"] == old["track"] and float(c["start"]) >= old_end - 1e-3:
                        c["start"] = round(max(0.0, float(c["start"]) + delta), 4)
            note = f"Voiced again ({dur:.2f} s) · {report}"
        else:
            track = req.get("track")
            if not track or not any(t["id"] == track for t in project["tracks"]):
                track = montage._voice_track(project, speaker)
            project["clips"].append({"id": montage._id("c"), "track": track, "start": round(float(req.get("at") or 0.0), 4),
                                     "gain": 0.0, "fade_in": 0.0, "fade_out": 0.0, **info})
            note = f"Line added at {float(req.get('at') or 0):.2f} s ({dur:.2f} s) · {report}"
        notify_done()
        yield answer(progress_bar_html(100, note, "", kind="done"), done=True, note=note)
    except Cancelled:
        yield answer(progress_bar_html(0, "Cancelled", "", kind="error"), error="Cancelled.")
    except Exception as exc:
        log(f"Error: {exc}")
        yield answer(progress_bar_html(0, f"Error: {exc}", "", kind="error"), error=str(exc))


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
    MANAGER.set_ref_max_seconds(ref_max_default())
    session_cleanup()

    with gr.Blocks(title=APP_TITLE, fill_width=True) as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.HTML(header_html("Higgs Audio Easy GUI"))
        with gr.Row(elem_classes="header-actions"):
            top_unload_btn = gr.Button("⏏ Unload models", size="sm", variant="secondary", scale=0)
            gr.Button("📖 Higgs Audio docs", size="sm", variant="secondary", scale=0, link="https://github.com/boson-ai/higgs-audio")
            gr.HTML(LIVE_PLAYER_HTML, css_template=LIVE_PLAYER_CSS,
                    js_on_load=LIVE_PLAYER_JS + "\nwindow.__hvTagOptions = (m) => server.voice_tag_options(m);",
                    server_functions=[live_poll, live_claim, voice_tag_options],
                    elem_id="live-player", scale=0, min_width=220)
            top_live = gr.Checkbox(value=True, label="📡 Live", container=False, scale=0, min_width=90,
                                   elem_classes=["top-live"],
                                   info=None)
            gui_preset_select = gr.Dropdown(list_gui_presets(), value=None, allow_custom_value=True, show_label=False,
                                            container=False, scale=0, min_width=220, elem_classes=["gui-preset-select"],
                                            info=None, label="GUI preset")
            gui_preset_load_btn = gr.Button("📂 Load preset", size="sm", variant="secondary", scale=0)
            gui_preset_save_btn = gr.Button("💾 Save preset", size="sm", variant="secondary", scale=0)
            gui_preset_delete_btn = gr.Button("🗑", size="sm", variant="secondary", scale=0, elem_classes=["red-btn"])
        with Modal("⚠️ Are you sure?") as danger:
            with danger.actions():
                danger_cancel = gr.Button("Cancel", variant="secondary", elem_classes=["hv-cancel"])
                danger_ok = gr.Button("🗑 Yes, delete", variant="stop")
        danger_kind = gr.State("")

        with gr.Tabs(elem_classes="main-tabs", selected="tab_studio") as main_tabs:
            # ================================================================ STUDIO
            with gr.Tab("🎙️ Inference", id="tab_studio"):
                with gr.Tabs(selected="tab_single") as studio_tabs:
                    with gr.Tab("Single voice", id="tab_single"):
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
                                            "Tip: type @ anywhere to insert a tag (emotion, style, prosody, sound effect) "
                                            "supported by the selected model, a library voice {voice:name} or a LoRA "
                                            "{lora:name}: several speakers in one text. {pause:800} sets the silence "
                                            "before what follows."
                                        ),
                                        lines=10,
                                        max_lines=30,
                                        value="",
                                        elem_classes=["script-box", "tag-ac"],
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
                                    infer_dir_op, infer_dir_instr, infer_dir_btn, infer_dir_note = director_bar(
                                        "🧠 LLM director: review, tags, pauses, continue, split direct speech, accents",
                                        ["review", "tags", "punctuate", "continue", "split", "accents"])
                                    infer_split_lines = gr.State([])
                                with gr.Column(elem_classes=["card", "output-card"]):
                                    gr.HTML(section("Output"))
                                    infer_progress = gr.HTML(IDLE_PROGRESS_HTML, elem_classes=["progress-card"])
                                    infer_blocks = gr.State([])
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
                                    infer_send = montage_send_bar()
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
                                    elem_id="infer-model-select",
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
                                infer_import_parts = clip_import_block("📥 Use a clip from a dataset or folder (audio + .txt)")
                                infer_youtube_parts = youtube_clip_block()
                                infer_ref_audio = gr.Audio(
                                    label="Reference audio · 5–30 s works best",
                                    type="filepath",
                                    value=None,
                                    sources=["upload", "microphone"],
                                    elem_classes=["audio-safe-space"],
                                )
                                with gr.Accordion("🎧 Separate the voice (music, noise, other voices)", open=False):
                                    with gr.Row(equal_height=True):
                                        infer_clean_engine = gr.Dropdown(REF_CLEAN_ENGINES, value=DEFAULT_SEP_CHAIN,
                                                                          label="Separation model or chain", scale=5)
                                        infer_clean_btn = gr.Button("🎧 Separate", variant="secondary", scale=1, min_width=110)
                                        infer_clean_undo = gr.Button("↩ Original", variant="secondary", scale=1, min_width=100)
                                    infer_clean_note = gr.Markdown(
                                        "Keeps only the voice of the reference above. RoFormer models download once "
                                        "(~0.2–0.9 GB each); the cleaned clip replaces the reference.", elem_classes="hint")
                                    infer_clean_orig = gr.State(None)
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
                                with gr.Accordion("💾 Save to the voice library", open=False):
                                    with gr.Row(equal_height=True):
                                        infer_save_name = gr.Textbox(label="Voice name", placeholder="e.g. news_anchor_1",
                                                                     scale=4)
                                        infer_save_btn = gr.Button("💾 Save", variant="primary", scale=1, min_width=110)
                                    infer_save_note = gr.Markdown(
                                        "Saves the reference audio above with its transcript (as they are now: trimmed, "
                                        "separated) to `samples/`, and selects it.", elem_classes="hint")

                            with gr.Column(min_width=320, elem_classes=["card", "studio-settings"]):
                                gr.HTML(section("Settings"))
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

                    with gr.Tab("Dialogue", id="tab_dialogue"):
                        with gr.Column(elem_classes="card"):
                            gr.HTML(section("Multi-speaker dialogue"))
                            gr.Markdown(
                                "Add as many lines as you need. Each line has its own voice: a library voice or its own "
                                "reference audio (uploading one wins), and its own LoRA. Drag ⠿ to reorder; ＋ adds a line below "
                                "with the same voice and LoRA, ⧉ clones it, ✕ deletes it. Type @ in a line for tags. "
                                "Pauses between lines are in ⚙️ Advanced speech settings → Pauses.",
                                elem_classes="hint",
                            )
                            dialogue_lora_init = dialogue_lora_choices("Higgs V3 TTS")
                            dialogue_editor = gr.HTML(
                                dialogue_value("{}", [dialogue_row("", default_sample_name), dialogue_row("", default_sample_name)],
                                               "Higgs V3 TTS", default_sample_name),
                                html_template=DIALOGUE_EDITOR_HTML, css_template=DIALOGUE_EDITOR_CSS, js_on_load=DIALOGUE_EDITOR_JS,
                                server_functions=[separate_ref_server], elem_id="dialogue-editor",
                            )
                            with gr.Row():
                                dialogue_refresh_lora_btn = gr.Button("🔄 Refresh LoRA lists", size="sm", variant="secondary")
                            dlg_dir_op, dlg_dir_instr, dlg_dir_btn, dlg_dir_note = director_bar(
                                "🧠 LLM director: review, tags, pauses, continue, accents",
                                ["review", "tags", "punctuate", "continue", "accents"])
                            dlg_only_changed = gr.Checkbox(label="♻ Voice only what changed", value=False,
                                                           info="Every line's take is kept by its text, voice, LoRA and sampling: after an edit only the changed lines are voiced again, the rest is reused. Off: everything is voiced again.")
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
                            dialogue_progress = gr.HTML(IDLE_PROGRESS_HTML, elem_classes=["progress-card"])
                            dialogue_blocks = gr.State([])
                            dialogue_audio_out = gr.Audio(
                                label="Generated dialogue",
                                type="filepath",
                                elem_id="dialogue-generated-audio",
                                elem_classes=["audio-safe-space", "output-clean"],
                            )
                            dialogue_file_out = gr.Textbox(label="Saved WAV", interactive=False, elem_classes=["output-path"])
                            dialogue_send = montage_send_bar()
                        gr.Markdown(
                            "Uses the model and the Single voice LoRA (for turns set to *Same as Single voice*); sampling, long-text mode, checks and pauses "
                            "come from ⚙️ Advanced speech settings (tags carry between the chunks of one line; each line keeps its own tags).",
                            elem_classes="hint",
                        )

                infer_console = gr.Textbox(label="Internal Status", lines=1, interactive=False, value="Idle.", visible=False)

            # ================================================================ PODCAST
            with gr.Tab("🎙 Podcast (LLM)", id="tab_podcast"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320, elem_classes="card"):
                        gr.HTML(section("Topic & cast"))
                        pc_topic = gr.Textbox(label="Topic or prompt", lines=4,
                                              placeholder="What the conversation is about, its angle, facts to include…")
                        with gr.Row(equal_height=True):
                            pc_language = gr.Dropdown(PODCAST_LANGS, value="auto", label="Language", allow_custom_value=True, min_width=90)
                            pc_minutes = gr.Slider(0.5, 30, value=3, step=0.5, label="Length (minutes)")
                            pc_count = gr.Slider(1, PODCAST_MAX_SPEAKERS, value=2, step=1, label="Speakers")
                        pc_speaker_inputs = []
                        pc_speaker_rows = []
                        pc_sep_parts = []
                        _pc_names = ["Host", "Guest", "Speaker 3", "Speaker 4", "Speaker 5", "Speaker 6"]
                        _pc_voices = [v for v in list_voice_names() if v != "None"]
                        for i in range(PODCAST_MAX_SPEAKERS):
                            with gr.Group(visible=i < 2) as grp:
                                with gr.Row(equal_height=True):
                                    pn = gr.Textbox(value=_pc_names[i], label=f"Speaker {i + 1} · name", scale=2)
                                    pm = gr.Textbox(label="Manner / character", placeholder="curious host, warm, quick", scale=3)
                                with gr.Row(equal_height=True):
                                    pv = gr.Dropdown(list_voice_names(), value=_pc_voices[i] if i < len(_pc_voices) else "None",
                                                     label="Voice", scale=2)
                                    pl = gr.Dropdown(dialogue_lora_init, value=DIALOGUE_INHERIT, label="LoRA", scale=2)
                                    pp = gr.Number(value=None, precision=0, minimum=0, maximum=30000, scale=1, min_width=110,
                                                   label="Pause before (ms)", placeholder="auto",
                                                   info="Before each line of this speaker; empty: the podcast pauses.")
                                with gr.Accordion("🎙 Own reference audio (optional, wins over the library voice)", open=False):
                                    pa = gr.Audio(label="Reference audio · 5–30 s works best", type="filepath",
                                                  sources=["upload", "microphone"], elem_classes=["audio-safe-space"])
                                    pt = gr.Textbox(label="Reference transcript", lines=1,
                                                    placeholder="What the reference says (empty: no transcript)")
                                    with gr.Row(equal_height=True):
                                        pse = gr.Dropdown(REF_CLEAN_ENGINES, value=DEFAULT_SEP_CHAIN, scale=5,
                                                          label="🎧 Separate the voice from music, noise or other voices")
                                        psb = gr.Button("🎧 Separate", variant="secondary", scale=1, min_width=110)
                                        psu = gr.Button("↩ Original", variant="secondary", scale=1, min_width=100)
                                    psn = gr.Markdown("", elem_classes="hint")
                                    pc_sep_parts.append((pa, pse, psb, psu, gr.State(None), psn))
                            pc_speaker_inputs.extend([pn, pm, pv, pl, pa, pt, pp])
                            pc_speaker_rows.append(grp)
                        with gr.Row():
                            pc_write_btn = gr.Button("✍ Write the script", variant="primary")
                            pc_clear_btn = gr.Button("🧹 Clear blocks", variant="secondary")
                        pc_direction = gr.Textbox(label="What comes next (for Continue)",
                                                  placeholder="e.g. they disagree about the price, then wrap up")
                        with gr.Row():
                            pc_continue_btn = gr.Button("➕ Continue from the last block", variant="secondary")
                        pc_note = gr.Markdown("", elem_classes="hint")
                        pc_dir_op, pc_dir_instr, pc_dir_btn, pc_dir_note = director_bar(
                            "🧠 LLM director: review, tags, pauses, accents", ["review", "tags", "punctuate", "accents"])
                    with gr.Column(scale=7, min_width=420):
                        with gr.Column(elem_classes="card"):
                            gr.HTML(section("Script blocks"))
                            gr.Markdown("Drag ⠿ (or ↑ ↓) to reorder, edit any text, pick another speaker; a block can override its "
                                        "speaker's voice and LoRA. ＋ adds a block below, ⧉ clones it, ✕ deletes it. Type @ for tags.",
                                        elem_classes="hint")
                            pc_editor = gr.HTML(
                                podcast_value("{}", podcast_speakers(2, ["Host", "", "None", DIALOGUE_INHERIT, None, "", None,
                                                                         "Guest", "", "None", DIALOGUE_INHERIT, None, "", None]), []),
                                html_template=PODCAST_EDITOR_HTML, css_template=PODCAST_EDITOR_CSS, js_on_load=PODCAST_EDITOR_JS,
                                elem_id="podcast-editor")
                            with gr.Row(equal_height=True):
                                pc_gen_btn = gr.Button("🎙 Voice the podcast", variant="primary", size="lg", scale=5,
                                                       elem_classes="button-primary")
                                pc_stop_btn = gr.Button("⏹ Stop", variant="stop", size="lg", scale=1, min_width=96,
                                                        elem_classes="button-stop", interactive=False)
                                pc_to_dialogue_btn = gr.Button("➡ Send to Dialogue", variant="secondary", scale=1, min_width=150)
                            with gr.Row(equal_height=True):
                                pc_pause_same = gr.Number(value=300, precision=0, minimum=0, maximum=10000,
                                                          label="Pause between lines of one speaker (ms)")
                                pc_pause_other = gr.Number(value=700, precision=0, minimum=0, maximum=10000,
                                                           label="Pause when the speaker changes (ms)")
                            pc_only_changed = gr.Checkbox(label="♻ Voice only what changed", value=False,
                                                          info="Every line's take is kept by its text, voice, LoRA and sampling: after an edit only the changed lines are voiced again, the rest is reused. Off: everything is voiced again.")
                        with gr.Column(elem_classes=["card", "output-card"]):
                            gr.HTML(section("Output"))
                            pc_progress = gr.HTML(IDLE_PROGRESS_HTML, elem_classes=["progress-card"])
                            pc_blocks = gr.State([])
                            pc_audio_out = gr.Audio(label="Podcast", type="filepath", elem_classes=["audio-safe-space", "output-clean"])
                            pc_file_out = gr.Textbox(label="Saved WAV", interactive=False, elem_classes=["output-path"])
                            pc_report = gr.Markdown("")
                            pc_send = montage_send_bar()

            # ================================================================ MONTAGE
            with gr.Tab("🎬 Montage", id="tab_montage"):
                with gr.Column(elem_classes="card"):
                    gr.HTML(section("Timeline"))
                    gr.Markdown(
                        "Space plays/stops · click, Ctrl/Shift+click or drag a box to select · Del deletes · drag a clip to move it "
                        "(also to another track), drag its edges to trim · S splits at the playhead · wheel zooms, Shift+wheel "
                        "scrolls · right-click for fades, gain and more · Ctrl+Z/Y undo/redo · drop audio files to import.",
                        elem_classes="hint")
                    montage_tl = gr.HTML(montage.initial_value(), html_template=montage.MONTAGE_HTML,
                                         css_template=montage.MONTAGE_CSS, js_on_load=montage.MONTAGE_JS,
                                         server_functions=[montage.montage_probe, montage_options, separate_ref_server],
                                         elem_id="montage-timeline")
                    montage_progress = gr.HTML(IDLE_PROGRESS_HTML, elem_classes=["progress-card"])
                with gr.Row(equal_height=False):
                    with gr.Column(elem_classes="card"):
                        gr.HTML(section("Project"))
                        with gr.Row(equal_height=True):
                            montage_name = gr.Textbox(label="Name", value="montage", scale=3)
                            montage_save_btn = gr.Button("💾 Save", variant="secondary", scale=1)
                        with gr.Row(equal_height=True):
                            montage_saved = gr.Dropdown(montage.list_saved(), value=None, label="Saved montages", scale=3)
                            montage_load_btn = gr.Button("📂 Load", variant="secondary", scale=1)
                            montage_new_btn = gr.Button("🆕 New", variant="secondary", scale=1, elem_classes=["red-btn"])
                        montage_note = gr.Markdown("")
                    with gr.Column(elem_classes=["card", "output-card"]):
                        gr.HTML(section("Export"))
                        with gr.Row(equal_height=True):
                            montage_fmt = gr.Dropdown(["44.1 kHz stereo WAV", "48 kHz stereo WAV", "24 kHz mono WAV"],
                                                      value="44.1 kHz stereo WAV", label="Format", scale=3)
                            montage_export_btn = gr.Button("⬇ Export the mix", variant="primary", scale=1)
                        montage_audio = gr.Audio(label="Mix", type="filepath", elem_classes=["audio-safe-space", "output-clean"])
                        montage_file = gr.Textbox(label="Exported file", interactive=False, elem_classes=["output-path"])

            # ================================================================ FIXER
            with gr.Tab("🩹 Dialogue Fixer", id="tab_fixer"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=380, elem_classes="card"):
                        gr.HTML(section("Recording"))
                        gr.Markdown(
                            "Import a recording, transcribe it with word timings, then edit the words you want said otherwise. "
                            "Only the changed words, grown to the nearest natural cut, are voiced again with the recording "
                            "itself as the voice reference; every seam is checked by ear (Descript-style). Delete words to cut them.",
                            elem_classes="hint")
                        fx_audio = gr.Audio(label="Recording to fix", type="filepath", sources=["upload", "microphone"],
                                            elem_classes=["audio-safe-space"])
                        with gr.Row(equal_height=True):
                            fx_sava = gr.Checkbox(value=True, label="Refine word edges with SAVA (voice activity)")
                            fx_transcribe_btn = gr.Button("🎙 Transcribe with word timings", variant="secondary")
                        fx_words_state = gr.State([])
                        fx_original = gr.Textbox(label="Transcript (as recorded)", lines=3, interactive=False)
                        fx_edited = gr.Textbox(label="Edited transcript · change the words to fix", lines=5, elem_classes=["tag-ac"])
                        fx_note = gr.Markdown("", elem_classes="hint")
                        fx_plan = gr.Markdown("", elem_classes="hint")
                    with gr.Column(scale=4, min_width=320, elem_classes="card"):
                        gr.HTML(section("Fix"))
                        fx_lora = gr.Dropdown(dialogue_lora_init[1:], value=NONE_ADAPTER, label="LoRA (optional)")
                        fx_match = gr.Checkbox(value=True, label="Match loudness to the speech around")
                        gr.Markdown("Everything else is automatic: each change grows to the nearest pause or punctuation "
                                    "of the recording (the pace of this speaker sets what a pause is), the voice is "
                                    "cloned from the speech around it, every take is heard and words said twice at a "
                                    "seam are cut off, and the crossfades fit the gap at each cut.", elem_classes="hint")
                        with gr.Row(equal_height=True):
                            fx_run_btn = gr.Button("🩹 Fix the changed words", variant="primary", size="lg", scale=5,
                                                   elem_classes="button-primary")
                            fx_stop_btn = gr.Button("⏹ Stop", variant="stop", size="lg", scale=1, min_width=96,
                                                    elem_classes="button-stop", interactive=False)
                        fx_progress = gr.HTML(IDLE_PROGRESS_HTML, elem_classes=["progress-card"])
                        fx_audio_out = gr.Audio(label="Fixed recording", type="filepath", elem_classes=["audio-safe-space", "output-clean"])
                        fx_report = gr.Markdown("")
                        fx_blocks = gr.State([])
                        fx_send = montage_send_bar()

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
                        prep_import_parts = clip_import_block("📥 Import from a dataset or folder (audio + .txt)")
                        prep_youtube_parts = youtube_clip_block()
                        prep_audio_player = gr.Audio(
                            label="Audio · 5–30 s works best (trim with the ✂ icon)",
                            type="filepath",
                            interactive=True,
                            value=None,
                            sources=["upload", "microphone"],
                            elem_classes=["audio-safe-space"],
                        )
                        with gr.Row(equal_height=True):
                            prep_clean_engine = gr.Dropdown(REF_CLEAN_ENGINES, value=DEFAULT_SEP_CHAIN,
                                                            label="🎧 Separate the voice from music, noise or other voices",
                                                            scale=5)
                            prep_clean_btn = gr.Button("🎧 Separate", variant="secondary", scale=1, min_width=110)
                            prep_clean_undo = gr.Button("↩ Original", variant="secondary", scale=1, min_width=100)
                        prep_clean_note = gr.Markdown("Clean the clip before saving it: the saved sample is the cleaned "
                                                      "audio. RoFormer models download once (~0.2–0.9 GB each).",
                                                      elem_classes="hint")
                        prep_clean_orig = gr.State(None)
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
                    "sidecar `.txt` / `.json` transcript are cleaned but kept whole. Already have clips + transcripts? "
                    "Choose **📦 Ready dataset** and point it at the folder or `.zip`. Optionally, Qwen3-ForcedAligner "
                    "checks every transcript against its audio and trims the clip to the spoken words.",
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
                        gr.Markdown("#### 📁 Source")
                        dataset_source_mode = gr.Radio(
                            [DATASET_MODE_RAW, DATASET_MODE_YOUTUBE, DATASET_MODE_READY],
                            value=DATASET_MODE_RAW,
                            label="What do you have?",
                            info="Raw recordings and YouTube audio are segmented and transcribed. "
                            "A ready dataset already has its clips and transcripts.",
                        )
                        with gr.Group(visible=False) as youtube_source_group:
                            youtube_urls = gr.Textbox(
                                label="YouTube links (one per line)",
                                lines=3,
                                placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/... 1:20-5:00\nhttps://www.youtube.com/playlist?list=...",
                                info="Videos, playlists or channels. Optional `start-end` after a link keeps only that part. "
                                "Use only content you have the rights to train on.",
                            )
                            with gr.Row():
                                youtube_download_btn = gr.Button("⬇️ Download audio", size="sm", min_width=130)
                                youtube_check_btn = gr.Button("🔍 Show downloaded", size="sm", min_width=130)
                            youtube_md = gr.Markdown(
                                "Audio is saved in `downloads/<project>` and reused on the next build. "
                                "**Build Higgs Dataset** downloads whatever is missing, then segments and transcribes it.",
                                elem_classes="hint",
                            )
                        with gr.Group(visible=True) as raw_source_group:
                            with gr.Row():
                                src_folder = gr.Textbox(label="Source Audio Folder", placeholder="J:\\path\\to\\audio_folder", scale=5)
                                browse_src_folder_btn = gr.Button("Search Folder", scale=1, min_width=130)
                            recursive_dataset = gr.Checkbox(label="Scan subfolders", value=True)
                        with gr.Group(visible=False) as ready_source_group:
                            ready_path = gr.Textbox(
                                label="Ready dataset (folder or .zip)",
                                placeholder="R:\\dataset   or   R:\\dataset.zip",
                                info="Audio + .txt pairs with the same name. metadata.jsonl / LICENSE are optional; "
                                "when metadata.jsonl exists it is used (text, keep flag, train/eval split).",
                            )
                            with gr.Row():
                                ready_browse_dir_btn = gr.Button("📁 Folder", size="sm", min_width=90)
                                ready_browse_zip_btn = gr.Button("🗜️ .zip", size="sm", min_width=90)
                                ready_scan_btn = gr.Button("🔍 Scan", size="sm", min_width=90)
                            ready_use_split = gr.Checkbox(
                                label="Use the train/eval split from metadata.jsonl",
                                value=True,
                                info="Off, or no split column: the Validation Split slider picks evenly spread eval clips.",
                            )
                            ready_scan_md = gr.Markdown(
                                "Enter or pick a folder / .zip with audio + `.txt` pairs (an optional `metadata.jsonl` is used when present).",
                                elem_classes="hint",
                            )
                    with gr.Column(scale=1, elem_classes="form-section"):
                        gr.Markdown("#### 🛰️ ASR & Split")
                        dataset_asr_model = gr.Dropdown(ASR_CHOICES, value=DEFAULT_ASR_MODEL, label="ASR Model", info="Not used for ready datasets.")
                        dataset_asr_lang = gr.Dropdown(list(WHISPER_LANGS), value="Auto-detect", label="Language")
                        dataset_asr_batch = gr.Slider(1, 32, value=8, step=1, label="ASR batch size", visible=False)
                        val_split_slider = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Validation Split")
                    with gr.Column(scale=1, elem_classes="form-section"):
                        gr.Markdown("#### 🎧 Clean-up & Segmentation")
                        dataset_isolate = gr.Dropdown(
                            CLEAN_ENGINES,
                            value=DEFAULT_SEP_CHAIN,
                            label="Voice clean-up / separation",
                            info="Removes music, noise or background voices. RoFormer models download once (~0.2–0.9 GB each).",
                        )
                        with gr.Group(visible=True) as raw_segment_group:
                            dataset_segment = gr.Checkbox(
                                label="Split into natural segments (Inworld SAVA)",
                                value=True,
                                info="Voice-activity + end-of-turn model; long turns are cut at the deepest pause.",
                            )
                        with gr.Row():
                            dataset_min_sec = gr.Slider(0.0, 600.0, value=2.0, step=0.5, label="Min duration (s)", info="Shorter clips are dropped.")
                            dataset_max_sec = gr.Slider(1.0, 600.0, value=15.0, step=0.5, label="Max duration (s)", info="Raw audio: longer turns are split. Ready dataset: only used when 'Keep the dataset's own lengths' is off.")
                        with gr.Group(visible=False) as ready_clean_group:
                            ready_auto_max = gr.Checkbox(
                                label="Keep the dataset's own lengths (max = the longest clip)",
                                value=True,
                                info="Every clip is kept as it is; the max shows the longest clip when scanned. Off: clips outside Min / Max duration are dropped.",
                            )
                            ready_vad_trim = gr.Checkbox(
                                label="Trim edge silence (SAVA voice activity)",
                                value=True,
                                info="Cuts only the silence before the first and after the last word, so the text stays aligned.",
                            )
                            ready_min_speech = gr.Slider(
                                0.0, 0.8, value=0.0, step=0.05,
                                label="Drop clips with less speech than",
                                info="Share of the clip with detected voice. 0 = keep all; 0.3–0.5 drops music/noise-only clips.",
                            )
                            ready_normalize = gr.Checkbox(label="Normalize loudness (the app's loudness, −20 LUFS by default)", value=True)
                        with gr.Row():
                            dataset_align = gr.Checkbox(
                                label="Align text to audio (Qwen3-ForcedAligner)",
                                value=True,
                                info="Places every word in time: trims each clip to its first / last word. "
                                "11 languages; downloads once (~1.8 GB).",
                            )
                            dataset_align_drop = gr.Checkbox(
                                label="Drop clips whose text does not match",
                                value=True,
                                info="Removes clips where the aligner cannot place the words (wrong or partial transcript).",
                            )
                        dataset_segment_hint = gr.Markdown(
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
                        # nothing is picked until a project (or the user) picks it
                        train_manifest = gr.Dropdown(choices=initial_datasets, value=None, label="Train Dataset Directory")
                        val_manifest = gr.Dropdown(choices=[""] + initial_datasets, value="", label="Validation Dataset Directory")
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
                        with gr.Column():
                            with gr.Row():
                                train_logging_steps = gr.Number(value=DEFAULT_TRAINING_24GB["logging_steps"], precision=0, label="Logging steps", min_width=90)
                                train_logging_epochs = gr.Number(value=0, minimum=0, step=0.5, label="Logging epochs", info="0 = use the steps · V3 only", min_width=90)
                            with gr.Row():
                                train_save_steps = gr.Number(value=DEFAULT_TRAINING_24GB["save_steps"], precision=0, label="Save steps", min_width=90)
                                train_save_epochs = gr.Number(value=0, minimum=0, step=0.5, label="Save epochs", info="0 = use the steps · V3 only", min_width=90)
                            with gr.Row():
                                train_eval_steps = gr.Number(value=DEFAULT_TRAINING_24GB["eval_steps"], precision=0, label="Eval steps", min_width=90)
                                train_eval_epochs = gr.Number(value=0, minimum=0, step=0.5, label="Eval epochs", info="0 = use the steps · V3 only", min_width=90)
                        with gr.Row():
                            resume_checkpoint_choice = gr.Dropdown(
                                choices=["None"],
                                value="None",
                                label="Resume checkpoint",
                                scale=3,
                            )
                            v3_patience = gr.Number(
                                value=3, precision=0, label="Early-stop patience (evals)", scale=2,
                                info="Evals without improvement before stopping (0 = never). "
                                "Can be changed on resume and while training: applied at the next eval.",
                            )
                        gr.Markdown(
                            "Existing projects auto-select their latest restorable checkpoint from `exp/<project>`. "
                            "Choose `None` to retrain from scratch. A resumed run keeps everything: optimizer state, "
                            "LR position (no new warmup), best eval, patience counter and the same charts. "
                            "Raise **Epochs** to train past the planned end; the LR then keeps decaying from where it was."
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
                        v3_max_clip = gr.Slider(0, 600, value=0, step=5, label="Skip clips longer than (s) · 0 = no limit",
                                                info="Every clip is trained by default. Long clips need more VRAM: "
                                                     "batch 1 and gradient checkpointing help.")
                    gr.Markdown("**Anti-overfitting**", elem_classes="hint")
                    with gr.Row():
                        v3_min_delta = gr.Number(
                            value=0.002, label="Min improvement (relative)",
                            info="0.002 = an eval must beat the best by 0.2 % to count as progress.",
                        )
                        v3_restore_best = gr.Checkbox(
                            label="Final adapter = best eval", value=True,
                            info="When training ends or stops, the exported LoRA is the best-eval checkpoint.",
                        )
                        v3_best_only = gr.Checkbox(
                            label="Keep only the best adapter (saves space, not resumable)", value=False,
                            info="No numbered checkpoints or optimizer state are written: when the run ends, the final LoRA is "
                            "the best-eval adapter (a few MB) and nothing else stays. The run cannot be resumed.",
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
                        elem_classes=["tag-ac"],
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

            # ================================================================ UTILS
            with gr.Tab("🧰 Utils", id="tab_utils"):
                gr.Markdown(
                    "Disk space used by the app. Everything lives inside the app folder. Deleting always asks for "
                    "confirmation first; deleted models download again when a feature needs them.",
                    elem_classes="hint",
                )
                with gr.Accordion("🧹 Output audios & voice library", open=False, elem_classes="accordion"):
                    gr.Markdown("Empty the **outputs** folder (every generated audio, the timeline's voiced lines and the "
                                "separated voices) or the **voice library** (every sample with its transcript). "
                                "Both ask first.", elem_classes="hint")
                    with gr.Row(equal_height=True):
                        delete_outputs_btn = gr.Button("🗑 Delete output audios", variant="secondary", elem_classes=["red-btn"])
                        delete_samples_btn = gr.Button("🗑 Delete reference samples", variant="secondary", elem_classes=["red-btn"])
                with gr.Accordion("📦 Extra models · downloads, status & cleanup", open=False, elem_classes="accordion"):
                    extras_md = gr.Markdown(extras_status())
                    with gr.Row(equal_height=True):
                        extras_refresh_btn = gr.Button("🔄 Refresh", size="sm", variant="secondary")
                        extras_wespeaker_btn = gr.Button("⬇ WeSpeaker voice model", size="sm", variant="secondary")
                        extras_llama_btn = gr.Button("⬇ llama.cpp engine", size="sm", variant="secondary")
                        extras_lava_btn = gr.Button("⬇ LavaSR voice enhancer", size="sm", variant="secondary")
                    with gr.Row(equal_height=True):
                        extras_pick = gr.Dropdown(extras_choices(), value=[], multiselect=True, label="Installed extras to delete",
                                                  scale=6)
                        extras_del_btn = gr.Button("🗑️ Delete selected", variant="stop", scale=1, min_width=150)
                    with Modal("🗑 Delete extra models") as extras_modal:
                        with extras_modal.actions():
                            extras_cancel_btn = gr.Button("Cancel", variant="secondary", elem_classes=["hv-cancel"])
                            extras_confirm_btn = gr.Button("🗑 Yes, delete", variant="stop")
                    extras_confirm_row = extras_modal.box
                    gr.Markdown("LLM models download from ⚙️ Advanced speech settings → 🧠 LLM (⬇ Download / check). "
                                "Check models (ASR, SAVA) download on first use. Deleting a model that is a hard link "
                                "to another copy only removes this link.", elem_classes="hint")
                storage_sections = {}
                for kind, title in (("models", "🧠 Models & caches"), ("loras", "🧬 LoRA runs"), ("datasets", "📂 Datasets")):
                    with gr.Accordion(title, open=kind == "models", elem_classes="accordion"):
                        with gr.Row(equal_height=True):
                            total_md = gr.Markdown("Press 🔄 Scan to measure.")
                            scan_btn = gr.Button("🔄 Scan", size="sm", variant="secondary", scale=0, min_width=110)
                        tbl = gr.Dataframe(headers=STORAGE_HEADERS, value=[], interactive=False, wrap=True)
                        with gr.Row(equal_height=True):
                            pick = gr.Dropdown([], value=[], multiselect=True, label="Select items to delete", scale=6)
                            del_btn = gr.Button("🗑️ Delete selected", variant="stop", scale=1, min_width=150)
                        storage_sections[kind] = (total_md, scan_btn, tbl, pick, del_btn)
                        if kind == "loras":
                            gr.Markdown("#### Checkpoints of one run")
                            with gr.Row(equal_height=True):
                                run_pick = gr.Dropdown([], value=None, label="LoRA run", scale=5)
                                keep_best_btn = gr.Button("⭐ Keep only best", variant="secondary", scale=1, min_width=150)
                            parts_md = gr.Markdown("")
                            parts_tbl = gr.Dataframe(headers=STORAGE_HEADERS, value=[], interactive=False, wrap=True)
                            with gr.Row(equal_height=True):
                                parts_pick = gr.Dropdown([], value=[], multiselect=True, label="Select checkpoints / folders to delete", scale=6)
                                parts_del_btn = gr.Button("🗑️ Delete selected", variant="stop", scale=1, min_width=150)
                        storage_pending = gr.State([])
                        with Modal("⚠️ Delete permanently") as storage_modal:
                            with storage_modal.actions():
                                cancel_btn = gr.Button("Cancel", variant="secondary", elem_classes=["hv-cancel"])
                                confirm_btn = gr.Button("🗑 Yes, delete permanently", variant="stop")
                        confirm_group, confirm_md = storage_modal.box, storage_modal.md
                        result_md = gr.Markdown("", elem_classes="hint")

                        scan_btn.click(lambda k=kind: storage_view(k), None, [tbl, total_md, pick])
                        if kind == "loras":
                            scan_btn.click(lambda: gr.update(choices=storage_run_choices()), None, run_pick)
                            run_pick.change(lambda r: storage_view("parts", r), run_pick, [parts_tbl, parts_md, parts_pick])
                            mode = gr.State("loras")
                            del_btn.click(lambda keys: ("loras", *storage_ask(keys, "loras")), pick, [mode, storage_pending, confirm_group, confirm_md])
                            parts_del_btn.click(lambda keys: ("parts", *storage_ask(keys, "parts")), parts_pick, [mode, storage_pending, confirm_group, confirm_md])
                            keep_best_btn.click(lambda r: ("parts", *storage_ask_keep_best(r)), run_pick, [mode, storage_pending, confirm_group, confirm_md])

                            def lora_confirm(keys, which, run_key):
                                done = storage_delete(keys, which, run_key)
                                runs = storage_view("loras")
                                parts = storage_view("parts", run_key) if run_key and (ROOT / run_key).exists() else ([], "", gr.update(choices=[], value=[]))
                                run_update = gr.update(choices=storage_run_choices(), value=run_key if run_key and (ROOT / run_key).exists() else None)
                                return (*done[:3], *runs, *parts, run_update)

                            confirm_btn.click(
                                lora_confirm,
                                [storage_pending, mode, run_pick],
                                [storage_pending, confirm_group, result_md, tbl, total_md, pick, parts_tbl, parts_md, parts_pick, run_pick],
                            )
                        else:
                            del_btn.click(lambda keys, k=kind: storage_ask(keys, k), pick, [storage_pending, confirm_group, confirm_md])
                            confirm_btn.click(
                                lambda keys, k=kind: storage_delete(keys, k),
                                storage_pending,
                                [storage_pending, confirm_group, result_md, tbl, total_md, pick],
                            )
                        cancel_btn.click(lambda: ([], gr.update(visible=False), "Cancelled: nothing was deleted."), None,
                                         [storage_pending, confirm_group, result_md], queue=False)

                with gr.Accordion("🎧 Quick checkpoint compare", open=False, elem_classes="accordion"):
                    gr.Markdown(
                        "Generate the same text with the same voice and seed on several checkpoints of one LoRA, "
                        "then listen side by side. Uses the Higgs V3 inference engine.",
                        elem_classes="hint",
                    )
                    with gr.Row(equal_height=True):
                        cmp_project = gr.Dropdown(lora_projects("v3"), value=None, label="LoRA run", scale=4)
                        cmp_refresh = gr.Button("🔄", scale=0, min_width=48, variant="secondary")
                    with gr.Row(equal_height=True):
                        cmp_checkpoints = gr.Dropdown([], value=[], multiselect=True, label="Checkpoints to compare", scale=6)
                        cmp_all = gr.Button("☑️ All", scale=0, min_width=80, variant="secondary")
                    cmp_text = gr.Textbox(
                        label="Text",
                        lines=2,
                        placeholder="A sentence that was NOT in the training data shows best how each checkpoint generalises. Type @ for tags.",
                        elem_classes=["tag-ac-v3"],
                    )
                    with gr.Row(equal_height=True):
                        cmp_voice = gr.Dropdown(list_voice_names(), value="None", label="Reference voice (optional)", scale=3)
                        cmp_seed = gr.Number(value=1234, precision=0, label="Seed", scale=1)
                        cmp_temp = gr.Slider(0.1, 1.5, value=1.0, step=0.05, label="Temperature", scale=2)
                    cmp_btn = gr.Button("▶️ Generate on each checkpoint", variant="primary")
                    cmp_status = gr.Markdown("")
                    cmp_results = gr.HTML(_compare_html([], "", 1234, "None"))
                    cmp_refresh.click(lambda: gr.update(choices=lora_projects("v3")), None, cmp_project, queue=False)
                    cmp_project.change(compare_checkpoint_choices, cmp_project, cmp_checkpoints, queue=False)
                    cmp_all.click(compare_select_all, cmp_project, cmp_checkpoints, queue=False)
                    cmp_btn.click(
                        compare_checkpoints,
                        [cmp_project, cmp_checkpoints, cmp_text, cmp_voice, cmp_seed, cmp_temp],
                        [cmp_status, cmp_results],
                    )

        with gr.Accordion("⚙️ Advanced speech settings · shared by every tab", open=False,
                          elem_classes=["accordion", "advanced-speech"]) as adv_accordion:
            with gr.Accordion("📏 Long text & seed", open=False):
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
                infer_carry_tags = gr.Checkbox(
                    label="Carry tags across segments",
                    value=True,
                    info="Repeats the sentence-level V3 tags (emotion, style, speed, pitch...) on the next "
                    "segments so the intonation doesn't reset after a split. New tags add up.",
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
                        0,
                        8192,
                        value=0,
                        step=64,
                        label="Max frames per pass (0 = auto)",
                        info="Safety cap per segment (25 frames = 1 s). Auto ends a take a second past the longest its text can last, so a runaway take stops early.",
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
                v3_ref_max = gr.Slider(
                    0, 600, value=ref_max_default(), step=5,
                    label="Reference length used (s) · 0 = no limit",
                    info="How much of a reference voice the model hears. 0 uses all of it (the model holds ~20 min "
                         "in total); a longer reference takes a bit more time and VRAM. Applies to the next generation.",
                )
            with gr.Accordion("🏁 Presets", open=True):
                with gr.Row(equal_height=True):
                    adv_quality = gr.Radio(
                        ["Draft", "Standard", "Optimal", "Custom"],
                        value="Draft",
                        label="Quality",
                        info="Draft: one take, not checked. Standard: every take heard by ASR and checked against the text. "
                        "Optimal: three takes, a strict text check and the voice compared with its reference; the best one stays.",
                        scale=3,
                    )
                    adv_sampling_defaults_btn = gr.Button("Balanced sampling (t 0.80 · k 50 · p 0.95)", variant="secondary", scale=1)
                    adv_easy_defaults_btn = gr.Button("Easy GUI sampling (t 1.00 · k 50 · p 0.95)", variant="secondary", scale=1)
            with gr.Accordion("⚡ Live playback & speed", open=True):
                with gr.Row(equal_height=True):
                    adv_live = gr.Checkbox(
                        label="Live playback while generating",
                        value=True,
                        info="Off: no streaming, the result plays when it is done (also the 📡 Live switch in the top bar). "
                        "On: plays the speech as it is made, block after block on one clock, in the top bar's player only "
                        "(the result player does not play it again). Esc stops it.",
                    )
                    adv_keep_models = gr.Checkbox(
                        label="Keep check models loaded",
                        value=True,
                        info="ASR and the voice model stay in memory between renders (faster, ~2.5 GB more VRAM with turbo).",
                    )
                    adv_clear_cache_btn = gr.Button("🗑 Clear cached takes", variant="secondary", elem_classes=["red-btn"])
            with gr.Accordion("🎧 Checks & retakes", open=False):
                with gr.Row(equal_height=True):
                    adv_asr_check = gr.Checkbox(label="Check every new take by ear (ASR)", value=False,
                                                info="Recognition hears the take and compares it with the text (WER).")
                    adv_similarity = gr.Checkbox(label="Check the voice matches its reference", value=False,
                                                 info="WeSpeaker voice print of the take against the reference audio.")
                    adv_similarity_min = gr.Slider(0.2, 0.9, value=0.5, step=0.01, label="Minimum voice similarity",
                                                   info="Cosine of the voice prints. Same voice ≈ 0.6–0.95, another voice < 0.3.")
                with gr.Row(equal_height=True):
                    adv_strictness = gr.Radio(["Strict", "Normal", "Lenient"], value="Normal", label="Strictness",
                                              info="Words allowed to differ: strict ≈ 10 %, normal ≈ 20 %, lenient ≈ 35 % (more in short blocks).")
                    adv_takes = gr.Number(value=1, precision=0, minimum=1, maximum=20, label="Takes per block",
                                          info="Takes made per block; the best one is kept.")
                    adv_retries = gr.Number(value=2, precision=0, minimum=0, maximum=20, label="Retries on a failed check",
                                            info="More takes when the best one fails a check.")
                with gr.Row(equal_height=True):
                    adv_asr_model = gr.Dropdown(ASR_CHOICES, value="CrisperWhisper 2.0 turbo", label="Check ASR model")
                    adv_language = gr.Dropdown(list(WHISPER_LANGS), value="Auto-detect", label="Speech language",
                                               info="For the ASR check and numbers in words.")
                adv_sound_checks = gr.CheckboxGroup(
                    choices=[(label, key) for key, label in tc.SOUND_CHECKS.items()],
                    value=list(tc.CHEAP_CHECKS),
                    label="Sound checks of every take",
                    info="A defect redoes the take; a suspect is decided by ASR when “Check every new take by ear” is on, "
                         "otherwise it is only shown, like a note. The last two are slower.",
                )
            with gr.Accordion("⏸ Pauses & finishing", open=False):
                with gr.Row(equal_height=True):
                    loud_on = gr.Checkbox(label="🔊 One loudness for every audio of the app", value=loudness.SETTINGS["on"], scale=3,
                                          info="The files themselves: references used to generate (once, when first used), the voice library, "
                                          "separated voices, every block and every output of every tab, "
                                          "and the dataset clips: all at the same loudness. Applies everywhere, at once.")
                    loud_target = gr.Slider(-30, -12, value=loudness.SETTINGS["target"], step=1, scale=2,
                                            label="Loudness (LUFS)", info="−20 speech / audiobook · −16 podcast · −14 video. "
                                            "Peaks stay under −1 dB.")
                with gr.Row(equal_height=True):
                    adv_pause_same = gr.Number(value=300, precision=0, minimum=0, maximum=10000, label="Pause, same speaker (ms)")
                    adv_pause_other = gr.Number(value=600, precision=0, minimum=0, maximum=10000, label="Pause, other speaker (ms)")
                    adv_final_loudness = gr.Dropdown(list(LOUDNESS_TARGETS), value="Off (as the model made it)",
                                                     label="Final loudness of this delivery",
                                                     info="A delivery target wins over the app's loudness for the final mix.")
                with gr.Row(equal_height=True):
                    adv_tidy = gr.Checkbox(label="Even pauses", value=False,
                                           info="No silence at the edges of a take, so pauses between blocks are exactly as set; "
                                           "a pause inside a take longer than 1.2 s is shortened.")
                    adv_level = gr.Checkbox(label="Even loudness of the pieces", value=False,
                                            info="Every take is brought to −20 LUFS before joining (already done when "
                                            "“One loudness” is on).")
                    adv_room_tone = gr.Checkbox(label="Room tone instead of digital silence", value=False,
                                                info="Pauses get a quiet room tone so the joins sound natural.")
                with gr.Row(equal_height=True):
                    adv_enhance = gr.Checkbox(label="✨ Enhance voice: restore high frequencies, save at 48 kHz (LavaSR)", value=False,
                                              info="A 56 MB model rebuilds the band above 12 kHz the speech model does not make; "
                                              "the voice below stays untouched. Fast (well over realtime on GPU). Downloads on first use.")
                    adv_enhance_denoise = gr.Checkbox(label="Denoise before enhancing", value=False,
                                                      info="Only for noisy results: the denoiser works at 16 kHz and the enhancer rebuilds the rest.")
            with gr.Accordion("🔢 Text before voicing", open=False):
                adv_words = gr.CheckboxGroup(
                    choices=[("Numbers", "numbers"), ("Years", "years"), ("Times (10:30)", "times"), ("Money ($ €)", "money"),
                             ("Ordinals (1º, 2nd)", "ordinals"), ("Percent", "percent")],
                    value=[],
                    label="Read with words",
                    info="Written out in the speech language before voicing (num2words). Tags stay untouched.",
                )
            with gr.Accordion("🧠 LLM · director & podcast writer", open=False):
                _llm_saved = llm_load_settings()
                with gr.Row(equal_height=True):
                    llm_provider = gr.Radio(["Local (llama.cpp)", "OpenAI-compatible server"],
                                            value=_llm_saved.get("provider", "Local (llama.cpp)"), label="Where the LLM runs",
                                            info="Local: llama.cpp on this GPU with a GGUF. Server: LM Studio, Ollama, vLLM… (OpenAI API).")
                    llm_free_tts = gr.Checkbox(value=bool(_llm_saved.get("free_tts", True)),
                                               label="Share the GPU: unload speech before the LLM loads, stop the LLM before voicing")
                with gr.Row(equal_height=True):
                    llm_local_model = gr.Dropdown(llm.local_choices(), value=_llm_saved.get("local_model") or llm.default_local_choice(),
                                                  label="Local model (GGUF)", allow_custom_value=True, scale=4,
                                                  info="Recommended Gemma 4 12B quants (✓ on disk, ⬇ downloads on first use); any .gguf in models/llm shows up too.")
                    llm_download_btn = gr.Button("⬇ Download / check", variant="secondary", scale=1)
                    llm_unload_btn = gr.Button("⏏ Unload LLM", variant="secondary", scale=1)
                with gr.Row(equal_height=True):
                    llm_url = gr.Textbox(value=_llm_saved.get("url", "http://127.0.0.1:1234/v1"), label="Server address", scale=3)
                    llm_api_key = gr.Textbox(value="", type="password", label="API key (optional, never saved)", scale=2)
                    llm_remote_model = gr.Dropdown(choices=[_llm_saved["remote_model"]] if _llm_saved.get("remote_model") else [],
                                                   value=_llm_saved.get("remote_model") or None, allow_custom_value=True,
                                                   label="Server model", scale=3)
                    llm_check_btn = gr.Button("🔌 Check remote connection", variant="secondary", scale=1)
                with gr.Row(equal_height=True):
                    llm_ctx = gr.Number(value=int(_llm_saved.get("ctx", 16384)), precision=0, minimum=2048, maximum=131072,
                                        label="Context (tokens)")
                    llm_temperature = gr.Slider(0.0, 1.5, value=float(_llm_saved.get("temperature", 0.7)), step=0.05,
                                                label="LLM temperature")
                llm_status = gr.Markdown("")
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
        adv_inputs = [
            adv_asr_check, adv_asr_model, adv_language, adv_similarity, adv_similarity_min, adv_strictness, adv_takes,
            adv_retries, adv_sound_checks, adv_pause_same, adv_pause_other, adv_tidy, adv_level, adv_room_tone,
            adv_final_loudness, adv_words, adv_live, adv_keep_models, adv_enhance, adv_enhance_denoise,
        ]
        assert len(adv_inputs) == len(ADV_KEYS)

        def apply_quality(mode):
            q = tc.QUALITY_PRESETS.get(mode)
            if not q:
                return gr.update(), gr.update(), gr.update(), gr.update()
            return q["asr_check"], q["takes"], q["similarity"], q["strictness"]

        adv_quality.input(apply_quality, adv_quality, [adv_asr_check, adv_takes, adv_similarity, adv_strictness], queue=False)

        for comp in (adv_asr_check, adv_takes, adv_similarity, adv_strictness):
            comp.input(lambda: "Custom", None, adv_quality, queue=False)
        adv_sampling_defaults_btn.click(lambda: (0.8, 50, 0.95), None, [infer_temperature, infer_top_k, infer_top_p], queue=False)
        adv_easy_defaults_btn.click(lambda: (1.0, 50, 0.95), None, [infer_temperature, infer_top_k, infer_top_p], queue=False)
        adv_clear_cache_btn.click(lambda: log(clear_take_cache()), None, logs_box, queue=False)
        voice_dropdowns = [sample_dropdown, infer_sample_select]
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
        ).then(dialogue_sync, [dialogue_editor, infer_model_select], dialogue_editor, queue=False)

        infer_sample_select.change(
            load_sample_for_ui,
            [infer_sample_select, infer_use_ref_text],
            [infer_ref_audio, infer_ref_text],
            queue=False,
        )
        refresh_infer_sample_btn.click(refresh_sample_dropdown, None, infer_sample_select, queue=False)
        for _src, _eng, _btn, _undo, _orig, _note in (
            (infer_ref_audio, infer_clean_engine, infer_clean_btn, infer_clean_undo, infer_clean_orig, infer_clean_note),
            (prep_audio_player, prep_clean_engine, prep_clean_btn, prep_clean_undo, prep_clean_orig, prep_clean_note),
            *pc_sep_parts,
        ):
            _clean = gr.State(None)  # the cleaned file: the player is emptied, then loaded with it (always redrawn)
            _btn.click(clean_reference_audio, [_src, _eng, _orig], [_clean, _orig, _note], concurrency_limit=1,
                       concurrency_id="ref_clean").then(
                lambda p: None if p else gr.update(), _clean, _src, queue=False, show_progress="hidden").then(
                lambda p: p or gr.update(), _clean, _src, queue=False, show_progress="hidden")
            _undo.click(restore_reference_audio, _orig, [_src, _orig, _note], queue=False)
        infer_sample_select.change(lambda: None, None, infer_clean_orig, queue=False)
        sample_dropdown.change(lambda: None, None, prep_clean_orig, queue=False)
        wire_clip_import(infer_import_parts, infer_ref_audio, infer_ref_text)
        wire_clip_import(prep_import_parts, prep_audio_player, prep_transcription, save_sample_name)
        wire_youtube_clip(infer_youtube_parts, infer_ref_audio, infer_ref_text, infer_clean_orig, infer_save_name)
        for trigger in (infer_save_btn.click, infer_save_name.submit):
            trigger(infer_save_to_library, [infer_ref_audio, infer_save_name, infer_ref_text],
                    [sample_dropdown, infer_sample_select, infer_save_note], queue=False).then(
                dialogue_sync, [dialogue_editor, infer_model_select], dialogue_editor, queue=False)
        wire_youtube_clip(prep_youtube_parts, prep_audio_player, prep_transcription, prep_clean_orig, save_sample_name)
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
            run_inference_ui,
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
                infer_carry_tags,
                *adv_inputs,
            ],
            [infer_audio_out, infer_file_out, infer_console, infer_gen_btn, infer_stop_btn, infer_progress, infer_blocks],
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
                *common_params,
                infer_save_outputs,
                infer_chunk_mode,
                infer_carry_tags,
                dialogue_editor,
                dlg_only_changed,
                *adv_inputs,
            ],
            [dialogue_audio_out, dialogue_file_out, infer_console, dialogue_gen_btn, dialogue_stop_btn, dialogue_progress, dialogue_blocks],
            show_progress="hidden",
        )
        dialogue_stop_btn.click(
            stop_dialogue_generation,
            None,
            [infer_console, dialogue_gen_btn, dialogue_stop_btn],
            queue=True,
            show_progress="hidden",
            cancels=[ev_dialogue],
        )
        dialogue_refresh_lora_btn.click(dialogue_sync, [dialogue_editor, infer_model_select], dialogue_editor, queue=False)
        infer_model_select.change(dialogue_sync, [dialogue_editor, infer_model_select], dialogue_editor, queue=False)

        # ================================================================ LLM, director, podcast, fixer, montage
        llm_inputs = [llm_provider, llm_local_model, llm_url, llm_api_key, llm_remote_model, llm_ctx, llm_temperature, llm_free_tts]
        for comp in llm_inputs:
            if comp is not llm_api_key:
                comp.change(llm_save_settings, llm_inputs, None, queue=False, show_progress="hidden")
        llm_check_btn.click(llm_check, [llm_provider, llm_url, llm_api_key, llm_local_model], [llm_status, llm_remote_model], queue=False)
        llm_download_btn.click(llm_download, llm_local_model, [llm_status, llm_local_model])
        llm_unload_btn.click(llm_unload, None, llm_status, queue=False)

        infer_dir_btn.click(
            director_inference, [infer_dir_op, infer_text, infer_dir_instr, adv_language, *llm_inputs],
            [infer_text, infer_dir_note, infer_split_lines, studio_tabs],
        ).then(dialogue_from_lines, [infer_split_lines, dialogue_editor, infer_model_select], [dialogue_editor, dlg_dir_note], queue=False)
        dlg_dir_btn.click(
            director_dialogue, [dlg_dir_op, dlg_dir_instr, adv_language, dialogue_editor, infer_model_select, *llm_inputs],
            [dialogue_editor, dlg_dir_note],
        )

        pc_count.change(lambda n: [gr.update(visible=i < int(n)) for i in range(PODCAST_MAX_SPEAKERS)], pc_count, pc_speaker_rows,
                        queue=False, show_progress="hidden")
        pc_sync_inputs = [pc_editor, infer_model_select, pc_count, *pc_speaker_inputs]
        pc_count.change(podcast_sync, pc_sync_inputs, pc_editor, queue=False, show_progress="hidden")
        for k, comp in enumerate(pc_speaker_inputs):
            (comp.blur if k % PODCAST_SPK_FIELDS in (0, 1, 5) else comp.change)(podcast_sync, pc_sync_inputs, pc_editor, queue=False,
                                                                                 show_progress="hidden")
        pc_write_btn.click(podcast_write, [pc_editor, pc_topic, pc_language, pc_minutes, infer_model_select, pc_count,
                                           *pc_speaker_inputs, *llm_inputs], [pc_editor, pc_note])
        pc_continue_btn.click(podcast_continue, [pc_editor, pc_direction, pc_language, pc_minutes, infer_model_select, pc_count,
                                                 *pc_speaker_inputs, *llm_inputs], [pc_editor, pc_note])
        pc_clear_btn.click(podcast_clear, pc_sync_inputs, [pc_editor, pc_note], queue=False)
        pc_dir_btn.click(podcast_director, [pc_dir_op, pc_dir_instr, pc_language, pc_editor, infer_model_select, pc_count,
                                            *pc_speaker_inputs, *llm_inputs], [pc_editor, pc_dir_note])
        pc_to_dialogue_btn.click(podcast_to_dialogue, [pc_editor, dialogue_editor, infer_model_select, pc_count, *pc_speaker_inputs],
                                 [dialogue_editor, pc_note, main_tabs, studio_tabs], queue=False)
        ev_pc = pc_gen_btn.click(
            lambda: (set_button_busy("Voicing…"), gr.update(interactive=True)), None, [pc_gen_btn, pc_stop_btn],
            queue=False, show_progress="hidden",
        ).then(
            run_podcast,
            [pc_editor, infer_model_select, lora_adapter, *common_params, infer_save_outputs, infer_chunk_mode, infer_carry_tags,
             pc_pause_same, pc_pause_other, pc_only_changed, pc_count, *pc_speaker_inputs, *adv_inputs],
            [pc_audio_out, pc_file_out, pc_report, pc_progress, pc_blocks, pc_gen_btn, pc_stop_btn],
            show_progress="hidden",
        )
        pc_stop_btn.click(lambda: stop_render("🎙 Voice the podcast"), None, [pc_gen_btn, pc_stop_btn], queue=False, cancels=[ev_pc])

        fx_transcribe_btn.click(fixer_transcribe, [fx_audio, fx_sava, *adv_inputs],
                                [fx_words_state, fx_original, fx_edited, fx_note])
        ev_fx = fx_run_btn.click(
            lambda: (set_button_busy("Fixing…"), gr.update(interactive=True)), None, [fx_run_btn, fx_stop_btn],
            queue=False, show_progress="hidden",
        ).then(
            fixer_run,
            [fx_audio, fx_words_state, fx_edited, fx_match, fx_lora,
             infer_model_select, *common_params,
             *adv_inputs],
            [fx_audio_out, fx_report, fx_progress, fx_blocks, fx_run_btn, fx_stop_btn],
            show_progress="hidden",
        )
        fx_edited.change(fixer_plan, [fx_words_state, fx_edited], fx_plan, queue=False, show_progress="hidden")
        fx_stop_btn.click(lambda: stop_render("🩹 Fix the changed words"), None, [fx_run_btn, fx_stop_btn], queue=False, cancels=[ev_fx])

        for parts, state in ((infer_send, infer_blocks), (dialogue_send, dialogue_blocks), (pc_send, pc_blocks), (fx_send, fx_blocks)):
            send, ask, append, replace, cancel, note = parts
            send.click(lambda: gr.update(visible=True), None, ask, queue=False, show_progress="hidden")
            cancel.click(lambda: gr.update(visible=False), None, ask, queue=False, show_progress="hidden")
            append.click(montage_send, [gr.State("append"), state, montage_tl], [montage_tl, ask, note, main_tabs], queue=False)
            replace.click(montage_send, [gr.State("replace"), state, montage_tl], [montage_tl, ask, note, main_tabs], queue=False)
        def _set_loudness(on, target):
            loudness.configure(on, target)
            _save_ui_settings(loudness_on=bool(on), loudness_target=float(target))
            print(f"[loudness] {'on' if on else 'off'} · {float(target):g} LUFS", flush=True)

        loud_on.change(_set_loudness, [loud_on, loud_target], None, queue=False, show_progress="hidden")
        loud_target.change(_set_loudness, [loud_on, loud_target], None, queue=False, show_progress="hidden")
        montage_tl.submit(
            montage_request,
            [montage_tl, infer_model_select, lora_adapter, *common_params, infer_chunk_mode, infer_carry_tags, *adv_inputs, *llm_inputs],
            [montage_tl, montage_progress], show_progress="hidden",
        )
        montage_save_btn.click(montage_save, [montage_tl, montage_name], [montage_saved, montage_note], queue=False)
        montage_load_btn.click(montage_load, montage_saved, [montage_tl, montage_note], queue=False)
        montage_new_btn.click(montage_new, montage_tl, [montage_tl, montage_note], queue=False)
        montage_export_btn.click(montage_export, [montage_tl, montage_name, montage_fmt], [montage_audio, montage_file, montage_note])

        extras_refresh_btn.click(lambda: (extras_status(), gr.update(choices=extras_choices(), value=[])), None,
                                 [extras_md, extras_pick], queue=False)
        extras_lava_btn.click(lambda: extras_download("lavasr"), None, extras_md)
        extras_del_btn.click(extras_ask, extras_pick, [extras_confirm_row, extras_modal.md], queue=False)
        extras_cancel_btn.click(lambda: gr.update(visible=False), None, extras_confirm_row, queue=False)
        extras_confirm_btn.click(extras_delete, extras_pick, [extras_md, extras_pick, extras_confirm_row], queue=False)
        extras_wespeaker_btn.click(lambda: extras_download("wespeaker"), None, extras_md)
        extras_llama_btn.click(lambda: extras_download("llama"), None, extras_md)

        dataset_source_mode.change(
            on_dataset_mode_change,
            [dataset_source_mode, dataset_isolate],
            [raw_source_group, ready_source_group, raw_segment_group, ready_clean_group, dataset_asr_model, dataset_isolate, youtube_source_group],
            queue=False,
            show_progress="hidden",
        )
        ev_youtube_download = youtube_download_btn.click(download_youtube_ui, [dataset_name_input, youtube_urls], youtube_md)
        youtube_check_btn.click(describe_youtube_ui, dataset_name_input, youtube_md, queue=False)
        ready_browse_dir_btn.click(lambda cur: browse_ready_dataset("dir", cur), ready_path, [ready_path, ready_scan_md], queue=False)
        ready_browse_zip_btn.click(lambda cur: browse_ready_dataset("zip", cur), ready_path, [ready_path, ready_scan_md], queue=False)
        ready_scan_btn.click(describe_ready_dataset, ready_path, ready_scan_md, queue=False)
        ready_path.submit(describe_ready_dataset, ready_path, ready_scan_md, queue=False)

        def ready_max_to_longest(path, auto):
            if not auto or not (path or "").strip():
                return gr.update()
            from higgs_local.ready_dataset import ready_longest_seconds

            longest = ready_longest_seconds(path)
            return gr.update(value=min(600.0, math.ceil(longest * 2) / 2)) if longest > 0 else gr.update()

        ready_scan_md.change(ready_max_to_longest, [ready_path, ready_auto_max], dataset_max_sec, show_progress="hidden")
        ready_auto_max.change(ready_max_to_longest, [ready_path, ready_auto_max], dataset_max_sec, show_progress="hidden")
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
                dataset_source_mode,
                ready_path,
                ready_use_split,
                ready_vad_trim,
                ready_min_speech,
                ready_normalize,
                dataset_align,
                dataset_align_drop,
                youtube_urls,
                ready_auto_max,
            ],
            [train_manifest, val_manifest, dataset_log_box, dataset_status_box, prep_btn, stop_prep_btn, output_name, dataset_project_card, dataset_name_input],
        )
        stop_prep_btn.click(
            stop_dataset_build,
            None,
            [dataset_log_box, dataset_status_box, prep_btn, stop_prep_btn],
            queue=True,
            show_progress="hidden",
            cancels=[ev_prep_build, ev_youtube_download],
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
                dataset_source_mode,
                ready_path,
                ready_use_split,
                ready_vad_trim,
                ready_min_speech,
                ready_normalize,
                dataset_align,
                dataset_align_drop,
                youtube_urls,
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
        v3_patience.input(set_live_patience, [v3_patience, output_name], None, queue=False, show_progress="hidden")
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
                v3_best_only,
                train_logging_epochs,
                train_save_epochs,
                train_eval_epochs,
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
        demo.load(fn=None, js=APP_JS.replace("__TAG_CATALOG__", json.dumps(tag_autocomplete_catalog())).replace(
            "__SEP_ENGINES__", json.dumps(REF_CLEAN_ENGINES)), queue=False)
        tb_btn.click(launch_tensorboard, output_name, [logs_box, training_status_box, training_log_box], queue=False)

        v3_precision.change(set_v3_runtime, [v3_precision, v3_attention_backend, v3_torch_compile], logs_box, queue=False)
        v3_attention_backend.change(set_v3_runtime, [v3_precision, v3_attention_backend, v3_torch_compile], logs_box, queue=False)
        v3_torch_compile.change(set_v3_runtime, [v3_precision, v3_attention_backend, v3_torch_compile], logs_box, queue=False)
        v3_ref_max.change(set_ref_max, v3_ref_max, logs_box, queue=False)  # also when a GUI preset sets it
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
        # the top bar's Live switch and the one in Advanced are the same setting
        top_live.input(lambda v: bool(v), top_live, adv_live, queue=False)
        adv_live.change(lambda v: bool(v), adv_live, top_live, queue=False)
        delete_outputs_btn.click(lambda: danger_ask("outputs"), None, [danger_kind, danger.box, danger.md], queue=False)
        delete_samples_btn.click(lambda: danger_ask("samples"), None, [danger_kind, danger.box, danger.md], queue=False)
        gui_preset_delete_btn.click(lambda name: danger_ask("preset", name), gui_preset_select,
                                    [danger_kind, danger.box, danger.md], queue=False)
        danger_cancel.click(lambda: ("", gr.update(visible=False)), None, [danger_kind, danger.box], queue=False)
        danger_ok.click(danger_run, [danger_kind, gui_preset_select], [danger_kind, danger.box, logs_box, gui_preset_select])

        # ---------------- GUI presets: every setting and selection of the page
        preset_comps = collect_preset_components(dict(locals()))
        preset_names = list(preset_comps)
        preset_list = list(preset_comps.values())
        print(f"[preset] {len(preset_names)} settings are saved in a GUI preset", flush=True)
        gui_preset_save_btn.click(lambda name, *v: save_gui_preset(preset_names, name, *v), [gui_preset_select, *preset_list],
                                  gui_preset_select, queue=False)
        gui_preset_load_btn.click(lambda name: load_gui_preset(preset_names, preset_list, name), gui_preset_select, preset_list,
                                  queue=False)

        # ---------------- this tab's settings survive a reload (an app restart reloads the page)
        _session_tab = gr.Textbox(visible=False)
        tab_id_js = ("(() => { let id = sessionStorage.getItem('hv-tab'); const fresh = !id; "
                     "if (!id) { id = crypto.randomUUID ? crypto.randomUUID() : ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, "
                     "(c) => (c ^ Math.random() * 16 >> c / 4).toString(16)); sessionStorage.setItem('hv-tab', id); } "
                     "return [id, fresh]; })()")
        demo.load(lambda tab: session_restore(preset_names, preset_list, tab), _session_tab, preset_list,
                  js=f"() => {{ const [id, fresh] = {tab_id_js}; return [fresh ? '' : id]; }}",
                  queue=False, show_progress="hidden")
        gr.on(
            [c.change for c in preset_list],
            lambda tab, *v: session_autosave(preset_names, tab, *v),
            [_session_tab, *preset_list],
            None,
            js=f"(tab, ...v) => [{tab_id_js}[0], ...v]",
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )

    return demo


if __name__ == "__main__":
    _install_cmd_mirror()
    from gradio.utils import get_upload_folder

    # Gradio's own file cache too: an audio a page got before the app was restarted (a library voice, a reference)
    # is still served, instead of an empty player ("File not allowed" for files an earlier run handed out)
    allowed = [ROOT / "exp", CACHE_DIR, OUTPUTS_DIR, ROOT / "projects", TEMP_DIR, SAMPLES_DIR, ROOT / "downloads",
               Path(get_upload_folder())]
    build_ui().queue(default_concurrency_limit=1).launch(inbrowser=True, css=UI_CSS, js=APP_JS, theme=build_theme(),
                                                          allowed_paths=[str(p) for p in allowed])
