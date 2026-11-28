"""Live playback of speech while it is generated.

The generation thread pushes each decoded audio block here; the page polls
``poll(job, cursor)`` through a ``gr.HTML`` server function (outside the Gradio
queue, so it answers while the queue is busy generating) and schedules every
block on one Web Audio clock right after the previous one, the way Speech
Studio's live player does. Blocks come from the streaming codec decoder with
left context and look-ahead, so they join without clicks, gaps or repeats, and
only this player sounds them: the result player gets the finished file without
autoplay.

Only one page plays a job: the page where a button was last pressed before the job started (it "claims" the
bus on every button press). Another tab or browser with the app open stays quiet, instead of playing the same
audio a few milliseconds apart, which sounds like a doubled voice.
"""

from __future__ import annotations

import base64
import threading
import time

import numpy as np

from .audio_utils import SAMPLE_RATE


class LiveBus:
    """One job at a time: a new job drops whatever the previous one queued."""

    def __init__(self, keep_seconds: float = 900.0):
        self.lock = threading.Lock()
        self.job = 0
        self.chunks: list[str] = []
        self.done = True
        self.label = ""
        self.restarts = 0
        self.keep = int(keep_seconds * SAMPLE_RATE)
        self.samples = 0
        self.started = 0.0
        self.claimant = ""  # the page that last pressed a button
        self.owner = ""  # the page that plays the current job ("" = every page)

    def claim(self, client: str) -> None:
        with self.lock:
            self.claimant = str(client or "")

    def start(self, label: str = "") -> int:
        with self.lock:
            self.job += 1
            self.owner = self.claimant
            self.chunks = []
            self.done = False
            self.label = label
            self.samples = 0
            self.started = time.time()
            return self.job

    def push(self, wav: np.ndarray) -> None:
        if wav is None or len(wav) == 0:
            return
        data = np.ascontiguousarray(np.asarray(wav, dtype=np.float32).reshape(-1))
        encoded = base64.b64encode(data.tobytes()).decode("ascii")
        with self.lock:
            if self.done or self.samples > self.keep:
                return
            self.chunks.append(encoded)
            self.samples += len(data)

    def restart(self) -> None:
        """The take being played was thrown away (a retake follows): the page drops what it queued of it."""
        with self.lock:
            self.chunks.append("restart")

    def end(self) -> None:
        with self.lock:
            self.done = True

    def poll(self, job: int, cursor: int, client: str = "") -> dict:
        with self.lock:
            if self.owner and client and str(client) != self.owner:
                # another page started this job: this one stays quiet
                return {"job": self.job, "cursor": len(self.chunks), "chunks": [], "done": self.done, "sr": SAMPLE_RATE,
                        "reset": False, "label": self.label, "elsewhere": True}
            if int(job) != self.job:
                # A new job (or the page just opened): start reading it from the top.
                return {"job": self.job, "cursor": 0, "chunks": [], "done": self.done, "sr": SAMPLE_RATE, "reset": True,
                        "label": self.label}
            cursor = max(0, int(cursor))
            new = self.chunks[cursor:]
            return {"job": self.job, "cursor": cursor + len(new), "chunks": new, "done": self.done, "sr": SAMPLE_RATE,
                    "reset": False, "label": self.label}


LIVE = LiveBus()


def _server_args(args: tuple, count: int) -> list:
    """Gradio hands a server function the page's arguments as one list: spread it back out."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = tuple(args[0])
    return (list(args) + [None] * count)[:count]


def live_poll(*args):
    """Server function for the page's live player: live_poll(job, cursor, client)."""
    job, cursor, client = _server_args(args, 3)
    try:
        return LIVE.poll(int(job or 0), int(cursor or 0), str(client or ""))
    except Exception as exc:  # the page shows it and keeps polling
        return {"error": f"{type(exc).__name__}: {exc}"}


def live_claim(*args):
    """Server function: this page pressed a button, so a job that starts now plays here (and only here)."""
    (client,) = _server_args(args, 1)
    LIVE.claim(str(client or ""))
    return {"ok": True}


LIVE_PLAYER_HTML = """
<div class="live-player">
  <span class="live-dot"></span>
  <span class="live-label">Live playback idle</span>
  <button class="live-stop" type="button" title="Stop live playback (Esc)">⏹</button>
</div>
"""

LIVE_PLAYER_CSS = """
.live-player { display:flex; align-items:center; gap:8px; font-size:12px; opacity:.85; padding:2px 4px; }
.live-dot { width:9px; height:9px; border-radius:50%; background:#666; flex:none; }
.live-player.on .live-dot { background:#e5484d; box-shadow:0 0 0 3px rgba(229,72,77,.25); }
.live-stop { border:1px solid var(--border-color-primary); border-radius:6px; padding:0 8px; background:transparent;
  cursor:pointer; color:inherit; }
"""

# Polls the live bus and schedules each block on one AudioContext clock: block n
# starts exactly where block n-1 ends, so the take plays as one continuous sound.
LIVE_PLAYER_JS = r"""
const box = element.querySelector('.live-player');
const label = element.querySelector('.live-label');
let ctx = null, cursorTime = 0, job = -1, cursor = 0, nodes = [], active = false, stopped = false, elsewhere = false;
// this page's id: a job started right after a button press here plays here only (not in other tabs / browsers)
const client = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)) ;
let lastClaim = 0;
const claim = () => { const now = Date.now(); if (now - lastClaim < 300) return; lastClaim = now; server.live_claim(client).catch(() => {}); };
document.addEventListener('pointerdown', (e) => { if (e.target.closest && e.target.closest('button')) claim(); }, true);
document.addEventListener('keydown', (e) => { if (e.key === 'Enter') claim(); }, true);
function decode(b64) {
  const bin = atob(b64); const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer, 0, bytes.length >> 2);
}
function dropQueued() {
  for (const n of nodes) { try { n.stop(); } catch (e) {} }
  nodes = []; cursorTime = 0;
}
function play(samples, sr) {
  ctx = ctx || new AudioContext();
  if (ctx.state === 'suspended') ctx.resume();
  const buf = ctx.createBuffer(1, samples.length, sr);
  buf.copyToChannel(samples, 0);
  const node = ctx.createBufferSource();
  node.buffer = buf; node.connect(ctx.destination);
  // the first block waits 150 ms so the next one is already here when it ends
  const at = Math.max(ctx.currentTime + (nodes.length ? 0.02 : 0.15), cursorTime);
  node.start(at); cursorTime = at + buf.duration; nodes.push(node);
  node.onended = () => { nodes = nodes.filter((x) => x !== node); };
}
window.higgsLive = {
  stop() { stopped = true; dropQueued(); },
  get playing() { return nodes.length > 0; },
};
element.querySelector('.live-stop').addEventListener('click', () => window.higgsLive.stop());
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && nodes.length) window.higgsLive.stop(); });
// a click anywhere unlocks audio in browsers that need a gesture before sound
document.addEventListener('pointerdown', () => { ctx = ctx || new AudioContext(); if (ctx.state === 'suspended') ctx.resume(); }, { once: true });
async function tick() {
  let wait = 900;
  try {
    const r = await server.live_poll(job, cursor, client);
    if (r && !r.error) {
      elsewhere = !!r.elsewhere;
      if (elsewhere) {
        job = r.job; cursor = r.cursor;
      } else if (r.reset) {
        if (job !== -1 && r.job !== job) { dropQueued(); stopped = false; }
        job = r.job; cursor = 0;
      } else {
        cursor = r.cursor;
        for (const c of r.chunks) {
          if (c === 'restart') { dropQueued(); continue; }
          if (!stopped) play(decode(c), r.sr);
        }
      }
      active = !r.done;
      box.classList.toggle('on', active || nodes.length > 0);
      label.textContent = elsewhere ? (active ? 'Live · playing in another window' : 'Live playback idle')
        : active ? ('Live · ' + (r.label || 'generating')) : (nodes.length ? 'Live · playing the rest' : 'Live playback idle');
      wait = active ? 120 : (nodes.length ? 400 : 900);
    }
  } catch (e) { wait = 2000; }
  setTimeout(tick, wait);
}
tick();
"""
