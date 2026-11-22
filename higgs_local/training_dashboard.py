"""In-app training monitor: charts, overfitting status and eval-audio evolution.

Reads ``exp/<project>/metrics.jsonl`` written by the V3 trainer and renders
self-contained HTML (inline SVG, no JS libraries). Charts and audio are two
separate renders so the periodic refresh never restarts a playing preview.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from urllib.parse import quote

from .paths import ROOT

EXP_DIR = ROOT / "exp"

STATUS_STYLE = {
    "baseline": ("Baseline measured", "info"),
    "warming up": ("Warming up", "info"),
    "improving": ("Improving", "ok"),
    "plateau": ("Plateau", "warn"),
    "overfitting": ("Overfitting", "bad"),
    "regressed": ("Worse than base model", "bad"),
}


def load_metrics(project: str | None) -> list[dict]:
    if not project:
        return []
    path = EXP_DIR / project / "metrics.jsonl"
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # half-written last line while training
    except OSError:
        return []
    return records


def _file_url(path: str) -> str:
    p = Path(path)
    try:
        version = int(p.stat().st_mtime)
    except OSError:
        version = 0
    return "/gradio_api/file=" + quote(p.as_posix(), safe="/:") + f"?v={version}"


# ------------------------------------------------------------------ charts
def _nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / count
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(round(v, 10))
        v += step
    return ticks


def _chart(series: list[dict], x_max: float, title: str, y_fmt: str = "{:.2f}", markers: list[dict] | None = None,
           hlines: list[dict] | None = None, log_y: bool = False, height: int = 220) -> str:
    """series: [{name, color, points:[(x,y)], dots:bool, width}]"""
    W, H = 760, height
    L, R, T, B = 52, 14, 26, 30
    pts = [p for s in series for p in s["points"] if p[1] is not None and math.isfinite(p[1])]
    for h in hlines or []:
        pts.append((0, h["y"]))
    if not pts:
        return f'<div class="tm-chart tm-empty"><div class="tm-chart-title">{html.escape(title)}</div><div class="tm-muted">Waiting for data…</div></div>'
    tf = (lambda v: math.log10(max(v, 1e-12))) if log_y else (lambda v: v)
    ys = [tf(p[1]) for p in pts]
    y_lo, y_hi = min(ys), max(ys)
    pad = (y_hi - y_lo) * 0.08 or 0.05
    y_lo, y_hi = y_lo - pad, y_hi + pad
    x_hi = max(x_max, max(p[0] for p in pts), 1)

    def X(x):
        return L + (W - L - R) * (x / x_hi)

    def Y(y):
        return T + (H - T - B) * (1 - (tf(y) - y_lo) / (y_hi - y_lo))

    out = [f'<svg viewBox="0 0 {W} {H}" class="tm-svg" preserveAspectRatio="none">']
    if log_y:
        ticks = [10 ** e for e in range(math.floor(y_lo), math.ceil(y_hi) + 1)]
    else:
        ticks = _nice_ticks(y_lo, y_hi)
    for t in ticks:
        if not (y_lo <= tf(t) <= y_hi):
            continue
        yy = Y(t)
        label = f"{t:.0e}" if log_y else y_fmt.format(t)
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{yy:.1f}" y2="{yy:.1f}" class="tm-grid"/>')
        out.append(f'<text x="{L - 6}" y="{yy + 4:.1f}" class="tm-axis" text-anchor="end">{label}</text>')
    for t in _nice_ticks(0, x_hi, 5):
        if 0 <= t <= x_hi:
            xx = X(t)
            out.append(f'<text x="{xx:.1f}" y="{H - 10}" class="tm-axis" text-anchor="middle">{int(t)}</text>')
    for h in hlines or []:
        yy = Y(h["y"])
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{h["color"]}" class="tm-dash"/>')
        out.append(f'<text x="{W - R - 4}" y="{yy - 5:.1f}" class="tm-axis" text-anchor="end" fill="{h["color"]}">{html.escape(h["label"])}</text>')
    for m in markers or []:
        xx = X(m["x"])
        out.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{T}" y2="{H - B}" stroke="{m["color"]}" class="tm-dash"/>')
        out.append(f'<text x="{xx + 4:.1f}" y="{T + 10}" class="tm-axis" fill="{m["color"]}">{html.escape(m["label"])}</text>')
    for s in series:
        p = [(x, y) for x, y in s["points"] if y is not None and math.isfinite(y)]
        if not p:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(p))
        out.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="{s.get("width", 2)}" stroke-linejoin="round"/>')
        if s.get("dots"):
            for x, y in p:
                out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.2" fill="{s["color"]}"/>')
    out.append("</svg>")
    legend = "".join(
        f'<span class="tm-leg"><i style="background:{s["color"]}"></i>{html.escape(s["name"])}</span>' for s in series
    )
    return f'<div class="tm-chart"><div class="tm-chart-head"><span class="tm-chart-title">{html.escape(title)}</span>{legend}</div>{"".join(out)}</div>'


# ------------------------------------------------------------------ main render
def monitor_html(project: str | None, running: bool = False) -> str:
    records = load_metrics(project)
    if not records:
        hint = "Select a project with a V3 run, or start training: charts appear after the first logged steps."
        return f'<div class="tm"><div class="tm-muted">{html.escape(hint)}</div></div>'
    config = next((r for r in reversed(records) if r["type"] == "config"), {})
    train = [r for r in records if r["type"] == "train"]
    evals = [r for r in records if r["type"] == "eval"]
    warns = [r for r in records if r["type"] == "warn"]
    resumes = [r for r in records if r["type"] == "resume"]
    # A continued run is one history: only a stop logged after the latest resume ends it.
    last_resume_at = max((i for i, r in enumerate(records) if r["type"] == "resume"), default=-1)
    stop = next((r for r in reversed(records[last_resume_at + 1:]) if r["type"] == "stop"), None)
    earlier_stops = [r for r in records[: last_resume_at + 1] if r["type"] == "stop"]
    patience_now = next(
        (r.get("patience") for r in reversed(records[last_resume_at + 1:]) if r["type"] in ("patience", "config")), None
    )
    total = float(config.get("total_steps") or 0) or max([r["step"] for r in train] + [1])
    base = next((r["eval_loss"] for r in evals if r.get("base")), None)
    run_evals = [r for r in evals if not r.get("base")]
    best = min(run_evals, key=lambda r: r["eval_loss"]) if run_evals else None
    last = run_evals[-1] if run_evals else None

    # ---- status
    if stop:
        reason = stop.get("reason", "finished")
        label, kind = ("Finished", "ok") if reason == "finished" else ("Stopped early", "warn")
        status_detail = reason if reason != "finished" else f"Final adapter = {stop.get('final', 'last')}"
    elif last and (not resumes or last["step"] > resumes[-1]["step"]):
        label, kind = STATUS_STYLE.get(last.get("status"), ("Training", "info"))
        status_detail = f"patience {last.get('bad_evals', 0)}/{patience_now or '—'}"
    elif resumes:
        label, kind = ("Resumed", "info") if running else ("Resumed (not running)", "info")
        status_detail = f"from step {resumes[-1]['step']} · first eval of this session pending · patience –/{patience_now or '—'}"
    else:
        label, kind, status_detail = "Training", "info", "first eval pending"
    if running and not stop and kind == "ok":
        label = "Improving"

    def card(title, value, sub="", tone=""):
        return (f'<div class="tm-card {tone}"><div class="tm-card-title">{html.escape(title)}</div>'
                f'<div class="tm-card-value">{value}</div><div class="tm-card-sub">{html.escape(sub)}</div></div>')

    cards = [
        card("Status", html.escape(label), status_detail, kind),
        card("Best eval", f"{best['eval_loss']:.3f}" if best else "—",
             f"step {best['step']} · epoch {best.get('epoch', 0):.1f}" if best else "", "ok" if best else ""),
        card("Last eval", f"{last['eval_loss']:.3f}" if last else "—", f"step {last['step']}" if last else ""),
    ]
    if base is not None and best:
        gain = (base - best["eval_loss"]) / base * 100
        cards.append(card("vs base model", f"{gain:+.1f}%", f"base {base:.3f}", "ok" if gain > 0 else "bad"))
    if last and last.get("train_loss") is not None:
        gap = last["eval_loss"] - last["train_loss"]
        gap_best = (best["eval_loss"] - best["train_loss"]) if best and best.get("train_loss") is not None else None
        tone = "bad" if gap_best is not None and gap > gap_best + 0.25 else ""
        cards.append(card("Train/eval gap", f"{gap:.2f}", f"{gap_best:.2f} at best" if gap_best is not None else "", tone))
    if train:
        t = train[-1]
        sub = f"epoch {t.get('epoch', 0):.1f} of {config.get('epochs', '?')}"
        if resumes:
            sub += f" · resumed ×{len(resumes)}"
        cards.append(card("Progress", f"{t['step']}/{int(total)}", sub))
        cards.append(card("LR now", f"{t['lr']:.1e}", config.get("optimizer", "")))

    # ---- charts
    loss_series = [
        {"name": "train loss", "color": "#8b5cf6", "points": [(r["step"], r["loss"]) for r in train], "width": 1.6},
        {"name": "eval loss", "color": "#22d3ee", "points": [(r["step"], r["eval_loss"]) for r in evals], "dots": True, "width": 2.4},
    ]
    markers = []
    if best:
        markers.append({"x": best["step"], "color": "#22c55e", "label": f"best {best['eval_loss']:.3f}"})
    for r in earlier_stops:
        if r.get("reason") not in (None, "finished") and not any(x["step"] == r["step"] for x in resumes):
            markers.append({"x": r["step"], "color": "#f97316", "label": "stopped"})
    for r in resumes:
        markers.append({"x": r["step"], "color": "#60a5fa", "label": "resumed"})
    if stop and stop.get("reason") not in (None, "finished"):
        markers.append({"x": stop["step"], "color": "#f97316", "label": "early stop"})
    hlines = [{"y": base, "color": "#a1a1aa", "label": f"base model {base:.3f}"}] if base is not None else []
    charts = _chart(loss_series, total, "Loss (lower is better) · eval above train and rising = memorising", markers=markers, hlines=hlines)
    resume_marks = [{"x": r["step"], "color": "#60a5fa", "label": "resumed"} for r in resumes]
    lr_chart = _chart([{"name": "learning rate", "color": "#f59e0b", "points": [(r["step"], r["lr"]) for r in train]}],
                      total, "Learning rate", log_y=True, height=170, markers=resume_marks)
    gn_chart = _chart([{"name": "grad norm", "color": "#ec4899", "points": [(r["step"], r.get("grad_norm")) for r in train]}],
                      total, "Gradient norm (spikes = instability)", height=170)

    warn_html = ""
    if warns:
        items = "".join(f'<li><b>step {w["step"]}</b> · {html.escape(w["message"])}</li>' for w in warns[-6:][::-1])
        warn_html = f'<div class="tm-warns"><div class="tm-chart-title">⚠️ Warnings</div><ul>{items}</ul></div>'

    return (
        f'<div class="tm"><div class="tm-cards">{"".join(cards)}</div>{charts}'
        f'<div class="tm-row2">{lr_chart}{gn_chart}</div>{warn_html}</div>'
    )


def audio_evolution_html(project: str | None) -> str:
    records = load_metrics(project)
    audios = [r for r in records if r["type"] == "audio" and Path(r.get("path", "")).exists()]
    if not audios:
        return '<div class="tm"><div class="tm-muted">Eval previews appear here (same text and seed at every eval).</div></div>'
    evals = {r["step"]: r for r in records if r["type"] == "eval"}
    best_step = None
    run_evals = [r for r in records if r["type"] == "eval" and not r.get("base")]
    if run_evals:
        best_step = min(run_evals, key=lambda r: r["eval_loss"])["step"]
    ref = next((r for r in records if r["type"] == "reference" and Path(r.get("path", "")).exists()), None)
    text = next((r for r in reversed(records) if r["type"] == "eval_text"), {})
    head = f'<div class="tm-text"><b>Text:</b> “{html.escape(text.get("text", ""))}” · seed {text.get("seed", "—")}</div>'
    rows = []
    if ref:
        rows.append(
            f'<div class="tm-audio ref"><div class="tm-audio-tag">Real recording</div>'
            f'<audio controls preload="none" src="{_file_url(ref["path"])}"></audio><div class="tm-audio-meta">ground truth (eval set)</div></div>'
        )
    for a in audios:
        ev = evals.get(a["step"])
        tag = "Base model" if a["step"] == 0 else f"Step {a['step']}"
        meta = f"eval {ev['eval_loss']:.3f}" if ev else ""
        if a.get("runaway"):
            meta += " · ⚠️ did not stop"
        badge = '<span class="tm-badge">best</span>' if a["step"] == best_step else ""
        rows.append(
            f'<div class="tm-audio{" best" if badge else ""}{" runaway" if a.get("runaway") else ""}"><div class="tm-audio-tag">{tag} {badge}</div>'
            f'<audio controls preload="none" src="{_file_url(a["path"])}"></audio>'
            f'<div class="tm-audio-meta">{html.escape(meta)} · {a.get("seconds", "?")} s</div></div>'
        )
    return f'<div class="tm">{head}<div class="tm-audios">{"".join(rows)}</div></div>'
