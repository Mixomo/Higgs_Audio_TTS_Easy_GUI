"""Visual theme, CSS and header for the Gradio app (light + dark aware)."""

from __future__ import annotations

import html

import gradio as gr


def build_theme() -> gr.themes.Base:
    sizes = gr.themes.sizes
    return gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
        radius_size=sizes.radius_lg,
        spacing_size=sizes.spacing_md,
        text_size=sizes.text_md,
    ).set(
        body_background_fill="#f5f5f8",
        body_background_fill_dark="#0b0b10",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#14141b",
        background_fill_secondary="#f5f5f8",
        background_fill_secondary_dark="#101016",
        block_background_fill="#ffffff",
        block_background_fill_dark="#14141b",
        block_border_width="1px",
        block_border_color="#e7e7ee",
        block_border_color_dark="#24242f",
        block_shadow="0 1px 2px rgba(15, 15, 30, 0.04)",
        block_shadow_dark="none",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_text_color="*neutral_500",
        block_label_text_color_dark="*neutral_400",
        block_title_text_weight="600",
        input_background_fill="#fafafc",
        input_background_fill_dark="#0f0f15",
        input_border_color="#e2e2ea",
        input_border_color_dark="#2a2a36",
        input_border_color_focus="*primary_400",
        input_border_color_focus_dark="*primary_500",
        button_primary_background_fill="linear-gradient(135deg, *primary_500, *primary_700)",
        button_primary_background_fill_hover="linear-gradient(135deg, *primary_400, *primary_600)",
        button_primary_background_fill_dark="linear-gradient(135deg, *primary_500, *primary_700)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, *primary_400, *primary_600)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_border_color="transparent",
        button_primary_border_color_dark="transparent",
        button_secondary_background_fill="#f1f1f5",
        button_secondary_background_fill_dark="#1d1d27",
        button_secondary_background_fill_hover="#e8e8ef",
        button_secondary_background_fill_hover_dark="#262633",
        button_secondary_border_color="transparent",
        button_secondary_border_color_dark="transparent",
        button_cancel_background_fill="#fde8e8",
        button_cancel_background_fill_dark="#2a1416",
        button_cancel_text_color="#b42318",
        button_cancel_text_color_dark="#f87171",
        slider_color="*primary_500",
        slider_color_dark="*primary_500",
        checkbox_background_color_selected="*primary_600",
        checkbox_background_color_selected_dark="*primary_500",
        checkbox_border_color_selected="*primary_600",
        checkbox_border_color_selected_dark="*primary_500",
        color_accent_soft="*primary_50",
        color_accent_soft_dark="rgba(139, 92, 246, 0.14)",
        border_color_accent="*primary_300",
        border_color_accent_dark="*primary_600",
        layout_gap="10px",
        # Gradio paints groups, form gaps and many borders with these: keep them in the app's palette
        border_color_primary="#e7e7ee",
        border_color_primary_dark="#24242f",
        panel_background_fill="#ffffff",
        panel_background_fill_dark="#14141b",
        panel_border_color="#e7e7ee",
        panel_border_color_dark="#24242f",
        code_background_fill="#f1f1f5",
        code_background_fill_dark="#101016",
        table_border_color="#e7e7ee",
        table_border_color_dark="#24242f",
        table_even_background_fill="#ffffff",
        table_even_background_fill_dark="#14141b",
        table_odd_background_fill="#fafafc",
        table_odd_background_fill_dark="#111118",
        table_row_focus="*primary_50",
        table_row_focus_dark="rgba(139, 92, 246, 0.14)",
        input_background_fill_hover_dark="#12121a",
        block_label_border_color_dark="#24242f",
        block_title_background_fill_dark="transparent",
        accordion_text_color_dark="*neutral_200",
    )


CSS = """
/* ---------- layout: use the screen width ----------
   Content spans the viewport up to 2560 px (beyond that lines get too long to
   scan and the app just centres). Outer padding scales with the viewport. */
.gradio-container { max-width: 2560px !important; width: 100% !important; margin: 0 auto !important; }
.gradio-container .fillable, .gradio-container .main, .gradio-container .contain {
  max-width: none !important;
  padding: clamp(6px, 0.8vw, 16px) clamp(8px, 1.1vw, 24px) !important;
}
.gradio-container main { padding: 0 !important; }
.app-header {
  display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap;
  padding: 12px 18px; border-radius: 14px; margin-bottom: 0;
  background: radial-gradient(120% 140% at 0% 0%, rgba(139,92,246,.18), transparent 55%),
              radial-gradient(120% 140% at 100% 100%, rgba(34,211,238,.12), transparent 50%),
              var(--block-background-fill);
  border: 1px solid var(--block-border-color);
}
.app-brand { display: flex; align-items: center; gap: 14px; }
.app-title { font-size: 1.32rem; font-weight: 700; letter-spacing: -.01em; color: var(--body-text-color); line-height: 1.2; }
.app-subtitle { font-size: .86rem; color: #5f5f6e; margin-top: 2px; }
.dark .app-subtitle { color: #a1a1aa; }
.hint, .text-stats, .result-meta { color: #5f5f6e !important; }
.dark .hint, .dark .text-stats, .dark .result-meta { color: #a1a1aa !important; }
.app-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
  font-size: .78rem; font-weight: 600; padding: 5px 11px; border-radius: 999px; white-space: nowrap;
  border: 1px solid var(--block-border-color); background: var(--background-fill-secondary); color: var(--body-text-color);
}
.chip.ok { border-color: rgba(34,197,94,.35); background: rgba(34,197,94,.10); color: #16a34a; }
.chip.accent { border-color: rgba(139,92,246,.40); background: rgba(139,92,246,.12); color: #7c3aed; }
.dark .chip.ok { color: #4ade80; }
.dark .chip.accent { color: #c4b5fd; }
.header-actions { justify-content: flex-end; gap: 8px !important; }
.header-actions { align-items: center; }
.header-actions .gui-preset-select { margin-left: auto; }
.gui-preset-select input::placeholder { opacity: .7; }
.header-actions button { flex: 0 0 auto !important; min-width: 0 !important; }

/* ---------- cards & sections ---------- */
.card, .form-section {
  background: var(--block-background-fill) !important;
  border: 1px solid var(--block-border-color) !important;
  border-radius: 14px !important; padding: 10px 12px !important;
  gap: 8px !important;
}
/* One frame per card: blocks and form groups inside a card drop their own
   border/background/padding, only the inputs keep their outline. */
.card .block, .card .form, .form-section .block, .form-section .form {
  border-color: transparent !important; box-shadow: none !important; background: transparent !important;
}
.card .block, .form-section .block { padding: 4px 4px !important; }
.card .form, .form-section .form { gap: 6px !important; }
/* Dataset / Training panels: short fields flow into as many columns as fit
   (min 210 px each) instead of one tall list; radios and text areas span the row. */
.form-section > .form {
  display: grid !important; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 6px 10px !important;
}
.form-section > .form > * { min-width: 0 !important; width: auto !important; justify-self: stretch; margin: 0 !important; }
.form-section > .form > *:has(input[type="radio"]),
.form-section > .form > *:has(textarea) { grid-column: 1 / -1; }
/* Icon buttons (🔄, ⏹) next to dropdowns: compact, aligned to the input. */
.card button.sm, .card .row > button { align-self: flex-end; }
.card .row > button:not(.lg) { min-height: 40px !important; max-height: 42px; margin-bottom: 4px; }
.section-title {
  display: flex; align-items: center; gap: 8px; margin: 2px 0 6px 0;
  font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  color: var(--body-text-color-subdued);
}
.section-title .dot { width: 7px; height: 7px; border-radius: 50%; background: linear-gradient(135deg, #8b5cf6, #22d3ee); }
.hint { font-size: .85rem; color: var(--body-text-color-subdued); line-height: 1.5; }
.text-stats { font-size: .82rem; color: var(--body-text-color-subdued); padding: 2px 4px 0 4px; min-height: 1.2em; }
.text-stats p { margin: 0 !important; }

/* ---------- script editor ---------- */
.script-box textarea {
  font-size: 1.02rem !important; line-height: 1.65 !important; min-height: 230px;
  font-family: var(--font) !important;
}

/* ---------- @ tag autocomplete ---------- */
#tag-ac-pop {
  display: none; position: fixed; z-index: 10000; width: 340px; max-width: calc(100vw - 16px); max-height: 320px;
  overflow-y: auto; padding: 4px; border-radius: 10px; border: 1px solid var(--border-color-primary);
  background: var(--background-fill-primary); box-shadow: 0 14px 36px rgba(0,0,0,.35); font-size: .86rem;
}
#tag-ac-pop .tag-ac-cat {
  padding: 6px 8px 2px; font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  color: var(--body-text-color-subdued);
}
#tag-ac-pop .tag-ac-item {
  display: flex; justify-content: space-between; gap: 10px; padding: 5px 8px; border-radius: 6px; cursor: pointer;
}
#tag-ac-pop .tag-ac-item.active, #tag-ac-pop .tag-ac-item:hover { background: rgba(139,92,246,.22); }
#tag-ac-pop .tag-ac-label { font-family: var(--font-mono, monospace); font-variant-ligatures: none; font-feature-settings: "liga" 0, "calt" 0; color: var(--body-text-color); white-space: nowrap; }
#tag-ac-pop .tag-ac-hint { color: var(--body-text-color-subdued); font-size: .74rem; white-space: nowrap; }
#tag-ac-pop .tag-ac-empty { padding: 8px; color: var(--body-text-color-subdued); }

/* ---------- buttons ---------- */
.button-primary { font-weight: 700 !important; letter-spacing: .01em; box-shadow: 0 8px 22px rgba(124,58,237,.28); }
.button-primary:disabled { box-shadow: none; opacity: .7; }
.button-stop:disabled, .red-btn:disabled { opacity: .45; }
.button-stop { font-weight: 600 !important; }
.green-btn { background: rgba(34,197,94,.14) !important; color: #16a34a !important; border: none !important; }
.red-btn { background: rgba(239,68,68,.12) !important; color: #dc2626 !important; border: none !important; }
.dark .green-btn { color: #4ade80 !important; }
.dark .red-btn { color: #f87171 !important; }

/* ---------- output ---------- */
.output-card { border-color: rgba(139,92,246,.35) !important; }
.live-player audio { width: 100%; }
.result-meta { font-size: .85rem; color: var(--body-text-color-subdued); }
.output-path textarea { font-family: var(--font-mono) !important; font-size: .78rem !important; min-height: 38px !important; }

/* ---------- progress bar ---------- */
.progress-card { padding: 2px 0 6px 0; }
.pb { display: flex; flex-direction: column; gap: 8px; }
.pb-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.pb-title { font-weight: 650; font-size: .95rem; color: var(--body-text-color); }
.pb-pct { font-weight: 700; font-variant-numeric: tabular-nums; color: #7c3aed; }
.dark .pb-pct { color: #c4b5fd; }
.pb-track { position: relative; height: 10px; border-radius: 999px; overflow: hidden; background: rgba(127,127,150,.18); }
.pb-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #8b5cf6, #22d3ee); transition: width .3s ease; }
.pb-indeterminate { position: absolute; width: 35%; animation: pb-slide 1.2s ease-in-out infinite; }
@keyframes pb-slide { 0% { left: -35%; } 100% { left: 100%; } }
.pb-detail { font-size: .82rem; font-variant-numeric: tabular-nums; color: #5f5f6e; }
.dark .pb-detail { color: #a1a1aa; }
.pb-done .pb-fill { background: linear-gradient(90deg, #22c55e, #10b981); }
.pb-done .pb-title { color: #16a34a; }
.dark .pb-done .pb-title { color: #4ade80; }
.pb-error .pb-title { color: #dc2626; }
.dark .pb-error .pb-title { color: #f87171; }
.pb-idle .pb-title { color: #5f5f6e; }
.dark .pb-idle .pb-title { color: #a1a1aa; }

/* ---------- project card (dataset -> training -> voice) ---------- */
.proj-card {
  border: 1px solid var(--block-border-color); border-radius: 14px; padding: 12px 16px;
  background: linear-gradient(90deg, rgba(139,92,246,.08), rgba(34,211,238,.05)), var(--block-background-fill);
}
.proj-empty { opacity: .85; }
.proj-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }
.proj-title { font-weight: 700; font-size: 1.05rem; color: var(--body-text-color); }
.proj-sub { font-size: .8rem; color: #5f5f6e; font-family: var(--font-mono); }
.dark .proj-sub { color: #a1a1aa; }
.proj-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
.proj-step { display: flex; gap: 10px; align-items: flex-start; padding: 8px 10px; border-radius: 10px; background: rgba(127,127,150,.07); }
.proj-dot {
  flex: none; width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center;
  font-size: .78rem; font-weight: 700; background: rgba(127,127,150,.22); color: var(--body-text-color);
}
.proj-step.done .proj-dot { background: linear-gradient(135deg, #22c55e, #10b981); color: #fff; }
.proj-step.active .proj-dot { background: linear-gradient(135deg, #8b5cf6, #22d3ee); color: #fff; animation: proj-pulse 1.4s ease-in-out infinite; }
@keyframes proj-pulse { 50% { opacity: .55; } }
.proj-step-title { font-weight: 650; font-size: .9rem; color: var(--body-text-color); }
.proj-step-detail { font-size: .8rem; color: #5f5f6e; line-height: 1.4; }
.dark .proj-step-detail { color: #a1a1aa; }

/* ---------- training monitor ---------- */
.tm { display: flex; flex-direction: column; gap: 12px; }
.tm-muted { color: #5f5f6e; font-size: .88rem; padding: 6px 2px; }
.dark .tm-muted { color: #a1a1aa; }
.tm-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }
.tm-card { border: 1px solid var(--block-border-color); border-radius: 12px; padding: 8px 12px; background: rgba(127,127,150,.06); }
.tm-card.ok { border-color: rgba(34,197,94,.45); background: rgba(34,197,94,.08); }
.tm-card.warn { border-color: rgba(245,158,11,.5); background: rgba(245,158,11,.09); }
.tm-card.bad { border-color: rgba(239,68,68,.5); background: rgba(239,68,68,.09); }
.tm-card.info { border-color: rgba(139,92,246,.45); background: rgba(139,92,246,.08); }
.tm-card-title { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: #5f5f6e; font-weight: 700; }
.dark .tm-card-title { color: #a1a1aa; }
.tm-card-value { font-size: 1.25rem; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--body-text-color); }
.tm-card-sub { font-size: .76rem; color: #5f5f6e; }
.dark .tm-card-sub { color: #a1a1aa; }
.tm-chart { border: 1px solid var(--block-border-color); border-radius: 12px; padding: 8px 10px 4px; background: rgba(127,127,150,.04); }
.tm-chart-head { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 2px; }
.tm-chart-title { font-weight: 650; font-size: .85rem; color: var(--body-text-color); margin-right: auto; }
.tm-leg { font-size: .76rem; color: #5f5f6e; display: inline-flex; align-items: center; gap: 5px; }
.dark .tm-leg { color: #a1a1aa; }
.tm-leg i { width: 12px; height: 3px; border-radius: 2px; display: inline-block; }
.tm-svg { width: 100%; height: auto; display: block; }
.tm-grid { stroke: rgba(127,127,150,.18); stroke-width: 1; }
.tm-dash { stroke-width: 1.4; stroke-dasharray: 5 4; }
.tm-axis { font-size: 11px; fill: #71717a; font-family: var(--font-mono); }
.tm-row2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
.tm-warns { border: 1px solid rgba(245,158,11,.45); background: rgba(245,158,11,.07); border-radius: 12px; padding: 8px 12px; }
.tm-warns ul { margin: 6px 0 0 0; padding-left: 18px; font-size: .84rem; }
.tm-text { font-size: .88rem; color: var(--body-text-color); }
.tm-audios { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px; }
.tm-audio { border: 1px solid var(--block-border-color); border-radius: 12px; padding: 8px 10px; display: flex; flex-direction: column; gap: 6px; }
.tm-audio.best { border-color: rgba(34,197,94,.6); background: rgba(34,197,94,.07); }
.tm-audio.ref { border-color: rgba(139,92,246,.5); background: rgba(139,92,246,.07); }
.tm-audio.runaway { border-color: rgba(239,68,68,.55); background: rgba(239,68,68,.07); }
.tm-audio audio { width: 100%; height: 34px; }
.tm-audio-tag { font-weight: 650; font-size: .85rem; color: var(--body-text-color); }
.tm-audio-meta { font-size: .76rem; color: #5f5f6e; }
.dark .tm-audio-meta { color: #a1a1aa; }
.tm-badge { font-size: .68rem; font-weight: 700; padding: 1px 7px; border-radius: 999px; background: #22c55e; color: #fff; margin-left: 6px; }

/* ---------- tabs ---------- */
.main-tabs > .tab-wrapper { margin-bottom: 6px; }
.main-tabs button[role="tab"] { font-weight: 600; font-size: .95rem; }

/* ---------- warnings ---------- */
.warn-box {
  border: 1px solid rgba(239,68,68,.35); background: rgba(239,68,68,.08);
  border-radius: 14px; padding: 16px 18px; margin-bottom: 14px; line-height: 1.6;
}
.warn-box .warn-title { color: #dc2626; font-weight: 700; margin-bottom: 6px; }
.dark .warn-box .warn-title { color: #f87171; }

/* ---------- legacy helpers kept for existing components ---------- */
.clips-count-mini { opacity: .8; font-size: 13px; }
.compact textarea { font-family: var(--font-mono); }
.global-inference-controls { margin-top: 4px; }
.audio-safe-space { overflow: visible !important; border: 0 !important; box-shadow: none !important; }
.audio-safe-space .wrap, .audio-safe-space [data-testid="audio"], .audio-safe-space [data-testid="waveform"] {
  overflow: visible !important; border: 0 !important; box-shadow: none !important;
}
.audio-safe-space .timestamps, .audio-safe-space .time, .audio-safe-space [class*="time"] {
  position: relative !important; z-index: 3 !important; bottom: auto !important;
}
.audio-safe-space *, .audio-safe-space *::before, .audio-safe-space *::after { scrollbar-width: none !important; -ms-overflow-style: none !important; }
.audio-safe-space ::-webkit-scrollbar { width: 0 !important; height: 0 !important; display: none !important; }
/* Long audio: wavesurfer adds a horizontal scrollbar inside its shadow DOM, which made the
   fixed-height #waveform box overflow onto the time labels. Let the box grow and keep the bar thin. */
div#waveform { height: auto !important; }
*::part(scroll) { scrollbar-width: thin; scrollbar-color: rgba(139,92,246,.55) transparent; }
.output-clean, .output-clean > div, .output-clean .wrap, .output-clean [data-testid="block-info"] { border: 0 !important; box-shadow: none !important; }
.console-accordion, .console-accordion > div { border-radius: 16px !important; }
#higgs-console-body::-webkit-scrollbar { width: 5px; }
#higgs-console-body::-webkit-scrollbar-track { background: transparent; }
#higgs-console-body::-webkit-scrollbar-thumb { background: #555; border-radius: 3px; }

/* ---------- Studio grid ----------
   < 900 px : one column.  900-1480 px : script | voice over settings.
   >= 1480 px (125 % zoom on 1080p and below): script | voice | settings. */
.studio-grid { display: grid !important; gap: 10px !important; align-items: start;
  grid-template-columns: minmax(0, 1fr); grid-template-areas: "main" "voice" "settings"; }
.studio-grid > .studio-main { grid-area: main; min-width: 0 !important; }
.studio-grid > .studio-voice { grid-area: voice; min-width: 0 !important; }
.studio-grid > .studio-settings { grid-area: settings; min-width: 0 !important; }
@media (min-width: 900px) {
  .studio-grid { grid-template-columns: minmax(0, 7fr) minmax(0, 5fr);
    grid-template-areas: "main voice" "main settings"; grid-template-rows: auto 1fr; }
}
@media (min-width: 1480px) {
  .studio-grid { grid-template-columns: minmax(0, 6fr) minmax(0, 4fr) minmax(0, 4fr);
    grid-template-areas: "main voice settings"; grid-template-rows: auto; }
  .script-box textarea { min-height: 300px; }
}
@media (max-width: 900px) {
  .app-header { padding: 10px 12px; }
  .app-title { font-size: 1.08rem; }
  .card, .form-section { padding: 8px 8px !important; }
}
"""


def _gpu_label() -> tuple[str, bool]:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{props.name.replace('NVIDIA ', '').replace('GeForce ', '')} · {props.total_memory / 1024**3:.0f} GB", True
    except Exception:
        pass
    return "CPU", False


def header_html(title: str = "Higgs Audio Studio") -> str:
    gpu, has_cuda = _gpu_label()
    engine = "Fast engine · CUDA graphs" if has_cuda else "Compatibility engine"
    chips = [
        f'<span class="chip {"ok" if has_cuda else ""}">{html.escape(gpu)}</span>',
        f'<span class="chip accent">{html.escape(engine)}</span>',
        '<span class="chip">LoRA voices</span>',
    ]
    return f"""
<div class="app-header">
  <div class="app-brand">
    <div>
      <div class="app-title">{html.escape(title)}</div>
      <div class="app-subtitle">Local Higgs Audio V2 / V3 · voice cloning, LoRA voices, dialogue and fine-tuning</div>
    </div>
  </div>
  <div class="app-chips">{''.join(chips)}</div>
</div>
"""


def section(title: str) -> str:
    return f'<div class="section-title"><span class="dot"></span>{html.escape(title)}</div>'


TRAINING_WARNING_HTML = """
<div class="warn-box">
  <div class="warn-title">⚠️ Experimental</div>
  <div>
    Higgs Audio V3 training is unofficial. There is no official implementation or guide for fine-tuning
    <strong>Higgs Audio V3 TTS</strong>; this app adapts the documented supervised workflow for
    <strong>Higgs Audio V2</strong> to the V3 Transformers architecture.
    <br><strong>GPU requirement:</strong> V2 and V3 training need a GPU with <strong>at least 24 GB of VRAM</strong>.
  </div>
</div>
"""


CSS += r'''

/* ---------------- modals: a card over the page, the page blurred behind it */
.hv-modal { position: fixed !important; inset: 0 !important; z-index: 9500 !important; display: flex; align-items: center;
  justify-content: center; padding: 24px !important; margin: 0 !important; width: auto !important; max-width: none !important;
  min-width: 0 !important; background: rgba(6, 6, 14, .55) !important; backdrop-filter: blur(8px) saturate(115%);
  -webkit-backdrop-filter: blur(8px) saturate(115%); border: 0 !important; border-radius: 0 !important; gap: 0 !important;
  animation: hv-fade .16s ease-out; }
.hv-modal > .hv-card, .hv-card.tl-modal { width: min(620px, calc(100vw - 32px)) !important; max-width: 620px; flex: 0 0 auto !important;
  max-height: calc(100vh - 48px); overflow: auto; background: var(--block-background-fill) !important; color: var(--body-text-color);
  border: 1px solid var(--border-color-primary) !important; border-radius: 18px !important; padding: 18px 20px 16px !important;
  box-shadow: 0 24px 64px rgba(0, 0, 0, .45), 0 2px 10px rgba(0, 0, 0, .25); gap: 12px !important;
  animation: hv-pop .2s cubic-bezier(.2, .9, .3, 1.15); }
.hv-card.tl-modal { width: min(720px, calc(100vw - 32px)) !important; max-width: 720px; }
.hv-modal.hide, .hv-modal.hidden, .hv-modal[hidden] { display: none !important; }
.hv-head { display: flex; align-items: center; gap: 10px; }
.hv-title { flex: 1; font-size: 16px; font-weight: 650; letter-spacing: -.01em; color: var(--body-text-color); }
.hv-x { width: 32px; height: 32px; border: 0; border-radius: 10px; background: transparent; color: var(--body-text-color-subdued);
  font-size: 15px; cursor: pointer; transition: background .12s, color .12s; }
.hv-x:hover { background: var(--background-fill-secondary); color: var(--body-text-color); }
.hv-text, .hv-text p, .hv-text li { color: var(--body-text-color-subdued) !important; font-size: 14px; }
.hv-text code { font-size: 12px; }
.hv-actions { justify-content: flex-end !important; gap: 8px !important; flex-wrap: wrap !important; margin-top: 4px; }
.hv-actions > button, .hv-actions button { flex: 0 0 auto !important; min-width: 104px !important; width: auto !important; padding-inline: 14px !important; }
.hv-card .hv-hidden-label { display: none; }
@keyframes hv-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes hv-pop { from { opacity: 0; transform: translateY(10px) scale(.97); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .hv-modal, .hv-modal > .hv-card, .hv-card.tl-modal { animation: none; } }
'''


CSS += r'''

/* ---------------- one palette end to end: no gray slabs from Gradio's defaults */
.gr-group, .gr-group > .styler, .gr-group .styler, .form, .form > .styler {
  background: var(--block-background-fill) !important; }
.gr-group { border: 1px solid var(--block-border-color) !important; border-radius: var(--block-radius) !important;
  overflow: hidden; }
.gr-group .form, .gr-group > .styler > .block, .gr-group .block { border-color: var(--block-border-color) !important; }
.form { border-color: var(--block-border-color) !important; }
.dark code, .dark pre { background: var(--code-background-fill) !important; border: 1px solid var(--border-color-primary); }
.dark th, .dark th.header-cell { background: #17171f !important; color: var(--body-text-color) !important; }
.dark td, .dark th, .dark table, .dark tr { border-color: var(--table-border-color) !important; }
.prose table { border-radius: 10px; overflow: hidden; border-collapse: separate; border-spacing: 0; border: 1px solid var(--table-border-color); }
.dark .warn-box { border-color: rgba(239, 68, 68, .28) !important; }
.dark .upload-container, .dark .wrap.default, .dark [data-testid="upload"], .dark .audio-container .empty {
  background: var(--input-background-fill) !important; }
.dark ::-webkit-scrollbar { width: 10px; height: 10px; }
.dark ::-webkit-scrollbar-track { background: transparent; }
.dark ::-webkit-scrollbar-thumb { background: #262633; border-radius: 10px; border: 2px solid var(--body-background-fill); }
.dark ::-webkit-scrollbar-thumb:hover { background: #33334a; }
input[type=range] { accent-color: var(--color-accent); }
'''
