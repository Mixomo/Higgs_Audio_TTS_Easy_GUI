"""Block editors that live in the page: the dialogue builder (any number of lines).

The editor's value is JSON: {"rows": [...], "voices": [...], "loras": [[label, value], ...], "rev": n}.
A row is {"id", "text", "voice", "lora", "audio", "audio_url", "audio_name", "ref_text"}; the page edits
it and sends it back, Python reads it when the dialogue is voiced.
"""

DIALOGUE_EDITOR_HTML = '<div class="dl-root"></div>'

DIALOGUE_EDITOR_CSS = r"""
.dl-root { display:flex; flex-direction:column; gap:6px; }
.dl-bar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.dl-bar button, .dl-add { font-size:12px; padding:4px 10px; border-radius:6px; border:1px solid var(--border-color-primary);
  background:transparent; color:var(--body-text-color); cursor:pointer; }
.dl-bar button:hover, .dl-add:hover { border-color:var(--color-accent); }
.dl-bar .dl-count { margin-left:auto; font-size:12px; opacity:.7; }
.dl-row { border:1px solid var(--border-color-primary); border-radius:8px; background:var(--block-background-fill); padding:6px; }
.dl-row.drag-over { border-color:var(--color-accent); box-shadow:0 -3px 0 var(--color-accent) inset; }
.dl-row.dragging { opacity:.45; }
.dl-main { display:grid; grid-template-columns: 22px 190px 1fr 34px; gap:6px; align-items:start; }
.dl-grip { cursor:grab; font-size:16px; opacity:.6; padding-top:6px; text-align:center; user-select:none; }
.dl-n { font-size:10px; opacity:.6; text-align:center; }
.dl-side { display:flex; flex-direction:column; gap:4px; }
.dl-side select, .dl-side input[type=number], .dl-ref input[type=text] { width:100%; font-size:12px; padding:3px 4px; border-radius:5px;
  border:1px solid var(--border-color-primary); background:var(--input-background-fill); color:var(--body-text-color); }
.dl-side label { font-size:10px; opacity:.65; }
.dl-text { width:100%; min-height:62px; resize:vertical; font-size:13px; padding:6px; border-radius:6px; border:1px solid var(--border-color-primary);
  background:var(--input-background-fill); color:var(--body-text-color); font-family:inherit; box-sizing:border-box; }
.dl-tools { display:flex; flex-direction:column; gap:3px; }
.dl-tools button { font-size:12px; padding:2px 0; border-radius:5px; border:1px solid var(--border-color-primary); background:transparent;
  color:var(--body-text-color); cursor:pointer; }
.dl-tools button:hover { border-color:var(--color-accent); }
.dl-tools button.del:hover { border-color:#e5484d; color:#e5484d; }
.dl-ref { margin:6px 0 0 28px; font-size:12px; }
.dl-ref summary { cursor:pointer; opacity:.8; }
.dl-ref .dl-refbody { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-top:4px; }
.dl-ref audio { height:30px; max-width:280px; }
.dl-ref .dl-up { font-size:12px; padding:3px 8px; border-radius:6px; border:1px solid var(--border-color-primary); cursor:pointer; }
.dl-ref input[type=text] { flex:1; min-width:220px; }
.dl-sep { display:inline-flex; gap:4px; align-items:center; }
.dl-sep select { font-size:12px; padding:3px 4px; border-radius:5px; max-width:230px;
  border:1px solid var(--border-color-primary); background:var(--input-background-fill); color:var(--body-text-color); }
.dl-sepnote { width:100%; font-size:11px; opacity:.75; }
.dl-empty { opacity:.65; padding:14px; border:1px dashed var(--border-color-primary); border-radius:8px; text-align:center; }
"""

DIALOGUE_EDITOR_JS = r"""
const root = element.querySelector('.dl-root');
let D = null, sent = null, dragId = null, timer = null;
function parse() { try { return JSON.parse(props.value || '{}'); } catch (e) { return {}; } }
function uid() { return Math.random().toString(16).slice(2, 10); }
function esc(s) { return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }
function push(now) {
  clearTimeout(timer);
  const go = () => { D.rev = (D.rev || 0) + 1; sent = JSON.stringify(D); props.value = sent; };
  if (now) go(); else timer = setTimeout(go, 350);
}
function opts(list, cur) {
  let h = '';
  for (const o of list) { const [label, val] = Array.isArray(o) ? o : [o, o]; h += `<option value="${esc(val)}"${val === cur ? ' selected' : ''}>${esc(label)}</option>`; }
  return h;
}
function blank(from) { return { id: uid(), text: '', voice: from?.voice || 'None', lora: from?.lora || '__inherit__', pause: from?.pause ?? '', audio: from?.audio || '',
  audio_url: from?.audio_url || '', audio_name: from?.audio_name || '', ref_text: from?.ref_text || '' }; }
function rowHtml(b, i) {
  const voices = D.voices || ['None'];
  const loras = D.loras || [['Same as Single voice', '__inherit__']];
  const engines = window.__hvSepEngines || [];
  const sep = b.audio ? `<span class="dl-sep"><select data-f="sep_engine" title="Separation model or chain">${opts(engines, b.sep_engine || 'voc_fv7+gabox_denoise+becruily_karaoke')}</select>
      <button class="dl-up" data-act="sep" title="Keep only the voice: remove music, noise or other voices"${b._sepBusy ? ' disabled' : ''}>${b._sepBusy ? '⏳ Separating…' : '🎧 Separate'}</button>
      ${b.audio_orig ? '<button class="dl-up" data-act="unsep" title="Bring back the audio before separation">↩ Original</button>' : ''}</span>` : '';
  const ref = b.audio ? `<audio controls preload="none" src="${esc(b.audio_url)}"></audio><span>${esc(b.audio_name || 'reference')}</span>
      <button class="dl-up" data-act="clearref" title="Use the library voice again">✕ Remove</button>${sep}` : '';
  return `<div class="dl-row" draggable="true" data-id="${b.id}">
    <div class="dl-main">
      <div><div class="dl-grip" title="Drag to reorder">⠿</div><div class="dl-n">${i + 1}</div></div>
      <div class="dl-side">
        <label>Voice</label><select data-f="voice">${opts(voices.includes(b.voice) ? voices : [...voices, b.voice], b.voice)}</select>
        <label>LoRA</label><select data-f="lora">${opts(loras.some((l) => l[1] === b.lora) ? loras : [...loras, [b.lora + ' (missing)', b.lora]], b.lora)}</select>
        <label title="Silence before this line; empty: the usual pause (same / other speaker)">Pause before (ms)</label>
        <input type="number" data-f="pause" min="0" max="30000" step="50" placeholder="auto" value="${esc(b.pause ?? '')}">
      </div>
      <textarea class="dl-text tag-ac" data-f="text" placeholder="What speaker ${i + 1} says… (type @ for tags)">${esc(b.text)}</textarea>
      <div class="dl-tools">
        <button data-act="up" title="Move up">↑</button><button data-act="down" title="Move down">↓</button>
        <button data-act="add" title="Add a line below with the same voice and LoRA">＋</button>
        <button data-act="dup" title="Clone this line">⧉</button><button data-act="del" class="del" title="Delete">✕</button>
      </div>
    </div>
    <details class="dl-ref"${b.audio || b._open ? ' open' : ''}><summary>🎙 Own reference audio (optional, wins over the library voice · 5–30 s works best)</summary>
      <div class="dl-refbody">${ref}
        <label class="dl-up">📎 ${b.audio ? 'Replace' : 'Upload'} audio<input type="file" accept="audio/*" hidden data-act="upload"></label>
        <input type="text" data-f="ref_text" value="${esc(b.ref_text)}" placeholder="What the reference says (empty: none or the library transcript)">
        ${b._sepNote ? `<span class="dl-sepnote">${esc(b._sepNote)}</span>` : ''}
      </div></details>
  </div>`;
}
function render() {
  D.rows = D.rows || [];
  const bar = `<div class="dl-bar">
      <button data-act="last">＋ Add a line</button><button data-act="reset" title="Two empty lines">Reset</button>
      <button data-act="cleartext" title="Empty every line's text, keep voices">Clear text</button>
      <button data-act="compact" title="Remove lines without text">Remove empty</button>
      <span class="dl-count">${D.rows.length} line(s) · no limit</span></div>`;
  root.innerHTML = bar + (D.rows.length ? D.rows.map(rowHtml).join('') : '<div class="dl-empty">No lines yet: ＋ Add a line.</div>')
    + (D.rows.length > 3 ? '<button class="dl-add" data-act="last">＋ Add a line at the end</button>' : '');
  root.querySelectorAll('.dl-text').forEach((t) => { t.style.height = 'auto'; t.style.height = Math.max(62, t.scrollHeight + 2) + 'px'; });
}
function rowOf(el) { const card = el.closest('.dl-row'); return card ? D.rows.find((x) => x.id === card.dataset.id) : null; }
root.addEventListener('input', (e) => {
  const el = e.target; const b = rowOf(el); if (!b || !el.dataset.f) return;
  b[el.dataset.f] = el.value;
  if (el.tagName === 'TEXTAREA') { el.style.height = 'auto'; el.style.height = Math.max(62, el.scrollHeight + 2) + 'px'; }
  push(el.tagName === 'SELECT');
});
root.addEventListener('change', async (e) => {
  const el = e.target;
  if (el.dataset.act !== 'upload' || !el.files?.length) return;
  const b = rowOf(el); if (!b) return;
  const f = el.files[0];
  try { const up = await upload(f); b.audio = up.path; b.audio_url = up.url; b.audio_name = f.name; b._open = true;
        delete b.audio_orig; b._sepNote = ''; }
  catch (err) { alert('Upload failed: ' + err); }
  render(); push(true);
});
root.addEventListener('toggle', (e) => { const b = e.target.tagName === 'DETAILS' ? rowOf(e.target) : null; if (b) b._open = e.target.open; }, true);
async function separate(b) {
  b._sepBusy = true; b._sepNote = 'Separating the voice… (models download once, 0.2–0.9 GB)'; render();
  try {
    const r = await server.separate_ref_server(b.audio, b.sep_engine || 'voc_fv7+gabox_denoise+becruily_karaoke');
    if (r && r.ok) {
      if (!b.audio_orig) b.audio_orig = { audio: b.audio, audio_url: b.audio_url, audio_name: b.audio_name };
      b.audio = r.path; b.audio_url = r.url; b.audio_name = r.name; b._sepNote = r.note;
    } else b._sepNote = '❌ ' + ((r && r.error) || 'Separation failed');
  } catch (err) { b._sepNote = '❌ ' + err; }
  b._sepBusy = false; render(); push(true);
}
root.addEventListener('click', (e) => {
  const act = e.target.closest('[data-act]')?.dataset.act; if (!act || act === 'upload') return;
  if (act === 'sep') { const b = rowOf(e.target); if (b && b.audio && !b._sepBusy) separate(b); return; }
  if (act === 'unsep') { const b = rowOf(e.target); if (b && b.audio_orig) { Object.assign(b, b.audio_orig); delete b.audio_orig; b._sepNote = '↩ Original audio back.'; render(); push(true); } return; }
  if (e.target.tagName === 'LABEL' || e.target.closest('label')) return;
  const b = rowOf(e.target); const i = b ? D.rows.indexOf(b) : -1;
  if (act === 'last') D.rows.push(blank(D.rows.at(-1)));
  else if (act === 'reset') D.rows = [blank({ voice: D.default_voice }), blank({ voice: D.default_voice })];
  else if (act === 'cleartext') D.rows.forEach((r) => { r.text = ''; });
  else if (act === 'compact') D.rows = D.rows.filter((r) => (r.text || '').trim());
  else if (act === 'add') D.rows.splice(i + 1, 0, { ...blank(b) });
  else if (act === 'dup') D.rows.splice(i + 1, 0, { ...JSON.parse(JSON.stringify(b)), id: uid() });
  else if (act === 'del') D.rows.splice(i, 1);
  else if (act === 'up' && i > 0) [D.rows[i - 1], D.rows[i]] = [D.rows[i], D.rows[i - 1]];
  else if (act === 'down' && i >= 0 && i < D.rows.length - 1) [D.rows[i + 1], D.rows[i]] = [D.rows[i], D.rows[i + 1]];
  else if (act === 'clearref' && b) { b.audio = ''; b.audio_url = ''; b.audio_name = ''; delete b.audio_orig; b._sepNote = ''; }
  else return;
  render(); push(true);
});
root.addEventListener('dragstart', (e) => {
  const card = e.target.closest?.('.dl-row'); if (!card) return;
  if (['TEXTAREA', 'INPUT', 'SELECT', 'AUDIO'].includes(e.target.tagName)) { e.preventDefault(); return; }
  dragId = card.dataset.id; card.classList.add('dragging'); e.dataTransfer.effectAllowed = 'move';
});
root.addEventListener('dragover', (e) => { const card = e.target.closest('.dl-row'); if (!card || !dragId) return; e.preventDefault();
  root.querySelectorAll('.drag-over').forEach((c) => c.classList.remove('drag-over')); card.classList.add('drag-over'); });
root.addEventListener('drop', (e) => {
  const card = e.target.closest('.dl-row'); if (!card || !dragId) return; e.preventDefault();
  const from = D.rows.findIndex((x) => x.id === dragId); const to = D.rows.findIndex((x) => x.id === card.dataset.id);
  const [b] = D.rows.splice(from, 1); D.rows.splice(to, 0, b);
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
