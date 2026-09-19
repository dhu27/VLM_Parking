"""Sign labeling tool (Phase 3): transcribe accepted signs into the schema, in the browser.

    uv run python scripts/label.py                 # every triage-accepted sign, by hand, blind (the default)
    uv run python scripts/label.py --set assisted  # (unused) pre-filled from data/labels/prefill/<id>.json
    uv run python scripts/label.py --round 2       # blind re-label for self-consistency (hides round 1)

Then open http://localhost:8766. Follow ANNOTATION_GUIDE.md.

- Left: the crop (zoom with the slider or the +/- keys) and a link to the full photo.
- Right: one card per panel, sign-level fields, and a "check a query" box that runs the evaluator
  on the form as it stands (use it for the plan's 30 hand-checks).
- Keys (when not typing in a field): a = save, r = reject, f = toggle ambiguous, n / p = next / previous,
  + = add panel. Cmd/Ctrl+Enter saves from anywhere.

Saved to data/labels/labels.sqlite: one row per (sign, round) plus an append-only history, with
provenance, time spent, and a timestamp. All labels are manual ("gold_manual"): model-assisted labeling was
dropped because manual labeling is fast (~23 s per labeled sign). The first 100 in the queue are the original
stratified sample in data/labels/gold_set.csv, drawn once with a fixed seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from vlm_parking import dataset
from vlm_parking.evaluator import evaluate
from vlm_parking.schema import Query, Sign

COLLECT = Path("data/collect")
LABELS = Path("data/labels")
DB = LABELS / "labels.sqlite"
GOLD = LABELS / "gold_set.csv"
PREFILL = LABELS / "prefill"

GOLD_TARGETS = {"1": 40, "2": 35, "3+": 25}  # detected-panel strata; oversample complex signs
REJECT_REASONS = ["illegible_panel", "illegible_fine_print", "cut_off", "multiple_signs", "unsupported_condition", "not_parking"]

ARGS: argparse.Namespace


# --- data -----------------------------------------------------------------------------------------


def accepted_candidates() -> pd.DataFrame:
    return dataset.accepted_candidates(COLLECT)  # triage accept AND OCR-confirmed parking text


def stratum(n_panels: int) -> str:
    return "3+" if n_panels >= 3 else str(n_panels)


def ensure_gold_set(seed: int = 0) -> pd.DataFrame:
    if GOLD.exists():
        return pd.read_csv(GOLD)
    acc = accepted_candidates().assign(stratum=lambda d: d.n_panels_detected.map(stratum))
    rng = random.Random(seed)
    picked = []
    for s, n in GOLD_TARGETS.items():
        ids = sorted(acc.loc[acc.stratum == s, "candidate_id"])
        rng.shuffle(ids)
        picked += [(cid, s) for cid in ids[:n]]
    rng.shuffle(picked)  # don't label all the hard ones in a row
    LABELS.mkdir(parents=True, exist_ok=True)
    gold = pd.DataFrame(picked, columns=["candidate_id", "stratum"])
    gold.to_csv(GOLD, index=False)
    return gold


def queue() -> pd.DataFrame:
    """Every sign accepted in triage, labeled by hand (all labels are gold).

    The original stratified 100 come first (in their fixed order), then the rest, most promising first:
    OCR-confirmed parking text, then more detected panels, then larger signs. In the gold sample, OCR-confirmed
    signs were fully transcribable 39% of the time vs 7% for OCR-unknown, so the tail is mostly quick rejects;
    it is still labeled, so hard-to-read signs aren't excluded. Signs accepted in later triage rounds appear on reload.
    """
    acc = accepted_candidates()
    gold_ids = [c for c in ensure_gold_set().candidate_id if c in set(acc.candidate_id)]
    rest = acc[~acc.candidate_id.isin(gold_ids)].assign(_p=lambda d: d.ocr_status.eq("parking"))
    rest = rest.sort_values(["_p", "n_panels_detected", "height_native", "candidate_id"], ascending=[False, False, False, True])
    ordered = pd.concat([acc.set_index("candidate_id").loc[gold_ids].reset_index(), rest.drop(columns="_p")])
    if ARGS.set == "assisted":  # kept for completeness; not used (all labeling is manual)
        return ordered[~ordered.candidate_id.isin(gold_ids)].reset_index(drop=True)
    return ordered.reset_index(drop=True)


def db() -> sqlite3.Connection:
    LABELS.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    cols = """candidate_id TEXT, round INTEGER, label_set TEXT, status TEXT, reject_reason TEXT,
              sign_json TEXT, provenance TEXT, seconds REAL, saved_at TEXT"""
    con.execute(f"CREATE TABLE IF NOT EXISTS labels ({cols}, PRIMARY KEY (candidate_id, round))")
    con.execute(f"CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
    return con


def saved_labels() -> dict[str, dict]:
    with db() as con:
        rows = con.execute(
            "SELECT candidate_id, status, reject_reason, sign_json, seconds FROM labels WHERE round = ?", (ARGS.round,)
        ).fetchall()
    return {r[0]: {"status": r[1], "reject_reason": r[2], "sign": json.loads(r[3]) if r[3] else None, "seconds": r[4]} for r in rows}


def prefill(cid: str) -> dict | None:
    if ARGS.set != "assisted":
        return None
    path = PREFILL / f"{cid.replace('/', '__')}.json"
    return json.loads(path.read_text()) if path.exists() else None


def write(cid: str, status: str, reason: str | None, sign: dict | None, provenance: str | None, seconds: float) -> None:
    row = (cid, ARGS.round, ARGS.set, status, reason, json.dumps(sign) if sign else None, provenance, seconds,
           datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with db() as con:
        con.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?,?,?)", row)
        con.execute("INSERT INTO history (candidate_id, round, label_set, status, reject_reason, sign_json, provenance, seconds, saved_at) VALUES (?,?,?,?,?,?,?,?,?)", row)


def validation_messages(err: ValidationError) -> list[str]:
    out = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"])
        out.append(f"{loc}: {e['msg']}" if loc else e["msg"])
    return out


def build_sign(cid: str, form: dict) -> Sign:
    return Sign(sign_id=cid, **{k: v for k, v in form.items() if k != "sign_id"})


# --- page -----------------------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Sign labeling</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#f6f6f4; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#dcdcd8; --accent:#1a66d6; --ok:#1e8e3e; --bad:#d93025; --warn:#b77900; }
  @media (prefers-color-scheme: dark) { :root { --bg:#171716; --fg:#ececea; --muted:#9a9a94; --card:#222221; --line:#3a3a38; --accent:#6aa3ff; } }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.4 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  header { display:flex; gap:14px; align-items:center; flex-wrap:wrap; padding:8px 16px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--bg); z-index:3; }
  header .muted { color:var(--muted); }
  main { display:grid; grid-template-columns: minmax(300px, 42%) 1fr; gap:16px; padding:12px 16px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .left { position:sticky; top:56px; align-self:start; }
  #viewer { height: calc(100vh - 150px); min-height:300px; overflow:auto; background:#111; border-radius:8px; display:flex; }
  #viewer img { margin:auto; image-rendering:auto; }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-bottom:10px; }
  .card h4 { margin:0 0 8px; display:flex; justify-content:space-between; align-items:center; font-size:13px; }
  label { font-size:12px; color:var(--muted); display:flex; flex-direction:column; gap:2px; }
  label.inline { flex-direction:row; align-items:center; gap:4px; color:var(--fg); font-size:13px; }
  input, select, textarea, button { font:inherit; color:var(--fg); background:var(--card); border:1px solid var(--line); border-radius:6px; padding:5px 8px; }
  input[type=text] { width:80px; } input.wide { width:100%; } textarea { width:100%; min-height:40px; }
  input[type=checkbox] { width:auto; }
  button { cursor:pointer; } button.primary { background:var(--ok); border-color:var(--ok); color:#fff; } button.danger { color:var(--bad); }
  .days button { padding:3px 7px; font-size:12px; } .days button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
  .days .preset { color:var(--muted); }
  .status { font-weight:600; } .status.labeled { color:var(--ok); } .status.rejected { color:var(--bad); }
  #errors { color:var(--bad); white-space:pre-wrap; font-size:13px; }
  #evalout { font-size:13px; margin-top:6px; }
  .kbd { font-size:11px; color:var(--muted); }
  .hidden { display:none; }
  .card.error { border-color:var(--bad); box-shadow:0 0 0 2px color-mix(in srgb, var(--bad) 30%, transparent); }
</style></head><body>
<header>
  <b>Sign labeling</b><span class="muted" id="setinfo"></span>
  <span id="progress" class="muted"></span>
  <button id="prev">← Prev (p)</button><button id="next">Next (n) →</button>
  <span class="status" id="status"></span>
  <span class="kbd">a save · r reject · f ambiguous · + panel · ⌘/Ctrl+Enter save</span>
</header>
<main>
  <div class="left">
    <div class="row" style="margin-bottom:6px">
      <span id="cid" class="muted"></span>
      <label class="inline">Zoom <input type="range" id="zoom" min="1" max="6" step="0.25" value="1"></label>
      <a id="full" target="_blank">full photo ↗</a>
    </div>
    <div id="viewer"><img id="img" alt="sign crop"></div>
  </div>
  <div>
    <div id="panels"></div>
    <div class="row" style="margin-bottom:10px"><button id="addpanel">+ Add panel</button><span class="kbd">Panels top to bottom. One rule per panel; two time windows = two panels.</span></div>

    <div class="card"><h4>Sign</h4>
      <div class="row">
        <label>Arrow <select id="arrow"><option value="none">none</option><option>left</option><option>right</option><option>both</option></select></label>
        <label>Angle <select id="angle"><option>frontal</option><option>oblique</option><option>severe</option></select></label>
        <label>Legibility <select id="legibility"><option>clear</option><option>hard</option></select></label>
        <label class="inline"><input type="checkbox" id="glare"> glare</label>
        <label class="inline"><input type="checkbox" id="occlusion"> occlusion</label>
        <label class="inline"><input type="checkbox" id="ambiguous"> ambiguous (f)</label>
      </div>
      <label style="margin-top:6px">Notes <textarea id="notes"></textarea></label>
    </div>

    <div class="row" style="margin-bottom:10px">
      <button class="primary" id="save">Save &amp; next (a)</button>
      <select id="reason"><option value="">Reject reason…</option>__REASONS__</select>
      <button class="danger" id="reject">Reject (r)</button>
    </div>
    <div id="errors"></div>

    <div class="card"><h4>Check a query against this form <span class="kbd">evaluator, unsaved state</span></h4>
      <div class="row">
        <select id="qday"><option>MON</option><option>TUE</option><option>WED</option><option>THU</option><option>FRI</option><option>SAT</option><option>SUN</option></select>
        <input type="text" id="qtime" placeholder="time" value="09:00">
        <label class="inline"><input type="text" id="qdur" value="60" style="width:60px"> min</label>
        <label class="inline">permit <input type="text" id="qpermit" placeholder="district" style="width:70px"></label>
        <button id="evalbtn">Check</button>
      </div>
      <div id="evalout"></div>
    </div>
  </div>
</main>
<template id="paneltpl">
  <div class="card panel">
    <h4><span class="ptitle"></span><span class="row"><button class="up">↑</button><button class="down">↓</button><button class="del danger">✕</button></span></h4>
    <div class="row">
      <label>Rule <select class="rule">
        <option>no_parking</option><option>no_stopping</option><option>no_standing</option><option>time_limited</option>
        <option>permit_only</option><option>metered</option><option>passenger_loading</option><option>commercial_loading</option><option>unrestricted</option>
      </select></label>
      <label>Start <input type="text" class="start" placeholder="8am"></label>
      <label>End <input type="text" class="end" placeholder="6pm"></label>
      <label class="inline"><input type="checkbox" class="allday"> all day</label>
      <label>Limit (min) <input type="text" class="limit" style="width:60px"></label>
      <label>District <input type="text" class="district" style="width:70px"></label>
    </div>
    <div class="row days" style="margin-top:6px"></div>
    <div class="row" style="margin-top:6px">
      <label class="inline"><input type="checkbox" class="holidays"> except holidays</label>
      <label class="inline"><input type="checkbox" class="tow"> tow-away</label>
    </div>
  </div>
</template>
<script>
const DAYS = ['MON','TUE','WED','THU','FRI','SAT','SUN'];
const PRESETS = {'All': DAYS, 'Mon–Fri': DAYS.slice(0,5), 'Mon–Sat': DAYS.slice(0,6), 'Sat–Sun': ['SAT','SUN'], 'None': []};
let items = [], done = {}, idx = 0, t0 = Date.now(), elapsed = 0;
const $ = s => document.querySelector(s);

function normTime(v, isEnd) {
  let s = (v || '').trim().toLowerCase().replace(/\s+/g, '').replace(/\./g, '');
  if (!s) return '';
  if (s === 'noon') return '12:00';
  if (s === 'midnight') return isEnd ? '24:00' : '00:00';
  const m = s.match(/^(\d{1,2})(?::?(\d{2}))?(am|pm|a|p)?$/);
  if (!m) return v.trim();
  let h = +m[1], mi = m[2] ? +m[2] : 0; const ap = m[3] ? m[3][0] : null;
  if (ap) { if (h === 12) h = 0; if (ap === 'p') h += 12; if (h === 0 && mi === 0 && isEnd && ap === 'a') return '24:00'; }
  return String(h).padStart(2,'0') + ':' + String(mi).padStart(2,'0');
}

function parseLimit(v) {
  const m = (v || '').trim().toLowerCase().match(/^(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)?$/);
  if (!m) return v.trim() ? NaN : null;
  return Math.round(+m[1] * (m[2] && m[2][0] === 'h' ? 60 : 1));
}
function addPanel(p) {
  const fresh = !p;
  p = p || {rule:'no_parking', days: DAYS.slice(), start:null, end:null};
  const el = $('#paneltpl').content.firstElementChild.cloneNode(true);
  el.querySelector('.rule').value = p.rule;
  el.querySelector('.start').value = p.start || ''; el.querySelector('.end').value = p.end || '';
  el.querySelector('.allday').checked = !fresh && !p.start;  // new cards start unticked; saved all-day panels stay ticked
  el.querySelector('.limit').value = p.limit_min ?? ''; el.querySelector('.district').value = p.district ?? '';
  el.querySelector('.holidays').checked = !!p.except_holidays; el.querySelector('.tow').checked = !!p.tow_away;
  const days = el.querySelector('.days'); const on = new Set(p.days || []);
  for (const d of DAYS) {
    const b = document.createElement('button'); b.textContent = d; b.dataset.day = d; if (on.has(d)) b.classList.add('on');
    b.onclick = () => b.classList.toggle('on'); days.appendChild(b);
  }
  for (const [name, list] of Object.entries(PRESETS)) {
    const b = document.createElement('button'); b.textContent = name; b.className = 'preset';
    b.onclick = () => days.querySelectorAll('[data-day]').forEach(x => x.classList.toggle('on', list.includes(x.dataset.day)));
    days.appendChild(b);
  }
  const sync = () => { const ad = el.querySelector('.allday').checked; el.querySelector('.start').disabled = ad; el.querySelector('.end').disabled = ad; };
  el.querySelector('.allday').onchange = sync;
  el.querySelector('.start').onblur = e => { e.target.value = normTime(e.target.value, false); if (e.target.value) { el.querySelector('.allday').checked = false; sync(); } };
  el.querySelector('.end').onblur = e => { e.target.value = normTime(e.target.value, true); };
  el.querySelector('.limit').onblur = e => { const n = parseLimit(e.target.value); if (Number.isFinite(n)) e.target.value = n; };
  el.querySelector('.del').onclick = () => { el.remove(); retitle(); };
  el.querySelector('.up').onclick = () => { if (el.previousElementSibling) el.parentNode.insertBefore(el, el.previousElementSibling); retitle(); };
  el.querySelector('.down').onclick = () => { if (el.nextElementSibling) el.parentNode.insertBefore(el.nextElementSibling, el); retitle(); };
  $('#panels').appendChild(el); sync(); retitle();
  return el;
}
function retitle() { document.querySelectorAll('.panel').forEach((el, i) => el.querySelector('.ptitle').textContent = `Panel ${i} ${i === 0 ? '(top)' : ''}`); }

function formSign() {
  const panels = [...document.querySelectorAll('.panel')].map((el, i) => {
    const allday = el.querySelector('.allday').checked, lim = el.querySelector('.limit').value.trim(), dist = el.querySelector('.district').value.trim();
    return {
      order: i, rule: el.querySelector('.rule').value,
      days: [...el.querySelectorAll('.days [data-day].on')].map(b => b.dataset.day),
      start: allday ? null : (normTime(el.querySelector('.start').value, false) || null),
      end: allday ? null : (normTime(el.querySelector('.end').value, true) || null),
      limit_min: parseLimit(lim), district: dist || null,
      except_holidays: el.querySelector('.holidays').checked, tow_away: el.querySelector('.tow').checked,
    };
  });
  return {
    panels, arrow: $('#arrow').value, ambiguous: $('#ambiguous').checked, notes: $('#notes').value,
    image_quality: {angle: $('#angle').value, glare: $('#glare').checked, occlusion: $('#occlusion').checked, legibility: $('#legibility').value},
  };
}

function fillForm(sign) {
  $('#panels').innerHTML = '';
  for (const p of (sign?.panels || [null])) addPanel(p);
  $('#arrow').value = sign?.arrow || 'none'; $('#ambiguous').checked = !!sign?.ambiguous; $('#notes').value = sign?.notes || '';
  const q = sign?.image_quality || {};
  $('#angle').value = q.angle || 'frontal'; $('#glare').checked = !!q.glare; $('#occlusion').checked = !!q.occlusion; $('#legibility').value = q.legibility || 'clear';
}

async function show(i) {
  idx = Math.max(0, Math.min(items.length - 1, i));
  const it = items[idx], saved = done[it.candidate_id];
  $('#cid').textContent = it.candidate_id; $('#full').href = it.source_url;
  $('#img').src = '/' + it.crop_path; $('#zoom').value = 1; applyZoom();
  $('#errors').textContent = ''; $('#evalout').textContent = ''; $('#reason').value = saved?.reject_reason || '';
  fillForm(saved?.sign || it.prefill || null);
  const st = saved ? saved.status : 'unlabeled';
  $('#status').textContent = st; $('#status').className = 'status ' + st;
  progress(); t0 = Date.now(); elapsed = 0;
}
function progress() {
  const n = Object.values(done).length, rej = Object.values(done).filter(d => d.status === 'rejected').length;
  $('#progress').textContent = `${idx + 1} / ${items.length} · ${n} done (${n - rej} labeled, ${rej} rejected)`;
}
function applyZoom() {
  const img = $('#img'), z = +$('#zoom').value, v = $('#viewer');
  if (!img.naturalWidth) return;
  const fit = Math.min(v.clientWidth / img.naturalWidth, v.clientHeight / img.naturalHeight);
  img.style.width = (img.naturalWidth * fit * z) + 'px';
}
$('#img').onload = applyZoom; $('#zoom').oninput = applyZoom; window.onresize = applyZoom;
document.addEventListener('visibilitychange', () => { if (document.hidden) { elapsed += Date.now() - t0; } else { t0 = Date.now(); } });
const seconds = () => (elapsed + (Date.now() - t0)) / 1000;

const FIELD_NAMES = {limit_min: 'Limit (min)', start: 'Start', end: 'End', days: 'Days', district: 'District', rule: 'Rule'};
function showErrors(target, errors, prefix) {
  document.querySelectorAll('.panel.error').forEach(el => el.classList.remove('error'));
  const panels = document.querySelectorAll('.panel');
  const lines = errors.map(msg => {
    const m = msg.match(/^panels\.(\d+)(?:\.(\w+))?: (?:Value error, )?(.*)$/);
    if (!m) return msg.replace(/^Value error, /, '');
    panels[+m[1]]?.classList.add('error');
    const field = m[2] ? ` (${FIELD_NAMES[m[2]] || m[2]})` : '';
    return `Panel ${m[1]}${field}: ${m[3]}`;
  });
  target.innerHTML = '<span style="color:var(--bad)">' + (prefix || '') + lines.join('<br>') + '</span>';
}
async function post(url, body) {
  const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  return r.json();
}
function nextUndone() {
  for (let k = 1; k <= items.length; k++) { const j = (idx + k) % items.length; if (!done[items[j].candidate_id]) return j; }
  return idx;
}
async function save() {
  const it = items[idx], r = await post('/api/save', {candidate_id: it.candidate_id, sign: formSign(), seconds: seconds()});
  if (!r.ok) { showErrors($('#errors'), r.errors, 'Not saved:<br>'); return; }
  done[it.candidate_id] = {status: 'labeled', sign: r.sign}; show(nextUndone());
}
async function reject() {
  const reason = $('#reason').value;
  if (!reason) { $('#reason').focus(); $('#errors').textContent = 'Pick a reject reason first.'; return; }
  const it = items[idx], r = await post('/api/reject', {candidate_id: it.candidate_id, reason, notes: $('#notes').value, seconds: seconds()});
  if (!r.ok) { $('#errors').textContent = r.errors.join('\n'); return; }
  done[it.candidate_id] = {status: 'rejected', reject_reason: reason}; show(nextUndone());
}
async function check() {
  const q = {day: $('#qday').value, time: normTime($('#qtime').value, false), duration_min: Number($('#qdur').value), permit_district: $('#qpermit').value.trim() || null};
  $('#qtime').value = q.time;
  const r = await post('/api/evaluate', {sign: formSign(), query: q});
  $('#evalout').innerHTML = r.ok
    ? `<b>${r.verdict.verdict.toUpperCase()}</b>${r.verdict.governing_panel !== null ? ' · governing panel ' + r.verdict.governing_panel : ''} — ${r.verdict.reason}`
    : '';
  if (!r.ok) showErrors($('#evalout'), r.errors);
}

$('#addpanel').onclick = () => addPanel(); $('#save').onclick = save; $('#reject').onclick = reject; $('#evalbtn').onclick = check;
$('#prev').onclick = () => show(idx - 1); $('#next').onclick = () => show(idx + 1);
document.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); save(); return; }
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) { if (e.key === 'Enter' && e.target.closest('.card') && e.target.id?.startsWith('q')) check(); return; }
  const k = e.key.toLowerCase();
  if (k === 'a') save(); else if (k === 'r') reject(); else if (k === 'n') show(idx + 1); else if (k === 'p') show(idx - 1);
  else if (k === 'f') $('#ambiguous').checked = !$('#ambiguous').checked;
  else if (e.key === '+' || e.key === '=') addPanel();
  else if (e.key === ']') { $('#zoom').value = Math.min(6, +$('#zoom').value + 0.5); applyZoom(); }
  else if (e.key === '[') { $('#zoom').value = Math.max(1, +$('#zoom').value - 0.5); applyZoom(); }
});

(async () => {
  const s = await (await fetch('/api/state')).json();
  items = s.items; done = s.done;
  $('#setinfo').textContent = `${s.set === 'gold' ? 'all accepted signs · manual, blind' : 'assisted (pre-filled)'} · round ${s.round}`;
  const first = items.findIndex(it => !done[it.candidate_id]);
  show(first === -1 ? 0 : first);
})();
</script></body></html>
"""


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(COLLECT), **kwargs)

    def log_message(self, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            options = "".join(f"<option>{r}</option>" for r in REJECT_REASONS)
            body = PAGE.replace("__REASONS__", options).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            q = queue()
            items = [
                {"candidate_id": r.candidate_id, "crop_path": r.crop_path, "source_url": r.source_url,
                 "area": r.area, "n_panels_detected": int(r.n_panels_detected), "prefill": prefill(r.candidate_id)}
                for r in q.itertuples()
            ]
            self._json({"set": ARGS.set, "round": ARGS.round, "items": items, "done": saved_labels()})
        elif self.path.startswith("/crops/"):
            super().do_GET()
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cid = body.get("candidate_id")
        try:
            if self.path == "/api/save":
                sign = build_sign(cid, body["sign"])
                data = sign.model_dump(mode="json")
                if ARGS.set == "gold":
                    provenance = "gold_manual"
                else:
                    pf = prefill(cid)
                    same = pf is not None and build_sign(cid, pf).model_dump(mode="json") == data
                    provenance = "assisted_accepted" if same else "assisted_corrected"
                data["provenance"] = provenance
                write(cid, "labeled", None, data, provenance, body.get("seconds", 0))
                self._json({"ok": True, "sign": data})
            elif self.path == "/api/reject":
                if body.get("reason") not in REJECT_REASONS:
                    self._json({"ok": False, "errors": ["unknown reject reason"]})
                    return
                write(cid, "rejected", body["reason"], {"notes": body.get("notes", "")}, None, body.get("seconds", 0))
                self._json({"ok": True})
            elif self.path == "/api/evaluate":
                sign = build_sign("check", body["sign"])
                verdict = evaluate(sign, Query(**body["query"]))
                self._json({"ok": True, "verdict": verdict.model_dump()})
            else:
                self.send_error(404)
        except ValidationError as err:
            self._json({"ok": False, "errors": validation_messages(err)})


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["gold", "assisted"], default="gold")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--port", type=int, default=8766)
    ARGS = ap.parse_args()
    gold = ensure_gold_set()
    print(f"Gold set: {len(gold)} signs ({gold.stratum.value_counts().to_dict()}) in {GOLD}")
    print(f"Labeling ({ARGS.set}, round {ARGS.round}) at http://localhost:{ARGS.port}  (Ctrl-C to stop; saved to {DB})")
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
