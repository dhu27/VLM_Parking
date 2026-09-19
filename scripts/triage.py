"""Fast accept/reject triage of collected sign crops, in the browser.

    uv run python scripts/triage.py            # then open http://localhost:8765

Shows a page of crops whose OCR text reads as a parking sign (--all to include the rest), largest stacks first. Click a crop to toggle reject; shift-click or
right-click for "maybe". X rejects the whole page (then click the good ones back), C clears it.
Press Enter (or the button) to accept everything unmarked on the page and move on. Decisions are saved to data/collect/triage.csv on every page, so you can stop any time.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path("data/collect")
DECISIONS = ROOT / "triage.csv"
# Only crops whose OCR text reads as parking are triaged (decision 2026-09-19): OCR-unknown crops were ~20%
# of accepts but only ~7% of those survived labeling. Pass --all to include them.
OCR_PARKING_ONLY = True


def load_candidates() -> list[dict]:
    frames = [pd.read_csv(p) for p in sorted(ROOT.glob("*/candidates.csv"))]
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    if OCR_PARKING_ONLY:
        df = df[df.ocr_status.eq("parking")]
    df["_p"] = df.ocr_status.eq("parking")
    df = df.sort_values(["_p", "n_panels_detected", "height_native"], ascending=False)
    cols = ["candidate_id", "area", "crop_path", "source_url", "n_panels_detected", "height_native", "ocr_status", "year"]
    return df[cols].fillna("").to_dict("records")


def load_decisions() -> dict[str, dict]:
    if not DECISIONS.exists():
        return {}
    with DECISIONS.open() as f:
        return {r["candidate_id"]: r for r in csv.DictReader(f)}


def save_decisions(decisions: dict[str, dict]) -> None:
    tmp = DECISIONS.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "decision", "decided_at"])
        w.writeheader()
        w.writerows(decisions.values())
    tmp.replace(DECISIONS)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Sign triage</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#f6f6f4; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#ddd; --reject:#d93025; --maybe:#e8a200; --accept:#1e8e3e; }
  @media (prefers-color-scheme: dark) { :root { --bg:#171716; --fg:#ececea; --muted:#9a9a94; --card:#232322; --line:#3a3a38; } }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  header { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  header b { font-size:15px; }
  .stats { color:var(--muted); }
  button, select { font:inherit; padding:6px 12px; border-radius:6px; border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer; }
  button.primary { background:var(--accept); color:#fff; border-color:var(--accept); }
  .help { color:var(--muted); font-size:12px; width:100%; }
  #grid { display:flex; flex-wrap:wrap; gap:8px; padding:16px; align-items:flex-end; }
  .tile { position:relative; background:var(--card); border:3px solid transparent; border-radius:6px; padding:2px; cursor:pointer; }
  .tile img { height:230px; display:block; border-radius:3px; }
  .tile .meta { font-size:11px; color:var(--muted); padding:2px 2px 0; display:flex; justify-content:space-between; gap:6px; }
  .tile.reject { border-color:var(--reject); } .tile.reject img { opacity:.35; }
  .tile.maybe { border-color:var(--maybe); }
  .tile .badge { position:absolute; top:6px; left:6px; font-size:11px; font-weight:600; padding:1px 6px; border-radius:4px; color:#fff; display:none; }
  .tile.reject .badge { display:block; background:var(--reject); } .tile.maybe .badge { display:block; background:var(--maybe); }
  .tile a { color:var(--muted); }
  #done { padding:40px 16px; font-size:16px; display:none; }
</style></head><body>
<header>
  <b>Sign triage</b><span class="stats">OCR-confirmed parking crops only</span>
  <select id="area"><option value="">All areas</option></select>
  <span class="stats" id="stats"></span>
  <button class="primary" id="next">Accept unmarked &amp; next page ⏎</button>
  <button id="rejectall">Reject all (X)</button>
  <button id="clearall">Clear all (C)</button>
  <button id="undo">Back one page</button>
  <div class="help">Click = toggle reject · shift-click / right-click = maybe (legible? unsure) · X = reject whole page, then click the good ones back · C = clear page · ⏎ accepts every unmarked crop.
  Accept only signs that are parking-related <i>and</i> look fully transcribable; "maybe" if you'd need the full image to decide.</div>
</header>
<div id="grid"></div><div id="done">All crops in this view are triaged. 🎉</div>
<script>
const PER_PAGE = 40;
let cands = [], decisions = {}, page = [], history = [];
const $ = s => document.querySelector(s);

async function load() {
  const r = await fetch('/api/state'); const s = await r.json();
  cands = s.candidates; decisions = s.decisions;
  const areas = [...new Set(cands.map(c => c.area))];
  for (const a of areas) $('#area').insertAdjacentHTML('beforeend', `<option>${a}</option>`);
  render();
}
function view() { const a = $('#area').value; return cands.filter(c => !a || c.area === a); }
function stats() {
  const v = view(), d = v.map(c => decisions[c.candidate_id]?.decision).filter(Boolean);
  const n = x => d.filter(y => y === x).length;
  $('#stats').textContent = `${d.length}/${v.length} triaged · ${n('accept')} accept · ${n('maybe')} maybe · ${n('reject')} reject`;
}
function render() {
  page = view().filter(c => !decisions[c.candidate_id]).slice(0, PER_PAGE);
  $('#grid').innerHTML = ''; $('#done').style.display = page.length ? 'none' : 'block';
  for (const c of page) {
    const t = document.createElement('div'); t.className = 'tile'; t.dataset.id = c.candidate_id; t.dataset.state = '';
    t.innerHTML = `<span class="badge"></span><img loading="lazy" src="/${c.crop_path}" alt="">
      <div class="meta"><span>${c.area} · ${c.n_panels_detected}p · ${c.year}</span><a href="${c.source_url}" target="_blank" onclick="event.stopPropagation()">full ↗</a></div>`;
    t.onclick = e => setState(t, e.shiftKey ? (t.dataset.state === 'maybe' ? '' : 'maybe') : (t.dataset.state ? '' : 'reject'));
    t.oncontextmenu = e => { e.preventDefault(); setState(t, t.dataset.state === 'maybe' ? '' : 'maybe'); };
    $('#grid').appendChild(t);
  }
  stats(); window.scrollTo(0, 0);
}
function setState(t, state) {
  t.dataset.state = state; t.className = 'tile ' + state; t.querySelector('.badge').textContent = state;
}
function setAll(state) { document.querySelectorAll('.tile').forEach(t => setState(t, state)); }
async function commit() {
  if (!page.length) return;
  const batch = [...document.querySelectorAll('.tile')].map(t => ({candidate_id: t.dataset.id, decision: t.dataset.state || 'accept'}));
  const r = await fetch('/api/decisions', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(batch)});
  if (!r.ok) { alert('Save failed; nothing lost, try again.'); return; }
  for (const b of batch) decisions[b.candidate_id] = b;
  history.push(batch.map(b => b.candidate_id)); render();
}
async function undo() {
  const ids = history.pop(); if (!ids) return;
  await fetch('/api/undo', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(ids)});
  for (const id of ids) delete decisions[id];
  render();
}
$('#next').onclick = commit; $('#undo').onclick = undo; $('#area').onchange = render;
$('#rejectall').onclick = () => setAll('reject'); $('#clearall').onclick = () => setAll('');
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === 'Enter') { e.preventDefault(); commit(); }
  else if (e.key === 'x' || e.key === 'X') setAll('reject');
  else if (e.key === 'c' || e.key === 'C') setAll('');
});
load();
</script></body></html>
"""


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, *args):  # keep the terminal quiet
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
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json({"candidates": load_candidates(), "decisions": load_decisions()})
        elif self.path.startswith("/crops/"):
            super().do_GET()
        else:
            self.send_error(404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        decisions = load_decisions()
        if self.path == "/api/decisions":
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for d in payload:
                if d["decision"] in ("accept", "reject", "maybe"):
                    decisions[d["candidate_id"]] = {"candidate_id": d["candidate_id"], "decision": d["decision"], "decided_at": now}
        elif self.path == "/api/undo":
            for cid in payload:
                decisions.pop(cid, None)
        else:
            self.send_error(404)
            return
        save_decisions(decisions)
        self._json({"ok": True, "n": len(decisions)})


def main() -> None:
    global ROOT, DECISIONS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--all", action="store_true", help="also show crops OCR couldn't read as parking")
    args = ap.parse_args()
    global OCR_PARKING_ONLY
    ROOT, DECISIONS, OCR_PARKING_ONLY = args.root, args.root / "triage.csv", not args.all
    print(f"Triage at http://localhost:{args.port}  (Ctrl-C to stop; decisions in {DECISIONS})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
