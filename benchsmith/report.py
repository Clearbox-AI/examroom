"""Static HTML report over the results ledger.

`benchsmith report` renders one self-contained HTML file (no server, no
dependencies, works offline) with the whole ledger embedded as JSON: a
sortable run table, per-run task scores and item results with the raw model
outputs, per-item history across runs, and a two-run comparison with
fixed/broken lists.  Regenerate it whenever the ledger changes.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from benchsmith.ledger import Ledger

_TRUNC = 2000


def build_data(ledger: Ledger, suite: str | None = None) -> dict:
    runs = []
    for row in ledger.runs(suite=suite):
        rid = row["run_id"]
        items = []
        for r in ledger.con.execute(
            "SELECT task, item_id, score, protocol, detail, seconds FROM results WHERE run_id=?",
            (rid,),
        ):
            detail = json.loads(r["detail"] or "{}")
            outputs = detail.pop("outputs", None)
            if outputs:
                outputs = [
                    o if len(o) <= _TRUNC else o[:_TRUNC] + f"\n…[truncated, {len(o)} chars]"
                    for o in outputs
                ]
            items.append(
                {"task": r["task"], "id": r["item_id"], "score": r["score"],
                 "protocol": r["protocol"], "sec": r["seconds"],
                 "outputs": outputs, "detail": detail or None}
            )
        runs.append(
            {"run_id": rid, "ts": row["ts"], "suite": row["suite"],
             "version": row["suite_version"], "hash": row["suite_hash"],
             "backend": row["backend"], "model": row["model"], "label": row["label"],
             "score": row["score"], "n": row["n_items"],
             "backend_info": json.loads(row["backend_json"]),
             "stats": json.loads(row["stats_json"]),
             "tasks": [{"task": t, "score": s, "n": n}
                       for t, (s, n) in sorted(ledger.task_scores(rid).items())],
             "items": items}
        )
    return {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "ledger": str(ledger.path),
        "runs": runs,
    }


def render_report(ledger: Ledger, out: str | Path, suite: str | None = None) -> Path:
    data = build_data(ledger, suite=suite)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = Path(out)
    out.write_text(_TEMPLATE.replace("__DATA__", payload))
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>benchsmith report</title>
<style>
:root{color-scheme:light;
  --bg:#fcfcfb;--surface:#ffffff;--border:#e4e3df;--ink:#0b0b0b;--muted:#52514e;
  --accent:#2a78d6;--good:#008300;--bad:#e34948;--chip:#f1f0ec}
@media (prefers-color-scheme: dark){:root{color-scheme:dark;
  --bg:#1a1a19;--surface:#232322;--border:#3a3a38;--ink:#ffffff;--muted:#c3c2b7;
  --accent:#3987e5;--good:#00a300;--bad:#e66767;--chip:#2e2e2c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 20px 80px}
h1{font-size:20px;margin:0 0 2px}
h2{font-size:15px;margin:26px 0 8px}
.sub{color:var(--muted);font-size:12px;margin-bottom:18px}
input[type=search]{background:var(--surface);border:1px solid var(--border);color:var(--ink);
  border-radius:6px;padding:6px 10px;width:280px;font:inherit}
table{border-collapse:collapse;width:100%;background:var(--surface);
  border:1px solid var(--border);border-radius:8px;overflow:hidden}
.tablewrap{overflow-x:auto}
th,td{padding:7px 10px;text-align:left;border-top:1px solid var(--border);white-space:nowrap}
thead th{border-top:none;color:var(--muted);font-weight:600;font-size:12px;cursor:pointer;
  user-select:none;background:var(--surface)}
tbody tr:hover{background:color-mix(in srgb,var(--accent) 6%,transparent)}
tr.sel{background:color-mix(in srgb,var(--accent) 12%,transparent)}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.muted{color:var(--muted)}
.bar{display:inline-block;height:8px;border-radius:0 4px 4px 0;background:var(--accent);
  vertical-align:middle;margin-right:6px}
.pass{color:var(--good)}.fail{color:var(--bad)}
.chip{display:inline-block;background:var(--chip);border:1px solid var(--border);
  border-radius:5px;padding:1px 8px;margin:2px 4px 2px 0;font-size:12px}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:14px 16px;margin:10px 0}
pre{background:var(--chip);border:1px solid var(--border);border-radius:6px;
  padding:8px 10px;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;margin:6px 0}
button{background:var(--surface);border:1px solid var(--border);color:var(--ink);
  border-radius:6px;padding:4px 10px;font:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:var(--accent)}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
details{margin:4px 0}
summary{cursor:pointer}
.taskrow{display:flex;align-items:center;gap:8px;margin:3px 0}
.taskname{width:220px;overflow:hidden;text-overflow:ellipsis;font-size:13px}
.delta-up{color:var(--good)}.delta-down{color:var(--bad)}
</style></head><body><main>
<h1>benchsmith report</h1>
<div class="sub" id="meta"></div>
<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
  <input type="search" id="filter" placeholder="filter runs (suite, model, label, hash)…">
  <span class="muted" style="font-size:12px">tick two runs to compare · click a run for details</span>
</div>
<h2>Runs</h2>
<div class="tablewrap"><table id="runs"><thead><tr>
<th data-k="_cmp"></th><th data-k="ts">when</th><th data-k="suite">suite</th>
<th data-k="model">model</th><th data-k="label">label</th><th data-k="score">score</th>
<th data-k="n">items</th><th data-k="tps">tok/s</th><th data-k="run_id">run</th>
</tr></thead><tbody></tbody></table></div>
<div id="compare"></div>
<div id="detail"></div>
<div id="history"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const runs = DATA.runs;
runs.forEach(r => r.tps = (r.stats && r.stats.tokens_per_second) || null);
const byId = Object.fromEntries(runs.map(r => [r.run_id, r]));
let sortK = 'ts', sortDir = -1, cmp = [], selected = null;

document.getElementById('meta').textContent =
  `${runs.length} runs · ledger ${DATA.ledger} · generated ${DATA.generated}`;

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (x, d=3) => x == null ? '–' : (+x).toFixed(d);
const scoreCell = s =>
  `<span class="bar" style="width:${Math.round(s*60)}px"></span>${fmt(s)}`;

function visibleRuns() {
  const q = document.getElementById('filter').value.toLowerCase();
  let rs = runs.filter(r =>
    !q || [r.suite, r.model, r.label, r.hash, r.run_id, r.version]
      .join(' ').toLowerCase().includes(q));
  rs.sort((a, b) => {
    const va = a[sortK], vb = b[sortK];
    if (va == null) return 1; if (vb == null) return -1;
    return (va < vb ? -1 : va > vb ? 1 : 0) * sortDir;
  });
  return rs;
}

function renderRuns() {
  const tb = document.querySelector('#runs tbody');
  tb.innerHTML = visibleRuns().map(r => `
    <tr data-id="${r.run_id}" class="${r.run_id === selected ? 'sel' : ''}">
    <td><input type="checkbox" data-cmp="${r.run_id}" ${cmp.includes(r.run_id) ? 'checked' : ''}></td>
    <td class="mono">${esc(r.ts.slice(0, 16).replace('T', ' '))}</td>
    <td>${esc(r.suite)}@${esc(r.version)} <span class="mono muted">${esc(r.hash.slice(0, 8))}</span></td>
    <td>${esc(r.backend)}:${esc(r.model)}</td>
    <td>${esc(r.label ?? '')}</td>
    <td>${scoreCell(r.score)}</td>
    <td>${r.n}</td><td>${fmt(r.tps, 1)}</td>
    <td class="mono muted">${esc(r.run_id.slice(0, 8))}</td></tr>`).join('');
}

document.querySelector('#runs thead').onclick = e => {
  const k = e.target.dataset.k;
  if (!k || k === '_cmp') return;
  sortDir = (sortK === k) ? -sortDir : (k === 'ts' || k === 'score' ? -1 : 1);
  sortK = k; renderRuns();
};
document.getElementById('filter').oninput = renderRuns;

document.querySelector('#runs tbody').onclick = e => {
  const box = e.target.closest('input[data-cmp]');
  if (box) {
    const id = box.dataset.cmp;
    cmp = cmp.includes(id) ? cmp.filter(x => x !== id) : [...cmp, id].slice(-2);
    renderRuns(); renderCompare(); return;
  }
  const tr = e.target.closest('tr[data-id]');
  if (tr) { selected = tr.dataset.id; renderRuns(); renderDetail(byId[selected]); }
};

function itemRow(r, it) {
  const out = (it.outputs || []).map(o => `<pre>${esc(o)}</pre>`).join('');
  const extra = it.detail ? `<pre>${esc(JSON.stringify(it.detail, null, 1))}</pre>` : '';
  return `<details><summary>
    <span class="${it.score > 0.5 ? 'pass' : 'fail'}">${it.score > 0.5 ? '✓' : '✗'} ${fmt(it.score, 2)}</span>
    <span class="mono">${esc(it.id)}</span>
    <span class="muted">${esc(it.task)} · ${esc(it.protocol)}</span>
    <button data-hist="${esc(it.id)}">history</button></summary>
    ${out}${extra}</details>`;
}

function renderDetail(r) {
  document.getElementById('history').innerHTML = '';
  const chips = Object.entries({label: r.label, ...r.backend_info.extra ?? {}, ...r.backend_info,
      ...r.stats})
    .filter(([k, v]) => v != null && typeof v !== 'object')
    .map(([k, v]) => `<span class="chip">${esc(k)}=${esc(typeof v === 'number' ? +(+v).toFixed(2) : v)}</span>`).join('');
  const maxW = 360;
  const tasks = r.tasks.map(t => `<div class="taskrow">
    <span class="taskname" title="${esc(t.task)}">${esc(t.task)}</span>
    <span class="bar" style="width:${Math.max(2, Math.round(t.score * maxW))}px"></span>
    <span class="mono">${fmt(t.score)} <span class="muted">n=${t.n}</span></span></div>`).join('');
  const el = document.getElementById('detail');
  el.innerHTML = `<h2>Run ${esc(r.run_id.slice(0, 12))} — ${esc(r.backend)}:${esc(r.model)}</h2>
    <div class="panel">${chips}</div>
    <div class="panel">${tasks}</div>
    <div class="panel">
      <button id="fonly">failures only</button>
      <input type="search" id="ifilter" placeholder="filter items…" style="width:200px;margin-left:8px">
      <div id="items" style="margin-top:8px"></div></div>`;
  let fail = false;
  const paint = () => {
    const q = (document.getElementById('ifilter').value || '').toLowerCase();
    document.getElementById('items').innerHTML = r.items
      .filter(it => (!fail || it.score <= 0.5) &&
                    (!q || (it.id + ' ' + it.task).toLowerCase().includes(q)))
      .map(it => itemRow(r, it)).join('') || '<span class="muted">no items match</span>';
  };
  document.getElementById('fonly').onclick = e => {
    fail = !fail; e.target.classList.toggle('on', fail); paint(); };
  document.getElementById('ifilter').oninput = paint;
  el.onclick = e => {
    const b = e.target.closest('button[data-hist]');
    if (b) { e.preventDefault(); renderHistory(b.dataset.hist); }
  };
  paint();
  el.scrollIntoView({behavior: 'smooth'});
}

function renderHistory(itemId) {
  const rows = runs.flatMap(r => {
    const it = r.items.find(i => i.id === itemId);
    return it ? [{r, it}] : [];
  }).sort((a, b) => a.r.ts < b.r.ts ? -1 : 1);
  document.getElementById('history').innerHTML =
    `<h2>Item ${esc(itemId)} across ${rows.length} runs</h2>
     <div class="tablewrap"><table><thead><tr><th>when</th><th>model</th><th>label</th>
     <th>score</th><th>protocol</th></tr></thead><tbody>` +
    rows.map(({r, it}) => `<tr><td class="mono">${esc(r.ts.slice(0, 16).replace('T', ' '))}</td>
      <td>${esc(r.backend)}:${esc(r.model)}</td><td>${esc(r.label ?? '')}</td>
      <td class="${it.score > 0.5 ? 'pass' : 'fail'}">${it.score > 0.5 ? '✓' : '✗'} ${fmt(it.score, 2)}</td>
      <td class="muted">${esc(it.protocol)}</td></tr>`).join('') +
    '</tbody></table></div>';
  document.getElementById('history').scrollIntoView({behavior: 'smooth'});
}

function renderCompare() {
  const el = document.getElementById('compare');
  if (cmp.length !== 2) { el.innerHTML = ''; return; }
  const [A, B] = cmp.map(id => byId[id]);
  const bItems = Object.fromEntries(B.items.map(i => [i.id, i]));
  const common = A.items.filter(i => bItems[i.id]);
  const fixed = common.filter(i => i.score <= 0.5 && bItems[i.id].score > 0.5).map(i => i.id);
  const broken = common.filter(i => i.score > 0.5 && bItems[i.id].score <= 0.5).map(i => i.id);
  const tasks = A.tasks.map(t => {
    const tb = B.tasks.find(x => x.task === t.task);
    if (!tb) return '';
    const d = tb.score - t.score;
    return `<tr><td>${esc(t.task)}</td><td>${fmt(t.score)}</td><td>${fmt(tb.score)}</td>
      <td class="${d > 0 ? 'delta-up' : d < 0 ? 'delta-down' : 'muted'}">${d >= 0 ? '+' : ''}${fmt(d)}</td></tr>`;
  }).join('');
  const lbl = r => esc(r.label || r.model);
  el.innerHTML = `<h2>Compare: A=${lbl(A)} vs B=${lbl(B)}
      <span class="muted" style="font-weight:400;font-size:12px">(${common.length} shared items)</span></h2>
    <div class="panel"><div class="tablewrap"><table><thead>
    <tr><th>task</th><th>A</th><th>B</th><th>Δ</th></tr></thead>
    <tbody>${tasks}</tbody></table></div>
    <p><span class="delta-up">fixed in B (${fixed.length}):</span>
       <span class="mono">${fixed.map(esc).join(', ') || '—'}</span></p>
    <p><span class="delta-down">broken in B (${broken.length}):</span>
       <span class="mono">${broken.map(esc).join(', ') || '—'}</span></p></div>`;
}

renderRuns();
</script></main></body></html>
"""
