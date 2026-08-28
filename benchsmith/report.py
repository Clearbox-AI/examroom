"""Static HTML report over the results ledger.

`benchsmith report` renders one self-contained HTML file (no server, no
dependencies, works offline).  The ledger holds scores and raw model outputs;
the suites on disk hold the questions and expected answers — the report joins
the two (matched by suite content hash, falling back to name@version) so each
result reads as *question → expected → what the model said*, understandable
without knowing benchsmith internals.  Views: a per-suite model leaderboard,
a sortable run table, per-run browsing with a failures-only toggle, per-item
history across runs, and a two-run comparison with side-by-side answers.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from benchsmith.items import Kind, load_suite
from benchsmith.ledger import Ledger

_TRUNC_OUT = 2000
_TRUNC_PROMPT = 1500


def _expected(it) -> tuple[str, str | None]:
    """(expected answer, rejected alternative or None) in reader terms."""
    if it.kind in (Kind.CHOICE_LOGLIK, Kind.CHOICE_GENERATE):
        others = [o.strip() for i, o in enumerate(it.options) if i != it.answer_idx]
        return it.options[it.answer_idx].strip(), " / ".join(others)
    if it.kind == Kind.PAIR_LOGLIK:
        return it.good.strip(), it.bad.strip()
    if it.kind == Kind.NUMERIC:
        exp = it.notes if (it.notes or "").startswith("book answer:") else str(it.target)
        tol = f" (tolerance ±{it.tolerance:g})" if it.tolerance else ""
        return exp + tol, None
    if it.kind == Kind.PROGRAM:
        return f"scored programmatically by {it.checker}", None
    return str(it.target), None  # exact / regex


def _suite_index(suites_dir: str | Path) -> dict[str, dict]:
    """content hash -> suite card with per-item question/expected/category."""
    out: dict[str, dict] = {}
    root = Path(suites_dir)
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not (d / "suite.json").exists():
            continue
        try:
            suite = load_suite(d)
        except Exception:
            continue  # unloadable suite dirs never block the report
        items = {}
        for task, it in suite.iter_items():
            exp, alt = _expected(it)
            p = it.prompt
            if len(p) > _TRUNC_PROMPT:
                p = p[:_TRUNC_PROMPT] + f" …[truncated, {len(it.prompt)} chars]"
            items[it.id] = {"q": p, "expected": exp, "alt": alt,
                            "cat": it.category, "kind": str(it.kind)}
        card = {"name": suite.name, "version": suite.version,
                "description": suite.meta.description, "dir": str(d), "items": items}
        out[suite.content_hash()] = card
        out.setdefault(f"{suite.name}@{suite.version}", card)  # fallback key
    return out


def build_data(ledger: Ledger, suite: str | None = None,
               suites_dir: str | Path = "suites") -> dict:
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
                    o if len(o) <= _TRUNC_OUT else o[:_TRUNC_OUT] + f"\n…[truncated, {len(o)} chars]"
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
        "suites": _suite_index(suites_dir),
        "runs": runs,
    }


def render_report(ledger: Ledger, out: str | Path, suite: str | None = None,
                  suites_dir: str | Path = "suites") -> Path:
    data = build_data(ledger, suite=suite, suites_dir=suites_dir)
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
  --accent:#2a78d6;--good:#008300;--bad:#e34948;--chip:#f1f0ec;
  --goodbg:color-mix(in srgb,#008300 8%,transparent);
  --badbg:color-mix(in srgb,#e34948 8%,transparent)}
@media (prefers-color-scheme: dark){:root{color-scheme:dark;
  --bg:#1a1a19;--surface:#232322;--border:#3a3a38;--ink:#ffffff;--muted:#c3c2b7;
  --accent:#3987e5;--good:#00a300;--bad:#e66767;--chip:#2e2e2c;
  --goodbg:color-mix(in srgb,#00a300 12%,transparent);
  --badbg:color-mix(in srgb,#e66767 12%,transparent)}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 20px 80px}
h1{font-size:20px;margin:0 0 2px}
h2{font-size:15px;margin:26px 0 8px}
h3{font-size:13px;margin:14px 0 4px;color:var(--muted)}
.sub{color:var(--muted);font-size:12px;margin-bottom:18px}
input[type=search]{background:var(--surface);border:1px solid var(--border);color:var(--ink);
  border-radius:6px;padding:6px 10px;width:280px;font:inherit}
table{border-collapse:collapse;width:100%;background:var(--surface);
  border:1px solid var(--border);border-radius:8px;overflow:hidden}
.tablewrap{overflow-x:auto}
th,td{padding:7px 10px;text-align:left;border-top:1px solid var(--border)}
thead th{border-top:none;color:var(--muted);font-weight:600;font-size:12px;
  user-select:none;background:var(--surface);white-space:nowrap}
#runs thead th{cursor:pointer}
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
  padding:8px 10px;font-size:12.5px;white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0}
pre.q{background:transparent}
pre.ans-good{border-left:3px solid var(--good);background:var(--goodbg)}
pre.ans-bad{border-left:3px solid var(--bad);background:var(--badbg)}
button{background:var(--surface);border:1px solid var(--border);color:var(--ink);
  border-radius:6px;padding:4px 10px;font:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:var(--accent)}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
details{margin:4px 0}
details.item{border:1px solid var(--border);border-radius:8px;background:var(--surface);
  padding:6px 10px;margin:6px 0}
summary{cursor:pointer}
.qsnip{overflow:hidden;text-overflow:ellipsis}
.field{margin:8px 0}
.fieldname{font-size:11px;text-transform:uppercase;letter-spacing:.04em;
  color:var(--muted);margin-bottom:2px}
.taskrow{display:flex;align-items:center;gap:8px;margin:3px 0}
.taskname{width:220px;overflow:hidden;text-overflow:ellipsis;font-size:13px}
.delta-up{color:var(--good)}.delta-down{color:var(--bad)}
.lb-row{display:flex;align-items:center;gap:10px;margin:4px 0}
.lb-model{width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.best{font-weight:600}
</style></head><body><main>
<h1>benchsmith report</h1>
<div class="sub" id="meta"></div>

<h2>Which model does best on each benchmark?</h2>
<div id="leader"></div>

<h2>All runs</h2>
<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
  <input type="search" id="filter" placeholder="filter runs (suite, model, label, hash)…">
  <span class="muted" style="font-size:12px">tick two runs to compare their answers · click a run to browse its questions</span>
</div>
<div class="tablewrap"><table id="runs"><thead><tr>
<th data-k="_cmp"></th><th data-k="ts">when</th><th data-k="suite">benchmark</th>
<th data-k="model">model</th><th data-k="label">label</th><th data-k="score">score</th>
<th data-k="n">questions</th><th data-k="tps">tok/s</th><th data-k="run_id">run</th>
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
const suiteFor = r => DATA.suites[r.hash] || DATA.suites[r.suite + '@' + r.version] || null;
let sortK = 'ts', sortDir = -1, cmp = [], selected = null;

document.getElementById('meta').textContent =
  `${runs.length} runs · ledger ${DATA.ledger} · generated ${DATA.generated}`;

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (x, d=3) => x == null ? '–' : (+x).toFixed(d);
const pct = x => x == null ? '–' : Math.round(x * 100) + '%';
const scoreCell = s =>
  `<span class="bar" style="width:${Math.round(s*60)}px"></span>${pct(s)}`;
const runName = r => `${r.model}${r.label ? ' · ' + r.label : ''}`;

/* ---------- leaderboard: best model per benchmark ---------- */
function renderLeader() {
  const groups = {};
  for (const r of runs) (groups[r.suite + '@' + r.version] ??= []).push(r);
  document.getElementById('leader').innerHTML = Object.entries(groups).map(([k, rs]) => {
    rs = [...rs].sort((a, b) => b.score - a.score);
    const card = suiteFor(rs[0]);
    const desc = card && card.description ? `<div class="muted" style="font-size:12px;margin-bottom:6px">${esc(card.description)}</div>` : '';
    const rows = rs.map((r, i) => `<div class="lb-row${i === 0 ? ' best' : ''}">
      <span class="lb-model">${i === 0 ? '★ ' : ''}${esc(runName(r))}</span>
      <span class="bar" style="width:${Math.max(2, Math.round(r.score * 260))}px"></span>
      <span class="mono">${pct(r.score)}</span>
      <span class="muted" style="font-size:12px">${r.n} questions · ${esc(r.ts.slice(0, 10))}</span>
    </div>`).join('');
    return `<div class="panel"><h3 style="margin-top:0">${esc(k)}</h3>${desc}${rows}</div>`;
  }).join('') || '<span class="muted">no runs recorded yet</span>';
}

/* ---------- runs table ---------- */
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

/* ---------- one result, told as question -> expected -> model ---------- */
function answerBlocks(it, ok) {
  return (it.outputs || []).map(o =>
    `<pre class="${ok ? 'ans-good' : 'ans-bad'}">${esc(o.trim() || '(empty answer)')}</pre>`).join('')
    || '<span class="muted">no text output recorded (likelihood-scored question)</span>';
}

function itemCard(r, it, meta) {
  const ok = it.score > 0.5;
  const snip = meta ? meta.q.replace(/\s+/g, ' ').slice(0, 110) : it.id;
  const cat = meta && meta.cat ? `<span class="chip">${esc(meta.cat)}</span>` : '';
  const qBlock = meta ? `<div class="field"><div class="fieldname">Question</div>
      <pre class="q">${esc(meta.q)}</pre></div>` : '';
  const altName = meta && meta.kind === 'pair_loglik' ? 'Bad alternative it must beat'
                                                      : 'Rejected options';
  const expBlock = meta ? `<div class="field"><div class="fieldname">Expected answer</div>
      <pre class="q">${esc(meta.expected)}</pre>
      ${meta.alt ? `<div class="fieldname">${altName}</div><pre class="q muted">${esc(meta.alt)}</pre>` : ''}</div>` : '';
  const tech = `<details><summary class="muted" style="font-size:12px">technical detail</summary>
      <div class="mono muted" style="margin:4px 0">${esc(it.id)} · ${esc(it.task)} · protocol ${esc(it.protocol)} · score ${fmt(it.score, 3)}</div>
      ${it.detail ? `<pre>${esc(JSON.stringify(it.detail, null, 1))}</pre>` : ''}</details>`;
  return `<details class="item"><summary>
      <span class="${ok ? 'pass' : 'fail'}">${ok ? '✓ correct' : '✗ wrong'}</span>
      ${cat} <span class="qsnip">${esc(snip)}…</span>
      <button data-hist="${esc(it.id)}">history</button></summary>
      ${qBlock}${expBlock}
      <div class="field"><div class="fieldname">Model answered</div>${answerBlocks(it, ok)}</div>
      ${tech}</details>`;
}

function renderDetail(r) {
  document.getElementById('history').innerHTML = '';
  const card = suiteFor(r);
  const meta = card ? card.items : {};
  const chips = Object.entries({label: r.label, ...r.backend_info.extra ?? {}, ...r.backend_info,
      ...r.stats})
    .filter(([k, v]) => v != null && typeof v !== 'object')
    .map(([k, v]) => `<span class="chip">${esc(k)}=${esc(typeof v === 'number' ? +(+v).toFixed(2) : v)}</span>`).join('');
  const note = card ? '' : `<div class="muted" style="font-size:12px;margin:4px 0">
      suite files not found on disk (hash ${esc(r.hash.slice(0, 8))}) — questions and expected
      answers can't be shown, only the model's outputs</div>`;
  const maxW = 360;
  const tasks = r.tasks.map(t => `<div class="taskrow">
    <span class="taskname" title="${esc(t.task)}">${esc(t.task)}</span>
    <span class="bar" style="width:${Math.max(2, Math.round(t.score * maxW))}px"></span>
    <span class="mono">${pct(t.score)} <span class="muted">of ${t.n}</span></span></div>`).join('');
  const el = document.getElementById('detail');
  el.innerHTML = `<h2>${esc(runName(r))} on ${esc(r.suite)}@${esc(r.version)} —
      ${pct(r.score)} of ${r.n} questions</h2>
    ${card && card.description ? `<div class="muted" style="margin:2px 0 8px">${esc(card.description)}</div>` : ''}
    ${note}
    <div class="panel">${chips}</div>
    <div class="panel">${tasks}</div>
    <div class="panel">
      <button id="fonly">wrong answers only</button>
      <input type="search" id="ifilter" placeholder="search questions…" style="width:220px;margin-left:8px">
      <div id="items" style="margin-top:8px"></div></div>`;
  let fail = false;
  const paint = () => {
    const q = (document.getElementById('ifilter').value || '').toLowerCase();
    const shown = r.items.filter(it => {
      if (fail && it.score > 0.5) return false;
      if (!q) return true;
      const m = meta[it.id];
      return (it.id + ' ' + it.task + ' ' + (m ? m.q + ' ' + (m.cat || '') : ''))
        .toLowerCase().includes(q);
    });
    document.getElementById('items').innerHTML =
      shown.map(it => itemCard(r, it, meta[it.id])).join('')
      || '<span class="muted">no questions match</span>';
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

/* ---------- one question across every run ---------- */
function renderHistory(itemId) {
  const rows = runs.flatMap(r => {
    const it = r.items.find(i => i.id === itemId);
    return it ? [{r, it}] : [];
  }).sort((a, b) => a.r.ts < b.r.ts ? -1 : 1);
  const meta = rows.length ? (suiteFor(rows[0].r) || {items: {}}).items[itemId] : null;
  document.getElementById('history').innerHTML =
    `<h2>One question, every model</h2>` +
    (meta ? `<div class="panel"><div class="fieldname">Question</div><pre class="q">${esc(meta.q)}</pre>
      <div class="fieldname">Expected answer</div><pre class="q">${esc(meta.expected)}</pre></div>` : '') +
    `<div class="tablewrap"><table><thead><tr><th>when</th><th>model</th>
     <th>result</th><th>model's answer</th></tr></thead><tbody>` +
    rows.map(({r, it}) => `<tr><td class="mono">${esc(r.ts.slice(0, 16).replace('T', ' '))}</td>
      <td>${esc(runName(r))}</td>
      <td class="${it.score > 0.5 ? 'pass' : 'fail'}">${it.score > 0.5 ? '✓ correct' : '✗ wrong'}</td>
      <td class="mono" style="white-space:normal">${esc(((it.outputs || [])[0] || '').trim().slice(0, 120))}</td></tr>`).join('') +
    '</tbody></table></div>';
  document.getElementById('history').scrollIntoView({behavior: 'smooth'});
}

/* ---------- two runs side by side ---------- */
function sideBySide(A, B, ids, meta) {
  const aBy = Object.fromEntries(A.items.map(i => [i.id, i]));
  const bBy = Object.fromEntries(B.items.map(i => [i.id, i]));
  return ids.map(id => {
    const m = meta[id], a = aBy[id], b = bBy[id];
    return `<details class="item"><summary><span class="qsnip">${esc(m ? m.q.replace(/\s+/g, ' ').slice(0, 110) : id)}…</span></summary>
      ${m ? `<div class="field"><div class="fieldname">Question</div><pre class="q">${esc(m.q)}</pre>
      <div class="fieldname">Expected answer</div><pre class="q">${esc(m.expected)}</pre></div>` : ''}
      <div class="field"><div class="fieldname">A · ${esc(runName(A))}</div>${answerBlocks(a, a.score > 0.5)}</div>
      <div class="field"><div class="fieldname">B · ${esc(runName(B))}</div>${answerBlocks(b, b.score > 0.5)}</div>
      </details>`;
  }).join('') || '<span class="muted">none</span>';
}

function renderCompare() {
  const el = document.getElementById('compare');
  if (cmp.length !== 2) { el.innerHTML = ''; return; }
  const [A, B] = cmp.map(id => byId[id]);
  const meta = (suiteFor(A) || {items: {}}).items;
  const bItems = Object.fromEntries(B.items.map(i => [i.id, i]));
  const common = A.items.filter(i => bItems[i.id]);
  const fixed = common.filter(i => i.score <= 0.5 && bItems[i.id].score > 0.5).map(i => i.id);
  const broken = common.filter(i => i.score > 0.5 && bItems[i.id].score <= 0.5).map(i => i.id);
  const tasks = A.tasks.map(t => {
    const tb = B.tasks.find(x => x.task === t.task);
    if (!tb) return '';
    const d = tb.score - t.score;
    return `<tr><td>${esc(t.task)}</td><td>${pct(t.score)}</td><td>${pct(tb.score)}</td>
      <td class="${d > 0 ? 'delta-up' : d < 0 ? 'delta-down' : 'muted'}">${d >= 0 ? '+' : ''}${fmt(d)}</td></tr>`;
  }).join('');
  el.innerHTML = `<h2>Compare — A: ${esc(runName(A))} vs B: ${esc(runName(B))}
      <span class="muted" style="font-weight:400;font-size:12px">(${common.length} shared questions)</span></h2>
    <div class="panel"><div class="tablewrap"><table><thead>
    <tr><th>task</th><th>A</th><th>B</th><th>Δ</th></tr></thead>
    <tbody>${tasks}</tbody></table></div>
    <h3 class="delta-up">B gets right what A got wrong (${fixed.length})</h3>${sideBySide(A, B, fixed, meta)}
    <h3 class="delta-down">B gets wrong what A got right (${broken.length})</h3>${sideBySide(A, B, broken, meta)}
    </div>`;
}

renderLeader();
renderRuns();
</script></main></body></html>
"""
