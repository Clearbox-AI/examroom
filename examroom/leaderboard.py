"""Static public leaderboard from exported runs, the model registry and a board file.

The board file is the curation: which suites (by exact content hash — normally the
held-out splits) and which run profiles are ranked.

    {"title": "examroom leaderboard",
     "profiles": ["nothink@1", "think@1"],
     "suites": [{"name": "crc_pilot-heldout", "version": "0.4", "hash": "443225e6a82424c0",
                 "label": "CRC pilot", "public": "hf://clearbox-ai/examroom-suites/crc_pilot_v0.4"}]}

A row is a (model, precision) pair; when several runs exist for a row, suite and
profile, the latest counts.  A suite score is the mean of its task scores (each
task weighs the same whatever its size); the overall score is the mean over the
board's suites and is only given to rows that ran all of them.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

from examroom.diff import wilson
from examroom.registry import TIERS, Registry, fmt_size


def load_board(path: str | Path) -> dict:
    board = json.loads(Path(path).read_text())
    for s in board.get("suites", []):
        missing = {"name", "version", "hash"} - s.keys()
        if missing:
            raise ValueError(f"board suite entry {s} lacks {', '.join(sorted(missing))}")
    if not board.get("profiles"):
        raise ValueError("board lists no profiles: unprofiled runs are never ranked")
    return board


def build_board(records: list[dict], board: dict, registry: Registry) -> dict:
    suites = board["suites"]
    views = []
    for profile in board["profiles"]:
        rows: dict[tuple, dict] = {}
        for si, sc in enumerate(suites):
            for rec in records:
                b = rec["backend"]
                if (rec["suite"], rec["suite_version"], rec["suite_hash"]) != (sc["name"], sc["version"], sc["hash"]) \
                        or b.get("profile") != profile:
                    continue
                ms = registry.size(b)
                key = (ms.model_id or f"{b.get('backend')}:{b.get('model')}", ms.precision)
                row = rows.setdefault(key, {
                    "model": key[0], "registered": ms.model_id is not None, "precision": ms.precision,
                    "n_params": ms.n_params, "active_params": ms.active_params, "weights_mb": ms.weights_mb,
                    "weights_source": ms.weights_source, "tiers": ms.tiers,
                    "hub": registry.models.get(ms.model_id, {}).get("hub", {}).get("repo"),
                    "suites": {},
                })
                prev = row["suites"].get(si)
                if prev and prev["ts"] >= rec["ts"]:
                    continue
                tasks = {}
                for t, v in rec["tasks"].items():
                    lo, hi = wilson(v["score"], v["n"])
                    tasks[t] = {"score": v["score"], "n": v["n"], "lo": lo, "hi": hi}
                row["suites"][si] = {"score": sum(v["score"] for v in tasks.values()) / len(tasks),
                                     "tasks": tasks, "run_id": rec["run_id"], "ts": rec["ts"],
                                     "served_by": b.get("base_url") or b.get("backend")}
        for row in rows.values():
            have = [row["suites"][i]["score"] for i in range(len(suites)) if i in row["suites"]]
            row["overall"] = sum(have) / len(have) if len(have) == len(suites) else None
        ranked = sorted(rows.values(), key=lambda r: (r["overall"] is None, -(r["overall"] or 0),
                                                      -sum(s["score"] for s in r["suites"].values())))
        views.append({"profile": profile, "rows": ranked})
    return {"title": board.get("title", "examroom leaderboard"), "suites": suites, "views": views,
            "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}


# --- rendering -------------------------------------------------------------------
def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _model_cell(r) -> str:
    name = _e(r["model"])
    if r["hub"]:
        name = f'<a href="https://huggingface.co/{_e(r["hub"])}">{name}</a>'
    prec = r["precision"] or "served (precision unstated)"
    return f'{name}<div class="sub">{_e(prec)}</div>'


def _size_cells(r) -> str:
    active = "dense" if r["active_params"] and r["active_params"] == r["n_params"] else fmt_size(r["active_params"])
    w = fmt_size(r["weights_mb"], "weights_mb") + ("~" if r["weights_source"] == "estimate" else "")
    chips = "".join(f'<span class="chip">{_e(t)}</span>' for t in r["tiers"])
    return (f'<td class="num">{fmt_size(r["n_params"])}</td><td class="num">{_e(active)}</td>'
            f'<td class="num">{_e(w)}</td><td>{chips}</td>')


def _row_attrs(r) -> str:
    return f'data-tiers="{_e(" ".join(r["tiers"]))}"'


def render_board(data: dict) -> str:
    suites = data["suites"]
    tier_rules = {name: (f"≤ {fmt_size(limit)} parameters" if key == "n_params"
                         else f"weights ≤ {limit / 1024:g} GB at the run's precision")
                  for name, key, limit in TIERS}
    tabs, panels = [], []
    for vi, v in enumerate(data["views"]):
        pid = f"p{vi}"
        tabs.append(f'<button class="tab" data-panel="{pid}" aria-selected="{str(vi == 0).lower()}">{_e(v["profile"])}</button>')
        head = ("<tr><th>#</th><th>model</th><th class='num'>params</th><th class='num'>active</th>"
                "<th class='num'>weights</th><th>tiers</th><th class='num sort'>overall</th>"
                + "".join(f"<th class='num sort'>{_e(s.get('label', s['name']))}</th>" for s in suites) + "</tr>")
        body = []
        for i, r in enumerate(v["rows"], 1):
            cells = "".join(
                f'<td class="num" title="run {_e(r["suites"][si]["run_id"])} · {_e(r["suites"][si]["ts"][:10])}">'
                f'{_pct(r["suites"][si]["score"])}</td>' if si in r["suites"] else '<td class="num">–</td>'
                for si in range(len(suites)))
            body.append(f'<tr {_row_attrs(r)}><td class="rank">{i}</td><td>{_model_cell(r)}</td>{_size_cells(r)}'
                        f'<td class="num strong">{_pct(r["overall"])}</td>{cells}</tr>')
        details = []
        for si, s in enumerate(suites):
            task_names = sorted({t for r in v["rows"] if si in r["suites"] for t in r["suites"][si]["tasks"]})
            if not task_names:
                continue
            th = "".join(f"<th class='num sort'>{_e(t)}</th>" for t in task_names)
            trs = []
            for r in sorted((r for r in v["rows"] if si in r["suites"]), key=lambda r: -r["suites"][si]["score"]):
                ts = r["suites"][si]["tasks"]
                tds = "".join(
                    f'<td class="num" title="{_pct(ts[t]["lo"])}–{_pct(ts[t]["hi"])} (95% CI, n={ts[t]["n"]})">'
                    f'{_pct(ts[t]["score"])}</td>' if t in ts else '<td class="num">–</td>' for t in task_names)
                trs.append(f'<tr {_row_attrs(r)}><td>{_model_cell(r)}</td>'
                           f'<td class="num strong">{_pct(r["suites"][si]["score"])}</td>{tds}</tr>')
            pub = f' · public split: <code>{_e(s["public"])}</code>' if s.get("public") else ""
            details.append(
                f'<details><summary>{_e(s.get("label", s["name"]))} <span class="sub">{_e(s["name"])}@{_e(s["version"])}'
                f' · hash {_e(s["hash"])}{pub}</span></summary><div class="scroll"><table class="sortable">'
                f'<thead><tr><th>model</th><th class="num sort">suite</th>{th}</tr></thead>'
                f'<tbody>{"".join(trs)}</tbody></table></div></details>')
        empty = '<p class="empty">No ranked runs under this profile yet.</p>' if not v["rows"] else ""
        panels.append(
            f'<section id="{pid}" class="panel"{"" if vi == 0 else " hidden"}>{empty}'
            f'<div class="scroll"><table class="sortable"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'
            f'<h2>Per task</h2>{"".join(details)}</section>')
    filters = "".join(f'<button class="filter" data-tier="{t}" title="{_e(tier_rules.get(t, ""))}">{t}</button>'
                      for t, _, _ in TIERS)
    return _PAGE.format(title=_e(data["title"]), tabs="".join(tabs), panels="".join(panels), filters=filters,
                        generated=_e(data["generated"][:16].replace("T", " ")),
                        rules=" · ".join(f"<b>{t}</b> {_e(d)}" for t, d in tier_rules.items()))


def render_leaderboard(records: list[dict], board: dict, registry: Registry, out: str | Path) -> Path:
    out = Path(out)
    out.write_text(render_board(build_board(records, board, registry)))
    return out


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#fbfaf8;--fg:#1d1d1b;--muted:#6b6862;--line:#e4e1db;--card:#fff;--accent:#2f5d8a;--chip:#eef2f6;color-scheme:light}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#151614;--fg:#e9e7e2;--muted:#9a978f;--line:#2e2f2b;--card:#1c1d1a;--accent:#8fb6dd;--chip:#24303b;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#151614;--fg:#e9e7e2;--muted:#9a978f;--line:#2e2f2b;--card:#1c1d1a;--accent:#8fb6dd;--chip:#24303b;color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1200px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:1.6rem;margin:0 0 4px}} h2{{font-size:1.1rem;margin:28px 0 8px}}
.lede{{color:var(--muted);margin:0 0 16px}}
.bar{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}}
button{{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:4px 12px;cursor:pointer}}
button[aria-selected="true"],button.on{{background:var(--accent);border-color:var(--accent);color:var(--bg)}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}}
th{{font-weight:600;font-size:.85rem;color:var(--muted);position:sticky;top:0;background:var(--card)}}
th.sort{{cursor:pointer}} th.sort:hover{{color:var(--fg)}}
.num{{text-align:right}} .strong{{font-weight:650}} .rank{{color:var(--muted)}}
.sub{{color:var(--muted);font-size:.8rem;font-weight:400}}
.chip{{display:inline-block;background:var(--chip);border-radius:4px;padding:0 6px;margin-right:4px;font-size:.8rem}}
a{{color:var(--accent);text-decoration:none}} a:hover{{text-decoration:underline}}
details{{margin:8px 0}} summary{{cursor:pointer;padding:6px 0;font-weight:600}}
code{{font-size:.8rem}} .empty{{color:var(--muted)}}
footer{{color:var(--muted);font-size:.8rem;margin-top:32px}}
</style></head><body><main>
<h1>{title}</h1>
<p class="lede">Tiny, small and local language models on curated held-out suites. Scores are percentages; a suite score
averages its tasks, the overall score averages the suites. Hover a score for its run or its 95% interval.</p>
<div class="bar" role="tablist">{tabs}</div>
<div class="bar"><span class="sub">tier:</span><button class="filter on" data-tier="">all</button>{filters}</div>
{panels}
<footer>Tiers: {rules}. Weights marked ~ are estimated from the parameter count (8 bits per weight when an API does not
state its precision). Rows without tiers are unregistered or closed-weight references. Generated {generated} UTC by examroom.</footer>
</main>
<script>
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{{
  document.querySelectorAll('.tab').forEach(x=>x.setAttribute('aria-selected',x===b));
  document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==b.dataset.panel);}}));
document.querySelectorAll('.filter').forEach(b=>b.addEventListener('click',()=>{{
  document.querySelectorAll('.filter').forEach(x=>x.classList.toggle('on',x===b));
  const t=b.dataset.tier;
  document.querySelectorAll('tr[data-tiers]').forEach(r=>r.hidden=!!t&&!r.dataset.tiers.split(' ').includes(t));}}));
const val=td=>{{const v=parseFloat(td.textContent);return isNaN(v)?-1:v}};
document.querySelectorAll('table.sortable th.sort').forEach(th=>th.addEventListener('click',()=>{{
  const tb=th.closest('table').tBodies[0],i=[...th.parentNode.children].indexOf(th);
  const desc=th.dataset.dir!=='desc';th.dataset.dir=desc?'desc':'asc';
  [...tb.rows].sort((a,b)=>(val(a.cells[i])-val(b.cells[i]))*(desc?-1:1)).forEach(r=>tb.appendChild(r));}}));
</script></body></html>
"""
