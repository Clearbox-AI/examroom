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

A board suite may name its public split ("public"); given that suite and its
exported runs, the page also shows, per model, the public items it failed: the
request, the expected answer, the model's output and the checker's reasons.  The
ranked held-out items are never shown (their exports carry no per-item data).
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
from pathlib import Path

from examroom.diff import wilson
from examroom.registry import TIERS, Registry, fmt_size
from examroom.splits import is_heldout


def load_board(path: str | Path) -> dict:
    board = json.loads(Path(path).read_text())
    for s in board.get("suites", []):
        missing = {"name", "version", "hash"} - s.keys()
        if missing:
            raise ValueError(f"board suite entry {s} lacks {', '.join(sorted(missing))}")
    if not board.get("profiles"):
        raise ValueError("board lists no profiles: unprofiled runs are never ranked")
    return board


def check_public(board_suite: dict, public) -> None:
    """The public split must be the other half of the ranked held-out suite."""
    if is_heldout(public.meta.name):
        raise ValueError(f"{board_suite['name']}: 'public' names a held-out suite ({public.meta.name})")
    manifest = public.root / "HELDOUT_MANIFEST.json"
    paired = json.loads(manifest.read_text()).get("content_hash") if manifest.exists() else None
    if paired != board_suite["hash"]:
        raise ValueError(f"{board_suite['name']}: public split {public.meta.name}@{public.meta.version} is paired "
                         f"with held-out hash {paired}, not {board_suite['hash']}")


def _row_key(b: dict, registry: Registry):
    ms = registry.size(b)
    return (ms.model_id or f"{b.get('backend')}:{b.get('model')}", ms.precision), ms


def _shared_prefix(prompts: list[str]) -> str:
    """The instructions every item of a task starts with, up to a line break."""
    p = os.path.commonprefix(prompts) if len(prompts) > 1 else ""
    return p[:p.rfind("\n") + 1]


def _examples(rec: dict, public, prefixes: dict[str, str], per_task: int) -> dict:
    items = {it.id: it for _, it in public.iter_items()}
    by_task: dict[str, list] = {}
    for r in rec.get("results", []):
        by_task.setdefault(r["task"], []).append(r)
    tasks = {}
    for t in [t.name for t in public.meta.tasks if t.name in by_task]:
        rs = by_task[t]
        failed = sorted((r for r in rs if r["score"] < 1), key=lambda r: (r["score"], r["item_id"]))
        shown = []
        for r in failed[:per_task]:
            it, d = items.get(r["item_id"]), r.get("detail") or {}
            if it is None:
                continue
            outs = d.get("outputs") or []
            why = (d.get("judge") or {}).get("failed") or []
            shown.append({"item_id": r["item_id"], "score": r["score"],
                          "request": it.prompt[len(prefixes.get(t, "")):], "expected": it.target,
                          "output": outs[0] if outs else "", "why": [str(w) for w in why]})
        tasks[t] = {"n": len(rs), "failed": len(failed), "shown": shown}
    return {"run_id": rec["run_id"], "tasks": tasks}


def build_board(records: list[dict], board: dict, registry: Registry, public: dict | None = None,
                per_task: int = 2) -> dict:
    """`public` maps a board suite's index to its loaded public split (see check_public);
    rows then carry up to `per_task` failed public items per task."""
    suites = board["suites"]
    public = public or {}
    prefixes = {si: {t.name: _shared_prefix([it.prompt for it in pub.items[t.name]]) for t in pub.meta.tasks}
                for si, pub in public.items()}
    pub_ids = {si: (pub.meta.name, pub.meta.version, pub.content_hash()) for si, pub in public.items()}
    views = []
    for profile in board["profiles"]:
        rows: dict[tuple, dict] = {}
        for si, sc in enumerate(suites):
            for rec in records:
                b = rec["backend"]
                if (rec["suite"], rec["suite_version"], rec["suite_hash"]) != (sc["name"], sc["version"], sc["hash"]) \
                        or b.get("profile") != profile:
                    continue
                key, ms = _row_key(b, registry)
                row = rows.setdefault(key, {
                    "model": key[0], "registered": ms.model_id is not None, "precision": ms.precision,
                    "n_params": ms.n_params, "active_params": ms.active_params, "weights_mb": ms.weights_mb,
                    "weights_source": ms.weights_source, "tiers": ms.tiers,
                    "hub": registry.models.get(ms.model_id, {}).get("hub", {}).get("repo"),
                    "suites": {}, "examples": {},
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
        for si, ident in pub_ids.items():
            latest = {}
            for rec in records:
                if (rec["suite"], rec["suite_version"], rec["suite_hash"]) != ident or is_heldout(rec["suite"]) \
                        or rec["backend"].get("profile") != profile or "results" not in rec:
                    continue
                key, _ = _row_key(rec["backend"], registry)
                if key in rows and (key not in latest or latest[key]["ts"] < rec["ts"]):
                    latest[key] = rec
            for key, rec in latest.items():
                rows[key]["examples"][si] = _examples(rec, public[si], prefixes[si], per_task)
        for row in rows.values():
            have = [row["suites"][i]["score"] for i in range(len(suites)) if i in row["suites"]]
            row["overall"] = sum(have) / len(have) if len(have) == len(suites) else None
        ranked = sorted(rows.values(), key=lambda r: (r["overall"] is None, -(r["overall"] or 0),
                                                      -sum(s["score"] for s in r["suites"].values())))
        views.append({"profile": profile, "rows": ranked})
    return {"title": board.get("title", "examroom leaderboard"), "suites": suites, "views": views,
            "instructions": prefixes, "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}


# --- rendering -------------------------------------------------------------------
def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _model_cell(r, fx_id: str | None = None) -> str:
    name = _e(r["model"])
    if r["hub"]:   # the Space shows the page in a frame, which huggingface.co refuses: open a new tab
        name = f'<a href="https://huggingface.co/{_e(r["hub"])}" target="_blank" rel="noopener">{name}</a>'
    prec = _e(r["precision"] or "served (precision unstated)")
    if fx_id:
        prec += f' · <a class="fx" href="#{fx_id}">failed examples</a>'
    return f'{name}<div class="sub">{prec}</div>'


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
            fx_id = f"fx-{pid}-{i}" if r["examples"] else None
            body.append(f'<tr {_row_attrs(r)}><td class="rank">{i}</td><td>{_model_cell(r, fx_id)}</td>{_size_cells(r)}'
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
            f'<h2>Per task</h2>{"".join(details)}{_examples_section(v["rows"], suites, data, pid)}</section>')
    filters = "".join(f'<button class="filter" data-tier="{t}" title="{_e(tier_rules.get(t, ""))}">{t}</button>'
                      for t, _, _ in TIERS)
    return _PAGE.format(title=_e(data["title"]), tabs="".join(tabs), panels="".join(panels), filters=filters,
                        generated=_e(data["generated"][:16].replace("T", " ")),
                        rules=" · ".join(f"<b>{t}</b> {_e(d)}" for t, d in tier_rules.items()))


_SENTINEL = re.compile(r"<object object at 0x[0-9a-f]+>")   # a checker's "argument missing" marker


def _pre(label: str, text: str) -> str:
    return f'<div class="lbl">{label}</div><pre>{_e(text)}</pre>' if text else ""


def _examples_section(rows: list[dict], suites: list[dict], data: dict, pid: str) -> str:
    if not any(r["examples"] for r in rows):
        return ""
    instr = "".join(
        f'<details><summary class="sub">Task instructions: {_e(s.get("label", s["name"]))}</summary>'
        + "".join(f'<div class="lbl">{_e(t)}</div><pre>{_e(text)}</pre>'
                  for t, text in data["instructions"].get(si, {}).items() if text)
        + "</details>"
        for si, s in enumerate(suites) if data["instructions"].get(si))
    blocks = []
    for i, r in enumerate(rows, 1):
        if not r["examples"]:
            continue
        parts = []
        for si, ex in sorted(r["examples"].items()):
            for t, te in ex["tasks"].items():      # in suite order
                if not te["failed"]:
                    parts.append(f'<h4>{_e(t)} <span class="sub">no failures in {te["n"]} items</span></h4>')
                    continue
                parts.append(f'<h4>{_e(t)} <span class="sub">{te["failed"]} of {te["n"]} items below full score'
                             f'{"; lowest scores shown" if te["failed"] > len(te["shown"]) else ""}</span></h4>')
                for e in te["shown"]:
                    why = "".join(f"<li>{_e(_SENTINEL.sub('(missing)', w))}</li>" for w in e["why"])
                    parts.append(
                        f'<div class="ex"><div class="sub">{_e(e["item_id"])} · score {e["score"]:.2f}</div>'
                        f'{_pre("request", e["request"])}{_pre("expected", e["expected"])}'
                        f'{_pre("model output", e["output"])}'
                        f'{f"<div class=lbl>checker</div><ul>{why}</ul>" if why else ""}</div>')
            label = suites[si].get("label", suites[si]["name"])
            parts.insert(0, f'<p class="sub">{_e(label)}: public-split run {_e(ex["run_id"])}</p>')
        failed = sum(te["failed"] for ex in r["examples"].values() for te in ex["tasks"].values())
        total = sum(te["n"] for ex in r["examples"].values() for te in ex["tasks"].values())
        blocks.append(f'<details id="fx-{pid}-{i}" class="fxm"><summary>{i}. {_e(r["model"])}'
                      f' <span class="sub">{failed} of {total} public items below full score</span></summary>'
                      f'{"".join(parts)}</details>')
    return ('<h2>Failed examples</h2><p class="sub">From each model\'s run on the public split: the same generator '
            'and tasks as the ranked suite, but different items (the held-out items are never published). '
            'Requests are shown without the instructions every item of a task shares; those are listed '
            f'once below.</p>{instr}{"".join(blocks)}')


def render_leaderboard(records: list[dict], board: dict, registry: Registry, out: str | Path,
                       public: dict | None = None, per_task: int = 2) -> Path:
    out = Path(out)
    out.write_text(render_board(build_board(records, board, registry, public, per_task)))
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
h4{{margin:16px 0 4px;font-size:.95rem}} .lbl{{color:var(--muted);font-size:.75rem;margin-top:6px;text-transform:uppercase;letter-spacing:.04em}}
pre{{white-space:pre-wrap;word-break:break-word;max-height:14em;overflow:auto;margin:2px 0;padding:6px 8px;background:var(--card);border:1px solid var(--line);border-radius:6px;font-size:.8rem}}
.ex{{border-left:3px solid var(--line);padding:2px 0 6px 10px;margin:8px 0}} .ex ul{{margin:2px 0;padding-left:20px;font-size:.85rem}}
details.fxm{{border-top:1px solid var(--line)}} details.fxm>summary{{padding:8px 0}}
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
document.querySelectorAll('a.fx').forEach(a=>a.addEventListener('click',()=>{{
  const d=document.getElementById(a.getAttribute('href').slice(1));if(d)d.open=true;}}));
const val=td=>{{const v=parseFloat(td.textContent);return isNaN(v)?-1:v}};
document.querySelectorAll('table.sortable th.sort').forEach(th=>th.addEventListener('click',()=>{{
  const tb=th.closest('table').tBodies[0],i=[...th.parentNode.children].indexOf(th);
  const desc=th.dataset.dir!=='desc';th.dataset.dir=desc?'desc':'asc';
  [...tb.rows].sort((a,b)=>(val(a.cells[i])-val(b.cells[i]))*(desc?-1:1)).forEach(r=>tb.appendChild(r));}}));
</script></body></html>
"""
