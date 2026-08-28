from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from benchsmith.items import load_suite


def cmd_validate(a):
    from benchsmith.validate import validate_suite

    suite = load_suite(a.suite)
    rep = validate_suite(suite, min_items=a.min_items, balance_tolerance=a.balance_tolerance,
                         corpus_dir=Path(a.corpus) if a.corpus else None, ngram=a.ngram)
    for t, m in rep.metrics.items():
        if t == "leakage":
            print(f"leakage: {m['n']}-grams, {m['docs_scanned']} docs scanned, {m['items_indexed']} items indexed "
                  f"({m['items_too_short']} shorter than n), {m['items_colliding']} colliding")
            continue
        print(f"{t:20s} n={m['n']:<5d}", end="")
        if "prior_accuracy" in m:
            print(f" chance={m['chance']:.3f} loo_prior={m['prior_accuracy']:.3f}", end="")
        if "mean_vocab_jaccard" in m:
            print(f" len_diff={m['mean_len_diff']:+.2f} good_longer={m['good_longer_frac']:.2f} jaccard={m['mean_vocab_jaccard']:.2f}", end="")
        if "max_conditional" in m:
            mc = m["max_conditional"]
            print(f" max P(correct|shown)={mc['p_correct_given_shown']:.2f} ({mc['option']!r}, shown {mc['shown']}x)", end="")
        print()
    for w in rep.warnings[:a.max_messages]:
        print("WARN ", w)
    if len(rep.warnings) > a.max_messages:
        print(f"... {len(rep.warnings) - a.max_messages} more warnings")
    for e in rep.errors[:a.max_messages]:
        print("ERROR", e)
    if len(rep.errors) > a.max_messages:
        print(f"... {len(rep.errors) - a.max_messages} more errors")
    print(f"hash={suite.content_hash()}  items={len(suite)}  {'OK' if rep.ok else 'FAILED'}")
    return 0 if rep.ok else 1


def cmd_leakage(a):
    from benchsmith.leakage import check_leakage

    rc = 0
    for sp in a.suite:
        suite = load_suite(sp)
        mode = a.mode or suite.meta.meta.get("leakage_check", "ngram")
        rep = check_leakage(suite, [Path(c) for c in a.corpus], n=a.ngram, text_column=a.text_column, mode=mode,
                            progress=lambda d, c: print(f"\r  {d} docs, {c} colliding items", end="", file=sys.stderr))
        print(file=sys.stderr)
        what = (f"{rep.items_indexed} items indexed ({rep.items_too_short} shorter than {a.ngram} words)"
                if mode == "ngram" else f"{rep.items_indexed} items, exact document/prefix identity")
        print(f"{suite.name}@{suite.version} [{mode}]: {rep.docs_scanned} docs scanned, {what}, {len(rep.collisions)} colliding")
        for iid, hits in sorted(rep.collisions.items()):
            for ref, gram in hits[: a.max_hits]:
                print(f"  {iid}  {ref}  {gram!r}")
        rc |= not rep.ok
    return rc


def cmd_freeze(a):
    suite = load_suite(a.suite)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    manifest = {
        "suite": suite.name, "version": suite.version, "content_hash": suite.content_hash(),
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), "git_commit": commit,
        "n_items": len(suite),
        "meta": suite.meta.meta,
        "files": {
            t.file: {
                "task": t.name, "kind": str(t.kind), "n_items": len(suite.items[t.name]),
                "provenance": dict(Counter(f"{i.provenance.method}:{i.provenance.source}" for i in suite.items[t.name])),
                "seeds": sorted({i.provenance.seed for i in suite.items[t.name] if i.provenance.seed is not None}),
                "categories": dict(Counter(i.category for i in suite.items[t.name])),
            } for t in suite.tasks
        },
    }
    out = suite.root / "MANIFEST.json"
    if out.exists() and not a.force:
        old = json.loads(out.read_text())
        if old.get("content_hash") != manifest["content_hash"]:
            print(f"REFUSING: {out} exists with hash {old.get('content_hash')} != {manifest['content_hash']}.\n"
                  f"A frozen suite must not change in place — bump the version (new directory) or pass --force.")
            return 1
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out} (hash {manifest['content_hash']}, {manifest['n_items']} items)")
    return 0


def cmd_run(a):
    from benchsmith.backends import load_backend
    from benchsmith.ledger import Ledger
    from benchsmith.runner import run_suite

    suite = load_suite(a.suite)
    if not a.allow_unfrozen:
        mf = suite.root / "MANIFEST.json"
        if not mf.exists() or json.loads(mf.read_text()).get("content_hash") != suite.content_hash():
            print("suite is not frozen or has changed since freeze; run `benchsmith freeze` or pass --allow-unfrozen")
            return 1
    kw = {}
    for kv in a.opt or []:
        k, v = kv.split("=", 1)
        kw[k] = int(v) if v.isdigit() else v
    backend = load_backend(a.backend, **kw)
    overrides = {k: v for k, v in {"max_tokens": a.max_tokens, "instruction": a.instruction}.items() if v}
    overrides = overrides or None

    def progress(i, n, r):
        if a.verbose or i % 100 == 0 or i == n:
            print(f"\r[{i}/{n}] {r.task} {r.item_id} {r.score:.0f}", end="", file=sys.stderr)

    run = run_suite(suite, backend, tasks=a.task, limit=a.limit, label=a.label, progress=progress, overrides=overrides)
    backend.close()
    print(file=sys.stderr)
    if not a.dry_run:
        Ledger(a.ledger).record(run)
    print(f"run {run.run_id}  {run.backend['backend']}:{run.backend['model']}  suite={run.suite}@{run.suite_version} "
          f"hash={run.suite_hash}")
    for t, (s, n) in run.task_scores().items():
        print(f"  {t:20s} {s:.3f}  (n={n})")
    print(f"  {'TOTAL':20s} {run.score:.3f}  (n={len(run.results)})")
    st = run.stats
    print(f"  tokens/s={st['tokens_per_second']}  wall={st['wall_seconds']}s  peak_rss={st['peak_rss_mb']}MB"
          f"  peak_device={st['peak_device_mem_mb']}MB")
    tiers: dict[str, dict[str, list[float]]] = {}
    for r in run.results:
        j = r.detail.get("judge")
        for jj in (j if isinstance(j, list) else [j]) if j else []:
            for k, v in (jj.get("tiers") or {}).items():
                tiers.setdefault(r.task, {}).setdefault(k, []).append(v)
    for t, tv in tiers.items():
        print(f"  tiers {t:14s} " + "  ".join(f"{k}={sum(v) / len(v):.3f}" for k, v in sorted(tv.items())))
    protocols = Counter(r.protocol for r in run.results)
    if any(r.protocol != str(suite.task(r.task).kind) for r in run.results):
        print(f"  NOTE protocol fallback used: {dict(protocols)}")
    return 0


def cmd_runs(a):
    from benchsmith.ledger import Ledger

    rows = Ledger(a.ledger).runs(suite=a.suite, model=a.model)
    print(f"{'run_id':12s} {'ts':20s} {'suite':14s} {'hash':10s} {'model':40s} {'quant':6s} {'score':6s} {'n':5s} label")
    for r in rows:
        print(f"{r['run_id']:12s} {r['ts'][:19]:20s} {r['suite'] + '@' + r['suite_version']:14s} {r['suite_hash'][:8]:10s} "
              f"{r['model'][:40]:40s} {(r['quantization'] or '-'):6s} {r['score']:.3f}  {r['n_items']:<5d}{r['label'] or ''}")
    return 0


def cmd_diff(a):
    from benchsmith.diff import diff_runs, wilson
    from benchsmith.ledger import Ledger

    d = diff_runs(Ledger(a.ledger), a.run_a, a.run_b)
    if not d.same_benchmark:
        print("WARNING: runs used different benchmark content (hash mismatch); comparison is not apples-to-apples")
    print(f"{'task':20s} {'n':>5s} {'A':>7s} {'B':>7s} {'delta':>7s} {'fixed':>6s} {'broken':>6s} {'p':>7s}")
    tot_n = 0
    for t in d.tasks:
        tot_n += t.n
        flag = " *" if t.p_value < 0.05 else ""
        print(f"{t.task:20s} {t.n:5d} {t.score_a:7.3f} {t.score_b:7.3f} {t.delta:+7.3f} {len(t.fixed):6d} {len(t.broken):6d} {t.p_value:7.3f}{flag}")
    if d.tasks:
        sa = sum(t.score_a * t.n for t in d.tasks) / tot_n
        sb = sum(t.score_b * t.n for t in d.tasks) / tot_n
        lo, hi = wilson(sb, tot_n)
        print(f"{'TOTAL':20s} {tot_n:5d} {sa:7.3f} {sb:7.3f} {sb - sa:+7.3f} {d.total_fixed:6d} {d.total_broken:6d}"
              f"   B 95% CI [{lo:.3f}, {hi:.3f}]")
    if a.items:
        for t in d.tasks:
            for i in t.fixed:
                print(f"FIXED  {i}")
            for i in t.broken:
                print(f"BROKEN {i}")
    return 0


def cmd_calibrate(a):
    """T4: KL between a run's sampled completions and the suite's reference_stats.json."""
    from benchsmith.checks.calibration import calibrate
    from benchsmith.ledger import Ledger

    led = Ledger(a.ledger)
    row = led.run(a.run_id)
    suite = load_suite(a.suite)
    if suite.content_hash() != row["suite_hash"]:
        print("WARNING: suite content hash differs from the run's")
    ref_path = suite.root / suite.meta.meta.get("reference_stats", "reference_stats.json")
    ref = json.loads(ref_path.read_text())
    items = {it.id: it for _, it in suite.iter_items()}
    rep = calibrate(items, led.results(row["run_id"]), ref)
    print(f"run {row['run_id']} {row['model']}: {rep['parsed']}/{rep['completions']} completions parsed; "
          f"mean KL = {rep['mean_kl']}")
    print(f"{'field':26s} {'KL':>8s} {'TV':>7s} {'n':>7s}")
    for k, f in sorted(rep["fields"].items(), key=lambda kv: -(kv[1]["kl"] or -1)):
        print(f"{k:26s} {f['kl'] if f['kl'] is not None else '-':>8} {f['tv'] if f['tv'] is not None else '-':>7} {f['n']:>7}")
    if not a.dry_run:
        led.set_metric(row["run_id"], "t4_mean_kl", rep["mean_kl"], rep)
        print("stored as metric t4_mean_kl")
    return 0


def cmd_history(a):
    from benchsmith.ledger import Ledger

    for r in Ledger(a.ledger).item_history(a.item_id):
        print(f"{r['ts'][:19]}  {r['model']:40s} {(r['quantization'] or '-'):6s} {r['score']:.2f}  {r['protocol']}  {r['label'] or ''}")
    return 0


def cmd_report(a):
    from benchsmith.ledger import Ledger
    from benchsmith.report import render_report

    ledger = Ledger(a.ledger)
    out = render_report(ledger, a.out, suite=a.suite)
    n = len(ledger.runs(suite=a.suite))
    print(f"wrote {out} ({n} runs) — open it in a browser")
    return 0


def cmd_show(a):
    import random

    suite = load_suite(a.suite)
    rng = random.Random(a.seed)
    for t in suite.tasks:
        if a.task and t.name not in a.task:
            continue
        items = suite.items[t.name]
        print(f"== {t.name} ({len(items)} items, kind={t.kind})")
        for it in rng.sample(items, min(a.n, len(items))):
            if it.options:
                tail = " | ".join(f"[{o}]" if i == it.answer_idx else o for i, o in enumerate(it.options))
            elif it.good:
                tail = f"good={it.good!r} bad={it.bad!r}"
            else:
                tail = f"target={(it.target or '')[:60]!r} checker={it.checker}"
            print(f"  {it.id}: {it.prompt!r} -> {tail}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchsmith", description="Custom LLM benchmarks with a regression ledger.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="check schema, duplicates, balance, conventions, leakage")
    p.add_argument("suite"); p.add_argument("--min-items", type=int, default=200)
    p.add_argument("--balance-tolerance", type=float, default=0.10)
    p.add_argument("--corpus", help="directory of training text for n-gram leakage check")
    p.add_argument("--ngram", type=int, default=13); p.add_argument("--max-messages", type=int, default=30)
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("leakage", help="streaming n-gram overlap check of suites against a training corpus")
    p.add_argument("suite", nargs="+"); p.add_argument("--corpus", action="append", required=True, help="file or directory (txt/md/jsonl/parquet)")
    p.add_argument("--ngram", type=int, default=13); p.add_argument("--text-column", default="text")
    p.add_argument("--max-hits", type=int, default=3)
    p.add_argument("--mode", choices=["ngram", "exact"], help="override the suite's declared check (meta.leakage_check)")
    p.set_defaults(fn=cmd_leakage)

    p = sub.add_parser("freeze", help="write MANIFEST.json with content hash and provenance")
    p.add_argument("suite"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("run", help="evaluate a model on a suite and record to the ledger")
    p.add_argument("suite"); p.add_argument("backend", help="hf:<model> | openai:<model> | dummy:uniform")
    p.add_argument("--opt", action="append", metavar="KEY=VAL", help="backend option (dtype=bf16, quantization=4bit, base_url=…, batch_size=8)")
    p.add_argument("--task", action="append"); p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, help="override generation budget for every task")
    p.add_argument("--instruction", help="append an instruction to every generative prompt "
                   "(chat models: e.g. 'Answer with only the number.'); recorded in run metadata")
    p.add_argument("--label"); p.add_argument("--ledger", default="benchsmith.sqlite")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--allow-unfrozen", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true"); p.set_defaults(fn=cmd_run)

    p = sub.add_parser("runs", help="list recorded runs")
    p.add_argument("--suite"); p.add_argument("--model"); p.add_argument("--ledger", default="benchsmith.sqlite")
    p.set_defaults(fn=cmd_runs)

    p = sub.add_parser("diff", help="compare two runs item by item")
    p.add_argument("run_a"); p.add_argument("run_b"); p.add_argument("--items", action="store_true")
    p.add_argument("--ledger", default="benchsmith.sqlite"); p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("calibrate", help="T4: KL of a run's sampled schedules vs the suite's reference_stats.json")
    p.add_argument("suite"); p.add_argument("run_id"); p.add_argument("--ledger", default="benchsmith.sqlite")
    p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("history", help="every run's result for one item")
    p.add_argument("item_id"); p.add_argument("--ledger", default="benchsmith.sqlite"); p.set_defaults(fn=cmd_history)

    p = sub.add_parser("report", help="render the ledger as a self-contained HTML page")
    p.add_argument("--out", default="report.html")
    p.add_argument("--suite", help="only include runs of this suite")
    p.add_argument("--ledger", default="benchsmith.sqlite")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("show", help="print a random sample of items for review")
    p.add_argument("suite"); p.add_argument("--task", action="append"); p.add_argument("-n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_show)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
