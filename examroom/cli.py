from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from examroom.items import load_suite


def cmd_validate(a):
    from examroom.validate import validate_suite

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
    from examroom.leakage import check_leakage

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
    from examroom.items import write_manifest

    suite = load_suite(a.suite)
    try:
        manifest = write_manifest(suite, force=a.force)
    except FileExistsError as e:
        print(e.args[0])
        return 1
    print(f"wrote {suite.root / 'MANIFEST.json'} (hash {manifest['content_hash']}, {manifest['n_items']} items)")
    return 0


def cmd_split(a):
    from examroom.splits import quoted_heldout, split_suite

    try:
        public, heldout = split_suite(a.suite, a.public, a.heldout, heldout_frac=a.frac, seed=a.seed,
                                      group_id=a.group_id)
    except (ValueError, FileExistsError) as e:
        print(e.args[0])
        return 1
    for s in (public, heldout):
        counts = "  ".join(f"{t.name}={len(s.items[t.name])}" for t in s.tasks)
        print(f"{s.name}@{s.version}  hash={s.content_hash()}  {len(s)} items  -> {s.root}\n    {counts}")
    hits = quoted_heldout(public.root, heldout)
    for f, iid in hits:
        print(f"WARN {public.root / f} quotes held-out item {iid}: edit it before publishing")
    print(f"public {public.root} carries HELDOUT_MANIFEST.json; keep {heldout.root} in a private repo only")
    return 1 if hits else 0


def cmd_run(a):
    from examroom.backends import load_backend
    from examroom.ledger import Ledger
    from examroom.runner import run_suite

    suite = load_suite(a.suite)
    if not a.allow_unfrozen:
        mf = suite.root / "MANIFEST.json"
        if not mf.exists() or json.loads(mf.read_text()).get("content_hash") != suite.content_hash():
            print("suite is not frozen or has changed since freeze; run `examroom freeze` or pass --allow-unfrozen")
            return 1
    from examroom.profiles import PROFILE_FIXED_OPTIONS, get_profile
    from examroom.selection import SIZE_KEYS, parse_size

    kw, size, model_id = {}, {}, None
    for kv in a.opt or []:
        k, v = kv.split("=", 1)
        if k in SIZE_KEYS:  # model metadata for `examroom select`, not a backend option
            size[k] = parse_size(k, v)
        elif k == "model_id":  # registry id, when the backend's model name is not one of its aliases
            model_id = v
        else:
            kw[k] = int(v) if v.isdigit() else v
    overrides = {k: v for k, v in {"max_tokens": a.max_tokens, "instruction": a.instruction}.items() if v}
    profile = None
    if a.profile:
        try:
            profile = get_profile(a.profile)
        except KeyError as e:
            print(e.args[0])
            return 1
        clash = sorted((PROFILE_FIXED_OPTIONS & kw.keys()) | overrides.keys())
        if not a.dry_run:  # a ranked run covers the whole suite
            clash += [f"--{k}" for k in ("limit", "task") if getattr(a, k)]
        if clash:
            print(f"profile {profile.id} fixes {', '.join(clash)}; drop them, or drop --profile for an "
                  f"exploratory run (unprofiled runs are not ranked)")
            return 1
        kw.update(profile.backend_options(a.backend.split(":", 1)[0]))
        overrides = profile.overrides()
    backend = load_backend(a.backend, **kw)
    for k, v in size.items():  # explicit values win over what the backend measured
        setattr(backend.info, k, v)
    if backend.info.quantization is None and isinstance(kw.get("quantization"), str):
        backend.info.quantization = kw["quantization"]  # e.g. the GGUF quant behind an openai: server
    if model_id:
        backend.info.extra["model_id"] = model_id
    if profile:
        backend.info.extra["profile"] = profile.id
        backend.info.extra["profile_spec"] = profile.as_dict()
    overrides = overrides or None

    def progress(i, n, r):
        if a.verbose or i % 100 == 0 or i == n:
            print(f"\r[{i}/{n}] {r.task} {r.item_id} {r.score:.0f}", end="", file=sys.stderr)

    run = run_suite(suite, backend, tasks=a.task, limit=a.limit, label=a.label, progress=progress,
                    overrides=overrides, workers=a.workers)
    backend.close()
    print(file=sys.stderr)
    if not a.dry_run:
        Ledger(a.ledger).record(run)
    print(f"run {run.run_id}  {run.backend['backend']}:{run.backend['model']}  suite={run.suite}@{run.suite_version} "
          f"hash={run.suite_hash}  profile={run.backend.get('profile') or '- (unranked)'}")
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
    from examroom.ledger import Ledger

    from examroom.selection import fmt_size

    rows = Ledger(a.ledger).runs(suite=a.suite, model=a.model)
    print(f"{'run_id':12s} {'ts':20s} {'suite':14s} {'hash':10s} {'model':40s} {'params':6s} {'quant':6s} "
          f"{'profile':10s} {'score':6s} {'n':5s} label")
    for r in rows:
        prof = json.loads(r["backend_json"]).get("profile") or "-"
        print(f"{r['run_id']:12s} {r['ts'][:19]:20s} {r['suite'] + '@' + r['suite_version']:14s} {r['suite_hash'][:8]:10s} "
              f"{r['model'][:40]:40s} {fmt_size(r['n_params']):6s} {(r['quantization'] or '-'):6s} {prof:10s} "
              f"{r['score']:.3f}  {r['n_items']:<5d}{r['label'] or ''}")
    return 0


def cmd_annotate(a):
    from examroom.ledger import Ledger
    from examroom.selection import SIZE_KEYS, parse_size

    fields = {}
    for kv in a.fields:
        k, _, v = kv.partition("=")
        if k == "quantization":
            fields[k] = v
        elif k in SIZE_KEYS:
            fields[k] = parse_size(k, v)
        else:
            print(f"unknown field {k!r}; expected quantization or one of {', '.join(SIZE_KEYS)}")
            return 1
    led = Ledger(a.ledger)
    led.annotate(a.run_id, **fields)
    print(f"run {led.run(a.run_id)['run_id']}: " + "  ".join(f"{k}={v}" for k, v in fields.items()))
    return 0


def cmd_select(a):
    from examroom.ledger import Ledger
    from examroom.selection import render, select

    suite = load_suite(a.suite)
    acc = suite.meta.meta.get("acceptance") or {}
    criteria = list(acc.items())
    if a.criteria:
        criteria += list(json.loads(Path(a.criteria).read_text()).items())
    for kv in a.min or []:
        pat, _, v = kv.rpartition("=")
        if not pat:
            print(f"--min expects TASK_GLOB=MIN, got {kv!r}")
            return 1
        criteria.append((pat, v))
    try:
        criteria = [(p, float(v)) for p, v in criteria]
    except ValueError as e:
        print(f"acceptance minimums must be numbers: {e}")
        return 1
    if bad := [p for p, v in criteria if not 0 <= v <= 1]:
        print(f"acceptance minimums must be in [0, 1]: {bad}")
        return 1
    from examroom.registry import Registry

    try:
        sel = select(Ledger(a.ledger), suite, criteria, gate=a.gate, size_key=a.size,
                     registry=Registry.load(a.registry), tier=a.tier, profile=a.profile)
    except KeyError as e:
        print(e.args[0])
        return 1
    print(render(sel))
    return 1 if sel.gated and not any(c.passed for c in sel.candidates) else 0


def cmd_profiles(a):
    from examroom.profiles import PROFILES

    for p in PROFILES.values():
        print(f"{p.id:12s} reasoning={p.reasoning:3s}  temperature={p.temperature if p.temperature is not None else 'suite'!s:5s}  "
              f"min_max_tokens={p.min_max_tokens or '-':<6}  {p.description}")
    return 0


def cmd_models_list(a):
    from examroom.registry import Registry, fmt_size, tiers_for

    reg = Registry.load(a.registry)
    if not reg.models:
        print(f"no models in {a.registry}; add one with `examroom models add`")
        return 0
    for mid, e in sorted(reg.models.items()):
        active = "dense" if e.get("active_params") == e.get("n_params") else fmt_size(e.get("active_params"))
        print(f"{mid}  params={fmt_size(e.get('n_params'))}  active={active}"
              + (f"  aliases={','.join(e['aliases'])}" if e.get("aliases") else ""))
        for prec, q in sorted(e.get("quants", {}).items()):
            tiers = tiers_for(e.get("n_params"), q.get("weights_mb"))
            print(f"    {prec:10s} {fmt_size(q.get('weights_mb'), 'weights_mb'):>8s}  tiers={','.join(tiers) or '-'}"
                  + (f"  {q['file']}" if q.get("file") else ""))
    return 0


def cmd_models_add(a):
    from examroom.registry import Registry, fmt_size

    reg = Registry.load(a.registry)
    if a.dense and a.active:
        print("--dense and --active are exclusive")
        return 1
    try:
        e = reg.add(a.model_id, a.params, a.params if a.dense else a.active, a.alias, a.source)
    except ValueError as err:
        print(err)
        return 1
    reg.save()
    print(f"{a.model_id}: params={fmt_size(e['n_params'])} active={fmt_size(e['active_params'])} -> {a.registry}")
    return 0


def cmd_models_add_hub(a):
    from examroom.hub import model_facts
    from examroom.registry import Registry, fmt_size

    reg = Registry.load(a.registry)
    if a.dense and a.active:
        print("--dense and --active are exclusive")
        return 1
    f = model_facts(a.repo)
    try:
        e = reg.add(a.repo, f["n_params"], f["n_params"] if a.dense else a.active, a.alias,
                    source=f"hf safetensors metadata @ {f['revision'][:12]}" + (f"; {a.source}" if a.source else ""),
                    card=a.card, hub={"repo": a.repo, "revision": f["revision"]})
    except ValueError as err:
        print(err)
        return 1
    if f["precision"]:
        reg.set_quant(a.repo, f["precision"], f["quant"])
    reg.save()
    print(f"{a.repo}: params={fmt_size(e['n_params'])} active={fmt_size(e['active_params'])} "
          f"{f['precision']}={fmt_size(f['quant']['weights_mb'], 'weights_mb')} -> {a.registry}")
    return 0


def cmd_models_quant_hub(a):
    from examroom.hub import quant_facts
    from examroom.registry import Registry, fmt_size

    reg = Registry.load(a.registry)
    try:
        q = reg.set_quant(a.model_id, a.precision, quant_facts(a.repo, a.match))
    except (KeyError, ValueError) as err:
        print(err.args[0])
        return 1
    reg.save()
    print(f"{a.model_id} {a.precision}: {fmt_size(q['weights_mb'], 'weights_mb')} from {a.repo}"
          + (f" ({q['file']})" if q.get("file") else "") + f" -> {a.registry}")
    return 0


def cmd_models_quant(a):
    from examroom.registry import Registry, fmt_size

    reg = Registry.load(a.registry)
    try:
        q = reg.add_quant(a.model_id, a.precision, weights_mb=a.weights_mb, file=a.file)
    except (KeyError, ValueError) as err:
        print(err.args[0])
        return 1
    reg.save()
    print(f"{a.model_id} {a.precision}: {fmt_size(q['weights_mb'], 'weights_mb')} -> {a.registry}")
    return 0


def cmd_export(a):
    from examroom.export import write_exports
    from examroom.ledger import Ledger

    led = Ledger(a.ledger)
    ids = list(a.run_id)
    if a.profiled:
        ids += [r["run_id"] for r in led.runs(suite=a.suite) if json.loads(r["backend_json"]).get("profile")]
    if not ids:
        print("no runs to export: name run ids or pass --profiled")
        return 1
    for path in write_exports(led, sorted(set(ids)), a.out):
        print(path)
    return 0


def cmd_leaderboard(a):
    from examroom.export import load_exports
    from examroom.leaderboard import load_board, render_leaderboard
    from examroom.registry import Registry

    records = load_exports(a.results)
    board = load_board(a.board)
    out = render_leaderboard(records, board, Registry.load(a.registry), a.out)
    print(f"wrote {out} ({len(records)} run records, {len(board['suites'])} suites, "
          f"profiles {', '.join(board['profiles'])})")
    return 0


def cmd_publish(a):
    from examroom import hub

    try:
        if a.what == "suite":
            print(hub.publish_suite(a.path, a.repo))
        elif a.what == "results":
            extra = {k: v for k, v in (("models.json", a.registry), ("board.json", a.board)) if v}
            print(hub.publish_results(a.path, a.repo, extra))
        else:
            print(hub.publish_page(a.path, a.repo))
    except (ValueError, PermissionError, FileExistsError) as e:
        print(f"REFUSING: {e}")
        return 1
    return 0


def cmd_diff(a):
    from examroom.diff import diff_runs, wilson
    from examroom.ledger import Ledger

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
    from examroom.ledger import Ledger
    from examroom.scoring import load_checker

    led = Ledger(a.ledger)
    row = led.run(a.run_id)
    suite = load_suite(a.suite)
    spec = suite.meta.meta.get("calibration", "./checker.py:calibrate")
    try:
        calibrate = load_checker(spec, suite.root)
    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
        print(f"suite has no calibration entry point ({spec}): {e}")
        return 1
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
    from examroom.ledger import Ledger

    for r in Ledger(a.ledger).item_history(a.item_id):
        print(f"{r['ts'][:19]}  {r['model']:40s} {(r['quantization'] or '-'):6s} {r['score']:.2f}  {r['protocol']}  {r['label'] or ''}")
    return 0


def cmd_report(a):
    from examroom.ledger import Ledger
    from examroom.report import render_report

    ledger = Ledger(a.ledger)
    out = render_report(ledger, a.out, suite=a.suite, suites_dir=a.suites_dir)
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
    ap = argparse.ArgumentParser(prog="examroom", description="Custom LLM benchmarks with a regression ledger.")
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

    p = sub.add_parser("split", help="split a frozen suite into a public and a held-out suite (both frozen)")
    p.add_argument("suite"); p.add_argument("--public", required=True, help="output directory of the public split")
    p.add_argument("--heldout", required=True, help="output directory of the held-out split (keep private)")
    p.add_argument("--frac", type=float, default=0.5, help="held-out share of each (task, category) stratum")
    p.add_argument("--group-id", metavar="REGEX",
                   help="items whose ids match once REGEX is removed stay together, e.g. --group-id='-(en|it)-' "
                        "keeps translations on the same side")
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_split)

    p = sub.add_parser("run", help="evaluate a model on a suite and record to the ledger")
    p.add_argument("suite"); p.add_argument("backend", help="hf:<model> | openai:<model> | dummy:uniform")
    # every command that takes a suite also accepts hf://<org>/<repo>/<suite dir>[@<revision>]
    p.add_argument("--opt", action="append", metavar="KEY=VAL", help="backend option (dtype=bf16, quantization=4bit, base_url=…, batch_size=8); "
                   "n_params=4B, active_params=3B, weights_mb=2500 record model size for `select`; "
                   "model_id=<registry id> when the backend's model name is not a registry alias")
    p.add_argument("--task", action="append"); p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, help="override generation budget for every task")
    p.add_argument("--instruction", help="append an instruction to every generative prompt "
                   "(chat models: e.g. 'Answer with only the number.'); recorded in run metadata")
    p.add_argument("--profile", help="run profile (nothink | think): the fixed protocol leaderboard runs are "
                   "ranked under; see `examroom profiles`")
    p.add_argument("--label"); p.add_argument("--ledger", default="examroom.sqlite")
    p.add_argument("--workers", type=int, default=1,
                   help="score items concurrently (API backends; hf runs stay sequential)")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--allow-unfrozen", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true"); p.set_defaults(fn=cmd_run)

    p = sub.add_parser("runs", help="list recorded runs")
    p.add_argument("--suite"); p.add_argument("--model"); p.add_argument("--ledger", default="examroom.sqlite")
    p.set_defaults(fn=cmd_runs)

    p = sub.add_parser("annotate", help="record model size / quantization on a past run (API backends cannot report it)")
    p.add_argument("run_id")
    p.add_argument("fields", nargs="+", metavar="KEY=VAL",
                   help="n_params=27B | active_params=3B | weights_mb=16500 | quantization=Q4_K_M")
    p.add_argument("--ledger", default="examroom.sqlite"); p.set_defaults(fn=cmd_annotate)

    p = sub.add_parser("select", help="smallest model whose runs clear every per-task minimum")
    p.add_argument("suite")
    p.add_argument("--min", action="append", metavar="TASK_GLOB=MIN",
                   help="per-task minimum, e.g. 'tool_hard_*=0.9'; applied after the suite's meta.acceptance "
                        "and --criteria (last matching pattern wins)")
    p.add_argument("--criteria", help="JSON file {task_glob: minimum}, e.g. a use-case profile")
    p.add_argument("--gate", choices=["lower", "point"], default="lower",
                   help="compare the minimum with the 95%% Wilson lower bound (default) or the raw score")
    p.add_argument("--size", choices=["n_params", "active_params", "weights_mb"], default="n_params",
                   help="what 'smallest' means")
    p.add_argument("--tier", choices=["tiny", "small", "local"], help="only runs of models in this tier")
    p.add_argument("--profile", help="only runs under this profile (a bare name means its current version)")
    p.add_argument("--registry", default="models.json", help="model registry: sizes and tiers")
    p.add_argument("--ledger", default="examroom.sqlite"); p.set_defaults(fn=cmd_select)

    p = sub.add_parser("profiles", help="list run profiles")
    p.set_defaults(fn=cmd_profiles)

    p = sub.add_parser("models", help="model registry: sizes and tiers for the leaderboard")
    p.add_argument("--registry", default="models.json")
    msub = p.add_subparsers(dest="models_cmd", required=True)
    q = msub.add_parser("list", help="registered models, quantisations and tiers"); q.set_defaults(fn=cmd_models_list)
    q = msub.add_parser("add", help="register a model (or update its parameter counts and aliases)")
    q.add_argument("model_id", help="canonical id, e.g. Qwen/Qwen3-30B-A3B")
    q.add_argument("--params", required=True, help="total parameters, e.g. 30.5B")
    q.add_argument("--active", help="active parameters per token (MoE), e.g. 3.3B")
    q.add_argument("--dense", action="store_true", help="not an MoE: active parameters = total")
    q.add_argument("--alias", action="append", default=[],
                   help="other names runs use for it: OpenRouter id, Ollama tag, local path")
    q.add_argument("--source", help="where the counts come from, e.g. 'config.json' or a model card URL")
    q.set_defaults(fn=cmd_models_add)
    q = msub.add_parser("quant", help="record the weight footprint of one precision of a model")
    q.add_argument("model_id"); q.add_argument("precision", help="bf16 | fp8 | Q4_K_M | Q8_0 | 4bit …")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights-mb", type=float, help="footprint in MiB")
    g.add_argument("--file", help="weights file (e.g. a GGUF): size and sha256 are recorded")
    q.set_defaults(fn=cmd_models_quant)
    q = msub.add_parser("add-hub", help="register a model from its Hugging Face repo: total parameters and "
                                        "native-precision footprint from the safetensors metadata")
    q.add_argument("repo", help="e.g. Qwen/Qwen3-30B-A3B-Instruct-2507 (also the registry id)")
    q.add_argument("--active", help="active parameters per token (MoE), from the model card, e.g. 3.3B")
    q.add_argument("--dense", action="store_true", help="not an MoE: active parameters = total")
    q.add_argument("--card", help="the model card's own statement of its size, kept verbatim")
    q.add_argument("--alias", action="append", default=[], help="other names runs use for it (OpenRouter id …)")
    q.add_argument("--source", help="extra provenance, e.g. where --active comes from")
    q.set_defaults(fn=cmd_models_add_hub)
    q = msub.add_parser("quant-hub", help="record a precision's footprint from a Hugging Face repo "
                                          "(all safetensors shards, or one GGUF; mmproj files excluded)")
    q.add_argument("model_id"); q.add_argument("precision", help="fp8 | Q4_0 | Q4_K_M …")
    q.add_argument("repo", help="e.g. Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    q.add_argument("--match", help="glob picking one GGUF when the repo holds several")
    q.set_defaults(fn=cmd_models_quant_hub)

    p = sub.add_parser("export", help="write runs as portable JSON records for a results repo "
                                      "(held-out suites: per-task aggregates only)")
    p.add_argument("run_id", nargs="*"); p.add_argument("--out", required=True)
    p.add_argument("--profiled", action="store_true", help="also every run recorded under a profile")
    p.add_argument("--suite", help="with --profiled: only this suite")
    p.add_argument("--ledger", default="examroom.sqlite"); p.set_defaults(fn=cmd_export)

    p = sub.add_parser("leaderboard", help="render the public leaderboard page from exported runs")
    p.add_argument("results", help="directory of exported run records (searched recursively)")
    p.add_argument("--board", required=True, help="board.json: ranked suites (by hash) and profiles")
    p.add_argument("--registry", default="models.json"); p.add_argument("--out", default="leaderboard.html")
    p.set_defaults(fn=cmd_leaderboard)

    p = sub.add_parser("publish", help="upload to the Hugging Face Hub (repos must exist; needs `hf auth login`)")
    p.add_argument("what", choices=["suite", "results", "page"])
    p.add_argument("path", help="suite directory | exported results directory | leaderboard.html")
    p.add_argument("--repo", required=True, help="dataset repo (suite, results) or static Space (page), "
                                                 "e.g. Clearbox-AI/examroom-suites")
    p.add_argument("--registry", help="results: also upload this registry as models.json")
    p.add_argument("--board", help="results: also upload this board file as board.json")
    p.set_defaults(fn=cmd_publish)

    p = sub.add_parser("diff", help="compare two runs item by item")
    p.add_argument("run_a"); p.add_argument("run_b"); p.add_argument("--items", action="store_true")
    p.add_argument("--ledger", default="examroom.sqlite"); p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("calibrate", help="T4: KL of a run's sampled schedules vs the suite's reference_stats.json")
    p.add_argument("suite"); p.add_argument("run_id"); p.add_argument("--ledger", default="examroom.sqlite")
    p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("history", help="every run's result for one item")
    p.add_argument("item_id"); p.add_argument("--ledger", default="examroom.sqlite"); p.set_defaults(fn=cmd_history)

    p = sub.add_parser("report", help="render the ledger as a self-contained HTML page")
    p.add_argument("--out", default="report.html")
    p.add_argument("--suite", help="only include runs of this suite")
    p.add_argument("--suites-dir", default="suites",
                   help="where to find suite dirs for question/expected-answer display")
    p.add_argument("--ledger", default="examroom.sqlite")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("show", help="print a random sample of items for review")
    p.add_argument("suite"); p.add_argument("--task", action="append"); p.add_argument("-n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_show)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
