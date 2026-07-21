#!/usr/bin/env python3
"""Triage GLM verification reports into a human-adjudication queue + base-rate stats.

Single-model design (GLM-5.2 only; DeepSeek dropped 2026-06-28). The precision signal is the
**self-verification** pass: after the whole-paper read, every major/critical finding is
re-examined by a fresh adversarial subagent (refute prompt) that tries equally hard to break
and to repair it (see verify.py / prompts/refute_v0.md). A finding the model itself *dissolves*
on re-check is almost certainly a false positive; one it *confirms* with a counterexample is
the prize. Aggregating these verdicts directly measures GLM's FP-rate on its own major flags —
which is exactly what the pilot needs to learn (it replaces the old cross-model agreement).

Usage:
  python scripts/triage.py --papers id1 id2 ...
  python scripts/triage.py --batch scripts/pilot_batch.txt --json reports/pilot_triage.json
  python scripts/triage.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

SEV_RANK = {"critical": 3, "major": 2, "minor": 1, "": 0, None: 0}
FIX_RANK = {"no_known_fix": 3, "fix_likely_but_unfound": 2, "sketchable_fix": 1, "trivial_fix": 0}
HARD_FIX = {"fix_likely_but_unfound", "no_known_fix"}


def load_latest(paper_ids=None):
    """Newest report per paper_id (GLM-only corpus, so one family); skips parse-failed runs."""
    best, best_ts = {}, {}
    for path in sorted(glob.glob(os.path.join(REPORTS_DIR, "*.json"))):
        try:
            rec = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        pid = rec.get("paper_id")
        if not pid or (paper_ids and pid not in paper_ids) or not rec.get("parse_ok"):
            continue
        ts = rec.get("generated_at", "")
        if ts >= best_ts.get(pid, ""):
            best_ts[pid] = ts
            best[pid] = rec
    return best


def is_major(it):
    return SEV_RANK.get(it.get("severity"), 0) >= 2 or it.get("fixability") in HARD_FIX


def bucket_of(it):
    """Priority bucket for a single finding, keyed on its self-verify verdict."""
    sv = it.get("self_verify") or {}
    verdict = sv.get("verdict")
    if not is_major(it):
        return "P4_minor"
    if verdict == "confirmed_error":
        return "P1_confirmed_major"
    if verdict == "dissolved":
        return "P3_dissolved_major"          # model retracted on re-check -> likely FP
    if verdict in ("inconclusive", None, "parse_error"):
        return "P2_unsettled_major"          # no self-verify, or it couldn't settle -> needs human
    return "P2_unsettled_major"


PRIORITY = ["P1_confirmed_major", "P2_unsettled_major", "P3_dissolved_major", "P4_minor"]
BLURB = {
    "P1_confirmed_major": "model confirmed on adversarial re-check (counterexample/impossibility) — ADJUDICATE FIRST",
    "P2_unsettled_major": "major, but self-verify was inconclusive or absent — needs human",
    "P3_dissolved_major": "model retracted on re-check — likely false positive, spot-check only",
    "P4_minor": "minor / trivial — noise",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--papers", nargs="+")
    g.add_argument("--batch")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    paper_ids = None
    if args.papers:
        paper_ids = set(args.papers)
    elif args.batch:
        txt = open(args.batch, encoding="utf-8").read()
        m = re.search(r'PAPERS="([^"]+)"', txt)
        paper_ids = set(m.group(1).split()) if m else {
            l.strip() for l in txt.splitlines() if l.strip() and not l.startswith("#")}

    reports = load_latest(paper_ids)
    if not reports:
        print("no parseable reports found for the requested papers in reports/")
        return 1

    # ---- base-rate stats ----
    n = len(reports)
    n_findings = 0
    assess_before, assess_after = defaultdict(int), defaultdict(int)
    sev_fix = defaultdict(int)
    sv_tally = defaultdict(int)
    n_self_verified = 0
    for pid, rec in reports.items():
        rep = rec.get("report", {}) or {}
        assess_before[rep.get("overall_assessment", "?")] += 1
        assess_after[rep.get("assessment_after_self_verify", rep.get("overall_assessment", "?"))] += 1
        for it in rep.get("issues", []) or []:
            n_findings += 1
            sev_fix[(it.get("severity"), it.get("fixability"))] += 1
            sv = it.get("self_verify")
            if sv:
                n_self_verified += 1
                sv_tally[sv.get("verdict", "?")] += 1

    print(f"=== BASE-RATE STATS — GLM-5.2, {n} papers ===")
    print(f"  findings: {n_findings} total ({n_findings / n:.1f}/paper)")
    print(f"  assessment (single pass)  : {dict(assess_before)}")
    print(f"  assessment (after self-verify): {dict(assess_after)}")
    print(f"  self-verify passes run: {n_self_verified} | verdicts: {dict(sv_tally)}")
    if n_self_verified:
        conf = sv_tally.get("confirmed_error", 0)
        diss = sv_tally.get("dissolved", 0)
        print(f"  --> of {n_self_verified} major flags re-checked: "
              f"{conf} confirmed, {diss} self-dissolved (apparent FP), "
              f"{n_self_verified - conf - diss} unsettled")
    print("  severity x fixability (single pass, all findings):")
    for (s, fx), c in sorted(sev_fix.items(), key=lambda kv: -(SEV_RANK.get(kv[0][0], 0) * 10 + FIX_RANK.get(kv[0][1], 0))):
        print(f"    {str(s):<9} {str(fx):<22} {c}")

    # ---- adjudication queue ----
    queue = defaultdict(list)
    for pid, rec in reports.items():
        for it in (rec.get("report", {}) or {}).get("issues", []) or []:
            b = bucket_of(it)
            sv = it.get("self_verify") or {}
            queue[b].append({
                "paper": pid, "id": it.get("id"),
                "severity": it.get("severity"), "fixability": it.get("fixability"),
                "self_verify": sv.get("verdict"), "sv_confidence": sv.get("confidence"),
                "location": it.get("location", "")[:90],
                "_rank": SEV_RANK.get(it.get("severity"), 0) * 10 + FIX_RANK.get(it.get("fixability"), 0),
            })

    print("\n=== ADJUDICATION QUEUE ===")
    for b in PRIORITY:
        items = sorted(queue[b], key=lambda e: -e["_rank"])
        print(f"\n-- {b} ({len(items)}) — {BLURB[b]} --")
        for e in items:
            sv = f" [self-verify:{e['self_verify']}/{e['sv_confidence']}]" if e["self_verify"] else ""
            print(f"  {e['paper']} {e['id']} {e['severity']}/{e['fixability']}{sv}: {e['location']}")

    if args.json:
        json.dump({"stats": {"n_papers": n, "n_findings": n_findings,
                             "assessment_before": dict(assess_before),
                             "assessment_after_self_verify": dict(assess_after),
                             "self_verify_verdicts": dict(sv_tally)},
                   "queue": {b: queue[b] for b in PRIORITY}},
                  open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
