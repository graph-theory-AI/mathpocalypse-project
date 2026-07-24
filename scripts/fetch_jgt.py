#!/usr/bin/env python3
"""Build the JGT leanification corpus: DBLP -> arXiv-matched -> .tex sources.

Phase 2 (leanification) targets *every* Journal of Graph Theory paper from the last few
years (any author) that has an arXiv version. Pipeline, all keyed on the DOI so matching
is exact (no fuzzy title matching -> no wrong-paper downloads):

  1. enumerate JGT via the DBLP stream API  (all authors, paginated)
  2. map each DOI -> arXiv id via the Semantic Scholar batch API
  3. write papers/jgt_registry.yaml          (same schema as registry.yaml)
  4. download .tex e-print sources into papers/src/<arxiv_base>/  (gitignored)

Steps are separable so the network-heavy download is opt-in and resumable:

    python scripts/fetch_jgt.py --since 2023 --metadata-only   # DBLP+SS -> registry, no downloads
    python scripts/fetch_jgt.py --since 2023                    # the above, then fetch sources
    python scripts/fetch_jgt.py --sources-only                 # (re)fetch sources from existing registry

The registry is the source of truth between steps: step 4 fills in each paper's detected
`main_tex` (and flags papers whose arXiv e-print has no LaTeX source) back into the YAML.
"""
from __future__ import annotations

import argparse
import io
import gzip
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_sources import UA, fetch_one  # noqa: E402  (reuse the proven e-print fetcher)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mathpoc.registry import REPO_ROOT  # noqa: E402

import yaml  # noqa: E402

JGT_REGISTRY = os.path.join(REPO_ROOT, "papers", "jgt_registry.yaml")
DBLP_STREAM = "https://dblp.org/search/publ/api"
SS_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
TODAY = "2026-06-26"


# ----------------------------------------------------------------------------- http helpers
def _get(url: str, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise
        except Exception:  # incl. ConnectionResetError when DBLP throttles bulk pages
            if i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def _post_json(url: str, payload: dict, tries: int = 5) -> list:
    body = json.dumps(payload).encode()
    headers = {**UA, "Content-Type": "application/json"}
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise
        except Exception:
            if i < tries - 1:
                time.sleep(3 * (i + 1))
                continue
            raise
    raise RuntimeError("unreachable")


# ----------------------------------------------------------------------------- step 1: DBLP
def dblp_jgt(since: int, page: int = 300) -> list[dict]:
    """Page the JGT stream (newest-first), return entries with year >= since.

    DBLP throttles big/rapid pages, so we use modest pages with backoff. The stream is
    returned newest-first, so we stop once an entire page falls below `since`.
    """
    out, f, total = [], 0, None
    while True:
        q = urllib.parse.urlencode(
            {"q": "stream:streams/journals/jgt:", "h": page, "f": f, "format": "json"}
        )
        res = json.loads(_get(f"{DBLP_STREAM}?{q}"))["result"]["hits"]
        if total is None:
            total = int(res["@total"])
            print(f"  DBLP: {total} JGT entries total")
        hits = res.get("hit", [])
        if not hits:
            break
        years_here = []
        for h in hits:
            info = h["info"]
            try:
                year = int(info.get("year", 0) or 0)
            except ValueError:
                year = 0
            years_here.append(year)
            if year < since:
                continue
            doi = info.get("doi")
            if not doi:
                continue
            out.append(
                {
                    "title": _clean(info.get("title", "")),
                    "authors": _authors(info.get("authors")),
                    "year": year,
                    "doi": doi,
                    "dblp_key": info.get("key", ""),
                }
            )
        f += len(hits)
        print(f"  DBLP: page to {f}/{total}, kept {len(out)} (page years "
              f"{min(years_here)}-{max(years_here)})")
        if f >= total or max(years_here) < since:
            break
        time.sleep(3)
    return out


def _authors(a) -> list[str]:
    if not a:
        return []
    au = a.get("author")
    if au is None:
        return []
    if isinstance(au, dict):
        au = [au]
    return [x.get("text", "") if isinstance(x, dict) else str(x) for x in au]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(".")


# ----------------------------------------------------------------- step 2: DOI -> arXiv (S2)
def map_arxiv(dois: list[str], chunk: int = 100) -> dict[str, str]:
    """DOI -> arXiv base id via Semantic Scholar batch. Missing -> absent from result."""
    found: dict[str, str] = {}
    for i in range(0, len(dois), chunk):
        part = dois[i : i + chunk]
        url = f"{SS_BATCH}?fields=externalIds,title"
        data = _post_json(url, {"ids": [f"DOI:{d}" for d in part]})
        for d, p in zip(part, data):
            ax = (p or {}).get("externalIds", {}).get("ArXiv") if p else None
            if ax:
                found[d] = ax.split("v")[0]  # strip any version suffix
        print(f"  S2: {i + len(part)}/{len(dois)} queried, {len(found)} with arXiv so far")
        time.sleep(1)
    return found


# ----------------------------------------------------------------------------- registry I/O
def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z]+", "", (s.split()[-1] if s.split() else "anon"))
    return s.lower() or "anon"


def to_entry(p: dict, arxiv: str) -> dict:
    last = slug(p["authors"][0]) if p["authors"] else "anon"
    return {
        "id": f"{last}-{p['year']}-{arxiv}",
        "title": p["title"],
        "authors": p["authors"],
        "venue": f"Journal of Graph Theory ({p['year']})",
        "doi": p["doi"],
        "year": p["year"],
        "arxiv": arxiv,
        "source": "tex",
        "src_dir": f"papers/src/{arxiv}",
        "main_tex": "",  # filled in by the sources step
        "dblp_key": p["dblp_key"],
        "added": TODAY,
        "status": "collected",
        "verdict": None,
        "report": None,
        "notes": "",
    }


HEADER = """\
# JGT leanification corpus -- auto-generated by scripts/fetch_jgt.py.
# Every Journal of Graph Theory paper (any author) since {since} that has an arXiv version,
# enumerated from DBLP and DOI-matched to arXiv via Semantic Scholar. This is the Phase-2
# (leanification) target set; the curated Phase-1 verification papers stay in registry.yaml.
# Sources live in papers/src/<arxiv>/ (gitignored); regenerate with scripts/fetch_jgt.py.
#
# status:   collected | source_missing | leanifying | ...
# main_tex: "" until the sources step downloads + detects it.
"""


def write_registry(entries: list[dict], since: int) -> None:
    with open(JGT_REGISTRY, "w", encoding="utf-8") as fh:
        fh.write(HEADER.format(since=since))
        yaml.safe_dump(
            {"papers": entries}, fh, allow_unicode=True, sort_keys=False, width=100
        )


def load_entries() -> list[dict]:
    if not os.path.isfile(JGT_REGISTRY):
        return []
    with open(JGT_REGISTRY, encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("papers", []) or []


# ----------------------------------------------------------------- step 4: sources + main_tex
def detect_main_tex(dest: str) -> str:
    """Pick the LaTeX entry point: a .tex with \\documentclass, preferring \\begin{document}."""
    cands = []
    for root, _, files in os.walk(dest):
        for fn in files:
            if not fn.endswith(".tex"):
                continue
            path = os.path.join(root, fn)
            try:
                txt = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            rel = os.path.relpath(path, dest)
            score = 0
            if "\\documentclass" in txt:
                score += 2
            if "\\begin{document}" in txt:
                score += 2
            if score:
                # tie-breakers: shallower path, then 'main.tex', then shorter name
                cands.append((score, -rel.count(os.sep), fn == "main.tex", -len(fn), rel))
    if not cands:
        return ""
    cands.sort(reverse=True)
    return cands[0][-1]


def fetch_sources(entries: list[dict], since: int, sleep: float = 3.0) -> None:
    n_ok = n_skip = n_fail = 0
    for p in entries:
        arxiv = p["arxiv"]
        dest = os.path.join(REPO_ROOT, p.get("src_dir") or f"papers/src/{arxiv}")
        # idempotent: already have a detected main_tex on disk -> skip
        if p.get("main_tex") and os.path.isfile(os.path.join(dest, p["main_tex"])):
            n_skip += 1
            continue
        try:
            fetch_one(arxiv, dest)
        except Exception as e:
            p["main_tex"], p["status"] = "", "source_missing"
            p["notes"] = f"e-print download failed: {e}"
            n_fail += 1
            print(f"  FAIL {p['id']}: {e}")
            write_registry(entries, since)  # persist progress each iteration
            time.sleep(sleep)
            continue
        main = detect_main_tex(dest)
        if main:
            p["main_tex"], p["status"] = main, "collected"
            n_ok += 1
            print(f"  ok   {p['id']}  main_tex={main}")
        else:
            p["main_tex"], p["status"] = "", "source_missing"
            p["notes"] = "no .tex in e-print (PDF-only submission?)"
            n_fail += 1
            print(f"  no-tex {p['id']} (PDF-only?)")
        write_registry(entries, since)  # checkpoint so the run is resumable
        time.sleep(sleep)
    print(f"\nsources: {n_ok} fetched, {n_skip} already present, {n_fail} without .tex")


# ----------------------------------------------------------------------------------- driver
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", type=int, default=2023, help="earliest publication year (default 2023)")
    ap.add_argument("--metadata-only", action="store_true", help="DBLP+S2 -> registry, no downloads")
    ap.add_argument("--sources-only", action="store_true", help="skip DBLP+S2, fetch from existing registry")
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between arXiv fetches (be polite)")
    args = ap.parse_args(argv)

    if args.sources_only:
        entries = load_entries()
        if not entries:
            print(f"no registry at {JGT_REGISTRY} -- run without --sources-only first")
            return 1
        # recover `since` from the smallest year present, for the header
        since = min((e.get("year", args.since) for e in entries), default=args.since)
        print(f"sources-only: {len(entries)} papers in registry")
        fetch_sources(entries, since, args.sleep)
        return 0

    print(f"[1/4] DBLP: enumerating JGT since {args.since} ...")
    papers = dblp_jgt(args.since)
    print(f"      {len(papers)} JGT papers with a DOI since {args.since}")

    print(f"[2/4] Semantic Scholar: mapping {len(papers)} DOIs -> arXiv ...")
    amap = map_arxiv([p["doi"] for p in papers])
    matched = [(p, amap[p["doi"]]) for p in papers if p["doi"] in amap]
    print(f"      {len(matched)}/{len(papers)} have an arXiv version "
          f"({100 * len(matched) // max(len(papers), 1)}%)")

    # de-dup on arXiv id (an arXiv paper occasionally maps from two DBLP records)
    seen, fresh = set(), []
    for p, ax in matched:
        if ax in seen:
            continue
        seen.add(ax)
        fresh.append(to_entry(p, ax))

    # merge into whatever's already on disk -- a re-run (e.g. widening --since) must NOT
    # clobber already-fetched main_tex/status/notes for papers we've already collected+verified.
    existing = {e["arxiv"]: e for e in load_entries()}
    n_new = 0
    for e in fresh:
        if e["arxiv"] in existing:
            continue
        existing[e["arxiv"]] = e
        n_new += 1
    entries = sorted(existing.values(), key=lambda e: (-e["year"], e["id"]))
    since = min(args.since, min((e.get("year", args.since) for e in entries), default=args.since))

    print(f"[3/4] merging: {n_new} new papers, {len(entries)} total (was {len(existing) - n_new})")
    write_registry(entries, since)

    if args.metadata_only:
        print("[4/4] --metadata-only: skipping source download.")
        print(f"      next: python scripts/fetch_jgt.py --sources-only")
        return 0

    print(f"[4/4] downloading {len(entries)} arXiv e-print sources ...")
    fetch_sources(entries, args.since, args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
