#!/usr/bin/env python3
"""Download arXiv .tex sources for registry papers into papers/src/<base>/.

Sources are gitignored, so this repopulates them from papers/registry.yaml (e.g. after a
fresh clone on the cluster). Idempotent: skips papers whose main_tex already exists.

    python scripts/fetch_sources.py            # all registry papers
    python scripts/fetch_sources.py <id> ...   # only these registry ids
"""
from __future__ import annotations

import io
import gzip
import os
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mathpoc.registry import REPO_ROOT, load_registry  # noqa: E402

UA = {"User-Agent": "mathpocalypse/0.1 (research; emanatale@gmail.com)"}


def fetch_one(arxiv_id: str, dest: str) -> str:
    base = arxiv_id.split("v")[0]
    os.makedirs(dest, exist_ok=True)
    raw = urllib.request.urlopen(
        urllib.request.Request(f"https://arxiv.org/e-print/{base}", headers=UA), timeout=120
    ).read()
    with open(os.path.join(dest, "source.bin"), "wb") as f:
        f.write(raw)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            tf.extractall(dest)  # noqa: S202 (trusted arXiv source)
        return "tar"
    except tarfile.ReadError:
        with open(os.path.join(dest, "main.tex"), "wb") as f:
            f.write(gzip.decompress(raw))
        return "gz-single"


def main(argv: list[str]) -> int:
    want = set(argv)
    for p in load_registry():
        if want and p.id not in want:
            continue
        if not p.arxiv:
            print(f"skip {p.id}: no arxiv id")
            continue
        dest = os.path.join(REPO_ROOT, p.src_dir or f"papers/src/{p.arxiv.split('v')[0]}")
        if os.path.isfile(os.path.join(dest, p.main_tex)):
            print(f"have {p.id}  ({p.main_tex})")
            continue
        how = fetch_one(p.arxiv, dest)
        ok = os.path.isfile(os.path.join(dest, p.main_tex))
        print(f"fetched {p.id}  [{how}]  main_tex={p.main_tex} {'OK' if ok else 'MISSING — check main_tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
