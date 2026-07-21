"""Load papers/registry.yaml into Paper objects."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default to the curated Phase-1 registry; override with MATHPOC_REGISTRY to point the whole
# harness at another index (e.g. papers/jgt_registry.yaml for the Phase-2 JGT corpus) without
# touching any call site. A relative value is resolved against the repo root.
def _default_registry_path() -> str:
    env = os.environ.get("MATHPOC_REGISTRY")
    if env:
        return env if os.path.isabs(env) else os.path.join(REPO_ROOT, env)
    return os.path.join(REPO_ROOT, "papers", "registry.yaml")


REGISTRY_PATH = _default_registry_path()


@dataclass
class Paper:
    id: str
    title: str = ""
    authors: list = field(default_factory=list)
    arxiv: str = ""
    venue: str = ""
    doi: str | None = None
    year: int | None = None
    src_dir: str = ""
    main_tex: str = "main.tex"
    status: str = "collected"
    raw: dict = field(default_factory=dict)

    @property
    def abs_src_dir(self) -> str:
        return os.path.join(REPO_ROOT, self.src_dir)

    @property
    def has_source(self) -> bool:
        return bool(self.src_dir) and os.path.isfile(
            os.path.join(self.abs_src_dir, self.main_tex)
        )


def load_registry(path: str = REGISTRY_PATH) -> list[Paper]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out = []
    for p in data.get("papers", []) or []:
        out.append(
            Paper(
                id=p["id"],
                title=p.get("title", ""),
                authors=p.get("authors", []) or [],
                arxiv=p.get("arxiv", ""),
                venue=p.get("venue", ""),
                doi=p.get("doi"),
                year=p.get("year"),
                src_dir=p.get("src_dir", ""),
                main_tex=p.get("main_tex", "main.tex"),
                status=p.get("status", "collected"),
                raw=p,
            )
        )
    return out


def get_paper(paper_id: str, path: str = REGISTRY_PATH) -> Paper:
    for p in load_registry(path):
        if p.id == paper_id:
            return p
    raise KeyError(f"paper id not in registry: {paper_id!r}")
