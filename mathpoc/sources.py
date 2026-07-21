"""Assemble a paper's LaTeX source into a single string.

We feed the model raw LaTeX (math survives unambiguously, unlike PDF text extraction).
`\\input`/`\\include` are inlined so multi-file papers arrive whole.
"""
from __future__ import annotations

import os
import re

from .registry import Paper

# \input{foo} / \include{foo}  (brace form — by far the most common on arXiv)
_INCLUDE_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")

# Optional, opt-in only: TikZ/pgf picture bodies. OFF by default — proofs often refer to
# their figures ("the construction in Fig. 3"), so dropping them can lose verifiable content.
# Reach for this only when a specific paper's source is too large to fit the context window.
_TIKZ_RE = re.compile(r"\\begin\{(tikzpicture|pgfpicture)\}.*?\\end\{\1\}", re.DOTALL)


def strip_tikz(text: str) -> tuple[str, int]:
    """Replace tikz/pgf picture bodies with a placeholder. Returns (text, n_stripped)."""
    n = 0

    def repl(_m: re.Match) -> str:
        nonlocal n
        n += 1
        return "% [tikz/pgf picture omitted by harness]"

    return _TIKZ_RE.sub(repl, text), n


def _resolve(base_dir: str, name: str) -> str | None:
    name = name.strip()
    for cand in (name, name + ".tex"):
        p = os.path.join(base_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def flatten_tex(src_dir: str, main_tex: str, _depth: int = 0, _seen: set | None = None) -> str:
    if _seen is None:
        _seen = set()
    path = os.path.join(src_dir, main_tex)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if _depth > 8:  # guard against include cycles
        return text

    def repl(m: re.Match) -> str:
        inc = _resolve(src_dir, m.group(1))
        if not inc or inc in _seen:
            return m.group(0)  # leave the command verbatim if unresolved
        _seen.add(inc)
        return flatten_tex(src_dir, os.path.relpath(inc, src_dir), _depth + 1, _seen)

    return _INCLUDE_RE.sub(repl, text)


def paper_source_text(paper: Paper, strip_figures: bool = False) -> tuple[str, str]:
    """Return (latex_source, assembly_note). The note flags suspect assembly.

    strip_figures: opt-in; removes tikz/pgf picture bodies (default keeps them, since
    proofs may reference their figures). Use only to make an oversized paper fit.
    """
    text = flatten_tex(paper.abs_src_dir, paper.main_tex)
    note = f"flattened from {paper.main_tex}"
    if strip_figures:
        text, n = strip_tikz(text)
        note += f"; stripped {n} tikz/pgf picture(s)"
    tex_files = [f for f in os.listdir(paper.abs_src_dir) if f.endswith(".tex")]
    if len(tex_files) > 1:
        total = sum(os.path.getsize(os.path.join(paper.abs_src_dir, f)) for f in tex_files)
        if len(text.encode("utf-8")) < 0.5 * total:
            note += (
                f"; WARNING: assembled {len(text)} chars but {len(tex_files)} .tex files "
                f"total {total} bytes — the main file may use a non-brace \\input style; "
                f"check that all sections were inlined"
            )
    return text, note
