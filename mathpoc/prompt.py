"""Build the chat messages sent to the model: prompt template + paper source."""
from __future__ import annotations

import os

from .registry import REPO_ROOT, Paper

DEFAULT_PROMPT = os.path.join(REPO_ROOT, "prompts", "verify_v0.md")


def load_prompt(path: str = DEFAULT_PROMPT) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_messages(paper: Paper, source_text: str, prompt_text: str) -> list[dict]:
    """System = the verification instructions; user = paper metadata + LaTeX source."""
    header = (
        "PAPER TO VERIFY\n"
        f"paper_id: {paper.id}\n"
        f"title: {paper.title}\n"
        f"authors: {', '.join(paper.authors)}\n"
        f"venue: {paper.venue}\n"
        f"arxiv: {paper.arxiv}\n\n"
        f"Use exactly paper_id={paper.id!r} in your JSON output.\n"
        "The full LaTeX source follows between the markers.\n"
        "=== BEGIN LATEX SOURCE ===\n"
    )
    user = header + source_text + "\n=== END LATEX SOURCE ==="
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user},
    ]
