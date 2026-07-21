"""Orchestrate one verification: paper -> messages -> backend -> parsed JSON report."""
from __future__ import annotations

import datetime
import json
import os
import re

from .backend import Backend, GenConfig
from .prompt import DEFAULT_PROMPT, build_messages, load_prompt
from .registry import REPO_ROOT, get_paper
from .sources import paper_source_text

REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
RAW_DIR = os.path.join(REPORTS_DIR, "raw")  # gitignored: full model dumps

# Self-verification: after the single whole-paper pass, re-examine each finding this severe
# with a fresh, adversarial subagent (reusing the refute prompt). With only one model in play
# (GLM-5.2) this is our false-positive guard — it replaces the cross-model agreement signal.
SELF_VERIFY_PROMPT = os.path.join(REPO_ROOT, "prompts", "refute_v0.md")
_SELF_VERIFY_SEVERITIES = frozenset({"critical", "major"})
_SEV_RANK = {"critical": 3, "major": 2, "minor": 1}
_HARD_FIX = {"fix_likely_but_unfound", "no_known_fix"}


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a model reply (handles ```json fences and stray prose)."""
    t = _strip_think(text)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL)
    cand = m.group(1) if m else None
    if cand is None:  # fall back to the first balanced {...}
        start = t.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(t)):
                if t[i] == "{":
                    depth += 1
                elif t[i] == "}":
                    depth -= 1
                    if depth == 0:
                        cand = t[start : i + 1]
                        break
    if cand is None:
        raise ValueError("no JSON object found in model output")
    return json.loads(cand)


def build_prompt_only(paper_id: str, prompt_path: str = DEFAULT_PROMPT, strip_figures: bool = False):
    """Assemble messages without contacting any backend (dry run / inspection)."""
    paper = get_paper(paper_id)
    source, note = paper_source_text(paper, strip_figures=strip_figures)
    messages = build_messages(paper, source, load_prompt(prompt_path))
    return paper, messages, note


def _prompt_tag(prompt_path: str) -> str:
    return os.path.splitext(os.path.basename(prompt_path))[0]  # verify_v0.md -> verify_v0


def _self_verify_messages(paper, source: str, issue: dict) -> list[dict]:
    """Adversarial re-check of ONE flagged finding, with the full paper as context."""
    user = (
        "PAPER\n"
        f"paper_id: {paper.id}\ntitle: {paper.title}\n"
        f"authors: {', '.join(paper.authors)}\narxiv: {paper.arxiv}\n"
        f"Use exactly target={issue.get('id', '?')!r} in your JSON output.\n\n"
        "A referee flagged the single step below as a possible error. Settle it on its own "
        "merits — try equally hard to break it and to repair it; let the evidence decide.\n\n"
        "THE CHALLENGED STEP (from the referee):\n"
        f"  location: {issue.get('location', '')}\n"
        f"  claim: {issue.get('claim', '')}\n"
        f"  type: {issue.get('type', '')}\n"
        f"  referee's reasoning: {issue.get('explanation', '')}\n"
        f"  referee's proposed resolution: {issue.get('what_would_resolve_it', '')}\n\n"
        "THE FULL PAPER (every definition and prior result it relies on is in here):\n\n"
        "=== BEGIN LATEX SOURCE ===\n" + source + "\n=== END LATEX SOURCE ==="
    )
    return [{"role": "system", "content": load_prompt(SELF_VERIFY_PROMPT)},
            {"role": "user", "content": user}]


def _should_self_verify(issue: dict, severities: frozenset) -> bool:
    return (issue.get("severity") in severities) or (issue.get("fixability") in _HARD_FIX)


def _assessment_after_self_verify(issues: list[dict]) -> str:
    """Recompute the verdict, dropping findings the adversarial pass dissolved."""
    present = set()
    for it in issues:
        sv = it.get("self_verify") or {}
        if sv.get("verdict") == "dissolved":
            continue  # retracted on re-check — does not drive the verdict
        sev = sv.get("severity") or it.get("severity")
        if sev in _SEV_RANK:
            present.add(sev)
    if "critical" in present:
        for it in issues:
            sv = it.get("self_verify") or {}
            if (sv.get("verdict") == "confirmed_error"
                    and (sv.get("severity") or it.get("severity")) == "critical"
                    and (sv.get("fixability") or it.get("fixability")) in _HARD_FIX):
                return "likely_wrong"
        return "major_issues"
    if "major" in present:
        return "major_issues"
    if "minor" in present:
        return "minor_issues"
    return "clean"


def verify_paper(
    paper_id: str,
    backend: Backend,
    prompt_path: str = DEFAULT_PROMPT,
    gen: GenConfig | None = None,
    out_dir: str = REPORTS_DIR,
    save: bool = True,
    strip_figures: bool = False,
    self_verify: bool = False,
    self_verify_severities: frozenset = _SELF_VERIFY_SEVERITIES,
    progress=lambda *_: None,
) -> dict:
    gen = gen or GenConfig()
    paper, messages, note = build_prompt_only(paper_id, prompt_path, strip_figures=strip_figures)
    comp = backend.complete(messages, gen)
    raw_dumps = ["=== REASONING ===\n" + comp.reasoning + "\n\n=== ANSWER ===\n" + comp.text]

    try:
        report = extract_json(comp.text)
        parse_ok = True
    except Exception as e:  # keep the raw text so a parse failure is recoverable
        report = {"paper_id": paper_id, "parse_error": str(e)}
        parse_ok = False

    # ---- self-verification: adversarial re-check of each major finding (one subagent each) ----
    sv_summary = None
    if self_verify and parse_ok:
        source, _ = paper_source_text(paper, strip_figures=strip_figures)
        issues = report.get("issues", []) or []
        targets = [it for it in issues if _should_self_verify(it, self_verify_severities)]
        sv_summary = {"checked": 0, "confirmed_error": 0, "dissolved": 0,
                      "inconclusive": 0, "parse_fail": 0}
        progress(f"  [self-verify] {len(targets)} major/critical finding(s) to re-check")
        for it in targets:
            scomp = backend.complete(_self_verify_messages(paper, source, it), gen)
            raw_dumps.append(
                f"=== SELF-VERIFY {it.get('id')} REASONING ===\n{scomp.reasoning}\n\n"
                f"=== SELF-VERIFY {it.get('id')} ANSWER ===\n{scomp.text}")
            try:
                verdict = extract_json(scomp.text)
            except Exception as e:
                verdict = {"verdict": "parse_error", "parse_error": str(e)}
                sv_summary["parse_fail"] += 1
            verdict["_usage"] = scomp.usage
            it["self_verify"] = verdict
            sv_summary["checked"] += 1
            v = verdict.get("verdict")
            if v in sv_summary:
                sv_summary[v] += 1
            progress(f"     {it.get('id')} [{it.get('severity')}] -> {v} "
                     f"(conf {verdict.get('confidence', '?')})")
        report["assessment_after_self_verify"] = _assessment_after_self_verify(issues)
        progress(f"  [self-verify] {sv_summary} | assessment "
                 f"{report.get('overall_assessment')} -> {report['assessment_after_self_verify']}")

    record = {
        "paper_id": paper_id,
        "prompt": os.path.relpath(prompt_path, REPO_ROOT),
        "model": comp.model,
        "source_note": note,
        "usage": comp.usage,
        "self_verify": sv_summary,
        "parse_ok": parse_ok,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "report": report,
    }

    if save:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(RAW_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{paper_id}.{_prompt_tag(prompt_path)}.{stamp}"
        with open(os.path.join(out_dir, base + ".json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        # full reasoning + answer (+ any self-verify passes) to gitignored raw/ for debugging
        with open(os.path.join(RAW_DIR, base + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(raw_dumps))
        record["_saved_as"] = base + ".json"

    return record
