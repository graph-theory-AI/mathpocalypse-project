"""Agent-based verification pipeline — the triage-first, adversarial alternative to a single
whole-paper pass (see docs/prompt_design.md).

    survey (master: global look + fragility triage + assemble per-target context)
        -> verify each fragile target (focused subagent, with its assembled context)
            -> refute each *substantial* finding (adversarial: try to build a counterexample
               OR repair the step)  -- the "are we reasonably sure?" gate
                -> aggregate (seams + consolidate; only confident findings drive the verdict)

There is no deterministic LaTeX parsing: a served model (the "master", DeepSeek-V4-Flash on the
cluster) reads the paper, decides where it is most likely to break, and assembles the
self-contained context each checker needs. Every stage goes through `Backend.complete`, exactly
like the single-pass path, so the same served model / transport is reused and the final report
shares the single-pass schema (the head-to-head is apples-to-apples).

`--dry-run` shows the master prompt without contacting a backend, for an offline preflight.
"""
from __future__ import annotations

import datetime
import json
import os

from .backend import Backend, GenConfig
from .prompt import load_prompt
from .registry import REPO_ROOT, Paper, get_paper
from .sources import paper_source_text
from .verify import REPORTS_DIR, RAW_DIR, extract_json

PROMPTS_DIR = os.path.join(REPO_ROOT, "prompts")
SURVEY_PROMPT = os.path.join(PROMPTS_DIR, "survey_v0.md")
UNIT_PROMPT = os.path.join(PROMPTS_DIR, "verify_unit_v0.md")
REFUTE_PROMPT = os.path.join(PROMPTS_DIR, "refute_v0.md")
AGGREGATE_PROMPT = os.path.join(PROMPTS_DIR, "aggregate_v0.md")

# A targeted finding is escalated to the adversarial refute stage iff it is at least this
# severe — we only spend a full counterexample hunt on errors that would actually matter.
_ESCALATE_SEVERITIES = {"critical", "major"}
_MAX_REFUTATIONS = 5  # safety cap on escalations per paper


def _paper_header(paper: Paper) -> str:
    return (
        "PAPER\n"
        f"paper_id: {paper.id}\ntitle: {paper.title}\n"
        f"authors: {', '.join(paper.authors)}\nvenue: {paper.venue}\narxiv: {paper.arxiv}\n"
        f"Use exactly paper_id={paper.id!r} in your JSON output.\n"
    )


def survey_messages(paper: Paper, source: str) -> list[dict]:
    user = (
        _paper_header(paper) +
        "\nRead the whole paper below, find where it is most likely to break, and assemble the "
        "context for each fragile target.\n\n=== BEGIN LATEX SOURCE ===\n" + source +
        "\n=== END LATEX SOURCE ==="
    )
    return [{"role": "system", "content": load_prompt(SURVEY_PROMPT)},
            {"role": "user", "content": user}]


def _call(backend: Backend, messages: list[dict], gen: GenConfig):
    """Run one stage; return (parsed_json_or_error_dict, completion, parse_ok)."""
    comp = backend.complete(messages, gen)
    try:
        return extract_json(comp.text), comp, True
    except Exception as e:
        return {"parse_error": str(e)}, comp, False


def _target_verify_messages(paper: Paper, target: dict) -> list[dict]:
    user = (
        _paper_header(paper) +
        f"\nYou are checking target '{target.get('id')}' — {target.get('location','')}.\n"
        f"Why it was flagged fragile: {target.get('why_fragile','')}\n"
        f"What depends on it: {target.get('load_bearing','')}\n\n"
        "Its self-contained context (assembled by the survey master) follows.\n\n"
        + (target.get("context") or "(no context provided)")
    )
    return [{"role": "system", "content": load_prompt(UNIT_PROMPT)},
            {"role": "user", "content": user}]


def _refute_messages(paper: Paper, target: dict, issue: dict) -> list[dict]:
    user = (
        _paper_header(paper) +
        f"\nSettle this one doubt about target '{target.get('id')}'.\n\n"
        "THE CHALLENGED STEP (from the referee):\n"
        f"  location: {issue.get('location','')}\n"
        f"  claim: {issue.get('claim','')}\n"
        f"  referee's reasoning: {issue.get('explanation','')}\n"
        f"  a counterexample idea, if the referee had one: {issue.get('counterexample_idea','')}\n\n"
        "THE CONTEXT (same as the referee had):\n\n"
        + (target.get("context") or "(no context provided)")
    )
    return [{"role": "system", "content": load_prompt(REFUTE_PROMPT)},
            {"role": "user", "content": user}]


def _aggregate_messages(paper: Paper, survey: dict, target_results: list[dict]) -> list[dict]:
    chunks = [_paper_header(paper)]
    chunks.append("\nSURVEY:\n" + json.dumps(
        {k: survey.get(k) for k in ("main_results", "proof_architecture",
                                    "fragility_assessment", "verifiable", "cleared")},
        ensure_ascii=False, indent=2))
    chunks.append("\nTARGET FINDINGS (and adversarial refute verdicts where escalated):")
    for tr in target_results:
        t = tr["target"]
        chunks.append(f"\n----- target {t.get('id')} : {t.get('location','')} -----")
        chunks.append("FRAGILITY: " + (t.get("why_fragile", "") or ""))
        chunks.append("TARGETED VERDICT:\n" + json.dumps(tr["verify"], ensure_ascii=False))
        if tr["refutations"]:
            chunks.append("REFUTE VERDICTS:\n" + json.dumps(
                [r["verdict"] for r in tr["refutations"]], ensure_ascii=False))
        else:
            chunks.append("REFUTE VERDICTS: (none escalated)")
    return [{"role": "system", "content": load_prompt(AGGREGATE_PROMPT)},
            {"role": "user", "content": "\n".join(chunks)}]


def _merge_usage(*usages: dict) -> dict:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    out = {k: 0 for k in keys}
    for u in usages:
        for k in keys:
            out[k] += (u or {}).get(k, 0) or 0
    return out


def verify_paper_pipeline(
    paper_id: str,
    backend: Backend,
    gen: GenConfig | None = None,
    strip_figures: bool = False,
    save: bool = True,
    out_dir: str = REPORTS_DIR,
    progress=print,
) -> dict:
    """Run survey -> targeted verify -> adversarial refute -> aggregate against a served model."""
    gen = gen or GenConfig()
    paper = get_paper(paper_id)
    source, note = paper_source_text(paper, strip_figures=strip_figures)
    raw: list[str] = []
    all_usage: list[dict] = []

    # Stage 1 — survey / fragility triage / context assembly (the master)
    progress("  [survey] global read -> fragile targets + assembled context")
    survey, scomp, sok = _call(backend, survey_messages(paper, source), gen)
    raw.append(f"=== SURVEY REASONING ===\n{scomp.reasoning}\n\n=== SURVEY ANSWER ===\n{scomp.text}")
    all_usage.append(scomp.usage)
    targets = survey.get("targets", []) if sok else []
    if not sok:
        progress(f"    !! survey JSON parse failed ({survey.get('parse_error')}); raw saved")
    progress(f"    survey: {len(targets)} fragile target(s); verifiable={survey.get('verifiable','?')}")

    # Stage 2 + 3 — verify each target, then adversarially refute substantial findings
    target_results: list[dict] = []
    n_refuted = 0
    for t in targets:
        tid = t.get("id", "?")
        progress(f"  [verify] target {tid}: {t.get('location','')[:70]}")
        verdict, vcomp, vok = _call(backend, _target_verify_messages(paper, t), gen)
        raw.append(f"=== VERIFY {tid} REASONING ===\n{vcomp.reasoning}\n\n=== VERIFY {tid} ANSWER ===\n{vcomp.text}")
        all_usage.append(vcomp.usage)
        issues = verdict.get("issues", []) if vok else []
        progress(f"           -> {verdict.get('unit_assessment','parse-fail')} ({len(issues)} issue(s))")

        refutations = []
        for iss in issues:
            if iss.get("severity") not in _ESCALATE_SEVERITIES or n_refuted >= _MAX_REFUTATIONS:
                continue
            n_refuted += 1
            progress(f"  [refute] {tid}/{iss.get('id','?')} [{iss.get('severity')}] "
                     "-> hunt counterexample / attempt repair")
            rv, rcomp, rok = _call(backend, _refute_messages(paper, t, iss), gen)
            raw.append(f"=== REFUTE {tid}/{iss.get('id')} REASONING ===\n{rcomp.reasoning}\n\n"
                       f"=== REFUTE {tid}/{iss.get('id')} ANSWER ===\n{rcomp.text}")
            all_usage.append(rcomp.usage)
            if rok:
                progress(f"           -> verdict: {rv.get('verdict','?')} (confidence {rv.get('confidence','?')})")
            refutations.append({"issue_id": iss.get("id"), "parse_ok": rok,
                                "usage": rcomp.usage, "verdict": rv})

        target_results.append({"target": t, "parse_ok": vok, "usage": vcomp.usage,
                               "verify": verdict, "refutations": refutations})

    # Stage 4 — aggregate / seam-check / final verdict
    progress("  [aggregate] seams + consolidate (reasonably-sure bar)")
    report, acomp, aok = _call(backend, _aggregate_messages(paper, survey, target_results), gen)
    raw.append(f"=== AGGREGATE REASONING ===\n{acomp.reasoning}\n\n=== AGGREGATE ANSWER ===\n{acomp.text}")
    all_usage.append(acomp.usage)

    record = {
        "paper_id": paper_id,
        "mode": "pipeline",
        "prompts": {"survey": "prompts/survey_v0.md", "verify_unit": "prompts/verify_unit_v0.md",
                    "refute": "prompts/refute_v0.md", "aggregate": "prompts/aggregate_v0.md"},
        "model": acomp.model,
        "source_note": note,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "parse_ok": aok,
        "n_targets": len(targets),
        "n_refutations": n_refuted,
        "survey": survey,
        "targets": target_results,
        "usage_total": _merge_usage(*all_usage),
        "report": report,
    }

    if save:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(RAW_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{paper_id}.pipeline_v0.{stamp}"
        with open(os.path.join(out_dir, base + ".json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        with open(os.path.join(RAW_DIR, base + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(raw))
        record["_saved_as"] = base + ".json"
    return record


def dry_run_survey(paper_id: str, strip_figures: bool = False) -> dict:
    """Assemble the survey (master) prompt without contacting any backend, for preflight."""
    paper = get_paper(paper_id)
    source, note = paper_source_text(paper, strip_figures=strip_figures)
    return {"paper": paper, "note": note, "messages": survey_messages(paper, source)}
