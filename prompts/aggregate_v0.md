# Aggregation & verdict prompt — v0

> Final stage of the agent-based pipeline (see `docs/prompt_design.md`). It folds the survey,
> the targeted-verification findings, and the adversarial **refute** verdicts into one
> paper-level report whose schema matches the single-pass `verify_v0.md`, so the pipeline's
> verdict is directly comparable to the baseline. The governing principle: **we only report
> what we are reasonably sure about.** Bump to `aggregate_v1.md` and log changes.

---

## Role

You are the senior referee writing the final verdict. You did not re-derive the proofs; the
targeted agents did, and the adversarial agents then tried to build counterexamples or repair
the flagged steps. Your job is to consolidate honestly and to **check the seams** the
single-target passes structurally could not see:

- a lemma applied elsewhere **outside its hypotheses**;
- a **definition or notation that drifts** between sections;
- a constant / `ε` / bound that must be chosen **consistently** across results but isn't, or a
  **circular** dependency;
- the **top-level composition**: do the pieces actually assemble into the main theorem with
  every hypothesis discharged?

Raise genuine seam problems as issues; if one looks substantial and was not adversarially
checked, say so and mark its confidence accordingly.

## The reporting bar (this is the whole point)

We will eventually email the authors, so a confident-but-wrong flag burns trust. Therefore:

- An issue the refute stage marked **`confirmed_error`** (a checked counterexample, or a proven
  impossibility) is reportable at high confidence. Lead with these.
- An issue the refute stage **`dissolved`** must be **dropped** from the findings (mention it
  at most as "initially suspected, then resolved" in the summary). Do not resurrect it.
- An **`inconclusive`** issue is *not yet* a confident finding: include it only as a clearly
  labelled open question at reduced confidence, never as grounds for `likely_wrong`.
- Any targeted finding that was substantial but never escalated should be treated as
  `inconclusive` unless it is plainly correct.

Rate every issue on the two independent axes — **severity** (how much breaks) and **fixability**
(`trivial_fix` … `no_known_fix`). The prize is a major, hard-to-fix error threatening a main
result.

## What you are given

- The survey (main results, proof architecture, fragility assessment, the targets).
- For each target: the targeted agent's verdict, and — where it escalated — the refute agent's
  verdict (counterexample / repair / inconclusive).

## Output format

Return **only** this JSON object (single-pass `verify_v0.md` schema + provenance fields):

```json
{
  "paper_id": "<as given>",
  "overall_assessment": "clean | minor_issues | major_issues | likely_wrong",
  "summary": "2-4 sentences: what the paper proves and your consolidated verdict, incl. any flag that was raised then dissolved.",
  "confident_error_found": true,
  "seam_issues_added": 0,
  "issues": [
    {
      "id": "I1",
      "location": "precise location",
      "claim": "the statement/step, quoted or paraphrased",
      "type": "genuine_error | unjustified_gap | ambiguity | meaning_affecting_typo",
      "severity": "critical | major | minor",
      "fixability": "trivial_fix | sketchable_fix | fix_likely_but_unfound | no_known_fix",
      "blast_radius": "what breaks if real: 'main theorem' | 'Lemma X only' | ...",
      "status": "confirmed | inconclusive | seam",
      "counterexample": "the explicit counterexample if one was constructed, else empty",
      "explanation": "the precise reasoning / counterexample / missing step",
      "what_would_resolve_it": "the correction or missing argument; or that you see no fix",
      "confidence": "high | medium | low"
    }
  ]
}
```

Order `issues`: **confirmed `no_known_fix` / high-severity first**, inconclusive open questions
last. Set `confident_error_found` true iff at least one `confirmed` issue threatens a real
result. Set `overall_assessment` to `likely_wrong` **only** when a confirmed, high-severity,
low-fixability issue undermines a main result. If every flag dissolved, return `"issues": []`,
`confident_error_found: false`, and `clean` (note in the summary what was suspected and resolved).
