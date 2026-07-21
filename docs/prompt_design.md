# Prompt & verification-architecture design

Living design note. The live prompt is `prompts/verify_v0.md`; this explains the *why* and
the open architectural choices.

## What we are optimizing for

The valuable output is **major errors that are hard or impossible to fix** in important
French-authored graph-theory / combinatorics papers. Typos and one-line slips are near-
worthless. Two consequences shape everything:

1. The report separates **severity** (how much breaks) from **fixability** (how hard to
   repair), and ranks `no_known_fix` / high-severity issues first. See the schema in
   `verify_v0.md`.
2. **False positives are expensive** — we will eventually email authors, so a confident-but-
   wrong flag burns trust. The prompt forces each flag to be justified to the point of a
   counterexample or a precisely located broken step.

## The central question: one pass, or subagents per proof unit?

Whether to split verification across subagents, each focusing on a different theorem /
proposition / lemma. Trade-offs:

**For splitting (per-proof-unit subagents):**
- **Focus & depth.** Each subagent spends its full reasoning budget on one proof instead of
  amortizing attention across a 30-page paper. For deep errors, depth-per-unit is exactly
  what we want.
- **Parallelism.** The served model on the GPU cluster has real aggregate throughput at high
  concurrency (thousands of tok/s) versus a single stream; independent units map naturally
  onto that.
- **Cleaner reports.** One structured verdict per unit, easy to aggregate and triage.

**Against / risks:**
- **Cross-unit errors get missed.** Many real errors live *between* results — Lemma 3 is
  used in Theorem 5 under hypotheses it doesn't actually guarantee; a definition drifts
  between sections. A subagent boxed into one unit can't see this.
- **Dependency context.** A unit's proof leans on definitions + earlier lemmas. Each
  subagent must be *given* those statements, or it flags "depends on unverified result" for
  everything and verifies nothing.

**Architecture we built (Phase 1): an agent-based, triage-first, adversarial pipeline.**
It is *agent-driven, not parser-driven* — a deterministic LaTeX parser was prototyped and
rejected as too brittle for the variability of human-written papers (custom `\def` theorem
environments, definitions buried in prose, macro figures). The served open-weight model (the
"master") does the structural reading. And it is *triage-first, not exhaustive* — we only care
about papers where we can become **reasonably sure** there is a real mistake, so effort goes to
the fragile, load-bearing steps, not to re-checking routine ones.

Four stages, all through the same `Backend` as the single pass (`mathpoc/pipeline.py`):
1. **Survey (master)** — `prompts/survey_v0.md`. One global read: understand the proof
   architecture, rank steps by *risk* = load-bearing × likely-to-be-wrong, and select the few
   genuinely fragile **targets**. For each target the master **assembles the self-contained
   context** a focused checker needs — PART A definitions/notation in force, PART B statements
   of results it depends on, PART C the target's statement + proof *verbatim* (the argument
   under test is never paraphrased; supporting material may be condensed but stays faithful).
   This is also where a paper is judged too informal to referee — a first-class outcome.
2. **Targeted verify** — `prompts/verify_unit_v0.md`. One subagent per target, with its
   assembled context, sharing the `verify_v0.md` issue schema so reports stay comparable.
3. **Adversarial refute** — `prompts/refute_v0.md`. Fires only on *substantial* findings
   (severity ≥ major). It works both sides at once: **try to construct an explicit
   counterexample**, and independently **try to repair the step**. This is the "reasonably
   sure" gate — `confirmed_error` (a checked counterexample / proven impossibility),
   `dissolved` (a repair was found — drop the flag), or `inconclusive` (not yet reportable).
4. **Aggregate** — `prompts/aggregate_v0.md`. Checks the **seams** (lemmas applied outside
   their hypotheses, drifting definitions, inconsistent constants, the top-level composition)
   that single-target passes cannot see, then consolidates — only `confirmed` findings drive
   the verdict; `dissolved` ones are dropped.

Why this shape serves our goal: false positives are expensive, so the counterexample/repair
gate (stage 3) is the heart of the design — a flag that cannot be made concrete is not
reported with confidence. Putting triage first (stage 1) keeps cost on the suspects.

**Baseline to beat.** The dead-simple **whole-paper single pass** (`verify_v0.md` over the
full text, `python -m mathpoc verify`) is the baseline the multi-stage pipeline must justify
itself against. The head-to-head question, on any paper where the single pass flags a
load-bearing gap: does the pipeline — which makes one agent stare only at that step with its
full dependency context, then *try to build a counterexample* — **confirm** the gap (it
survives the refute stage / a counterexample appears) or **dissolve** it (the refute stage
reconstructs the missing argument, i.e. the single pass had hallucinated a gap in a terse
peer-reviewed step)? `scripts/azzurra/job_pipeline.sh` runs the pipeline in one served
session. Record each head-to-head in the lab notebook.

## Other open questions

- **Paper delivery.** Full LaTeX source (best — unambiguous math) vs. PDF text extraction
  (lossy on formulas) vs. PDF-as-images to a multimodal model. Prefer arXiv `.tex` source
  when available.
- **One-shot vs. iterative.** Let the model ask itself follow-up questions / attempt a
  counterexample search in a scratchpad before committing the JSON?

Validation is **human-in-the-loop**: once we believe we've found a real mistake, we take it
to the authors and record their confirmation or rebuttal. There is no pre-built calibration
set — author response is the ground truth.
