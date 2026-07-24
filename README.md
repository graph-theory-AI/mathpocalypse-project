# Mathpocalypse

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21499917-1682d4)](https://doi.org/10.5281/zenodo.21499917)

**Can a frontier, fully open-source language model — run entirely on public research
computers — help find genuine mistakes in the published mathematics literature?**

Mathpocalypse is a pilot study that runs an open-weight LLM over published mathematics /
theoretical-computer-science papers and asks it to **carefully re-check the proofs and flag
anything that looks potentially wrong** — a missing case, an unjustified step, a broken lemma.

The long-term vision — the half-joking "math-pocalypse" — is models that can reliably
re-check almost any paper and verify its correctness with high confidence, surfacing errors in
the literature at scale.

## Contents

- [How it works](#how-it-works)
- [Scope](#scope)
- [Results so far](#results-so-far)
- [Reproduce it](#reproduce-it)
- [What is *not* in this repo](#what-is-intentionally-not-in-this-repo)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [People](#people)
- [Acknowledgements](#acknowledgements)

## How it works

The **detection pipeline is sovereign** by design: open-source models only, run exclusively
on university / national computing resources, with all author correspondence kept as private
personal communication — never sent to any commercial service.

- **Model.** `GLM-5.2` (open weights), run at **full reasoning power**, served on the
  **Jean Zay** (IDRIS/GENCI) and **Azzurra** (Université Côte d'Azur) GPU clusters. *(The
  pilot briefly used DeepSeek-V4-Flash; it was dropped on 2026-06-28 in favor of GLM-5.2.
  Each report names the model that produced it.)*
- **Input.** We prefer arXiv **`.tex` source** over PDF text extraction, so the mathematics
  survives unambiguously.
- **Prompt.** A versioned verification prompt (`prompts/`) asks the model to localize each
  concern to an exact step and rate it on two independent axes: **severity** (how much breaks
  if it's real) and **fixability** (`trivial_fix` … `no_known_fix`). The finding we care about
  is a **major error that is hard to fix** — a typo is near-worthless.
- **Self-verification.** Every major/critical flag is handed to a fresh adversarial subagent
  (`prompts/refute_v0.md`) that tries equally hard to *break* and to *repair* it. A flag the
  model dissolves on re-check is almost certainly a false positive.
- **Light auditing.** We are not experts in these areas, so our own check of a surviving flag
  is limited: we read it against the source and, for this last step only, may occasionally ask
  a proprietary model for a quick second opinion. This never touches detection and never shares
  author correspondence — it is just a sanity check before we decide whether a flag is worth an
  author's time.

See [`docs/prompt_design.md`](docs/prompt_design.md) for the architecture rationale,
including the map → verify-per-unit → aggregate pipeline idea.

> ### These are flags, not verdicts
>
> An AI reading a proof produces a **hypothesis that a step might be wrong** — nothing more.
> Many such flags are false alarms (the model misreads a figure, or a subtlety it missed
> resolves the concern). We treat every flag as *"the model thinks this might be a problem;
> it may or may not be."* We write to authors only after a flag survives the adversarial pass
> and our own (non-expert) check, we frame it as a question, and we defer entirely to the
> authors' judgment. Nothing in this repo should be read as a claim that any paper is wrong.

## Scope

French-authored papers in graph theory and combinatorics, with the pilot corpus drawn from
the **_Journal of Graph Theory_**. Working in a focused, well-understood slice keeps human
adjudication tractable while we get the verification prompt right.

## Results so far

Aggregate base-rate statistics over the pilot corpus (GLM-5.2, latest run per paper):

| Metric | Value |
|---|---|
| Papers scanned | **306** |
| Total flags raised | **886** (~2.9 per paper) |
| Whole-paper assessment (after self-verify) | 47 clean · 183 minor · **75 with a major flag** · 1 likely-wrong |
| Major/critical flags sent to adversarial self-verify | **204** |
| → confirmed on re-check | **119** |
| → self-dissolved (apparent false positive) | **78** |
| → unsettled | 7 |
| Author teams contacted (after light auditing) | **23** |
| Replies so far | **9 of 23** — 5 confirmed the flag · 2 refuted · 2 under review |

**No confirmed hard-to-fix theorem-breaker yet** — the real prize. Every real flag so far has
been a genuine, precisely-located gap that turned out to be fixable or that leaves an open
question, not a result that collapses. That is itself an informative finding about both the
model and the health of this slice of the literature.

We share the specifics of each flag privately with the paper's authors and keep the per-author
outreach record (who we contacted and how they responded) private; only the aggregate counts
above are published.

## Reproduce it

No install needed; run from the repo root.

```bash
# 1. See the tracked papers and whether their sources are present
python -m mathpoc list

# 2. Fetch arXiv .tex sources for the registry (sources are gitignored: size + copyright)
python scripts/fetch_sources.py

# 3. Inspect exactly what the model receives — no backend needed
python -m mathpoc verify --dry-run <paper-id>

# 4. Run verification against a served model (see scripts/ for the cluster jobs)
MATHPOC_BASE_URL=... MATHPOC_MODEL=glm-5.2 python -m mathpoc verify <paper-id>

# 5. Triage a batch of reports into a base-rate summary
python scripts/triage.py --all
```

The cluster serving recipe (vLLM, tensor-parallel over 4× H100) lives in
`scripts/azzurra/` and `scripts/jeanzay/`.

## What is intentionally *not* in this repo

To publish responsibly, this public mirror **excludes**:

- **All author correspondence** — emails, addresses, and reply threads are private personal
  communication and are never published.
- **The per-paper AI reports** — the raw model output flags *potential* issues in named
  papers, including known false positives. Publishing 300+ such per-paper claims would
  misrepresent hypotheses as verdicts, so we publish **aggregate statistics only** and share
  specific findings privately with the authors.

The harness, prompts, registry, and method are fully public so the work can be reproduced.

## Repository layout

| Path | What |
|---|---|
| `mathpoc/` | The Python harness (registry → source assembly → prompt → backend → report). |
| `prompts/` | Versioned verification / refutation / aggregation prompts. |
| `scripts/` | Source fetching, report triage, and cluster (SLURM) serving jobs. |
| `papers/*.yaml` | Tracked paper metadata registries (sources themselves are gitignored). |
| `docs/prompt_design.md` | Verification-architecture rationale. |

## Citation

If Mathpocalypse is relevant to your work, please cite it. GitHub's **"Cite this repository"**
button (below the About section, top right) exports ready-made BibTeX and APA.

```bibtex
@software{mathpocalypse,
  author  = {Natale, Emanuele and Oyallon, Édouard},
  title   = {{Mathpocalypse}: detecting errors in published mathematics with open-source LLMs},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.21499917},
  url     = {https://github.com/natema/mathpocalypse-project}
}
```

## People

[Emanuele Natale](https://natema.github.io/) (CNRS, Université Côte d'Azur) and
[Édouard Oyallon](https://edouardoyallon.github.io/) (CNRS, Sorbonne Université).

## Acknowledgements

This work was granted access to the HPC resources of IDRIS under the allocation
2026-AD011018098 made by GENCI.

Experiments presented in this work were carried out using the Grid'5000 testbed, supported by a
scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as
well as other organizations (see https://www.grid5000.fr).

The authors are grateful to the OPAL infrastructure from Université Côte d'Azur for providing
resources and support (the Azzurra GPU cluster is operated within OPAL).
