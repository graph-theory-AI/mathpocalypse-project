"""CLI: python -m mathpoc <list|build-prompt|verify|verify-pipeline> ...

Examples:
  python -m mathpoc list
  python -m mathpoc build-prompt bonamy-2013-1303.4025        # inspect, no GPU
  python -m mathpoc verify bonamy-2013-1303.4025              # single-pass; needs a served model
  python -m mathpoc verify --all
  python -m mathpoc verify-pipeline --dry-run 1408.1964       # show each subagent's bundle, no GPU
  python -m mathpoc verify-pipeline bonamy-2014-1408.1964     # map->per-unit->aggregate run
"""
from __future__ import annotations

import argparse
import sys

from .backend import GenConfig
from .prompt import DEFAULT_PROMPT
from .registry import load_registry
from .verify import build_prompt_only, verify_paper


def _est_tokens(messages) -> int:
    chars = sum(len(m["content"]) for m in messages)
    return chars // 4  # rough; just to gauge context budget


def cmd_list(_args) -> int:
    for p in load_registry():
        src = "src✓" if p.has_source else "src✗"
        print(f"{p.id:<28} {src}  [{p.status}]  {p.year}  {p.venue}")
        print(f"    {p.title}")
    return 0


def cmd_build_prompt(args) -> int:
    paper, messages, note = build_prompt_only(args.paper, args.prompt, strip_figures=args.strip_figures)
    print(f"# paper: {paper.id} — {paper.title}")
    print(f"# source: {note}")
    print(f"# est. input tokens: ~{_est_tokens(messages):,}\n")
    for m in messages:
        print(f"----- {m['role'].upper()} ({len(m['content'])} chars) -----")
        if args.full:
            print(m["content"])
        else:
            body = m["content"]
            print(body if len(body) < 2000 else body[:2000] + f"\n... [+{len(body)-2000} chars; use --full]")
        print()
    return 0


def _thinking_kwargs(style: str, effort: str) -> dict | None:
    """Map (model family, effort) -> the `chat_template_kwargs` the server's chat template wants.
    Returns None to send no thinking block at all.

    Both DeepSeek-V4 and GLM-5.2 read a `reasoning_effort` field; they differ only in how
    thinking is toggled:
      - deepseek: thinking is opt-IN  -> {"thinking": true, "reasoning_effort": ...}
      - glm     : thinking is ON by default -> {"reasoning_effort": ...}; disable via
                  {"enable_thinking": false}
    """
    if style == "deepseek":
        return None if effort == "off" else {"thinking": True, "reasoning_effort": effort}
    if style == "glm":
        return {"enable_thinking": False} if effort == "off" else {"reasoning_effort": effort}
    raise ValueError(f"unknown thinking style: {style!r}")


def _gen_from_args(args) -> GenConfig:
    extra_body = {}
    kwargs = _thinking_kwargs(args.thinking_style, args.reasoning_effort)
    if kwargs is not None:
        extra_body["chat_template_kwargs"] = kwargs
    return GenConfig(temperature=args.temperature, top_p=args.top_p,
                     max_tokens=args.max_tokens, extra_body=extra_body)


def _make_backend(args):
    """Build the transport. `http` = OpenAI-compatible endpoint (server, e.g. Azzurra single
    node); `vllm` = in-process offline batch (no server — the model is loaded in this process,
    e.g. the Jean Zay multi-node GLM-5.2 run). Imported lazily so dry-run needs neither installed."""
    if getattr(args, "backend", "http") == "vllm":
        from .backend import VLLMOfflineBackend
        return VLLMOfflineBackend(model=args.model)
    from .backend import OpenAICompatBackend
    return OpenAICompatBackend(base_url=args.base_url, model=args.model)


def cmd_verify(args) -> int:
    papers = [p.id for p in load_registry()] if args.all else args.paper
    if not papers:
        print("error: give one or more paper ids, or --all", file=sys.stderr)
        return 2

    gen = _gen_from_args(args)

    if args.dry_run:
        for pid in papers:
            paper, messages, note = build_prompt_only(pid, args.prompt, strip_figures=args.strip_figures)
            print(f"{pid}: {note}; ~{_est_tokens(messages):,} input tokens "
                  f"(max_tokens={gen.max_tokens}, temp={gen.temperature}) [DRY RUN, not sent]")
        return 0

    backend = _make_backend(args)
    base = getattr(backend, "base_url", "in-process (vllm offline)")
    print(f"# backend: {base}  model={backend.model}")
    rc = 0
    sev = frozenset(s.strip() for s in args.self_verify_severities.split(",") if s.strip())
    for pid in papers:
        print(f"\n=== verifying {pid} ===")
        try:
            rec = verify_paper(pid, backend, prompt_path=args.prompt, gen=gen,
                               strip_figures=args.strip_figures, self_verify=args.self_verify,
                               self_verify_severities=sev, progress=print)
        except Exception as e:
            # Isolate per-paper failures: a single bad paper (e.g. a 400 context-overflow, a
            # transient server error) must NOT abort a whole batch and lose the remaining papers.
            print(f"  !! FAILED {pid}: {type(e).__name__}: {e}")
            rc = 1
            continue
        rep = rec["report"]
        if not rec["parse_ok"]:
            print(f"  !! JSON parse failed: {rep.get('parse_error')} (raw saved)")
            rc = 1
            continue
        issues = rep.get("issues", [])
        print(f"  assessment: {rep.get('overall_assessment','?')}  | {len(issues)} issue(s)")
        for it in issues:
            sv = it.get("self_verify")
            tag = f"  =>self-verify: {sv.get('verdict')}" if sv else ""
            print(f"   - [{it.get('severity','?')}/{it.get('fixability','?')}] "
                  f"{it.get('location','?')}: {it.get('claim','')[:80]}{tag}")
        if rec.get("self_verify"):
            print(f"  self-verify: {rec['self_verify']}  | corrected assessment: "
                  f"{rep.get('assessment_after_self_verify','?')}")
        print(f"  saved: reports/{rec.get('_saved_as')}  (usage: {rec.get('usage')})")
    return rc


def _resolve_id(token: str) -> str:
    """Accept a full registry id or a bare arxiv base (e.g. '1408.1964')."""
    ids = [p.id for p in load_registry()]
    if token in ids:
        return token
    hits = [i for i in ids if token in i]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"error: no registry paper matches {token!r}")
    raise SystemExit(f"error: {token!r} is ambiguous: {hits}")


def cmd_verify_pipeline(args) -> int:
    pid = _resolve_id(args.paper)

    if args.dry_run:
        from .pipeline import dry_run_survey
        d = dry_run_survey(pid, strip_figures=args.strip_figures)
        msgs = d["messages"]
        print(f"# paper: {pid} — {d['paper'].title}")
        print(f"# source: {d['note']}")
        print("# agent pipeline: survey(master) -> verify each fragile target -> "
              "refute(counterexample/repair) -> aggregate")
        print(f"# stage-1 survey prompt (sent to the master; ~{_est_tokens(msgs):,} input tokens):\n")
        for m in msgs:
            body = m["content"]
            print(f"----- {m['role'].upper()} ({len(body)} chars) -----")
            print(body if args.full or len(body) < 2400
                  else body[:2400] + f"\n... [+{len(body)-2400} chars; use --full]")
            print()
        print("# (targets, per-target context, and the verify/refute prompts are produced by "
              "the master at run time — they need the served model)")
        return 0

    gen = _gen_from_args(args)
    from .pipeline import verify_paper_pipeline

    backend = _make_backend(args)
    base = getattr(backend, "base_url", "in-process (vllm offline)")
    print(f"# backend: {base}  model={backend.model}")
    print(f"\n=== pipeline-verifying {pid} ===")
    rec = verify_paper_pipeline(pid, backend, gen=gen, strip_figures=args.strip_figures)
    rep = rec["report"]
    if not rec["parse_ok"]:
        print(f"  !! aggregate JSON parse failed: {rep.get('parse_error')} (raw saved)")
    else:
        issues = rep.get("issues", [])
        print(f"\n  AGGREGATE: {rep.get('overall_assessment','?')}  | {len(issues)} issue(s)  "
              f"| confident error: {rep.get('confident_error_found','?')}")
        for it in issues:
            print(f"   - [{it.get('severity','?')}/{it.get('fixability','?')}/"
                  f"{it.get('status','?')}] {it.get('location','?')}: {it.get('claim','')[:70]}")
    print(f"  targets={rec.get('n_targets')} refutations={rec.get('n_refutations')}  "
          f"saved: reports/{rec.get('_saved_as')}  (usage: {rec.get('usage_total')})")
    return 0 if rec["parse_ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mathpoc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list registry papers").set_defaults(func=cmd_list)

    bp = sub.add_parser("build-prompt", help="assemble + print the prompt (no backend)")
    bp.add_argument("paper")
    bp.add_argument("--prompt", default=DEFAULT_PROMPT)
    bp.add_argument("--full", action="store_true", help="print full message bodies")
    bp.add_argument("--strip-figures", action="store_true",
                    help="drop tikz/pgf picture bodies (default keeps them; use only to fit an oversized paper)")
    bp.set_defaults(func=cmd_build_prompt)

    v = sub.add_parser("verify", help="run verification against the served model")
    v.add_argument("paper", nargs="*")
    v.add_argument("--all", action="store_true")
    v.add_argument("--prompt", default=DEFAULT_PROMPT)
    v.add_argument("--backend", choices=["http", "vllm"], default="http",
                   help="http = OpenAI-compatible endpoint (served model); "
                        "vllm = in-process offline batch (no server; multi-node via Ray)")
    v.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (or MATHPOC_BASE_URL); http backend only")
    v.add_argument("--model", default=None, help="model id (or MATHPOC_MODEL; else auto-detect)")
    v.add_argument("--max-tokens", type=int, default=32768)
    v.add_argument("--temperature", type=float, default=1.0, help="DeepSeek-V4 recommends 1.0")
    v.add_argument("--top-p", type=float, default=1.0)
    v.add_argument("--reasoning-effort", choices=["off", "high", "max"], default="max",
                   help="thinking effort (chat_template_kwargs); 'max' = Think Max")
    v.add_argument("--thinking-style", choices=["deepseek", "glm"], default="deepseek",
                   help="how to encode thinking: deepseek (thinking:true+effort) | glm (effort; "
                        "enable_thinking:false to disable). Use 'glm' for GLM-5.2.")
    v.add_argument("--strip-figures", action="store_true",
                   help="drop tikz/pgf picture bodies (default keeps them; use only to fit an oversized paper)")
    v.add_argument("--self-verify", action="store_true",
                   help="after the single pass, re-check each major/critical finding with a fresh "
                        "adversarial subagent (refute prompt) — the false-positive guard")
    v.add_argument("--self-verify-severities", default="critical,major",
                   help="comma-separated severities that trigger a self-verify pass (default critical,major)")
    v.add_argument("--dry-run", action="store_true", help="assemble only; do not contact the model")
    v.set_defaults(func=cmd_verify)

    vp = sub.add_parser("verify-pipeline",
                        help="subagent-per-proof-unit pipeline (map -> per-unit -> aggregate)")
    vp.add_argument("paper", help="registry id or bare arxiv base, e.g. 1408.1964")
    vp.add_argument("--backend", choices=["http", "vllm"], default="http",
                    help="http = served endpoint; vllm = in-process offline batch (no server)")
    vp.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (or MATHPOC_BASE_URL); http backend only")
    vp.add_argument("--model", default=None, help="model id (or MATHPOC_MODEL; else auto-detect)")
    vp.add_argument("--max-tokens", type=int, default=32768)
    vp.add_argument("--temperature", type=float, default=1.0)
    vp.add_argument("--top-p", type=float, default=1.0)
    vp.add_argument("--reasoning-effort", choices=["off", "high", "max"], default="max")
    vp.add_argument("--thinking-style", choices=["deepseek", "glm"], default="deepseek",
                    help="how to encode thinking in chat_template_kwargs; use 'glm' for GLM-5.2")
    vp.add_argument("--strip-figures", action="store_true",
                    help="drop tikz/pgf picture bodies (default keeps them)")
    vp.add_argument("--full", action="store_true", help="(dry-run) print full bundle text")
    vp.add_argument("--dry-run", action="store_true",
                    help="assemble + print every subagent bundle; do not contact the model")
    vp.set_defaults(func=cmd_verify_pipeline)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
