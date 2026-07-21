"""mathpoc — harness for LLM-based error detection in math/TCS papers.

Pipeline: registry (papers/registry.yaml) -> assemble LaTeX source -> build prompt
(prompts/verify_v*.md) -> Backend (served model) -> parse JSON -> report (reports/).

The transport to the model is behind `backend.Backend`; the rest is transport-agnostic.
"""
