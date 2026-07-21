"""Transport to the served model, behind a small interface.

The rest of the harness only knows `Backend.complete(messages, gen) -> Completion`.
Swap the implementation to change *how* we reach the model without touching anything else.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class GenConfig:
    """Generation knobs. Defaults aim at 'full power': deterministic, room for long
    reasoning + the final JSON. `extra_body` carries server-specific knobs (e.g. a
    reasoning-effort field) without hard-coding any one server's API here."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 32768  # reasoning models are verbose; don't truncate before the JSON
    # Server-specific knobs. For DeepSeek-V4-Flash, max reasoning ("Think Max") is:
    #   extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}}
    # (requires the server started with --max-model-len >= 393216).
    extra_body: dict = field(default_factory=dict)


@dataclass
class Completion:
    text: str
    reasoning: str = ""
    usage: dict = field(default_factory=dict)
    model: str = ""


class Backend:
    def complete(self, messages: list[dict], gen: GenConfig) -> Completion:
        raise NotImplementedError


class OpenAICompatBackend(Backend):
    """Chat against an OpenAI-compatible endpoint.

    This is how DeepSeek-V4-Flash is currently served on Azzurra (vLLM's OpenAI server,
    see ../bigLLM-azzurra and scripts/azzurra/). It is ONE concrete Backend — the harness
    does not assume it. Config via args or env: MATHPOC_BASE_URL, MATHPOC_API_KEY,
    MATHPOC_MODEL. If MATHPOC_MODEL is unset, the served model id is auto-detected.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        from openai import OpenAI  # lazy: dry-run/build-prompt need no client installed

        self.base_url = base_url or os.environ.get("MATHPOC_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("MATHPOC_API_KEY", "EMPTY")
        # A max-reasoning pass (e.g. GLM-5.2 Think-Max) streams tens of thousands of tokens; the
        # response is non-streaming, so the WHOLE generation counts against one read-timeout window.
        # The SDK default (600 s, ×2 silent retries that each re-run the full generation) tripped at
        # ~30 min and lost the near-complete report (job 922436). Set a generous explicit timeout and
        # drop the wasteful retries. Override with MATHPOC_TIMEOUT (seconds).
        self.timeout = float(os.environ.get("MATHPOC_TIMEOUT", "7200"))
        self.client = OpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=0
        )
        self.model = model or os.environ.get("MATHPOC_MODEL") or self._autodetect()

    def _autodetect(self) -> str:
        return self.client.models.list().data[0].id

    def complete(self, messages: list[dict], gen: GenConfig) -> Completion:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_tokens,
            extra_body=gen.extra_body or None,
        )
        msg = resp.choices[0].message
        # DeepSeek/R1-style servers expose chain-of-thought separately from the answer.
        # The OpenAI client v2.x doesn't surface the non-standard `reasoning_content` as a
        # plain attribute — it lands in `model_extra` — so check both.
        reasoning = getattr(msg, "reasoning_content", None)
        if not reasoning:
            extra = getattr(msg, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
        return Completion(
            text=msg.content or "",
            reasoning=reasoning,
            usage=resp.usage.model_dump() if resp.usage else {},
            model=resp.model,
        )


class VLLMOfflineBackend(Backend):
    """Run the model IN-PROCESS with vLLM's offline API — no HTTP server, no localhost.

    This is a batch driver, not a service: the model is loaded once, generated for each paper,
    then the process exits. Nothing enters the node after launch. For a model too large for one
    node (GLM-5.2-FP8 spread over 3-4 H100 nodes) the surrounding SLURM job still boots a Ray
    cluster so vLLM can place the TP×PP ranks — this backend is the *driver* on the head node,
    not an endpoint. Same `Backend.complete` contract as the HTTP backend, so verify.py and
    pipeline.py are unchanged.

    Config via env (or `--model`): MATHPOC_MODEL (required, path or HF id), MATHPOC_TP,
    MATHPOC_PP, MATHPOC_MAX_MODEL_LEN, MATHPOC_KV_DTYPE, MATHPOC_GPU_MEM_UTIL,
    MATHPOC_DIST_BACKEND.
    """

    def __init__(self, model: str | None = None):
        from vllm import LLM, SamplingParams  # lazy: only this backend needs vllm/GPUs

        self._SamplingParams = SamplingParams
        self.model = model or os.environ.get("MATHPOC_MODEL")
        if not self.model:
            raise ValueError("VLLMOfflineBackend needs a model (MATHPOC_MODEL or --model)")
        tp = int(os.environ.get("MATHPOC_TP", "4"))
        pp = int(os.environ.get("MATHPOC_PP", "1"))
        max_len = int(os.environ.get("MATHPOC_MAX_MODEL_LEN", "0")) or None
        dist = os.environ.get("MATHPOC_DIST_BACKEND") or ("ray" if pp > 1 else "mp")
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            kv_cache_dtype=os.environ.get("MATHPOC_KV_DTYPE", "auto"),
            max_model_len=max_len,
            gpu_memory_utilization=float(os.environ.get("MATHPOC_GPU_MEM_UTIL", "0.90")),
            distributed_executor_backend=dist,
            trust_remote_code=True,
        )

    def complete(self, messages: list[dict], gen: GenConfig) -> Completion:
        sp = self._SamplingParams(
            temperature=gen.temperature, top_p=gen.top_p, max_tokens=gen.max_tokens
        )
        # gen.extra_body carries {"chat_template_kwargs": {...}} (the thinking knob) — vLLM's
        # offline chat() takes chat_template_kwargs directly, so the same GenConfig drives both
        # backends identically.
        ctk = (gen.extra_body or {}).get("chat_template_kwargs")
        out = self.llm.chat([messages], sampling_params=sp, chat_template_kwargs=ctk)[0]
        gen_out = out.outputs[0]
        text = gen_out.text or ""
        # Offline mode returns one raw stream; no server-side reasoning parser splits it. GLM/
        # DeepSeek wrap chain-of-thought in <think>…</think>, which verify.extract_json already
        # strips from the answer — we just lift it out here so raw/ keeps the reasoning too.
        m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        reasoning = m.group(1).strip() if m else ""
        usage = {
            "prompt_tokens": len(out.prompt_token_ids or []),
            "completion_tokens": len(gen_out.token_ids or []),
        }
        return Completion(text=text, reasoning=reasoning, usage=usage, model=self.model)
