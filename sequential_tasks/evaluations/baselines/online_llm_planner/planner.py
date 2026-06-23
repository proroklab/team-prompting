"""Groq API client for the online-LLM-planner baseline.

Uses ``llama-3.3-70b-versatile`` via Groq's OpenAI-compatible chat-completions
endpoint with forced tool calling so the response is always structured JSON.

Why Groq + Llama 3.3 70B:
* Free tier (30 RPM, 1000 RPD for 70B) is sufficient for the smoke test and
  spread-over-a-day full eval.
* OpenAI-compatible API: standard tool-calling shape,
  reliable JSON output.
* Open-weights frontier-class model — defensible as a paper baseline
  ("we used a strong open-weights LLM as the online planner").

API key resolution: ``sequence_models.api_keys.GROQ_API_KEY`` first, then
the ``GROQ_API_KEY`` env var.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from groq import Groq

from evaluations.baselines.online_llm_planner.prompts import (
    SYSTEM_PROMPT, ACTIONS,
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 4
PER_CALL_TIMEOUT_S = 60


# OpenAI-style tool schema for forced structured output.
ASSIGN_ACTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "assign_actions",
        "description": (
            "Assign one action to each agent for the next planning interval. "
            "Reasoning is short (≤ 2 sentences) — used for logs only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Brief reasoning (≤ 2 sentences)"},
                "agent_0": {"type": "string", "enum": list(ACTIONS), "description": "Action for agent 0 (scout)"},
                "agent_1": {"type": "string", "enum": list(ACTIONS), "description": "Action for agent 1 (scout)"},
                "agent_2": {"type": "string", "enum": list(ACTIONS), "description": "Action for agent 2 (carrier)"},
                "agent_3": {"type": "string", "enum": list(ACTIONS), "description": "Action for agent 3 (carrier)"},
            },
            "required": ["reasoning", "agent_0", "agent_1", "agent_2", "agent_3"],
        },
    },
}


@dataclass
class PlanResult:
    actions: Dict[str, str]
    reasoning: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0       # Groq doesn't expose cache reads in the same way
    raw_text: str = ""


@dataclass
class CumulativePlannerStats:
    n_calls: int = 0
    total_latency_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    failures: int = 0
    log: List[Dict] = field(default_factory=list)

    def add(self, result: PlanResult) -> None:
        self.n_calls += 1
        self.total_latency_s += result.latency_s
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        self.total_cache_read_tokens += result.cache_read_tokens

    def summary(self) -> Dict:
        return {
            "n_calls": self.n_calls,
            "total_latency_s": round(self.total_latency_s, 3),
            "mean_latency_s": round(self.total_latency_s / max(1, self.n_calls), 3),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "failures": self.failures,
        }


def _resolve_api_key() -> str:
    try:
        from sequence_models.api_keys import GROQ_API_KEY
        if GROQ_API_KEY:
            return GROQ_API_KEY
    except Exception:
        pass
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key
    raise RuntimeError(
        "Groq API key not found. Either set sequence_models/api_keys.py "
        "GROQ_API_KEY, or export GROQ_API_KEY in the environment."
    )


class OnlineLLMPlanner:
    """Calls Groq per planning step. Optional rate limiter for free-tier RPM."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = 512,
        min_interval_s: float = 0.0,
    ) -> None:
        api_key = _resolve_api_key()
        self.client = Groq(api_key=api_key, timeout=PER_CALL_TIMEOUT_S)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.min_interval_s = min_interval_s
        self._last_call_at = 0.0

    def plan(self, user_text: str) -> PlanResult:
        if self.min_interval_s > 0.0:
            wait = self.min_interval_s - (time.time() - self._last_call_at)
            if wait > 0:
                time.sleep(wait)

        last_err: Optional[Exception] = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                t0 = time.time()
                self._last_call_at = t0
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    tools=[ASSIGN_ACTIONS_TOOL],
                    tool_choice={"type": "function", "function": {"name": "assign_actions"}},
                    temperature=0.0,
                    max_completion_tokens=self.max_output_tokens,
                )
                latency = time.time() - t0
                actions, reasoning, raw_text = self._extract_actions(response)
                usage = response.usage
                return PlanResult(
                    actions=actions,
                    reasoning=reasoning,
                    latency_s=latency,
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    cache_read_tokens=0,
                    raw_text=raw_text,
                )
            except Exception as exc:
                last_err = exc
                if attempt < MAX_RETRIES:
                    backoff = min(60.0, 2.0 ** attempt + 5.0)
                    time.sleep(backoff)
        raise RuntimeError(f"LLM planning failed after {MAX_RETRIES + 1} attempts: {last_err}")

    @staticmethod
    def _extract_actions(response) -> tuple[Dict[str, str], str, str]:
        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            raise ValueError(
                f"Groq response missing tool_calls. finish_reason={choice.finish_reason}, "
                f"content={getattr(message, 'content', None)!r}"
            )
        call = tool_calls[0]
        if call.function.name != "assign_actions":
            raise ValueError(f"Unexpected tool name: {call.function.name}")
        raw = call.function.arguments  # JSON string
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool arguments not valid JSON: {exc}\n--- raw ---\n{raw[:500]}")

        actions: Dict[str, str] = {}
        for k in ("agent_0", "agent_1", "agent_2", "agent_3"):
            if k not in parsed:
                raise ValueError(f"Tool args missing key {k!r}: {parsed}")
            v = parsed[k]
            if v not in ACTIONS:
                raise ValueError(f"Tool returned unknown action {v!r} for {k}")
            actions[k] = v
        reasoning = parsed.get("reasoning", "")
        return actions, reasoning, raw
