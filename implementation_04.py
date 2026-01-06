from __future__ import annotations

from collections import defaultdict
from typing import Any

from openai import OpenAI, AsyncOpenAI

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class OpenAIResponsesClient(BaseLM):
    """
    OpenAI client using the Responses API.
    - Supports both string prompts and OpenAI-style message lists.
    - Tracks token usage when available.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def _to_input(self, prompt: str | list[dict[str, Any]]) -> Any:
        # Responses API accepts `input` as text or structured items.
        # The simplest faithful mapping is:
        # - str -> str
        # - messages -> list of input items (we keep it as "messages" item)
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
            # Represent as a single "message" input item sequence:
            # OpenAI docs: Responses "input" can contain message items.
            return prompt
        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAIResponsesClient.")

        resp = self.client.responses.create(
            model=model,
            input=self._to_input(prompt),
        )

        text = getattr(resp, "output_text", None)
        if text is None:
            # Fallback: attempt to extract from output items
            text = str(resp)

        self._track_usage(resp, model)
        return text

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAIResponsesClient.")

        resp = await self.async_client.responses.create(
            model=model,
            input=self._to_input(prompt),
        )

        text = getattr(resp, "output_text", None)
        if text is None:
            text = str(resp)

        self._track_usage(resp, model)
        return text

    def _track_usage(self, resp: Any, model: str) -> None:
        self.model_call_counts[model] += 1

        usage = getattr(resp, "usage", None)
        if usage is None:
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0
            return

        # Usage objects vary slightly; handle defensively.
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0

        self.model_input_tokens[model] += prompt_tokens
        self.model_output_tokens[model] += completion_tokens

        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_summaries: dict[str, ModelUsageSummary] = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
```

This uses the Responses API, which OpenAI recommends as the forward path. ([OpenAI Platform][2])

---

### B) Modify: `rlm/clients/__init__.py` to register it

Add an `elif` branch:

```python
    elif backend == "openai_responses":
        from rlm.clients.openai_responses import OpenAIResponsesClient
        return OpenAIResponsesClient(**backend_kwargs)
```

---

### C) Add a simple budget governor: `rlm/utils/budget.py`

```python
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Budget:
    max_iterations: int = 30
    max_subcalls: int = 200
    max_wall_seconds: float = 180.0

    def start(self) -> float:
        return time.perf_counter()

    def check_wall(self, t0: float) -> None:
        if (time.perf_counter() - t0) > self.max_wall_seconds:
            raise TimeoutError(f"Budget exceeded: wall time > {self.max_wall_seconds}s")

    def check_iterations(self, iteration: int) -> None:
        if iteration >= self.max_iterations:
            raise RuntimeError(f"Budget exceeded: iterations >= {self.max_iterations}")
```

---

### D) Minimal change in `rlm/core/rlm.py` to enforce budgets

In `completion()`, at the top:

```python
from rlm.utils.budget import Budget
```

Add an optional ctor arg `budget: Budget | None = None`, default `None`, store `self.budget = budget or Budget(max_iterations=max_iterations)`.

Then in the loop:

```python
t0 = self.budget.start()
subcall_count = 0

for i in range(self.max_iterations):
    self.budget.check_wall(t0)
    self.budget.check_iterations(i)

    iteration = self._completion_turn(...)
    # count subcalls by reading code_block.result.rlm_calls
    for cb in iteration.code_blocks:
        subcall_count += len(getattr(cb.result, "rlm_calls", []))
    if subcall_count > self.budget.max_subcalls:
        raise RuntimeError(f"Budget exceeded: subcalls > {self.budget.max_subcalls}")
```

That’s enough to prevent “death spirals” while keeping their existing structure.

---

### E) Usage: your target models

Examples (root model = best model; add cheaper “subcall” model if you want):

```python
import os
from rlm import RLM
from rlm.utils.budget import Budget

rlm = RLM(
    backend="openai_responses",
    backend_kwargs={"api_key": os.getenv("OPENAI_API_KEY"), "model_name": "gpt-5.2"},
    other_backends=["anthropic", "gemini"],
    other_backend_kwargs=[
        {"api_key": os.getenv("ANTHROPIC_API_KEY"), "model_name": "claude-opus-4-5"},
        {"api_key": os.getenv("GEMINI_API_KEY"), "model_name": "gemini-3-pro-preview"},
    ],
    environment="docker",  # recommended if any prompt is untrusted
    max_iterations=25,
    budget=Budget(max_iterations=25, max_subcalls=120, max_wall_seconds=180),
    verbose=True,
)

result = rlm.completion(
    prompt=open("huge.txt", "r", encoding="utf-8").read(),
    root_prompt="Answer: what are the top 5 claims, with quotes and locations?",
)
print(result.response)
