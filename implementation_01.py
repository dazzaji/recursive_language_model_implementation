import os
import re
import io
from contextlib import redirect_stdout
from typing import Callable, Optional

# === Provider-specific imports (install: pip install openai anthropic google-generativeai) ===
from openai import OpenAI as OpenAIClient
from anthropic import Anthropic
from google.generativeai import GenerativeModel, configure as genai_configure
import google.generativeai.types as genai_types

class LLMProvider:
    def __init__(self, root_model: str, sub_model: str):
        self.root_model = root_model
        self.sub_model = sub_model

    def query(self, system: str, user: str, is_sub: bool = False) -> str:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, root_model="gpt-5.2", sub_model="gpt-5-mini"):
        super().__init__(root_model, sub_model)
        self.client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

    def query(self, system: str, user: str, is_sub: bool = False) -> str:
        model = self.sub_model if is_sub else self.root_model
        resp = self.client.responses.create(
            model=model,
            instructions=system or None,
            input=user,
            reasoning={"effort": "high" if not is_sub else "medium"}
        )
        return resp.output_text

class AnthropicProvider(LLMProvider):
    def __init__(self, root_model="claude-opus-4-5-20251101", sub_model="claude-sonnet-4-5-20250929"):
        super().__init__(root_model, sub_model)
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def query(self, system: str, user: str, is_sub: bool = False) -> str:
        model = self.sub_model if is_sub else self.root_model
        msg = self.client.messages.create(
            model=model,
            max_tokens=8192,
            system=system or None,
            messages=[{"role": "user", "content": user}],
            extra_body={"effort": "high" if not is_sub else "medium"}
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

class GeminiProvider(LLMProvider):
    def __init__(self, root_model="gemini-3-pro-preview", sub_model="gemini-3-flash-preview"):
        super().__init__(root_model, sub_model)
        genai_configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.root_gen = GenerativeModel(self.root_model)
        self.sub_gen = GenerativeModel(self.sub_model)

    def query(self, system: str, user: str, is_sub: bool = False) -> str:
        model = self.sub_gen if is_sub else self.root_gen
        config = genai_types.GenerateContentConfig(
            thinking_level="high" if not is_sub else "medium"
        )
        resp = model.generate_content(
            user,
            system_instruction=system or None,
            generation_config=config
        )
        return resp.text or ""

# === Safe REPL ===
class SafeREPL:
    SAFE_BUILTINS = {
        "print": print, "len": len, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "tuple": tuple, "set": set, "bool": bool,
        "range": range, "enumerate": enumerate, "zip": zip, "min": min, "max": max,
        "sum": sum, "abs": abs, "round": round, "sorted": sorted, "reversed": reversed,
        "any": any, "all": all,
    }

    def __init__(self, llm_query_fn: Callable[[str], str], context: str):
        self.env = {
            "context": context,
            "buffers": [],
            "llm_query": llm_query_fn,
        }
        # Pre-import useful modules (common in paper trajectories)
        pre_code = """
import re
import json
import math
from collections import Counter, defaultdict
from itertools import chain, groupby
"""
        exec(pre_code, {}, self.env)
        self.env["__builtins__"] = self.SAFE_BUILTINS

    def execute(self, code: str) -> str:
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, self.env)
            output = buf.getvalue()
            result = "Success"
            if output:
                result += f"\nOutput:\n{output}"
        except Exception as e:
            result = f"Error: {type(e).__name__}: {str(e)}"
        # Truncate per-block
        return result[:20000] + ("\n...[truncated]" if len(result) > 20000 else "")

# === RLM Core ===
SYSTEM_PROMPT = """You are a Recursive Language Model (RLM). The full context is in the variable `context` ({length} characters).

Guidelines:
- Inspect structure first (print samples, len(context)).
- Chunk intelligently (newlines, headers, regex).
- Use llm_query(prompt) heavily for semantic tasks — it handles large chunks (~500K+ chars). Batch aggressively rather than many tiny calls.
- Store intermediates in variables / buffers list.
- Print useful info for continuation.

Execute code in ```repl

When finished:
- FINAL(your answer) for direct string answer
- FINAL_VAR(var_name) to return a REPL variable

Query: {query}
"""

class RecursiveLanguageModel:
    def __init__(self, provider: LLMProvider, max_iters: int = 30, verbose: bool = True):
        self.provider = provider
        self.max_iters = max_iters
        self.verbose = verbose

    def _sub_query(self, prompt: str) -> str:
        return self.provider.query("", prompt, is_sub=True)

    def run(self, context: str, query: str) -> str:
        system = SYSTEM_PROMPT.format(length=len(context), query=query)
        repl = SafeREPL(self._sub_query, context)

        last_output = "Initial state — no previous execution."
        for it in range(self.max_iters):
            if self.verbose:
                print(f"\n=== Iteration {it+1}/{self.max_iters} ===")

            user_msg = f"""Last REPL output (truncated):
{last_output}

Next action (reasoning + code in ```repl or FINAL if complete):"""

            model_out = self.provider.query(system, user_msg)
            if self.verbose:
                print(f"Model response preview: {model_out[:500]}...")

            # Check termination
            final_match = re.search(r"FINAL\s*\(\s*(.*?)\s*\)", model_out, re.DOTALL)
            if final_match:
                return final_match.group(1).strip()

            var_match = re.search(r"FINAL_VAR\s*\(\s*(\w+)\s*\)", model_out)
            if var_match:
                var_name = var_match.group(1)
                val = repl.env.get(var_name)
                return str(val) if val is not None else f"[Missing variable: {var_name}]"

            # Execute code blocks
            blocks = re.findall(r"```repl\s*(.*?)\s*```", model_out, re.DOTALL)
            outputs = []
            for i, code in enumerate(blocks):
                if self.verbose:
                    print(f"Executing block {i+1}")
                out = repl.execute(code)
                outputs.append(out)
                if self.verbose:
                    print(f"Block result: {out[:500]}...")

            last_output = "\n---\n".join(outputs)
            if len(last_output) > 30000:
                last_output = last_output[:30000] + "\n...[full output truncated to prevent overflow]"

        return "[Max iterations reached — no final answer]"

# === Factory ===
def create_rlm(provider: str = "openai", root_model: Optional[str] = None, sub_model: Optional[str] = None):
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
    }
    cls = providers[provider.lower()]
    return RecursiveLanguageModel(cls(root_model=root_model, sub_model=sub_model))

# === Example Usage ===
if __name__ == "__main__":
    # Load huge context from file
    with open("huge_document.txt", "r", encoding="utf-8") as f:
        huge_context = f.read()

    query = "Summarize the key findings on long-context tasks and compare methods."

    rlm = create_rlm("openai", verbose=True)  # or "anthropic" / "gemini"
    answer = rlm.run(huge_context, query)
    print("\nFINAL ANSWER:\n", answer)
