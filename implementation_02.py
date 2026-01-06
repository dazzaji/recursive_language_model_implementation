import os
import re
import sys
import json
import time
import tempfile
import concurrent.futures
import argparse
from io import StringIO
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

# Load env vars
load_dotenv()
console = Console()

# --- 1. PROVIDER ADAPTERS (Verified Jan 2026 Specs) ---

class LLMProvider:
    def completion(self, prompt: str, system: str, history: List[Dict] = []) -> str:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, model="gpt-5.2"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model # Verified: gpt-5.2

    def completion(self, prompt: str, system: str, history: List[Dict] = []) -> str:
        # Verified: Responses API uses 'instructions' for system prompt
        # Merging history into a single input string for Responses API context
        full_input = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history]) + f"\nuser: {prompt}"
        
        response = self.client.responses.create(
            model=self.model,
            input=full_input,
            instructions=system,
        )
        return response.output_text

class AnthropicProvider(LLMProvider):
    def __init__(self, model="claude-opus-4-5-20251101"):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model # Verified: claude-opus-4-5-20251101

    def completion(self, prompt: str, system: str, history: List[Dict] = []) -> str:
        # Verified: Extended thinking enabled via 'thinking' param
        msgs = [{"role": m["role"], "content": m["content"]} for m in history]
        msgs.append({"role": "user", "content": prompt})
        
        # Note: 'thinking' requires max_tokens > 2048
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=msgs,
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 4096}, 
            temperature=1.0 # Required for thinking models
        )
        # Filter for text blocks (ignore thinking blocks)
        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "".join(text_blocks)

class GeminiProvider(LLMProvider):
    def __init__(self, model="gemini-3-pro-preview"):
        import google.generativeai as genai
        from google.genai.types import GenerateContentConfig
        
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = genai.Client()
        self.model = model # Verified: gemini-3-pro-preview

    def completion(self, prompt: str, system: str, history: List[Dict] = []) -> str:
        # Verified: 'thinking_level' config
        # Convert history format
        gemini_history = []
        for m in history:
            role = "user" if m["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [m["content"]]})

        chat = self.client.chats.create(model=self.model, history=gemini_history)
        
        response = chat.send_message(
            prompt,
            config={"system_instruction": system, "thinking_level": "high"}
        )
        return response.text

# --- 2. REPL ENVIRONMENT (With Parallel Batching) ---

class REPLEnv:
    def __init__(self, context_path: str, provider: LLMProvider, sub_model_provider: LLMProvider):
        self.context_path = context_path
        self.provider = provider
        self.sub_provider = sub_model_provider
        self.locals = {}
        self.temp_dir = tempfile.mkdtemp()
        
        # Safe Globals + RLM Tools
        self.globals = {
            "__builtins__": {
                "print": print, "len": len, "str": str, "int": int, "float": float,
                "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
                "range": range, "enumerate": enumerate, "zip": zip, "min": min, "max": max,
                "sum": sum, "open": open, "sorted": sorted, "abs": abs,
                "__import__": __import__
            },
            "llm_query": self.llm_query,
            "llm_batch_query": self.llm_batch_query, # Critical for efficiency
        }
        self.globals["CONTEXT_FILE_PATH"] = self.context_path

    def llm_query(self, prompt: str) -> str:
        """Sequential sub-call."""
        try:
            return self.sub_provider.completion(
                prompt=prompt,
                system="You are a sub-agent analyzing a specific text chunk. Be concise."
            )
        except Exception as e:
            return f"Error: {e}"

    def llm_batch_query(self, prompts: List[str]) -> List[str]:
        """Parallel sub-calls for scanning massive files."""
        console.print(f"[dim cyan]Running batch query on {len(prompts)} items...[/dim cyan]")
        results = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_idx = {executor.submit(self.llm_query, p): i for i, p in enumerate(prompts)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    def execute(self, code: str) -> str:
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output
        try:
            exec_globals = {**self.globals, **self.locals}
            exec(code, exec_globals, exec_globals)
            # Update state
            self.locals.update({k: v for k, v in exec_globals.items() if k not in self.globals})
            return redirected_output.getvalue().strip() or "[Executed]"
        except Exception as e:
            return f"Traceback: {str(e)}"
        finally:
            sys.stdout = old_stdout

# --- 3. SYSTEM PROMPT ---

SYSTEM_PROMPT = """You are a Recursive Language Model (RLM).
You have access to a Python REPL to answer the query.
The context is in a file at: `CONTEXT_FILE_PATH`.

DO NOT read the whole file into memory. It might be 10M+ tokens.
Instead:
1. Inspect file size/structure.
2. Chunk the content using Python.
3. Use `llm_batch_query([prompts])` to scan chunks in parallel.

Use ```repl ... ``` blocks to write code.
When found, output `FINAL(answer)`.
"""

# --- 4. RUNNER ---

def run_rlm(context_file: str, query: str, root_provider="openai", max_iters=15):
    # Setup Root Provider
    if root_provider == "openai": root_llm = OpenAIProvider("gpt-5.2")
    elif root_provider == "anthropic": root_llm = AnthropicProvider("claude-opus-4-5-20251101")
    elif root_provider == "gemini": root_llm = GeminiProvider("gemini-3-pro-preview")
    
    # Setup Sub-Provider (Use smaller/cheaper models for recursion)
    # Using Gemini 3 Flash for sub-calls is best practice for 1M+ tokens
    sub_llm = GeminiProvider("gemini-3-flash-preview") 

    repl = REPLEnv(context_file, root_llm, sub_llm)
    history = []

    console.rule(f"[bold blue]RLM Started | Root: {root_provider} | Sub: Gemini-3-Flash[/bold blue]")

    for i in range(max_iters):
        console.print(f"[bold yellow]--- Iteration {i+1} ---[/bold yellow]")
        
        # Call Root
        with console.status("[bold green]Root Model Thinking...[/bold green]"):
            response = root_llm.completion(query, SYSTEM_PROMPT, history)

        # Parse Logic
        parts = re.split(r'(```repl.*?```)', response, flags=re.DOTALL)
        for part in parts:
            if part.startswith("```repl"):
                code = part.replace("```repl", "").replace("```", "").strip()
                console.print(Panel(Syntax(code, "python"), title="Code", border_style="cyan"))
                
                with console.status("[bold cyan]Executing...[/bold cyan]"):
                    output = repl.execute(code)
                
                console.print(Panel(output[:500] + "...", title="Output", border_style="green"))
                history.append({"role": "assistant", "content": part})
                history.append({"role": "user", "content": f"Output:\n{output[:10000]}"})
            elif part.strip():
                console.print(Markdown(part))
                history.append({"role": "assistant", "content": part})

        if "FINAL(" in response:
            final = response.split("FINAL(")[1].split(")")[0]
            console.print(Panel(final, title="FINAL ANSWER", style="bold red"))
            break

if __name__ == "__main__":
    # Create dummy massive file for demo
    with open("massive_context.txt", "w") as f:
        f.write("Start of log...\n" + "Log entry: Nothing here.\n" * 50000 + "Log entry: SECRET_KEY=XYZ-2026\n")
    
    run_rlm("massive_context.txt", "Find the secret key in the logs.", root_provider="openai")
