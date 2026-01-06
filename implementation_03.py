#!/usr/bin/env python3
"""
Recursive Language Model (RLM) - Production Implementation
============================================================

A hybrid implementation combining:
- Architecture patterns from alexzhang13/rlm (paper authors)
- Current January 2026 API specifications
- Enhanced safety, batching, and error handling

Based on: Zhang, Kraska, & Khattab (2025) - "Recursive Language Models"
Paper: arXiv:2512.24601v1

Requirements:
    pip install openai anthropic google-genai rich requests dill

Environment Variables:
    OPENAI_API_KEY - Your OpenAI API key
    ANTHROPIC_API_KEY - Your Anthropic API key  
    GEMINI_API_KEY - Your Google Gemini API key

Usage:
    # As a script
    python rlm_final.py --provider openai --context document.txt --query "Summarize this"
    
    # As a library
    from rlm_final import RLM
    rlm = RLM(backend="openai", backend_kwargs={"model_name": "gpt-5.2"})
    result = rlm.completion("Your long context here")
    print(result.response)

Author: Hybrid implementation based on paper methodology + current APIs
Date: January 2026
Version: 2.0.0
"""

from __future__ import annotations

import os
import re
import io
import sys
import json
import time
import uuid
import socket
import struct
import asyncio
import argparse
import textwrap
import threading
import tempfile
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union, Literal, Tuple
from contextlib import redirect_stdout, contextmanager
from collections import defaultdict
from socketserver import ThreadingTCPServer, StreamRequestHandler
from pathlib import Path

# Optional rich console for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich.style import Style
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# Type Definitions & Configuration
# =============================================================================

ClientBackend = Literal["openai", "anthropic", "gemini", "portkey", "litellm", "vllm"]
EnvironmentType = Literal["local", "docker"]


@dataclass
class ModelUsageSummary:
    """Track token usage for a model."""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelUsageSummary":
        return cls(
            total_calls=data.get("total_calls", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
        )


@dataclass
class UsageSummary:
    """Aggregate usage across multiple models."""
    model_usage_summaries: Dict[str, ModelUsageSummary] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "model_usage_summaries": {
                model: usage.to_dict() 
                for model, usage in self.model_usage_summaries.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "UsageSummary":
        return cls(
            model_usage_summaries={
                model: ModelUsageSummary.from_dict(usage)
                for model, usage in data.get("model_usage_summaries", {}).items()
            }
        )


@dataclass
class REPLResult:
    """Result from REPL code execution."""
    stdout: str = ""
    stderr: str = ""
    locals: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    llm_calls: List["RLMChatCompletion"] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: repr(v) for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            "llm_calls": [c.to_dict() for c in self.llm_calls],
        }


@dataclass
class CodeBlock:
    """A code block extracted from LLM response."""
    code: str
    result: REPLResult
    
    def to_dict(self) -> Dict:
        return {"code": self.code, "result": self.result.to_dict()}


@dataclass
class RLMIteration:
    """Record of a single RLM iteration."""
    prompt: Union[str, Dict, List]
    response: str
    code_blocks: List[CodeBlock]
    final_answer: Optional[str] = None
    iteration_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "code_blocks": [cb.to_dict() for cb in self.code_blocks],
            "final_answer": self.final_answer,
            "iteration_time": self.iteration_time,
        }


@dataclass
class RLMChatCompletion:
    """Final result of an RLM completion call."""
    root_model: str
    prompt: Union[str, Dict, List]
    response: str
    usage_summary: UsageSummary
    execution_time: float
    
    def to_dict(self) -> Dict:
        return {
            "root_model": self.root_model,
            "prompt": self.prompt if isinstance(self.prompt, str) else str(self.prompt)[:500],
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RLMChatCompletion":
        return cls(
            root_model=data.get("root_model", "unknown"),
            prompt=data.get("prompt", ""),
            response=data.get("response", ""),
            usage_summary=UsageSummary.from_dict(data.get("usage_summary", {})),
            execution_time=data.get("execution_time", 0.0),
        )


@dataclass
class RLMMetadata:
    """Metadata about an RLM configuration."""
    root_model: str
    max_depth: int
    max_iterations: int
    backend: str
    backend_kwargs: Dict[str, Any]
    environment_type: str
    environment_kwargs: Dict[str, Any]
    other_backends: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        return {
            "root_model": self.root_model,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "backend": self.backend,
            "backend_kwargs": {k: v for k, v in self.backend_kwargs.items() if "key" not in k.lower()},
            "environment_type": self.environment_type,
            "environment_kwargs": self.environment_kwargs,
            "other_backends": self.other_backends,
        }


# =============================================================================
# Default Model Configurations (January 2026)
# =============================================================================

DEFAULT_MODELS = {
    "openai": {
        "root": "gpt-5.2",        # Flagship model
        "sub": "gpt-5-mini",      # Cost-efficient for recursion
    },
    "anthropic": {
        "root": "claude-opus-4-5-20251101",
        "sub": "claude-sonnet-4-5-20250929",
    },
    "gemini": {
        "root": "gemini-3-pro-preview",
        "sub": "gemini-3-flash-preview",
    },
    "portkey": {
        "root": "@openai/gpt-5.2",
        "sub": "@openai/gpt-5-mini",
    },
    "litellm": {
        "root": "gpt-5.2",
        "sub": "gpt-5-mini",
    },
    "vllm": {
        "root": "meta-llama/Llama-3-70b",
        "sub": "meta-llama/Llama-3-8b",
    },
}


# =============================================================================
# System Prompt (Based on Paper's Appendix D with enhancements)
# =============================================================================

RLM_SYSTEM_PROMPT = textwrap.dedent("""
You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:

1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.

2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.

3. A `llm_query_batched` function that allows you to query multiple prompts concurrently: `llm_query_batched(prompts: List[str]) -> List[str]`. This is much faster than sequential `llm_query` calls when you have multiple independent queries. Results are returned in the same order as the input prompts.

4. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example:

```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {chunk}")
print(answer)
```

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:

1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
""").strip()


# =============================================================================
# Socket Communication Protocol
# =============================================================================

@dataclass
class LMRequest:
    """Request to LM Handler."""
    prompt: Optional[Union[str, Dict]] = None
    prompts: Optional[List[Union[str, Dict]]] = None  # For batched requests
    model: Optional[str] = None
    
    @property
    def is_batched(self) -> bool:
        return self.prompts is not None and len(self.prompts) > 0
    
    def to_dict(self) -> Dict:
        d = {}
        if self.prompt is not None:
            d["prompt"] = self.prompt
        if self.prompts is not None:
            d["prompts"] = self.prompts
        if self.model is not None:
            d["model"] = self.model
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LMRequest":
        return cls(
            prompt=data.get("prompt"),
            prompts=data.get("prompts"),
            model=data.get("model"),
        )


@dataclass
class LMResponse:
    """Response from LM Handler."""
    error: Optional[str] = None
    chat_completion: Optional[RLMChatCompletion] = None
    chat_completions: Optional[List[RLMChatCompletion]] = None
    
    @property
    def success(self) -> bool:
        return self.error is None
    
    def to_dict(self) -> Dict:
        if self.error is not None:
            return {"error": self.error}
        if self.chat_completions is not None:
            return {"chat_completions": [c.to_dict() for c in self.chat_completions]}
        if self.chat_completion is not None:
            return {"chat_completion": self.chat_completion.to_dict()}
        return {"error": "No response"}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LMResponse":
        if data.get("error"):
            return cls(error=data["error"])
        if data.get("chat_completions"):
            return cls(chat_completions=[RLMChatCompletion.from_dict(c) for c in data["chat_completions"]])
        if data.get("chat_completion"):
            return cls(chat_completion=RLMChatCompletion.from_dict(data["chat_completion"]))
        return cls(error="Invalid response format")


def socket_send(sock: socket.socket, data: Dict) -> None:
    """Send length-prefixed JSON over socket."""
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def socket_recv(sock: socket.socket) -> Dict:
    """Receive length-prefixed JSON from socket."""
    raw_len = sock.recv(4)
    if not raw_len:
        return {}
    length = struct.unpack(">I", raw_len)[0]
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("Connection closed")
        payload += chunk
    return json.loads(payload.decode("utf-8"))


def send_lm_request(address: Tuple[str, int], request: LMRequest, timeout: int = 300) -> LMResponse:
    """Send request to LM Handler and get response."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(address)
            socket_send(sock, request.to_dict())
            response_data = socket_recv(sock)
            return LMResponse.from_dict(response_data)
    except Exception as e:
        return LMResponse(error=f"Request failed: {e}")


def send_lm_request_batched(
    address: Tuple[str, int], 
    prompts: List[str], 
    model: Optional[str] = None,
    timeout: int = 300
) -> List[LMResponse]:
    """Send batched request to LM Handler."""
    try:
        request = LMRequest(prompts=prompts, model=model)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(address)
            socket_send(sock, request.to_dict())
            response_data = socket_recv(sock)
            response = LMResponse.from_dict(response_data)
            
            if not response.success:
                return [LMResponse(error=response.error) for _ in prompts]
            if response.chat_completions:
                return [LMResponse(chat_completion=c) for c in response.chat_completions]
            return [LMResponse(error="No completions") for _ in prompts]
    except Exception as e:
        return [LMResponse(error=f"Batch request failed: {e}") for _ in prompts]


# =============================================================================
# LM Client Base Class & Implementations
# =============================================================================

class BaseLM(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.model_call_counts: Dict[str, int] = defaultdict(int)
        self.model_input_tokens: Dict[str, int] = defaultdict(int)
        self.model_output_tokens: Dict[str, int] = defaultdict(int)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
    
    @abstractmethod
    def completion(self, prompt: Union[str, List[Dict]], model: Optional[str] = None) -> str:
        """Generate a completion."""
        pass
    
    @abstractmethod
    async def acompletion(self, prompt: Union[str, List[Dict]], model: Optional[str] = None) -> str:
        """Async completion."""
        pass
    
    def get_usage_summary(self) -> UsageSummary:
        """Get aggregated usage statistics."""
        summaries = {}
        for model in self.model_call_counts:
            summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=summaries)
    
    def get_last_usage(self) -> ModelUsageSummary:
        """Get usage from the last call."""
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )


class OpenAIClient(BaseLM):
    """
    OpenAI client supporting GPT-5.x via Responses API (January 2026).
    
    The Responses API provides improved intelligence through CoT passthrough
    and is the recommended API for GPT-5.x models.
    """
    
    def __init__(
        self, 
        model_name: str = None, 
        api_key: str = None,
        base_url: str = None,
        reasoning_effort: str = "medium",
        **kwargs
    ):
        super().__init__(model_name or DEFAULT_MODELS["openai"]["root"], **kwargs)
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
        self.reasoning_effort = reasoning_effort
        self._use_responses_api = "gpt-5" in self.model_name.lower()
    
    def completion(self, prompt: Union[str, List[Dict]], model: str = None) -> str:
        model = model or self.model_name
        
        if self._use_responses_api:
            try:
                # Try Responses API for GPT-5.x
                input_text = prompt if isinstance(prompt, str) else self._messages_to_text(prompt)
                response = self.client.responses.create(
                    model=model,
                    input=input_text,
                    reasoning={"effort": self.reasoning_effort},
                )
                self._track_cost(response, model)
                return response.output_text
            except AttributeError:
                pass  # Fall through to Chat Completions
        
        # Standard Chat Completions API
        messages = self._to_messages(prompt)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        self._track_cost_chat(response, model)
        return response.choices[0].message.content
    
    async def acompletion(self, prompt: Union[str, List[Dict]], model: str = None) -> str:
        model = model or self.model_name
        messages = self._to_messages(prompt)
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        self._track_cost_chat(response, model)
        return response.choices[0].message.content
    
    def _to_messages(self, prompt: Union[str, List[Dict]]) -> List[Dict]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt
    
    def _messages_to_text(self, messages: List[Dict]) -> str:
        return "\n\n".join(m.get("content", "") for m in messages)
    
    def _track_cost(self, response, model: str):
        self.model_call_counts[model] += 1
        if hasattr(response, "usage"):
            self.model_input_tokens[model] += response.usage.input_tokens
            self.model_output_tokens[model] += response.usage.output_tokens
            self.last_prompt_tokens = response.usage.input_tokens
            self.last_completion_tokens = response.usage.output_tokens
    
    def _track_cost_chat(self, response, model: str):
        self.model_call_counts[model] += 1
        if hasattr(response, "usage") and response.usage:
            self.model_input_tokens[model] += response.usage.prompt_tokens
            self.model_output_tokens[model] += response.usage.completion_tokens
            self.last_prompt_tokens = response.usage.prompt_tokens
            self.last_completion_tokens = response.usage.completion_tokens


class AnthropicClient(BaseLM):
    """
    Anthropic client for Claude Opus 4.5 and Sonnet 4.5 (January 2026).
    """
    
    def __init__(
        self, 
        model_name: str = None, 
        api_key: str = None,
        max_tokens: int = 8192,
        **kwargs
    ):
        super().__init__(model_name or DEFAULT_MODELS["anthropic"]["root"], **kwargs)
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
            
        self.client = anthropic.Anthropic(**client_kwargs)
        self.async_client = anthropic.AsyncAnthropic(**client_kwargs)
        self.max_tokens = max_tokens
    
    def completion(self, prompt: Union[str, List[Dict]], model: str = None) -> str:
        model = model or self.model_name
        messages, system = self._prepare_messages(prompt)
        
        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        
        response = self.client.messages.create(**kwargs)
        self._track_cost(response, model)
        return self._extract_text(response)
    
    async def acompletion(self, prompt: Union[str, List[Dict]], model: str = None) -> str:
        model = model or self.model_name
        messages, system = self._prepare_messages(prompt)
        
        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        
        response = await self.async_client.messages.create(**kwargs)
        self._track_cost(response, model)
        return self._extract_text(response)
    
    def _prepare_messages(self, prompt: Union[str, List[Dict]]) -> Tuple[List[Dict], Optional[str]]:
        system = None
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}], None
        
        messages = []
        for msg in prompt:
            if msg.get("role") == "system":
                system = msg.get("content")
            else:
                messages.append(msg)
        return messages, system
    
    def _extract_text(self, response) -> str:
        return "".join(
            block.text for block in response.content 
            if hasattr(block, "text")
        )
    
    def _track_cost(self, response, model: str):
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += response.usage.input_tokens
        self.model_output_tokens[model] += response.usage.output_tokens
        self.last_prompt_tokens = response.usage.input_tokens
        self.last_completion_tokens = response.usage.output_tokens


class GeminiClient(BaseLM):
    """
    Google Gemini client for Gemini 3 Pro/Flash (January 2026).
    """
    
    def __init__(
        self, 
        model_name: str = None, 
        api_key: str = None,
        thinking_level: str = "medium",
        **kwargs
    ):
        super().__init__(model_name or DEFAULT_MODELS["gemini"]["root"], **kwargs)
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Please install google-genai: pip install google-genai")
        
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.thinking_level = thinking_level
    
    def completion(self, prompt: Union[str, List[Dict]], model: str = None) -> str:
        model = model or self.model_name
        contents, system = self._prepare_contents(prompt)
        
        config = None
        if system:
            config = self.types.GenerateContentConfig(system_instruction=system)
        
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        self._track_cost(response, model)
        return response.text
    
    async def acompletion(self, prompt: Union[str, List[Dict]], model: str = None) -> str:
        model = model or self.model_name
        contents, system = self._prepare_contents(prompt)
        
        config = None
        if system:
            config = self.types.GenerateContentConfig(system_instruction=system)
        
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        self._track_cost(response, model)
        return response.text
    
    def _prepare_contents(self, prompt: Union[str, List[Dict]]) -> Tuple[Any, Optional[str]]:
        system = None
        if isinstance(prompt, str):
            return prompt, None
        
        contents = []
        for msg in prompt:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                contents.append(self.types.Content(role="user", parts=[self.types.Part(text=content)]))
            elif role == "assistant":
                contents.append(self.types.Content(role="model", parts=[self.types.Part(text=content)]))
        return contents, system
    
    def _track_cost(self, response, model: str):
        self.model_call_counts[model] += 1
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.model_input_tokens[model] += response.usage_metadata.prompt_token_count or 0
            self.model_output_tokens[model] += response.usage_metadata.candidates_token_count or 0
            self.last_prompt_tokens = response.usage_metadata.prompt_token_count or 0
            self.last_completion_tokens = response.usage_metadata.candidates_token_count or 0


def get_client(backend: ClientBackend, backend_kwargs: Dict[str, Any]) -> BaseLM:
    """Factory function to create the appropriate LM client."""
    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "gemini": GeminiClient,
        "vllm": OpenAIClient,  # vLLM uses OpenAI-compatible API
    }
    
    if backend == "vllm":
        if "base_url" not in backend_kwargs:
            raise ValueError("base_url is required for vLLM backend")
    
    if backend not in clients:
        raise ValueError(f"Unknown backend: {backend}. Supported: {list(clients.keys())}")
    
    return clients[backend](**backend_kwargs)


# =============================================================================
# LM Handler (Socket Server)
# =============================================================================

class LMRequestHandler(StreamRequestHandler):
    """Handle LLM completion requests over socket."""
    
    def handle(self):
        try:
            request_data = socket_recv(self.connection)
            if not isinstance(request_data, dict):
                response = LMResponse(error="Request must be JSON object")
                socket_send(self.connection, response.to_dict())
                return
            
            request = LMRequest.from_dict(request_data)
            handler: LMHandler = self.server.lm_handler
            
            if request.is_batched:
                response = self._handle_batched(request, handler)
            elif request.prompt:
                response = self._handle_single(request, handler)
            else:
                response = LMResponse(error="Missing prompt or prompts")
            
            socket_send(self.connection, response.to_dict())
        except Exception as e:
            socket_send(self.connection, LMResponse(error=str(e)).to_dict())
    
    def _handle_single(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        client = handler.get_client(request.model)
        start = time.perf_counter()
        content = client.completion(request.prompt)
        
        return LMResponse(chat_completion=RLMChatCompletion(
            root_model=request.model or client.model_name,
            prompt=request.prompt,
            response=content,
            usage_summary=client.get_last_usage(),
            execution_time=time.perf_counter() - start,
        ))
    
    def _handle_batched(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        client = handler.get_client(request.model)
        start = time.perf_counter()
        
        async def run_all():
            tasks = [client.acompletion(p) for p in request.prompts]
            return await asyncio.gather(*tasks)
        
        results = asyncio.run(run_all())
        elapsed = time.perf_counter() - start
        
        completions = [
            RLMChatCompletion(
                root_model=request.model or client.model_name,
                prompt=prompt,
                response=content,
                usage_summary=client.get_last_usage(),
                execution_time=elapsed / len(request.prompts),
            )
            for prompt, content in zip(request.prompts, results)
        ]
        return LMResponse(chat_completions=completions)


class LMHandler:
    """Multi-threaded socket server for LM requests."""
    
    def __init__(self, client: BaseLM, host: str = "127.0.0.1", port: int = 0):
        self.default_client = client
        self.clients: Dict[str, BaseLM] = {client.model_name: client}
        self.host = host
        self._server = None
        self._thread = None
        self._port = port
    
    def register_client(self, model_name: str, client: BaseLM):
        """Register additional model client."""
        self.clients[model_name] = client
    
    def get_client(self, model: str = None) -> BaseLM:
        """Get client by model name."""
        if model and model in self.clients:
            return self.clients[model]
        return self.default_client
    
    @property
    def port(self) -> int:
        if self._server:
            return self._server.server_address[1]
        return self._port
    
    @property
    def address(self) -> Tuple[str, int]:
        return (self.host, self.port)
    
    def start(self) -> Tuple[str, int]:
        """Start the server in a background thread."""
        if self._server is not None:
            return self.address
        
        class Server(ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True
        
        self._server = Server((self.host, self._port), LMRequestHandler)
        self._server.lm_handler = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.address
    
    def stop(self):
        """Stop the server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
    
    def completion(self, prompt: str, model: str = None) -> str:
        """Direct completion (for main process use)."""
        return self.get_client(model).completion(prompt)
    
    def get_usage_summary(self) -> UsageSummary:
        """Get merged usage from all clients."""
        merged = {}
        for client in self.clients.values():
            for model, usage in client.get_usage_summary().model_usage_summaries.items():
                merged[model] = usage
        return UsageSummary(model_usage_summaries=merged)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
        return False


# =============================================================================
# Safe Builtins for REPL
# =============================================================================

_SAFE_BUILTINS = {
    # Core types
    "print": print, "len": len, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
    "type": type, "isinstance": isinstance, "issubclass": issubclass,
    "bytes": bytes, "bytearray": bytearray, "memoryview": memoryview,
    "complex": complex, "object": object, "super": super,
    
    # Iterators
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "range": range, "reversed": reversed, "sorted": sorted,
    "iter": iter, "next": next, "slice": slice,
    
    # Math
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "pow": pow, "divmod": divmod,
    
    # Sequences
    "all": all, "any": any,
    
    # String
    "chr": chr, "ord": ord, "repr": repr, "ascii": ascii,
    "format": format, "hash": hash, "id": id,
    
    # Attributes
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "delattr": delattr,
    "dir": dir, "vars": vars, "callable": callable,
    
    # Properties
    "property": property, "staticmethod": staticmethod, "classmethod": classmethod,
    
    # Allow controlled imports
    "__import__": __import__, "open": open,
    
    # Exceptions
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "FileNotFoundError": FileNotFoundError, "IOError": IOError,
    
    # Blocked (set to None)
    "eval": None, "exec": None, "compile": None, "input": None,
    "globals": None, "locals": None,
}


# =============================================================================
# REPL Environments
# =============================================================================

class BaseEnv(ABC):
    """Abstract base for REPL environments."""
    
    @abstractmethod
    def setup(self): pass
    
    @abstractmethod
    def load_context(self, context: Union[str, Dict, List]): pass
    
    @abstractmethod
    def execute_code(self, code: str) -> REPLResult: pass
    
    @abstractmethod
    def cleanup(self): pass


class LocalREPL(BaseEnv):
    """
    Local REPL environment with sandboxed execution.
    
    Provides:
    - Restricted builtins (no eval/exec/input)
    - Controlled module imports
    - llm_query and llm_query_batched functions
    - FINAL_VAR helper
    """
    
    ALLOWED_MODULES = frozenset({
        "re", "json", "math", "collections", "itertools",
        "functools", "operator", "string", "textwrap",
    })
    
    def __init__(
        self,
        lm_handler_address: Tuple[str, int] = None,
        context_payload: Union[str, Dict, List] = None,
        setup_code: str = None,
        **kwargs
    ):
        self.lm_handler_address = lm_handler_address
        self.temp_dir = tempfile.mkdtemp(prefix=f"rlm_repl_{uuid.uuid4().hex[:8]}_")
        self._lock = threading.Lock()
        self._pending_llm_calls: List[RLMChatCompletion] = []
        
        self.setup()
        if context_payload is not None:
            self.load_context(context_payload)
        if setup_code:
            self.execute_code(setup_code)
    
    def setup(self):
        """Initialize the execution environment."""
        self.globals: Dict[str, Any] = {
            "__builtins__": self._create_safe_builtins(),
            "__name__": "__main__",
        }
        self.locals: Dict[str, Any] = {}
        
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
    
    def _create_safe_builtins(self) -> Dict:
        safe = _SAFE_BUILTINS.copy()
        allowed = self.ALLOWED_MODULES
        original_import = __import__
        
        def restricted_import(name, *args, **kwargs):
            if name.split(".")[0] not in allowed:
                raise ImportError(f"Import of '{name}' not allowed. Allowed: {allowed}")
            return original_import(name, *args, **kwargs)
        
        safe["__import__"] = restricted_import
        return safe
    
    def _final_var(self, variable_name: str) -> str:
        """Return variable value as final answer."""
        name = variable_name.strip().strip("\"'")
        if name in self.locals:
            return str(self.locals[name])
        return f"Error: Variable '{name}' not found"
    
    def _llm_query(self, prompt: str, model: str = None) -> str:
        """Query sub-LLM via socket."""
        if not self.lm_handler_address:
            return "Error: No LM handler configured"
        try:
            request = LMRequest(prompt=prompt, model=model)
            response = send_lm_request(self.lm_handler_address, request)
            if not response.success:
                return f"Error: {response.error}"
            self._pending_llm_calls.append(response.chat_completion)
            return response.chat_completion.response
        except Exception as e:
            return f"Error: {e}"
    
    def _llm_query_batched(self, prompts: List[str], model: str = None) -> List[str]:
        """Query sub-LLM with multiple prompts concurrently."""
        if not self.lm_handler_address:
            return [f"Error: No LM handler configured"] * len(prompts)
        try:
            responses = send_lm_request_batched(self.lm_handler_address, prompts, model)
            results = []
            for resp in responses:
                if not resp.success:
                    results.append(f"Error: {resp.error}")
                else:
                    self._pending_llm_calls.append(resp.chat_completion)
                    results.append(resp.chat_completion.response)
            return results
        except Exception as e:
            return [f"Error: {e}"] * len(prompts)
    
    def load_context(self, context: Union[str, Dict, List]):
        """Load context into the environment."""
        if isinstance(context, str):
            path = os.path.join(self.temp_dir, "context.txt")
            with open(path, "w") as f:
                f.write(context)
            self.execute_code(f"with open(r'{path}', 'r') as f:\n    context = f.read()")
        else:
            path = os.path.join(self.temp_dir, "context.json")
            with open(path, "w") as f:
                json.dump(context, f)
            self.execute_code(f"import json\nwith open(r'{path}', 'r') as f:\n    context = json.load(f)")
    
    @contextmanager
    def _capture_output(self):
        """Capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
    
    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the sandboxed environment."""
        start = time.perf_counter()
        self._pending_llm_calls = []
        
        with self._capture_output() as (stdout_buf, stderr_buf):
            try:
                combined = {**self.globals, **self.locals}
                exec(code, combined, combined)
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"
        
        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start,
            llm_calls=self._pending_llm_calls.copy(),
        )
    
    def cleanup(self):
        """Clean up resources."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
        self.globals.clear()
        self.locals.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()
        return False


def get_environment(env_type: EnvironmentType, env_kwargs: Dict[str, Any]) -> BaseEnv:
    """Factory to create environment."""
    if env_type == "local":
        return LocalREPL(**env_kwargs)
    elif env_type == "docker":
        # Docker implementation would go here
        raise NotImplementedError("Docker environment requires additional setup")
    else:
        raise ValueError(f"Unknown environment: {env_type}")


# =============================================================================
# Logging
# =============================================================================

class RLMLogger:
    """Logger for RLM trajectories."""
    
    def __init__(self, log_dir: str = "./logs", file_name: str = "rlm"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = uuid.uuid4().hex[:8]
        self.log_file = self.log_dir / f"{file_name}_{timestamp}_{run_id}.jsonl"
        self._iteration_count = 0
        self._metadata_logged = False
    
    def log_metadata(self, metadata: RLMMetadata):
        """Log RLM configuration metadata."""
        if self._metadata_logged:
            return
        entry = {"type": "metadata", "timestamp": datetime.now().isoformat(), **metadata.to_dict()}
        with open(self.log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        self._metadata_logged = True
    
    def log(self, iteration: RLMIteration):
        """Log an iteration."""
        self._iteration_count += 1
        entry = {"type": "iteration", "iteration": self._iteration_count, "timestamp": datetime.now().isoformat(), **iteration.to_dict()}
        with open(self.log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")


# =============================================================================
# Verbose Printer
# =============================================================================

class VerbosePrinter:
    """Pretty console output for RLM execution."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and RICH_AVAILABLE
    
    def print_metadata(self, metadata: RLMMetadata):
        if not self.enabled:
            return
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Backend", metadata.backend)
        table.add_row("Model", metadata.root_model)
        table.add_row("Environment", metadata.environment_type)
        table.add_row("Max Iterations", str(metadata.max_iterations))
        console.print(Panel(table, title="[bold blue]RLM Configuration[/bold blue]"))
    
    def print_iteration(self, iteration: RLMIteration, num: int):
        if not self.enabled:
            return
        console.print(Rule(f"[bold]Iteration {num}[/bold]", style="blue"))
        preview = iteration.response[:500] + "..." if len(iteration.response) > 500 else iteration.response
        console.print(Panel(preview, title="[cyan]LLM Response[/cyan]", border_style="dim"))
        for i, cb in enumerate(iteration.code_blocks):
            console.print(f"\n[green]Code Block {i+1}:[/green]")
            console.print(Panel(cb.code, border_style="green"))
            if cb.result.stdout:
                console.print(f"[yellow]Output:[/yellow] {cb.result.stdout[:300]}")
            if cb.result.stderr:
                console.print(f"[red]Error:[/red] {cb.result.stderr[:300]}")
    
    def print_final_answer(self, answer: str):
        if not self.enabled:
            return
        console.print()
        console.print(Panel(answer, title="[bold green]FINAL ANSWER[/bold green]", border_style="green"))
    
    def print_summary(self, iterations: int, time_sec: float, usage: Dict):
        if not self.enabled:
            return
        console.print(Rule("[bold]Summary[/bold]", style="blue"))
        console.print(f"Iterations: {iterations}")
        console.print(f"Total Time: {time_sec:.2f}s")
        if usage.get("model_usage_summaries"):
            for model, stats in usage["model_usage_summaries"].items():
                console.print(f"{model}: {stats['total_calls']} calls, {stats['total_input_tokens']} in, {stats['total_output_tokens']} out")


# =============================================================================
# Parsing Utilities
# =============================================================================

def find_code_blocks(text: str) -> List[str]:
    """Extract ```repl code blocks from text."""
    pattern = r"```repl\s*\n(.*?)\n```"
    return [m.strip() for m in re.findall(pattern, text, re.DOTALL)]


def find_final_answer(text: str, environment: BaseEnv = None) -> Optional[str]:
    """Find FINAL() or FINAL_VAR() in response."""
    # Check FINAL_VAR first
    match = re.search(r"^\s*FINAL_VAR\((.*?)\)", text, re.MULTILINE | re.DOTALL)
    if match:
        var_name = match.group(1).strip().strip("\"'")
        if environment:
            result = environment.execute_code(f"print(FINAL_VAR({var_name!r}))")
            return result.stdout.strip() or result.stderr.strip()
        return None
    
    # Check FINAL
    match = re.search(r"^\s*FINAL\((.*?)\)", text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None


def format_iteration(iteration: RLMIteration, max_chars: int = 20000) -> List[Dict]:
    """Format iteration for conversation history."""
    messages = [{"role": "assistant", "content": iteration.response}]
    
    for cb in iteration.code_blocks:
        result_text = cb.result.stdout
        if cb.result.stderr:
            result_text += f"\nError: {cb.result.stderr}"
        
        if len(result_text) > max_chars:
            result_text = result_text[:max_chars] + f"... [{len(result_text) - max_chars} chars truncated]"
        
        messages.append({
            "role": "user",
            "content": f"Code executed:\n```python\n{cb.code}\n```\n\nREPL output:\n{result_text or 'No output'}",
        })
    
    return messages


# =============================================================================
# Query Metadata
# =============================================================================

@dataclass
class QueryMetadata:
    """Metadata about the input context."""
    context_lengths: List[int]
    context_total_length: int
    context_type: str
    
    def __init__(self, prompt: Union[str, List, Dict]):
        if isinstance(prompt, str):
            self.context_lengths = [len(prompt)]
            self.context_type = "str"
        elif isinstance(prompt, list):
            self.context_type = "list"
            self.context_lengths = [len(str(c)) for c in prompt]
        elif isinstance(prompt, dict):
            self.context_type = "dict"
            self.context_lengths = [len(json.dumps(v, default=str)) for v in prompt.values()]
        else:
            self.context_lengths = [0]
            self.context_type = "unknown"
        self.context_total_length = sum(self.context_lengths)


# =============================================================================
# Main RLM Class
# =============================================================================

class RLM:
    """
    Recursive Language Model.
    
    Replaces standard llm.completion(prompt, model) with rlm.completion(prompt, model)
    to handle near-infinite length contexts through programmatic decomposition.
    
    Example:
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "gpt-5.2"})
        result = rlm.completion("Your very long context here", root_prompt="Summarize")
        print(result.response)
    """
    
    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: Dict[str, Any] = None,
        environment: EnvironmentType = "local",
        environment_kwargs: Dict[str, Any] = None,
        max_depth: int = 1,
        max_iterations: int = 30,
        custom_system_prompt: str = None,
        other_backends: List[ClientBackend] = None,
        other_backend_kwargs: List[Dict[str, Any]] = None,
        logger: RLMLogger = None,
        verbose: bool = False,
    ):
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.environment_type = environment
        self.environment_kwargs = environment_kwargs.copy() if environment_kwargs else {}
        self.other_backends = other_backends
        self.other_backend_kwargs = other_backend_kwargs
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.system_prompt = custom_system_prompt or RLM_SYSTEM_PROMPT
        self.logger = logger
        self.verbose = VerbosePrinter(enabled=verbose)
        
        # Log metadata
        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=self.backend_kwargs.get("model_name", DEFAULT_MODELS.get(backend, {}).get("root", "unknown")),
                max_depth=max_depth,
                max_iterations=max_iterations,
                backend=backend,
                backend_kwargs={k: v for k, v in self.backend_kwargs.items() if "key" not in k.lower()},
                environment_type=environment,
                environment_kwargs=self.environment_kwargs,
                other_backends=other_backends,
            )
            if self.logger:
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)
    
    @contextmanager
    def _spawn_completion_context(self, prompt: Union[str, Dict, List]):
        """Create LM handler and environment for a completion."""
        # Create primary client
        client = get_client(self.backend, self.backend_kwargs)
        lm_handler = LMHandler(client)
        
        # Register sub-model for recursive calls
        if "sub" in DEFAULT_MODELS.get(self.backend, {}):
            sub_kwargs = self.backend_kwargs.copy()
            sub_kwargs["model_name"] = DEFAULT_MODELS[self.backend]["sub"]
            sub_client = get_client(self.backend, sub_kwargs)
            lm_handler.register_client(sub_client.model_name, sub_client)
        
        # Register other backends
        if self.other_backends and self.other_backend_kwargs:
            for backend, kwargs in zip(self.other_backends, self.other_backend_kwargs):
                other_client = get_client(backend, kwargs)
                lm_handler.register_client(other_client.model_name, other_client)
        
        lm_handler.start()
        
        # Create environment
        env_kwargs = self.environment_kwargs.copy()
        env_kwargs["lm_handler_address"] = lm_handler.address
        env_kwargs["context_payload"] = prompt
        
        environment = get_environment(self.environment_type, env_kwargs)
        
        try:
            yield lm_handler, environment
        finally:
            lm_handler.stop()
            if hasattr(environment, "cleanup"):
                environment.cleanup()
    
    def _build_system_prompt(self, prompt: Union[str, Dict, List]) -> List[Dict]:
        """Build initial message history with system prompt."""
        metadata = QueryMetadata(prompt)
        
        # Truncate length display if too long
        lengths = metadata.context_lengths
        if len(lengths) > 100:
            lengths = str(lengths[:100]) + f"... [{len(lengths) - 100} more]"
        
        context_info = f"Your context is a {metadata.context_type} with {metadata.context_total_length} total characters."
        
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "assistant", "content": context_info},
        ]
    
    def _build_user_prompt(self, root_prompt: str = None, iteration: int = 0) -> Dict:
        """Build the user prompt for each iteration."""
        base = "Think step-by-step on what to do using the REPL environment (which contains the context)"
        
        if root_prompt:
            base += f' to answer the original prompt: "{root_prompt}"'
        
        base += ".\n\nContinue using the REPL environment and querying sub-LLMs. Your next action:"
        
        if iteration == 0:
            prefix = "You have not interacted with the REPL yet. Look through the context first.\n\n"
            return {"role": "user", "content": prefix + base}
        else:
            prefix = "The above shows your previous interactions.\n\n"
            return {"role": "user", "content": prefix + base}
    
    def completion(
        self, 
        prompt: Union[str, Dict, List], 
        root_prompt: str = None
    ) -> RLMChatCompletion:
        """
        Run RLM completion on the given context.
        
        Args:
            prompt: The input context (string, list, or dict)
            root_prompt: Optional short prompt visible to root LLM
            
        Returns:
            RLMChatCompletion with the final answer
        """
        start_time = time.perf_counter()
        
        with self._spawn_completion_context(prompt) as (lm_handler, environment):
            message_history = self._build_system_prompt(prompt)
            
            for i in range(self.max_iterations):
                # Build current prompt
                current_prompt = message_history + [self._build_user_prompt(root_prompt, i)]
                
                # Get LLM response
                iter_start = time.perf_counter()
                response = lm_handler.completion(current_prompt)
                
                # Extract and execute code blocks
                code_block_strs = find_code_blocks(response)
                code_blocks = []
                
                for code_str in code_block_strs:
                    result = environment.execute_code(code_str)
                    code_blocks.append(CodeBlock(code=code_str, result=result))
                
                iteration = RLMIteration(
                    prompt=current_prompt,
                    response=response,
                    code_blocks=code_blocks,
                    iteration_time=time.perf_counter() - iter_start,
                )
                
                # Check for final answer
                final_answer = find_final_answer(response, environment)
                iteration.final_answer = final_answer
                
                # Log
                if self.logger:
                    self.logger.log(iteration)
                
                self.verbose.print_iteration(iteration, i + 1)
                
                if final_answer:
                    elapsed = time.perf_counter() - start_time
                    usage = lm_handler.get_usage_summary()
                    
                    self.verbose.print_final_answer(final_answer)
                    self.verbose.print_summary(i + 1, elapsed, usage.to_dict())
                    
                    return RLMChatCompletion(
                        root_model=self.backend_kwargs.get("model_name", DEFAULT_MODELS.get(self.backend, {}).get("root", "unknown")),
                        prompt=prompt,
                        response=final_answer,
                        usage_summary=usage,
                        execution_time=elapsed,
                    )
                
                # Update message history
                message_history.extend(format_iteration(iteration))
            
            # Max iterations reached - try to get an answer
            elapsed = time.perf_counter() - start_time
            fallback_prompt = message_history + [{
                "role": "user",
                "content": "You've reached max iterations. Provide a final answer now based on what you've learned.",
            }]
            final_response = lm_handler.completion(fallback_prompt)
            usage = lm_handler.get_usage_summary()
            
            self.verbose.print_final_answer(f"[MAX ITERATIONS] {final_response[:500]}...")
            self.verbose.print_summary(self.max_iterations, elapsed, usage.to_dict())
            
            return RLMChatCompletion(
                root_model=self.backend_kwargs.get("model_name", "unknown"),
                prompt=prompt,
                response=final_response,
                usage_summary=usage,
                execution_time=elapsed,
            )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recursive Language Model - Process near-infinite contexts with LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rlm_final.py --provider openai --context doc.txt --query "Summarize"
  python rlm_final.py --provider anthropic --max-iter 50 --verbose --query "Find X"
  cat huge_file.txt | python rlm_final.py --provider gemini --query "Extract facts"
"""
    )
    
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default="openai")
    parser.add_argument("--context", type=str, help="Path to context file")
    parser.add_argument("--query", type=str, required=True, help="Query to answer")
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-dir", type=str, help="Save trajectory logs")
    
    args = parser.parse_args()
    
    # Load context
    if args.context:
        with open(args.context, "r") as f:
            context = f.read()
    elif not sys.stdin.isatty():
        context = sys.stdin.read()
    else:
        parser.error("Provide --context FILE or pipe via stdin")
    
    print(f"Context: {len(context)} characters")
    print(f"Provider: {args.provider}")
    print(f"Query: {args.query}")
    print()
    
    # Setup
    logger = RLMLogger(log_dir=args.log_dir) if args.log_dir else None
    
    rlm = RLM(
        backend=args.provider,
        backend_kwargs={"model_name": DEFAULT_MODELS[args.provider]["root"]},
        max_iterations=args.max_iter,
        logger=logger,
        verbose=args.verbose,
    )
    
    # Run
    try:
        result = rlm.completion(context, root_prompt=args.query)
        
        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)
        print(result.response)
        print()
        print(f"Execution time: {result.execution_time:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
