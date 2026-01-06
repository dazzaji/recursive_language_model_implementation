# RLM Project: Complete Development Sprint Plan

Premised on implementation_03.py in root.

## 🎯 PROJECT NORTH STAR

### What We're Building
**Recursive Language Models (RLM)** - A production-grade Python library that enables Large Language Models to process near-infinite length contexts by treating prompts as external environment objects that the LLM can programmatically examine, decompose, and recursively call itself on.

### Why We're Building This
Traditional LLMs are constrained by fixed context windows (even 1M token models hit practical limits). RLM breaks this barrier by enabling:
- **Unlimited context processing** - Handle 100M+ character inputs
- **Intelligent decomposition** - LLM decides how to chunk and process
- **Recursive analysis** - Sub-LLM calls for semantic tasks
- **Better results** - 20-500x+ improvement on information-dense tasks (per paper benchmarks)

### How It Works
1. Context is loaded into a REPL environment as a `context` variable
2. Root LLM examines the context, plans a strategy, and writes Python code
3. Code executes in sandboxed REPL, can call `llm_query()` for semantic analysis
4. Results are captured and fed back to root LLM
5. Process iterates until LLM signals `FINAL(answer)` or `FINAL_VAR(variable)`

### What It's Good For
- **Document analysis** - Legal contracts, research papers, codebases
- **Information extraction** - Finding needles in haystacks across thousands of documents
- **Comparison tasks** - Analyzing multiple sources for discrepancies
- **Long-form summarization** - Books, reports, meeting transcripts
- **Code understanding** - Large codebases beyond context window limits

### Expected Results
- **S-NIAH benchmark**: ~95% accuracy on single-needle retrieval
- **OOLONG benchmark**: 3-4x improvement over base LLM
- **OOLONG-Pairs**: 500x+ improvement (0.04% → 58% with GPT-5)
- **Cost**: Comparable median, higher variance tails (mitigation: sub-model optimization)

### How to Measure Against Other Approaches
1. **Baseline**: Standard LLM with truncation
2. **RAG**: Retrieval-augmented generation with vector search
3. **Sliding window**: Sequential context processing
4. **Map-reduce**: Parallel chunk processing with aggregation

**Metrics:**
- Accuracy on benchmark tasks (S-NIAH, OOLONG)
- Total token usage / cost
- Latency (time to answer)
- Trajectory length (iterations to converge)

### How to Use and Extend
```python
# Basic usage
from rlm import RLM
rlm = RLM(backend="openai", backend_kwargs={"model_name": "gpt-5.2"})
result = rlm.completion(massive_document, root_prompt="Find all mentions of X")
print(result.response)

# Custom environment (Docker for isolation)
rlm = RLM(
    backend="anthropic",
    environment="docker",
    environment_kwargs={"image": "my-custom-image"},
    verbose=True,
)

# Custom system prompt
rlm = RLM(
    backend="gemini",
    custom_system_prompt="You are a legal document analyst...",
)
```

---

## 📁 Project Structure

```
rlm/
├── .github/workflows/
│   ├── test.yml
│   ├── style.yml
│   └── publish.yml
├── docs/
│   ├── getting-started.md
│   ├── api-reference.md
│   └── environments.md
├── examples/
│   ├── quickstart.py
│   ├── docker_example.py
│   └── multi_provider.py
├── rlm/
│   ├── __init__.py
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── base_lm.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── gemini.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── rlm.py
│   │   ├── types.py
│   │   ├── lm_handler.py
│   │   └── comms_utils.py
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── base_env.py
│   │   ├── local_repl.py
│   │   └── docker_repl.py
│   ├── logger/
│   │   ├── __init__.py
│   │   ├── rlm_logger.py
│   │   └── verbose.py
│   └── utils/
│       ├── __init__.py
│       ├── parsing.py
│       └── prompts.py
├── tests/
│   ├── conftest.py
│   ├── test_types.py
│   ├── test_comms.py
│   ├── test_clients.py
│   ├── test_lm_handler.py
│   ├── test_local_repl.py
│   ├── test_parsing.py
│   ├── test_rlm.py
│   └── integration/
│       └── test_e2e.py
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 🏃 SPRINT BREAKDOWN

### Sprint 0: Project Setup (Day 1)
**Goal:** Initialize repository with all configuration files

#### Files to Create:

**`.gitignore`**
```gitignore
__pycache__/
*.py[cod]
.env
.venv/
venv/
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
logs/
*.jsonl
.DS_Store
```

**`.env.example`**
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

**`pyproject.toml`**
```toml
[project]
name = "rlm"
version = "2.0.0"
description = "Recursive Language Models for near-infinite context processing"
requires-python = ">=3.11"
dependencies = [
    "openai>=2.14.0",
    "anthropic>=0.75.0",
    "google-genai>=1.56.0",
    "rich>=13.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.24.0", "ruff>=0.4.0", "pre-commit>=3.5.0"]
docker = ["dill>=0.3.7"]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

#### Definition of Done:
- [ ] Git repository initialized
- [ ] All config files created
- [ ] `pip install -e ".[dev]"` succeeds
- [ ] `pytest` runs (even with no tests)

---

### Sprint 1: Core Types (Day 2)
**Goal:** Define all data types

#### Key Files:
- `rlm/core/types.py` - All dataclasses
- `rlm/__init__.py` - Package exports

#### Tests:
```python
# tests/test_types.py
def test_model_usage_summary_round_trip():
    usage = ModelUsageSummary(5, 1000, 500)
    d = usage.to_dict()
    restored = ModelUsageSummary.from_dict(d)
    assert restored.total_calls == 5

def test_rlm_chat_completion():
    completion = RLMChatCompletion(...)
    assert completion.to_dict()["root_model"] == "gpt-5.2"
```

#### Definition of Done:
- [ ] All dataclasses implemented
- [ ] to_dict/from_dict working
- [ ] Tests pass

---

### Sprint 2: Socket Protocol (Day 3)
**Goal:** TCP communication for LM Handler

#### Key Files:
- `rlm/core/comms_utils.py`

#### Tests:
```python
def test_socket_send_recv():
    # Server/client round-trip test
```

---

### Sprint 3: LM Clients (Day 4-5)
**Goal:** Provider abstraction

#### Key Files:
- `rlm/clients/base_lm.py`
- `rlm/clients/openai.py`
- `rlm/clients/anthropic.py`
- `rlm/clients/gemini.py`

#### Tests:
```python
@patch("rlm.clients.openai.OpenAI")
def test_openai_completion(mock_openai):
    # Mock and test
```

---

### Sprint 4: LM Handler (Day 6)
**Goal:** Multi-threaded socket server

#### Key Files:
- `rlm/core/lm_handler.py`

#### Tests:
```python
def test_handler_single_request():
    with LMHandler(MockLM()) as handler:
        # Socket request test
```

---

### Sprint 5: Local REPL (Day 7-8)
**Goal:** Sandboxed Python execution

#### Key Files:
- `rlm/environments/local_repl.py`

#### Tests:
```python
def test_execution():
    repl = LocalREPL()
    result = repl.execute_code("x = 1 + 2")
    assert repl.locals["x"] == 3

def test_llm_query():
    with LMHandler(MockLM()) as handler:
        repl = LocalREPL(lm_handler_address=handler.address)
        repl.execute_code("r = llm_query('test')")
        assert "Mock" in repl.locals["r"]
```

---

### Sprint 6: Parsing & Prompts (Day 9)
**Goal:** Code extraction, system prompts

#### Key Files:
- `rlm/utils/parsing.py`
- `rlm/utils/prompts.py`

#### Tests:
```python
def test_find_code_blocks():
    text = "```repl\nx=1\n```"
    blocks = find_code_blocks(text)
    assert "x=1" in blocks[0]
```

---

### Sprint 7: Main RLM Class (Day 10-11)
**Goal:** Core orchestration

#### Key Files:
- `rlm/core/rlm.py`

#### Tests:
```python
def test_completion_returns_final():
    rlm = RLM(backend="mock")
    result = rlm.completion("context", "query")
    assert result.response is not None
```

---

### Sprint 8: Logging (Day 12)
**Goal:** Trajectory logging, verbose output

#### Key Files:
- `rlm/logger/rlm_logger.py`
- `rlm/logger/verbose.py`

---

### Sprint 9: CLI (Day 13)
**Goal:** Command-line interface

#### Key Files:
- `rlm/cli.py`

---

### Sprint 10: Docker Environment (Day 14)
**Goal:** Isolated execution

#### Key Files:
- `rlm/environments/docker_repl.py`

---

### Sprint 11: Integration Tests (Day 15)
**Goal:** E2E testing

---

### Sprint 12: Documentation (Day 16-17)
**Goal:** README, docs, examples

---

### Sprint 13: CI/CD (Day 18)
**Goal:** GitHub Actions, PyPI

---

## 📊 Test Summary

| Sprint | Tests |
|--------|-------|
| 1 | test_types.py |
| 2 | test_comms.py |
| 3 | test_clients.py |
| 4 | test_lm_handler.py |
| 5 | test_local_repl.py |
| 6 | test_parsing.py |
| 7 | test_rlm.py |
| 8 | test_logger.py |
| 11 | test_e2e.py |

**Target: 80+ tests, >80% coverage**

---

## 🚀 Quick Start After Sprints

```bash
git clone https://github.com/yourusername/rlm.git
cd rlm
pip install -e ".[dev]"
cp .env.example .env
# Add API keys
pytest tests/ -v
python -m rlm --provider openai --context doc.txt --query "Summarize"
```
