# Workbench

A minimal, completely local AI workbench: a terminal harness (pi-inspired) driving local
models served by `llama.cpp`, with a fast local classifier (`laya`) deciding intent and model
complexity per turn, running an autonomous plan → approve → execute → reflect agent loop over
sandboxed tools, with first-class local observability.

---

## 1. Key Principles & Constraints

1. **Strictly Offline Runtime:** Zero external API calls at runtime. All inference runs against localhost (`llama-server`). Network access is restricted exclusively to explicit `workbench models download` commands and the local Prometheus/Grafana containers. The runtime automatically enforces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` process-wide.
2. **llama.cpp Runtime via Router Mode:** Driven via `llama-serve --router --models-dir <store>`. A single server process manages loaded models; model switching is an in-request parameter rather than a process restart.
3. **Laya Intent & Complexity Routing:** Employs upstream PyTorch `laya` (ModernBERT-based) running natively on CUDA/CPU for fast (~30ms) batched intent classification, difficulty scoring (tiers: `small` vs `mid`), and tool requirement detection, backed by a deterministic heuristic router fallback.
4. **CLI & TUI Harness:** Interactive terminal interface built with `agentic-tui` (Rich + `prompt_toolkit`), scrollback-preserving, with inline streaming markdown, diffs, live step progress, and slash commands. Also supports headless scriptable execution via `-p/--print` and `--json` event streaming.
5. **Core Four Tools:** `read`, `write`, `edit`, `bash`.
6. **Multi-layer Sandbox:** Path jail (workspace boundary enforcement) + `bubblewrap` (OS namespace isolation, private tmp, dropped caps, network unshared) + command guardrail policy with interactive approval.
7. **Local Observability:** Pre-configured Prometheus and Grafana dashboards as code (`workbench obs up/down`) scraping latency, routing decisions, TTFT, token metrics, and llama-server internals.
8. **Pluggable Architecture:** Extension points for custom tools (`workbench.tools`), routers (`workbench.routers`), model providers (`workbench.providers`), sandboxes (`workbench.sandboxes`), and TUI slash commands (`workbench.commands`), loaded via Python entry points or `~/.config/workbench/plugins/*.py`.

---

## 2. Requirements & Installation

### Requirements

- **Linux x86_64** (kernel with user namespaces enabled for bubblewrap)
- **Python >= 3.14**
- **uv** package manager
- **bubblewrap** (`bwrap`) >= 0.8
- **llama.cpp** (`llama-server` / `llama-serve` wrapper on `PATH` or in `~/.local/bin`)
- **Docker** (optional, required for local Prometheus/Grafana stack)

### Setup

```bash
# Clone and enter directory
cd workbench

# Install dependencies with uv
uv sync --group dev

# Verify installation
uv run workbench --help
```

---

## 3. Quickstart

### Step 1: Configure Models

Create a project configuration `.workbench.toml` (or global `~/.config/workbench/config.toml`):

```toml
[models]
search_paths = ["~/models", "~/.local/share/workbench/models"]

[models.profiles.minicpm2b]
path = "~/models/MiniCPM5-2B-Q4_K_M.gguf"
ctx_len = 8192
tool_calling = "auto"

[models.profiles.coder7b]
path = "~/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
ctx_len = 8192
tool_calling = "auto"

[routing.tiers]
small = "minicpm2b"
mid = "coder7b"
```

### Step 2: Download Models & Warm Cache

Download models into your store. This is the **only** workbench command that touches the network:

```bash
# Download a GGUF from Hugging Face
uv run workbench models download hf:Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf

# Or pull via Ollama shim
uv run workbench models download ollama:qwen2.5-coder:7b

# Inspect store and configured profiles
uv run workbench models list
```

The download command also prefetches the `laya` classifier checkpoint so the entire runtime stays completely offline.

### Step 3: Run the Interactive TUI

```bash
uv run workbench
```

Inside the TUI:
- Type your prompt. The router classifies intent and selects the optimal model tier.
- If planning is suggested, a numbered step plan is displayed for interactive approval (`y`/`N` or feedback).
- File operations render inline syntax-highlighted unified diffs.
- Slash commands are available:
  - `/model <auto|small|mid|profile>`: Inspect or pin model tier.
  - `/models`: List configured model profiles and context lengths.
  - `/plan`: View current plan steps and statuses.
  - `/router`: Explain the last routing decision and confidence.
  - `/stats`: Display session metrics (rounds, tokens, tool calls).
  - `/sessions` & `/resume <id>`: Manage and resume previous sessions.
  - `/clear`: Clear terminal viewport.
  - `/quit`: Exit workbench.

### Step 4: Headless Execution (Scripting & Automation)

```bash
# Print answer to stdout
uv run workbench -p "Write a python script in test.py that calculates fibonacci"

# Stream structured NDJSON events
uv run workbench -p "Explain quantum computing" --json
```

---

## 4. Architecture & Subsystems

```
workbench/
├── pyproject.toml              # uv project; scripts, entry-point groups
├── src/workbench/
│   ├── cli.py                  # CLI dispatcher: TUI, -p/--print, --json, subcommands
│   ├── config.py               # Layered TOML config -> typed frozen Config model
│   ├── logging_setup.py        # Central JSONL rotating file + console logging
│   ├── core/
│   │   ├── registry.py         # Plugin discovery (entry points + ~/.config/workbench/plugins/*.py)
│   │   ├── bootstrap.py        # Process bootstrap: tools, routers, sandboxes, providers
│   │   ├── events.py           # In-process typed event bus
│   │   ├── session.py          # Chat history, plan state, JSONL persistence
│   │   └── protocols.py        # Protocol definitions (Router, LLMClient, Tool, Sandbox, ModelProvider)
│   ├── llm/
│   │   ├── client.py           # Streaming OpenAI-compatible HTTP client
│   │   └── server.py           # llama-serve lifecycle manager (lazy spawn, health polling, idle-stop)
│   ├── routing/
│   │   ├── laya_router.py      # Upstream PyTorch/CUDA laya classification backend
│   │   ├── heuristic.py        # Fast deterministic fallback router
│   │   └── policy.py           # Tier policy: thresholds, hysteresis, failure escalation
│   ├── agent/
│   │   ├── loop.py             # Plan -> approve -> execute -> reflect state machine
│   │   ├── planning.py         # Structured step-plan generation and parser
│   │   └── json_calls.py       # JSON tool-calling fallback parser
│   ├── tools/
│   │   ├── files.py            # read, write, edit tools
│   │   ├── bash.py             # bash execution via sandbox backend
│   │   ├── jail.py             # PathJail workspace boundary enforcement
│   │   └── guardrails.py       # Dangerous command filters and interactive confirmation
│   ├── sandbox/
│   │   └── bwrap.py            # Bubblewrap OS namespace isolation
│   ├── models/
│   │   ├── store.py            # GGUF search paths and index.json catalog
│   │   ├── download.py         # Download orchestration & laya prefetch
│   │   ├── huggingface.py      # HuggingFace Hub snapshot provider
│   │   ├── ollama.py           # Ollama content-addressed blob shim
│   │   └── offline.py          # Process-wide offline enforcement guard
│   ├── tui/
│   │   ├── app.py              # agentic-tui main loop & event bridge
│   │   ├── transcript.py       # Message, plan, tool diff, and error renderers
│   │   ├── commands.py         # Built-in and plugin slash command dispatch
│   │   └── status.py           # Context, tier, rounds, and token status bar
│   └── observability/
│       ├── metrics.py          # Prometheus metrics definitions
│       ├── server.py           # Background exposition HTTP server (:9600)
│       └── stack.py            # docker-compose orchestration (obs up/down)
├── docker/
│   ├── observability/          # docker-compose.yml, prometheus.yml
│   └── grafana/                # Grafana provisioning & pre-built JSON dashboards
├── plugins/
│   └── examples/               # Reference plugin implementations for each extension point
└── tests/                      # Unit, integration, e2e, regression test suite
```

---

## 5. Local Observability Stack

Launch the local Prometheus + Grafana stack:

```bash
# Start Prometheus (:9090) and Grafana (:3000)
uv run workbench obs up

# Open Grafana in your browser: http://localhost:3000 (admin / admin)
# Dashboards are auto-provisioned:
#   - Overview: TTFT, turn latency, token throughput, active tier
#   - Routing: Tier distribution, intent breakdown, fallback & escalation rate
#   - Agent & Tools: Execution rounds, tool latency, guardrail block reasons
#   - Server: llama-server memory, slots, processing metrics

# Stop the stack
uv run workbench obs down
```

---

## 6. Plugins & Extensibility

Workbench provides 5 extension point groups (PLAN §3):

| Group | Interface | Purpose |
|---|---|---|
| `workbench.tools` | `Tool` protocol | Register custom tools available to the agent loop |
| `workbench.routers` | `Router` protocol | Custom classification backends (e.g., ONNX, heuristic) |
| `workbench.providers` | `ModelProvider` protocol | Download providers (custom repositories, local mirrors) |
| `workbench.sandboxes` | `Sandbox` protocol | Alternative execution environments (Docker, custom jails) |
| `workbench.commands` | `async (ui, args) -> None` | Extra slash commands in the interactive TUI |

Check [plugins/examples/](file:///home/ojas/projects/workbench/plugins/examples/) for complete, working sample implementations of each extension point.

---

## 7. Testing & Quality Assurance

The codebase maintains strict type annotations (`mypy` strict) and comprehensive test coverage across unit, integration, pty/TUI, sandboxed execution, and regression tests.

```bash
# Run complete test suite (unit, integration, pty, regressions)
uv run pytest

# Run CI-safe tests (excluding local GPU model weights and OS bubblewrap execution)
uv run pytest -m "not model and not sandbox"

# Linting and formatting checks
uv run ruff check src tests plugins

# Static type checking
uv run mypy
```
