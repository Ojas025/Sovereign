# PLAN — Local AI Workbench (offline, llama.cpp + laya, pi-like TUI harness)

**Status:** finalized design, ready for implementation
**Target location:** repo root `~/projects/workbench/PLAN.md` (written to the plan directory while in plan mode; copy to repo root on approval / mode switch)
**Repo:** `~/projects/workbench` (greenfield) · **Branch:** `main` · **Commits:** plain messages, no prefixes

---

## 1. Vision & constraints

A minimal, completely local AI workbench: a terminal harness (pi-inspired) driving local
models served by llama.cpp, with a fast local classifier deciding *intent* and *model
complexity* per turn, running a plan → approve → execute → reflect agent loop over a
sandboxed tool set, with first-class observability.

**Design-wide constraints (user-mandated):**

1. **Offline runtime.** Zero external API calls while running. Only localhost traffic
   (llama-server, our metrics port, local Prometheus/Grafana). The *only* network code lives in
   `models/download.py` (explicit `workbench models download` command) and the local
   Prometheus/Grafana containers. Runtime sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
   process-wide so stray hub calls fail fast instead of reaching the network.
2. **llama.cpp as the runtime** via the existing **`llama-serve` wrapper** — primarily in
   **`--router` mode** (`llama-server --models-dir <store>`): one long-lived server exposing
   all models; model switching is a per-request `model` field, not a process restart.
3. **laya (upstream, PyTorch)** for intent classification + complexity routing — runs
   natively on Linux/CUDA (RTX 3050 6GB). The MLX port (`laya-mlx`) is Apple-Silicon-oriented
   and was dropped in favor of the original package.
4. **CLI-only interface.** No HTTP API surface of our own beyond the localhost metrics port.
   TUI is the primary surface; `-p/--print` and `--json` event stream exist for scripting/tests.
5. **pi-like minimal harness** with exactly four tools: `read`, `write`, `edit`, `bash`.
6. **Model-agnostic harness.** Models are opaque profiles (GGUF path + runtime config);
   the user picks models. No model-specific code paths (a `tool_calling = auto|native|json|none`
   profile flag covers models without native tool templates).
7. **Sandboxed tool execution** with layered guardrails (path jail + bubblewrap + command policy).
8. **Local Prometheus + Grafana from day one**, plus central config and central logging.
9. **Core kernel + plugin registries** architecture (entry points), so tools/routers/providers/
   commands are extensible without touching core.
10. **Testing:** unit, integration, e2e, and regression tests written alongside each feature;
    no workarounds to make tests pass; every functionality hands-on validated before done.
11. **Git:** work directly on `main`, plain commit messages (`Initialize project`, not `feat(x): …`).

**Hardware/runtime facts:** Linux x86_64, 16 cores, 23GB RAM, RTX 3050 **6GB** (SM 86),
docker 29 (daemon up), `uv`, `hf`, `ollama` installed; `llama`/`llama-serve` in `~/.local/bin`
(llama.cpp b10200, server bin at `~/repos/llama.cpp/build/bin/llama-server`); models in
`~/models` (e.g. `MiniCPM5-2B-Q4_K_M.gguf` 1.5G, `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` 4.4G).

---

## 2. Decision log

| # | Decision | Choice |
|---|---|---|
| 1 | Implementation stack | **Python** (uv-managed, venv pinned to a torch/`agentic-tui`-compatible Python — verify ≥3.13) |
| 2 | Model serving | **Own manager around `llama-serve`, router mode primary** |
| 3 | "CLI harness endpoint" | **CLI is the only interface** (TUI + `-p`/`--json`) |
| 4 | Tool set | **pi core four: read / write / edit / bash** |
| 5 | Offline rule | **Offline runtime; network only on explicit download command** |
| 6 | Plan/reflect loop | **Plan → y/N human approval → autonomous execute/reflect loop** |
| 7 | Model tiers | **Two tiers: small + mid** (config-defined profile names) |
| 8 | GPU residency | **Lazy single-active** (start on demand, idle-stop) — provided by llama.cpp router mode |
| 9 | Observability | **Prometheus + Grafana local stack** (user-amended from generic OTel collector) |
| 10 | Downloads | **`huggingface_hub` lib + ollama CLI shim** |
| 11 | TUI | **`agentic-tui`** (Rich + prompt_toolkit, scrollback-preserving, CSI 2026) |
| 12 | Architecture | **B: core kernel + plugin registries** |
| 13 | Classification/routing engine | **Original `laya` package (PyTorch/CUDA)**, not laya-mlx |
| 14 | Bash sandbox | **bubblewrap** (verified working, 0.11.2) — Docker backend as optional plugin |
| 15 | Models | **User-chosen from `~/models`; harness stays model-agnostic** |

---

## 3. Architecture (Section 1)

```
workbench/
├── pyproject.toml              # uv; console script `workbench`; entry-point groups
├── src/workbench/
│   ├── cli.py                  # dispatch: default → TUI; -p/--print; --json;
│   │                           #   subcommands: models, obs, route (debug)
│   ├── config.py               # central config: layered TOML → one typed frozen model, get_config()
│   ├── logging_setup.py        # central logging: get_logger(), JSONL rotating file + console, correlation ids
│   ├── core/
│   │   ├── registry.py         # plugin discovery: entry points (workbench.*) + ~/.config/workbench/plugins/*.py
│   │   ├── events.py           # typed in-process event bus (pi-style vocabulary)
│   │   ├── session.py          # message/plan/reflect state model + JSONL persistence
│   │   └── protocols.py        # Protocol interfaces: Router, LLMClient, Tool, Sandbox, ModelProvider
│   ├── llm/                    # streaming OpenAI-compatible client → localhost only
│   ├── router/                 # laya backend, heuristic fallback, tier policy
│   ├── agent/                  # plan→approve→execute→reflect loop + tools/
│   ├── server/                 # llama-serve manager (lifecycle, health, metrics)
│   ├── models/                 # store/registry + download (hf lib, ollama shim, laya prefetch)
│   ├── tui/                    # agentic-tui wiring: transcript, blocks, status line, slash cmds
│   └── observability/          # prometheus instrumentation + exposition server + dashboards
├── docker/observability/docker-compose.yml   # Prometheus + Grafana
├── docker/grafana/dashboards/*.json          # dashboards as code
├── plugins/examples/           # sample plugins proving each extension point
└── tests/                      # unit / integration / e2e / regression (see §9)
```

**Plugin entry-point groups** (all optional; core ships defaults):

| Group | Extends | Defaults shipped |
|---|---|---|
| `workbench.tools` | tool set beyond the four | read, write, edit, bash |
| `workbench.routers` | routing backends | `laya`, `heuristic` |
| `workbench.providers` | download sources | `huggingface`, `ollama` |
| `workbench.sandboxes` | bash sandbox backends | `bwrap` |
| `workbench.commands` | extra TUI slash commands | — |

**Data flow (one turn):**

```
TUI input
  → Router (laya: intent + difficulty + needs_tools, batched, one forward pass
            + prior-phase signal + confidence gate + hysteresis + escalation)
  → tier policy → model profile
  → [plan phase if policy suggests] → y/N approval (in TUI)
  → agent loop: llm call ─HTTP→ llama-serve (localhost, model=<profile-id>)
                ⇄ sandboxed tools (bwrap / path jail)
                → reflect at plan-step checkpoints → correction or completion
  → events → TUI render + JSONL log + Prometheus metrics
```

---

## 4. Routing subsystem (Section 2)

### 4.1 Backend: `LayaRouter`

- Upstream `laya` package (PyTorch ≥2.0, CUDA on the 3050). Checkpoint configurable;
  default `laya` (English, ModernBERT 421M); `laya-multilingual` / `laya-typed-decisions` selectable.
- **One batched forward pass per turn** (~30-40ms; checkpoint kept resident for the session):

| Question | Type | Purpose |
|---|---|---|
| `intent` | choice | `chat_qa · file_edit · code_gen · shell_task · search_analysis · planning · meta` → plan-phase trigger + tool hints |
| `difficulty` | score (ordinal 0-3: trivial/easy/moderate/hard) | **complexity routing** between tiers |
| `needs_tools` | noul (P(true)) | feeds plan-phase decision |

- **State = current user message only** (transcript excluded — respects laya's context budget,
  avoids stale-context overconfidence).
- Prefetch checkpoint in `workbench models download`; runtime forces offline env vars.

### 4.2 Fallback: `HeuristicRouter`

Deterministic length/keyword/code-fence heuristic. Auto-used when laya is unavailable,
inference errors, or confidence < gate. Same `Router` Protocol → swappable via config or plugin.

### 4.3 Tier policy (`policy.py`) — research-updated

```
inputs:  laya difficulty + intent + needs_tools      (learned, ~30ms)
       + prior-phase (explore|implement|verify|none) (deterministic, from tool history,
                                                      2-consecutive-call stability rule)
       + consecutive tool-error count

rules:
  difficulty ≥ T_high (default 2.0)            → mid
  difficulty < T_high                          → small
  confidence < gate (default 0.6)              → config default tier (mid)   [default-to-larger]
  intent == planning OR needs_tools high       → mid + suggest plan phase
  ≥2 consecutive tool failures                 → escalate small→mid, ONCE per task (route_escalated)
  config rules[]: intent → tier overrides (e.g. intent=meta → small)

stability: keep current tier unless new decision clears threshold+δ or intent class changes
           (hysteresis — no tier flapping mid-session)
override:  /model small|mid|auto|<profile> pins tier; router advisory-only in pinned mode
```

- **Per-turn routing** with session affinity (above). Every decision logged as a structured
  record (input, decision, confidence, outcome) for later auditing/calibration.
- **Tier → thinking passthrough:** profile may carry `reasoning_effort`/sampler args sent to
  llama-server (ignored if unsupported — keeps harness model-agnostic).

### 4.4 Metrics

`route_decisions_total{intent,tier,backend,confidence_band}`,
`route_fallback_total`, `route_escalated_total`, `route_duration_seconds` (histogram).

### 4.5 Plugin seam

`workbench.routers` — laya + heuristic are core defaults; alternatives (e.g. tiny ONNX
classifier) drop in without core changes.

---

## 5. Agent loop, sandboxed tools, guardrails (Section 3)

### 5.1 Loop state machine

```
route → [plan phase] → y/N approval → EXECUTE: llm_call ⇄ sandboxed tools
                                          ↓ at each plan-step checkpoint
                                       REFLECT: on-track? next? done?
                                          ↓ no                      ↓ yes
                                    correct & continue        final summary → user
```

- **Plan phase** — triggered by `planning` intent / high `needs_tools` / difficulty ≥ T /
  explicit request. Model emits a structured plan (ordered steps). Rejection → free-text
  feedback → regenerate (max 3) → or cancel.
- **Execute** — standard turn loop; tool calls run in the sandbox; results appended; emits
  `turn_start → message_start/update/end → tool_execution_start/end → turn_end`.
- **Reflect** — at *plan-step boundaries* (not every tool call): bounded self-check
  (step done? aligned? plan obsolete?). Off-track → inject correction. Model claims done →
  **completion check vs plan steps**; unfinished → nudge (cap in config).
- **Budgets (config):** max tool rounds/task (default 15), per-turn token cap, wall-clock
  timeout, reflection-nudge cap (default 3). Breach → graceful stop + state summary.
- **Escalation hook:** ≥2 consecutive tool failures → policy may bump tier (§4.3).

### 5.2 Tools

| Tool | Semantics | Caps |
|---|---|---|
| `read(path, offset?, limit?)` | line-numbered file read | max bytes/call |
| `write(path, content)` | create/overwrite | max size |
| `edit(path, old_string, new_string, replace_all?)` | exact match, must be unique unless `replace_all` | — |
| `bash(command, timeout?)` | guarded shell | timeout (def 60s), output truncation (64KB) |

### 5.3 Sandbox layers

1. **Path jail (all file tools):** `realpath()` must resolve inside workspace root;
   `..`, symlink, absolute-path escapes rejected; writes workspace-only; extra read roots
   declared in config.
2. **bash sandbox — bubblewrap (default):** one-shot namespace jail per command —
   workspace bound rw, remainder ro/absent, `--unshare-net` (no network ⇒ offline enforced
   at OS level), `--die-with-parent`, private `/tmp`, no D-Bus/X11/Wayland, dropped caps,
   minimal env (no secrets). Optional Docker backend via `workbench.sandboxes` plugin.
3. **Guardrails (independent of sandbox):** deny-pattern filter (`rm -rf /`, `mkfs`,
   `dd of=/dev/*`, fork bombs, …), optional allowlist mode, **interactive TUI confirmation**
   for flagged-but-allowed commands, timeouts, output truncation, env scrubbing.
   Blocked/allowed outcomes emit `tool_blocked_total{reason}`.

---

## 6. Model management + llama-server manager (Section 4)

### 6.1 Model profiles (model-agnostic source of truth)

```toml
[models.profiles.minicpm2b]
path    = "~/models/MiniCPM5-2B-Q4_K_M.gguf"
ctx_len = 8192
n_gpu_layers = "auto"        # computed from GGUF size vs free VRAM (nvidia-smi), KV reserve subtracted
tool_calling  = "auto"       # auto | native | json | none
reasoning_effort = ""        # optional passthrough

[models.profiles.coder7b]
path    = "~/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
ctx_len = 8192
n_gpu_layers = "auto"
tool_calling  = "auto"

[routing.tiers]
small = "minicpm2b"
mid   = "coder7b"
```

- `tool_calling` is the model-agnosticism guarantee: `native` → llama-server `tools` param;
  `json` → harness embeds a JSON tool-call protocol in prompts and parses structured output;
  `none` → chat only. `auto` probes GGUF metadata.
- **Store:** configurable search paths (default `~/models`, `~/.local/share/workbench/models`),
  auto-scanned `index.json` registry (name, size, sha, quant, family).
  `workbench models list` shows profiles ↔ tiers ↔ store entries.

### 6.2 Downloads — the only network code

`workbench models download <spec>`:
- **HF:** `huggingface_hub` snapshot with `*.gguf` filter, resume + checksum, into store.
- **Ollama:** `ollama pull <name>` subprocess → resolve layer blob from
  `~/.ollama/models/manifests` → register/symlink GGUF blob in our store.
- Same command **prefetches the laya checkpoint**, then the runtime stays offline.

### 6.3 llama-serve manager

- Wraps the existing `llama-serve` wrapper; **primary: `llama-serve --router
  --models-dir <store> --port <cfg>`** — one long-lived child exposing all models.
- **Model switching = request field** (`model: "<profile-id>"`) — llama.cpp router mode
  loads/switches models from the dir; the router can switch between *any* store models,
  not just the two tiers.
- Lifecycle: lazy start on first LLM call → poll `/health` + `/v1/models` until registered →
  warm → **idle-stop** (default 300s) → SIGTERM → VRAM freed.
- Crash resilience: auto-restart, backoff, max 3, then surface to TUI.
- Profile ↔ llama.cpp model-id registry via `/v1/models` scan.
- Metrics: `server_starts_total`, `server_ready_seconds`, `server_active`,
  `server_restarts_total`; llama-server's own `/metrics` scraped as a second target.

**Spike items (M2):**
1. Router-mode **VRAM eviction** on 6GB when switching models (must honor single-active
   budget). Fallback: manager-driven single-model mode (`llama-serve <model>`) with
   restart-on-switch, behind the same interface.
2. Router mode's support for **per-model server args** (ctx_len, n_gpu_layers) —
   global flags vs per-model config files.

---

## 7. TUI harness (Section 5)

**Framework:** `agentic-tui` (Rich + prompt_toolkit; scrollback-preserving live regions;
CSI 2026 synchronized output; streaming markdown, tool-call blocks, diffs, prompts, slash
commands). Native copy/paste, tmux scrollback and terminal search keep working.

**Transcript:**
- user turns (styled echo); assistant **streaming markdown** (in-place, commits on turn end)
- **plan block** — numbered steps with live status `○ pending → ● active → ✓ done → ✗ failed`;
  reflection notes as a dim collapsible footer
- **tool-call block** — name + pretty JSON args, status badge
  (pending/running/done/error/blocked), auto-collapse on completion; `edit`/`write` render
  **inline unified diffs**; `bash` shows truncated output
- **routing notice** — `↷ escalated small→mid: 2 tool failures`
- **confirmations** — plan y/N (+ free-text feedback), dangerous-command approval

**Status line:**

```
tier:mid · model:coder7b · ctx:12.4k/32k (38%) · intent:code_gen · rounds:3/15 · tok:4.1k · esc:0 · sandbox:bwrap · ses:a1b2
```

- `ctx` from `usage.prompt_tokens` vs profile `ctx_len`; green→amber→red shift near limit
  with a compaction/`/clear` hint.

**Slash commands (MVP):** `/model <auto|small|mid|profile>` · `/models` · `/plan` ·
`/stats` · `/router` (explain last decision) · `/sessions` · `/resume` · `/clear` ·
`/help` · `/quit` + plugin commands (`workbench.commands`).

**Concurrency:** agent loop + router + HTTP streaming in a worker thread; the TUI thread
only consumes the typed event bus. Cancel (Esc/Ctrl-C) propagates an abort signal to the
in-flight llama-server request and tool timeouts.

**Headless (CLI-only compliant):** `workbench -p "<prompt>"` (print mode) and
`--json` (JSON event stream) for scripting and e2e tests.

---

## 8. Observability, central config & logging (Section 6)

### 8.1 Central config (`config.py`)

- Layered TOML: `~/.config/workbench/config.toml` ← project `.workbench.toml` ←
  env (`WB_*`) ← CLI flags → **one typed frozen model**, loaded once via `get_config()`.
  Subsystems receive it as a dependency; no module reads files/env directly.
  Ships with complete defaults (zero-config start).

### 8.2 Central logging (`logging_setup.py`)

- `get_logger(name)` only. Rotating **JSONL** file handler
  (`~/.local/state/workbench/logs/run-<id>.jsonl`) + optional Rich console (`--verbose`).
- `run_id`/`turn_id` correlation ids in `extra`; configured once at startup.
  No module sets its own handlers or levels.

### 8.3 Prometheus metrics (localhost metrics port, alive only while workbench runs)

| Area | Metrics |
|---|---|
| Routing | `route_decisions_total{intent,tier,backend,band}`, `route_fallback_total`, `route_escalated_total`, `route_duration_seconds` |
| LLM | `llm_requests_total{model,tier,status}`, `llm_ttft_seconds`, `llm_duration_seconds`, `llm_tokens_total{direction,model}` |
| Agent | `agent_rounds_total`, `agent_tasks_total{outcome}`, `agent_reflections_total{verdict}`, `agent_plan_approvals_total{decision}` |
| Tools | `tool_calls_total{tool,status}`, `tool_duration_seconds{tool}`, `tool_blocked_total{reason}` |
| Server | `server_starts_total`, `server_ready_seconds`, `server_active`, `server_restarts_total` |
| System | `event_loop_lag_seconds`, process metrics |

### 8.4 Local stack (`docker/observability/docker-compose.yml`)

- **Prometheus** — pre-wired scrape targets (our metrics port + llama-server `/metrics`),
  15s interval, local TSDB.
- **Grafana** — provisioned datasource + **dashboards as JSON code**
  (`docker/grafana/dashboards/`): *Overview* (turn rate, latency p50/p95, TTFT, active tier),
  *Routing* (tier/intent mix, fallback/escalation rates), *Agent & Tools* (rounds/task,
  tool latency, block reasons), *Server* (llama-server metrics).
- Managed by `workbench obs up` / `workbench obs down` — fully local.

---

## 9. Testing & validation (Section 7)

| Layer | Coverage | Technique |
|---|---|---|
| **Unit** | config layering, registry/plugin discovery, tier policy (gate/hysteresis/escalation), path-jail escapes, bash guardrail classifier, `edit` uniqueness, JSON tool-call parser, llama-serve arg builder, event bus | pytest, pure/fast |
| **Integration** | fake `llama-server` stub (`/health`, SSE `/v1/chat/completions`, `/metrics`); llm client vs local OpenAI stub; **real bwrap** execution; downloads vs mocked HF endpoints; laya vs real checkpoint (`@pytest.mark.model`, local-only; CI uses protocol fakes) | tmp dirs, localhost sockets |
| **E2E** | `workbench -p/--json` drives fake server + real sandbox + real files → assert transcript, plan artifact, metrics exposition | subprocess + tmp workspace |
| **Regression** | every bug → failing-first test *before* the fix, kept forever | red→green, real fixes only |
| **TUI** | component render-output assertions + one pty smoke test | no GPU |
| **Observability** | scripted run → parse `/metrics` for required names; `docker compose config` validity | CI-safe |

**Process rules:**
- Tests written **with** each feature; never weakened/skipped/xfailed to go green —
  a failing test gets a proper fix.
- **Hands-on validation checklist per milestone** (real model load, real route decision,
  real sandboxed bash, real plan→approve→reflect run, Grafana showing live data) executed
  and reported before moving on.
- CI: `ruff` + `mypy` + `pytest` on every push to `main`.

---

## 10. Error handling

Typed exceptions per layer (`ConfigError`, `ServerStartError`, `RouterUnavailableError`,
`ToolBlockedError`, `BudgetExceeded`, …), caught at the loop boundary → TUI error block +
JSONL log + metric increment. The TUI never crashes from an internal error; the llama-server
manager degrades (restart with backoff → surface → keep TUI alive).

---

## 11. Milestones

| # | Deliverable | Includes | Validation |
|---|---|---|---|
| **M1** | Skeleton | uv project, central config + logging, event bus, plugin registry, CLI scaffold, CI (ruff/mypy/pytest) | config layering + registry tests green; `workbench --help` runs |
| **M2** | LLM layer | streaming client, fake-server fixture, `llama-serve` manager (**spikes:** router-mode VRAM eviction, per-model args) | real router-mode server starts; streaming completion against real small model |
| **M3** | Router | heuristic → laya backend → tier policy (+ checkpoint prefetch in download cmd) | real laya predicts intent/difficulty; policy unit tests (gate/hysteresis/escalation) green |
| **M4** | Tools & sandbox | 4 tools, path jail, bwrap, guardrails, confirmation flow | escape attempts blocked (tests + manual); sandboxed bash runs; denied command prompts |
| **M5** | Agent loop | plan→approve→execute→reflect, budgets, `-p`/`--json` headless | e2e: fake server full plan cycle; real-model task completes with reflection |
| **M6** | TUI | agentic-tui wiring, plan/tool blocks, status line (incl. ctx), slash cmds, sessions | pty smoke; interactive run: plan y/N, tool blocks, `/model` switching |
| **M7** | Observability | metrics on all layers, compose stack, Grafana dashboards | `workbench obs up` → live dashboards; metric-name test green |
| **M8** | Downloads | hf provider, ollama shim, store registry, laya prefetch | downloads a model end-to-end; runtime stays offline (guard test) |
| **M9** | Hardening & docs | e2e + regression suite complete, README, full validation pass | whole-suite green on `main`; checklist §9 executed |

---

## 12. Risk register

| Risk | Mitigation |
|---|---|
| torch wheels vs Python version (agentic-tui needs ≥3.13) | verify first (M1): pick venv Python with both `torch` and `agentic-tui` wheels (likely 3.13); pin in `.python-version` |
| laya thresholds uncalibrated | confidence gate + default-to-larger; every decision logged for later calibration; thresholds in config |
| MiniCPM (or any model) lacks tool templates | `tool_calling = json` prompt-protocol fallback per profile |
| llama.cpp router-mode VRAM eviction unknown on 6GB | **spike M2**; fallback single-model mode behind same interface |
| Router mode per-model args unsupported | **spike M2**; fallback to global launch args / restart-on-switch |
| laya checkpoint on CUDA unverified | M3 validation; heuristic router is the designed fallback |
| agentic-tui is a young dependency | narrow surface (we use its primitives only); wrap behind `tui/` layer so it's swappable |

---

## 13. Out of scope (MVP)

HTTP/API endpoint, web UI, sub-agents, MCP, multi-user, compaction UX beyond a
`/clear` hint, remote model providers, model fine-tuning, automatic threshold
self-calibration, non-Linux platforms.
