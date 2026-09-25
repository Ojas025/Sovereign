# workbench

Minimal, fully local AI workbench: llama.cpp runtime (`llama-serve --router`),
laya-based intent/complexity routing, a pi-like CLI/TUI harness with four
sandboxed tools (`read`, `write`, `edit`, `bash`), a plan → approve → execute →
reflect agent loop, and local Prometheus + Grafana observability.

Zero external API calls at runtime; models are fetched only via an explicit
`workbench models download` command.

## Status

Implementation follows `PLAN.md` (design, decision log, milestones M1–M9).

## Development

```bash
uv sync                 # install deps
uv run pytest           # tests
uv run ruff check src tests
uv run mypy
uv run workbench --help
```
