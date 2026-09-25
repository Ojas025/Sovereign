# Workbench Plugin Examples

This directory provides working examples for all 5 extension point groups supported by Workbench (PLAN §3):

| Group | Protocol / Signature | Example File | Description |
|---|---|---|---|
| `workbench.tools` | `Tool` protocol | [custom_tool.py](file:///home/ojas/projects/workbench/plugins/examples/custom_tool.py) | Safe arithmetic calculator tool (`calc`) |
| `workbench.routers` | `Router` protocol | [custom_router.py](file:///home/ojas/projects/workbench/plugins/examples/custom_router.py) | Fast keyword-based intent and difficulty classifier |
| `workbench.providers` | `ModelProvider` protocol | [custom_provider.py](file:///home/ojas/projects/workbench/plugins/examples/custom_provider.py) | Local filesystem model file importer (`localfile:`) |
| `workbench.sandboxes` | `Sandbox` protocol | [custom_sandbox.py](file:///home/ojas/projects/workbench/plugins/examples/custom_sandbox.py) | Alternative sandbox backend (`passthrough`) |
| `workbench.commands` | `async (ui, args) -> None` | [custom_command.py](file:///home/ojas/projects/workbench/plugins/examples/custom_command.py) | Interactive TUI slash command (`/echo`) |

## How to Load Plugins

Plugins can be registered through two methods:

### Method 1: Local Plugin Directory (`~/.config/workbench/plugins/*.py`)

Drop any `.py` file into `~/.config/workbench/plugins/`. Each file defines a `register(registry)` function that registers components with `registry.register(group, name, object)`:

```python
def register(registry):
    registry.register("tools", "calc", CalcTool())
```

### Method 2: Python Packaging Entry Points (`pyproject.toml`)

In any installed Python package, declare entry points matching `workbench.<group>`:

```toml
[project.entry-points."workbench.tools"]
calc = "my_package.tools:CalcTool"

[project.entry-points."workbench.routers"]
my_router = "my_package.routers:MyRouter"

[project.entry-points."workbench.commands"]
echo = "my_package.commands:echo_command"
```

Core tools and commands take precedence, preventing plugins from accidentally shadowing built-ins.
