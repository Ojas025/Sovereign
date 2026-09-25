"""Example tool plugin proving the ``workbench.tools`` extension point (PLAN §3).

Defines an example ``calc`` tool that evaluates simple mathematical expressions.
Can be registered either via entry point in pyproject.toml:
    [project.entry-points."workbench.tools"]
    calc = "plugins.examples.custom_tool:CalcTool"

Or placed in ~/.config/workbench/plugins/calc.py with a register() hook.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping

from workbench.core.protocols import ToolContext, ToolResult
from workbench.core.registry import Registry


class CalcTool:
    """Safe arithmetic calculator tool."""

    name = "calc"
    description = (
        "Evaluate a simple arithmetic expression (addition, subtraction, multiplication, division)."
    )
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression to evaluate (e.g. '2 + 2 * 10')",
            }
        },
        "required": ["expression"],
    }

    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def _eval(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._OPERATORS:
            left = self._eval(node.left)
            right = self._eval(node.right)
            return self._OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._OPERATORS:
            operand = self._eval(node.operand)
            return self._OPERATORS[type(node.op)](operand)
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    async def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        expr = str(arguments.get("expression", "")).strip()
        if not expr:
            return ToolResult(content="expression parameter is required", is_error=True)
        try:
            tree = ast.parse(expr, mode="eval")
            result = self._eval(tree.body)
            return ToolResult(content=str(result), is_error=False)
        except Exception as exc:
            return ToolResult(content=f"calculation error: {exc}", is_error=True)


def register(registry: Registry) -> None:
    """Registration hook when loaded from ~/.config/workbench/plugins/."""
    registry.register("tools", "calc", CalcTool())
