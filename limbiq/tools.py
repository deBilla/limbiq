"""
Tool Use Framework -- safe, sandboxed tools the LLM can invoke.

Supported tools:
  /file <path>   -- read a file (max 5000 chars)
  /run <cmd>     -- run a safe read-only shell command
  /calc <expr>   -- evaluate a math expression

The ToolRegistry also detects auto-invocation patterns in LLM output
so tools can be called transparently.
"""

import ast
import logging
import operator
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    name: str
    output: str
    success: bool
    error: Optional[str] = None


# ── Base tool ────────────────────────────────────────────────────────────────

class BaseTool:
    name: str = "base"
    description: str = ""

    def execute(self, args: str) -> ToolResult:  # pragma: no cover
        raise NotImplementedError


# ── FileReaderTool ───────────────────────────────────────────────────────────

class FileReaderTool(BaseTool):
    name = "file"
    description = "Read the contents of a file. Usage: /file <path>"
    MAX_CHARS = 5000

    def execute(self, args: str) -> ToolResult:
        path = args.strip()
        if not path:
            return ToolResult(name=self.name, output="", success=False, error="No path provided.")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(self.MAX_CHARS)
            truncated = len(content) >= self.MAX_CHARS
            if truncated:
                content += "\n... (truncated)"
            return ToolResult(name=self.name, output=content, success=True)
        except FileNotFoundError:
            return ToolResult(name=self.name, output="", success=False,
                              error=f"File not found: {path}")
        except PermissionError:
            return ToolResult(name=self.name, output="", success=False,
                              error=f"Permission denied: {path}")
        except Exception as e:
            return ToolResult(name=self.name, output="", success=False, error=str(e))


# ── TerminalTool ─────────────────────────────────────────────────────────────

_SAFE_COMMANDS = {
    "ls", "cat", "head", "tail", "pwd", "which", "echo",
    "date", "whoami", "uname", "df", "free",
}

def _is_safe_command(cmd: str) -> bool:
    """Allow only whitelisted read-only shell commands."""
    parts = cmd.strip().split()
    if not parts:
        return False
    base = parts[0].split("/")[-1]  # handle /bin/ls etc.
    return base in _SAFE_COMMANDS


class TerminalTool(BaseTool):
    name = "run"
    description = (
        "Run a safe read-only shell command. "
        f"Allowed: {', '.join(sorted(_SAFE_COMMANDS))}. "
        "Usage: /run <command>"
    )
    TIMEOUT = 10
    MAX_OUTPUT = 5000

    def execute(self, args: str) -> ToolResult:
        cmd = args.strip()
        if not cmd:
            return ToolResult(name=self.name, output="", success=False, error="No command provided.")
        if not _is_safe_command(cmd):
            base = cmd.split()[0].split("/")[-1]
            return ToolResult(
                name=self.name, output="", success=False,
                error=f"Command '{base}' is not allowed. Allowed: {', '.join(sorted(_SAFE_COMMANDS))}"
            )
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=self.TIMEOUT,
            )
            output = (result.stdout + result.stderr)[: self.MAX_OUTPUT]
            return ToolResult(name=self.name, output=output, success=result.returncode == 0,
                              error=None if result.returncode == 0 else f"Exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            return ToolResult(name=self.name, output="", success=False,
                              error=f"Command timed out after {self.TIMEOUT}s.")
        except Exception as e:
            return ToolResult(name=self.name, output="", success=False, error=str(e))


# ── CalculatorTool ───────────────────────────────────────────────────────────

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """Recursively evaluate an AST node using only safe numeric operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported literal: {node.value!r}")
    elif isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if not op_fn:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large (> 100)")
        return op_fn(left, right)
    elif isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if not op_fn:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))
    elif isinstance(node, ast.Num):  # Python 3.7 compat
        return node.n
    else:
        raise ValueError(f"Unsupported node type: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = "calc"
    description = "Evaluate a math expression safely. Usage: /calc <expression>"

    def execute(self, args: str) -> ToolResult:
        expr = args.strip()
        if not expr:
            return ToolResult(name=self.name, output="", success=False, error="No expression provided.")
        try:
            tree = ast.parse(expr, mode="eval")
            result = _safe_eval(tree)
            # Format nicely
            if isinstance(result, float) and result == int(result):
                output = str(int(result))
            elif isinstance(result, float):
                output = f"{result:.6g}"
            else:
                output = str(result)
            return ToolResult(name=self.name, output=f"{expr} = {output}", success=True)
        except ZeroDivisionError:
            return ToolResult(name=self.name, output="", success=False, error="Division by zero.")
        except Exception as e:
            return ToolResult(name=self.name, output="", success=False,
                              error=f"Could not evaluate '{expr}': {e}")


# ── ToolRegistry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """Registry of available tools with detection helpers."""

    # User-facing command patterns: /tool args
    _COMMAND_RE = re.compile(
        r"^/(file|run|calc)\s+(.*)", re.IGNORECASE | re.DOTALL
    )

    # Auto-detect phrases in LLM output → map to tool
    _AUTO_PATTERNS = [
        (re.compile(r"let me (?:read|check|open|look at) (?:the )?file\s+(\S+)", re.I), "file"),
        (re.compile(r"let me (?:run|execute|check)\s+`?([^`\n]+)`?", re.I), "run"),
        (re.compile(r"let me (?:calculate|compute|figure out)\s+([0-9+\-*/().^ \t]+)", re.I), "calc"),
    ]

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._register_defaults()

    def _register_defaults(self):
        for t in [FileReaderTool(), TerminalTool(), CalculatorTool()]:
            self._tools[t.name] = t

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def detect_tool_request(self, text: str) -> Optional[Tuple[str, str]]:
        """Detect explicit /tool command. Returns (tool_name, args) or None."""
        m = self._COMMAND_RE.match(text.strip())
        if m:
            return m.group(1).lower(), m.group(2).strip()
        return None

    def detect_auto(self, text: str) -> Optional[Tuple[str, str]]:
        """Detect implicit tool invocation in LLM output. Returns (tool_name, args) or None."""
        for pattern, tool_name in self._AUTO_PATTERNS:
            m = pattern.search(text)
            if m:
                return tool_name, m.group(1).strip()
        return None

    def execute(self, tool_name: str, args: str) -> ToolResult:
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(name=tool_name, output="", success=False,
                              error=f"Unknown tool: {tool_name!r}")
        logger.info(f"Tool '{tool_name}' executing: {args[:80]!r}")
        result = tool.execute(args)
        logger.info(f"Tool '{tool_name}' result: success={result.success}, output_len={len(result.output)}")
        return result


# ── Formatting ───────────────────────────────────────────────────────────────

def format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results for injection into the LLM context."""
    if not results:
        return ""
    parts = []
    for r in results:
        if r.success:
            parts.append(f"[TOOL: {r.name}]\n{r.output}")
        else:
            parts.append(f"[TOOL: {r.name} — ERROR: {r.error}]")
    return "\n\n".join(parts)
