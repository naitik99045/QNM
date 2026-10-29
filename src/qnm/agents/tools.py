"""
Autonomous Agent Tool Ecosystem.
Provides safe tools for File System manipulation, Terminal execution, Web Search,
Symbol Guard literal protection, and SQLite Concept Memory querying.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel
from qnm.guard.symbol_table import SymbolGuard
from qnm.memory.store import MemoryStore


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class FileSystemTool:
    """Safe Workspace File Operations."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()

    def _resolve(self, target_path: str) -> Path:
        p = Path(target_path)
        if not p.is_absolute():
            p = (self.workspace_root / p).resolve()
        return p

    def read_file(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Read text from a file with optional line boundaries."""
        target = self._resolve(path)
        if not target.exists():
            return f"Error: File not found: {path}"
        if not target.is_file():
            return f"Error: Path is a directory: {path}"

        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            s = (start_line - 1) if (start_line and start_line > 0) else 0
            e = end_line if (end_line and end_line > 0) else len(lines)
            selected = lines[s:e]
            numbered = [f"{s + idx + 1}: {line}" for idx, line in enumerate(selected)]
            return "".join(numbered)
        except Exception as e:
            return f"Error reading file {path}: {e}"

    def write_file(self, path: str, content: str, overwrite: bool = True) -> str:
        """Create or overwrite a file with given text content."""
        target = self._resolve(path)
        if target.exists() and not overwrite:
            return f"Error: File already exists and overwrite is False: {path}"

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing to file {path}: {e}"

    def edit_file(self, path: str, target_string: str, replacement_string: str) -> str:
        """Perform an exact target string replacement in a file."""
        target = self._resolve(path)
        if not target.exists():
            return f"Error: File not found: {path}"

        try:
            with open(target, "r", encoding="utf-8") as f:
                content = f.read()

            if target_string not in content:
                return f"Error: Target string not found in {path}"

            new_content = content.replace(target_string, replacement_string)
            with open(target, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"Successfully replaced target in {path}"
        except Exception as e:
            return f"Error editing {path}: {e}"

    def list_dir(self, path: str = ".") -> str:
        """List files and directories in a given path."""
        target = self._resolve(path)
        if not target.exists():
            return f"Error: Directory not found: {path}"

        try:
            items = []
            for item in sorted(target.iterdir()):
                kind = "DIR " if item.is_dir() else "FILE"
                items.append(f"[{kind}] {item.name}")
            return "\n".join(items) if items else "(empty directory)"
        except Exception as e:
            return f"Error listing directory {path}: {e}"


class TerminalTool:
    """Safe Terminal Subprocess Execution."""

    def __init__(self, timeout: int = 30):
        self.default_timeout = timeout

    def run_command(self, command: str, timeout: Optional[int] = None) -> str:
        """Execute a shell command with timeout and output capture."""
        t = timeout or self.default_timeout
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=t
            )
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            code = proc.returncode

            result_lines = [f"[Exit Code]: {code}"]
            if stdout:
                result_lines.append(f"[STDOUT]:\n{stdout[:2000]}")
            if stderr:
                result_lines.append(f"[STDERR]:\n{stderr[:2000]}")
            if not stdout and not stderr:
                result_lines.append("(No output produced)")

            return "\n".join(result_lines)
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {t} seconds."
        except Exception as e:
            return f"Error running command: {e}"


class WebSearchTool:
    """Live Web Search via DuckDuckGo."""

    def __init__(self, max_results: int = 4):
        self.max_results = max_results

    def search(self, query: str, max_results: Optional[int] = None) -> str:
        """Search the web for up-to-date documentation or facts."""
        limit = max_results or self.max_results
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=limit))

            if not results:
                return f"No search results found for query: {query}"

            formatted = []
            for i, r in enumerate(results):
                title = r.get("title", "")
                snippet = r.get("body", "")
                href = r.get("href", "")
                formatted.append(f"[{i+1}] {title}\nURL: {href}\nSnippet: {snippet}\n")

            return "\n".join(formatted)
        except Exception as e:
            return f"Search engine query notice: {e}"


class ToolRegistry:
    """Central registry of agent tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._definitions: Dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        self._tools[name] = func
        self._definitions[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters
        )

    def execute(self, name: str, **kwargs) -> str:
        if name not in self._tools:
            return f"Error: Unknown tool '{name}'."
        try:
            return str(self._tools[name](**kwargs))
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    def get_definitions(self) -> List[ToolDefinition]:
        return list(self._definitions.values())

    def get_prompt_description(self) -> str:
        lines = ["Available Tools:"]
        for d in self._definitions.values():
            params = ", ".join(f"{k}: {v.get('type', 'str')}" for k, v in d.parameters.items())
            lines.append(f"- {d.name}({params}): {d.description}")
        return "\n".join(lines)
