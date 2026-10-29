"""Pre-installed AI Agent Suite and Tool Ecosystem."""
from qnm.agents.tools import ToolRegistry, FileSystemTool, TerminalTool, WebSearchTool
from qnm.agents.base import BaseAgent
from qnm.agents.code_agent import CodeAgent
from qnm.agents.research_agent import ResearchAgent
from qnm.agents.shell_agent import ShellAgent
from qnm.agents.custom_agent import CustomAgent, resolve_agent, get_available_agents

__all__ = [
    "ToolRegistry",
    "FileSystemTool",
    "TerminalTool",
    "WebSearchTool",
    "BaseAgent",
    "CodeAgent",
    "ResearchAgent",
    "ShellAgent",
    "CustomAgent",
    "resolve_agent",
    "get_available_agents",
]
