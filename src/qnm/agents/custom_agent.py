"""
Custom Agent Loader and Unified Agent Resolver.
Loads arbitrary agent presets from config/agents.yaml and pairs them
with the requested LM Studio model and sandboxed tools.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

from qnm.agents.base import BaseAgent
from qnm.agents.code_agent import CodeAgent
from qnm.agents.research_agent import ResearchAgent
from qnm.agents.shell_agent import ShellAgent
from qnm.agents.tools import FileSystemTool, TerminalTool, WebSearchTool
from qnm.settings import get_settings


class CustomAgent(BaseAgent):
    """Dynamically configured autonomous agent instantiated from YAML configuration."""

    def __init__(
        self,
        name: str,
        role_description: str,
        system_prompt: str,
        enabled_tools: List[str],
        max_iterations: int = 6,
        model: Optional[str] = None
    ):
        self.enabled_tools = enabled_tools
        super().__init__(
            name=name,
            role_description=role_description,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            model=model
        )

    def _register_default_tools(self):
        """Register only tools explicitly enabled for this custom agent."""
        fs = FileSystemTool()
        term = TerminalTool()
        web = WebSearchTool()

        tool_map = {
            "read_file": ("read_file", "Read content of a file", {"path": {"type": "string"}}, fs.read_file),
            "write_file": ("write_file", "Write text to a file", {"path": {"type": "string"}, "content": {"type": "string"}}, fs.write_file),
            "edit_file": ("edit_file", "Replace exact text target in a file", {"path": {"type": "string"}, "target_string": {"type": "string"}, "replacement_string": {"type": "string"}}, fs.edit_file),
            "list_dir": ("list_dir", "List contents of a directory", {"path": {"type": "string"}}, fs.list_dir),
            "run_command": ("run_command", "Execute shell command", {"command": {"type": "string"}}, term.run_command),
            "search_web": ("search_web", "Search the web for real-time information", {"query": {"type": "string"}}, web.search),
        }

        # If enabled_tools not provided, enable all standard tools
        tools_to_add = self.enabled_tools if hasattr(self, "enabled_tools") and self.enabled_tools else list(tool_map.keys())

        for t_name in tools_to_add:
            if t_name in tool_map:
                name, desc, params, func = tool_map[t_name]
                self.tools.register(name=name, description=desc, parameters=params, func=func)


def load_yaml_presets() -> Dict[str, Dict[str, Any]]:
    """Load custom agent presets from config/agents.yaml."""
    settings = get_settings()
    agents_file = settings.resolve_path("config/agents.yaml")
    if not agents_file.exists():
        return {}

    try:
        with open(agents_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "agents" in data and isinstance(data["agents"], dict):
                return data["agents"]
    except Exception:
        pass
    return {}


def get_available_agents() -> Dict[str, Dict[str, Any]]:
    """Return catalog of all available agents (built-in and custom YAML presets)."""
    catalog: Dict[str, Dict[str, Any]] = {
        "code": {
            "name": "QNM-CodeAgent",
            "role": "Autonomous Software Engineer & Code Architect",
            "type": "builtin",
            "tools": ["read_file", "write_file", "edit_file", "list_dir", "run_command", "search_web"]
        },
        "research": {
            "name": "QNM-ResearchAgent",
            "role": "Deep Technical Researcher & Document Synthesizer",
            "type": "builtin",
            "tools": ["read_file", "write_file", "edit_file", "list_dir", "run_command", "search_web"]
        },
        "shell": {
            "name": "QNM-ShellAgent",
            "role": "DevOps, System Diagnostic & Shell Automation Specialist",
            "type": "builtin",
            "tools": ["read_file", "write_file", "edit_file", "list_dir", "run_command", "search_web"]
        }
    }

    yaml_presets = load_yaml_presets()
    for key, spec in yaml_presets.items():
        catalog[key.lower()] = {
            "name": f"QNM-{key.capitalize()}Agent",
            "role": spec.get("role", "Specialist Assistant"),
            "type": "custom",
            "tools": spec.get("tools", ["read_file", "search_web"]),
            "system_prompt": spec.get("system_prompt", "")
        }

    return catalog


def resolve_agent(agent_name: str, model: Optional[str] = None) -> BaseAgent:
    """Instantiate the requested agent by name, configured with the specified model."""
    name_clean = agent_name.lower().strip()

    if name_clean == "code":
        return CodeAgent(model=model)
    elif name_clean == "research":
        return ResearchAgent(model=model)
    elif name_clean == "shell":
        return ShellAgent(model=model)

    yaml_presets = load_yaml_presets()
    if name_clean in yaml_presets:
        spec = yaml_presets[name_clean]
        return CustomAgent(
            name=f"QNM-{name_clean.capitalize()}Agent",
            role_description=spec.get("role", "Specialist Assistant"),
            system_prompt=spec.get("system_prompt", ""),
            enabled_tools=spec.get("tools", []),
            model=model
        )

    raise ValueError(f"Unknown agent '{agent_name}'. Available: {', '.join(get_available_agents().keys())}")
