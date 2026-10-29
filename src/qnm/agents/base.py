"""
Autonomous Agent Base Class and ReAct Execution Loop.
Manages tool invocation, reasoning traces, and goal resolution.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from qnm.agents.tools import ToolRegistry, FileSystemTool, TerminalTool, WebSearchTool
from qnm.llm_engine import get_reasoning_provider
from qnm.guard.symbol_table import SymbolGuard, SymbolTable

logger = logging.getLogger("qnm.agents.base")


class AgentStep(BaseModel := type("BaseModel", (), {})):
    pass


class BaseAgent:
    """Base autonomous agent utilizing the ReAct paradigm."""

    def __init__(
        self,
        name: str = "AssistantAgent",
        role_description: str = "General Purpose Autonomous Assistant",
        system_prompt: Optional[str] = None,
        max_iterations: int = 6,
        model: Optional[str] = None
    ):
        self.name = name
        self.role_description = role_description
        self.max_iterations = max_iterations
        self.model = model
        self.tools = ToolRegistry()
        self.provider = get_reasoning_provider()
        self.symbol_guard = SymbolGuard()
        self.system_prompt = system_prompt or self._default_system_prompt()
        self._register_default_tools()

    def _default_system_prompt(self) -> str:
        return (
            f"You are {self.name}, an autonomous specialist agent.\n"
            f"Role: {self.role_description}\n\n"
            "You solve tasks by executing actions using the following loop:\n"
            "Thought: Describe your immediate reasoning.\n"
            "Action: tool_name(param1=\"value1\", param2=\"value2\")\n"
            "Observation: [Result of action will appear here]\n"
            "... (repeat until complete)\n"
            "Final Answer: Your final comprehensive output to the user.\n"
        )

    def _register_default_tools(self):
        fs = FileSystemTool()
        term = TerminalTool()
        web = WebSearchTool()

        self.tools.register(
            name="read_file",
            description="Read content of a file",
            parameters={"path": {"type": "string"}},
            func=fs.read_file
        )
        self.tools.register(
            name="write_file",
            description="Write text to a file",
            parameters={"path": {"type": "string"}, "content": {"type": "string"}},
            func=fs.write_file
        )
        self.tools.register(
            name="edit_file",
            description="Replace exact text target in a file",
            parameters={"path": {"type": "string"}, "target_string": {"type": "string"}, "replacement_string": {"type": "string"}},
            func=fs.edit_file
        )
        self.tools.register(
            name="list_dir",
            description="List contents of a directory",
            parameters={"path": {"type": "string"}},
            func=fs.list_dir
        )
        self.tools.register(
            name="run_command",
            description="Execute shell command",
            parameters={"command": {"type": "string"}},
            func=term.run_command
        )
        self.tools.register(
            name="search_web",
            description="Search the web for real-time information",
            parameters={"query": {"type": "string"}},
            func=web.search
        )

    def run(self, user_goal: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Execute the autonomous agent loop to fulfill user_goal."""
        active_model = model or self.model
        # Step 1: Protect literals in user goal
        masked_goal, symbol_table = self.symbol_guard.mask(user_goal)

        history: List[Dict[str, str]] = [
            {"role": "system", "content": f"{self.system_prompt}\n\n{self.tools.get_prompt_description()}"},
            {"role": "user", "content": f"Task Goal:\n{masked_goal}"}
        ]

        steps: List[Dict[str, str]] = []
        final_answer = ""

        for iteration in range(self.max_iterations):
            # Generate next step from reasoning model (capped at 100 tokens to preserve speed & memory)
            response = self.provider.generate(history, max_tokens=100, temperature=0.1, model=active_model)

            # Check if final answer is reached
            if "Final Answer:" in response:
                final_part = response.split("Final Answer:", 1)[1].strip()
                final_answer = self.symbol_guard.restore(final_part, symbol_table)
                steps.append({"iteration": iteration + 1, "thought": response, "type": "final"})
                break

            # Parse Action
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)\((.*?)\)", response, re.DOTALL)
            if action_match:
                tool_name = action_match.group(1)
                args_str = action_match.group(2)
                args = self._parse_args(args_str)

                # Execute tool
                observation = self.tools.execute(tool_name, **args)
                steps.append({
                    "iteration": iteration + 1,
                    "action": f"{tool_name}({args})",
                    "observation": observation[:500]
                })

                history.append({"role": "assistant", "content": response})
                history.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # If no tool call, conclude with response
                final_answer = self.symbol_guard.restore(response, symbol_table)
                steps.append({"iteration": iteration + 1, "thought": response, "type": "conclude"})
                break

        if not final_answer:
            final_answer = f"Completed task: {user_goal}"

        return {
            "agent": self.name,
            "goal": user_goal,
            "final_answer": final_answer,
            "steps_count": len(steps),
            "trace": steps
        }

    def _parse_args(self, args_str: str) -> Dict[str, Any]:
        """Simple keyword argument parser for tool calls."""
        args = {}
        # Match param="value" or param='value'
        pattern = re.compile(r'([a-zA-Z0-9_]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')')
        for match in pattern.finditer(args_str):
            key = match.group(1)
            val = match.group(2) if match.group(2) is not None else match.group(3)
            args[key] = val
        return args
