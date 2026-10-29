"""
Shell & DevOps Specialist Agent.
Diagnoses system resources, inspects LM Studio server state, manages background daemons,
and automates shell tasks.
"""

from typing import Optional
from qnm.agents.base import BaseAgent


class ShellAgent(BaseAgent):
    """Specialist Agent for System Diagnostics, DevOps & Command-Line Automation."""

    def __init__(self, max_iterations: int = 6, model: Optional[str] = None):
        system_prompt = (
            "You are QNM ShellAgent, an autonomous DevOps and systems engineer.\n"
            "Your responsibilities:\n"
            "1. Diagnose local machine resources (CPU, RAM, GPU/VRAM, disk).\n"
            "2. Inspect LM Studio daemon state, loaded models, and ports.\n"
            "3. Execute terminal commands safely with error handling and output checks.\n"
            "4. Monitor background processes and network connectivity.\n"
            "Conclude with 'Final Answer: <Diagnostic findings and recommended actions>'."
        )
        super().__init__(
            name="QNM-ShellAgent",
            role_description="DevOps, System Diagnostic & Shell Automation Specialist",
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            model=model
        )
