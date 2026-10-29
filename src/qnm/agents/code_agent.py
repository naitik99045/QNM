"""
Code Specialist Agent.
Analyzes codebases, edits files with symbol preservation, runs tests, and debugs failures.
"""

from typing import Optional
from qnm.agents.base import BaseAgent


class CodeAgent(BaseAgent):
    """Specialist Agent for Software Engineering & Code Verification."""

    def __init__(self, max_iterations: int = 8, model: Optional[str] = None):
        system_prompt = (
            "You are QNM CodeAgent, an elite autonomous software engineer.\n"
            "Your responsibilities:\n"
            "1. Read and inspect source files before making modifications.\n"
            "2. Preserve exact variable names, file paths, and syntax.\n"
            "3. Use edit_file or write_file to apply precise, complete fixes.\n"
            "4. Run tests using run_command to verify all changes.\n"
            "5. If a test fails, diagnose the error and remediate immediately.\n"
            "Conclude with 'Final Answer: <Summary of changes and verification result>'."
        )
        super().__init__(
            name="QNM-CodeAgent",
            role_description="Autonomous Software Engineer & Code Architect",
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            model=model
        )
