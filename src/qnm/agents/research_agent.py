"""
Research & Documentation Specialist Agent.
Digests massive documents via NCM multiscale concept compression,
queries live web search, and synthesizes grounded technical insights.
"""

from typing import Optional
from qnm.agents.base import BaseAgent
from qnm.ncm_engine.compressor import NCMCompressor


class ResearchAgent(BaseAgent):
    """Specialist Agent for Deep Research, Document Ingestion & Fact Grounding."""

    def __init__(self, max_iterations: int = 6, model: Optional[str] = None):
        self.compressor = NCMCompressor()
        system_prompt = (
            "You are QNM ResearchAgent, an expert technical researcher and knowledge synthesizer.\n"
            "Your responsibilities:\n"
            "1. Digest large technical documents, files, and specifications.\n"
            "2. Identify key concepts, architectural patterns, and core findings.\n"
            "3. Query live web search using search_web to verify facts and external APIs.\n"
            "4. Provide crisp, high-signal, fully grounded syntheses.\n"
            "Conclude with 'Final Answer: <Comprehensive synthesis with citations>'."
        )
        super().__init__(
            name="QNM-ResearchAgent",
            role_description="Deep Technical Researcher & Document Synthesizer",
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            model=model
        )

    def digest_document(self, content: str) -> str:
        """Compress large text using NCM multiscale VQ-VAE and return the concept gist."""
        masked_content, table = self.symbol_guard.mask(content)
        gist, codes, metrics = self.compressor.encode(masked_content)
        restored_gist = self.symbol_guard.restore(gist, table)
        return f"[NCM Gist ({metrics['saved_percent']}% compression savings)]:\n{restored_gist}"
