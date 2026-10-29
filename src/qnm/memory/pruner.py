"""
Context Pruner and Relevance Budget Manager.
Extracts top relevant historical coarse concept gists under a strict token budget.
"""

from typing import List, Dict, Any, Optional
from qnm.memory.store import MemoryStore


class ContextPruner:
    """Selects and prunes historical turns to fit within a strict token budget."""

    def __init__(self, memory_store: MemoryStore, max_tokens: int = 512):
        self.store = memory_store
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token estimator (~4 characters per token)."""
        return max(1, len(text.strip()) // 4)

    def retrieve_context(self, session_id: str, new_query: str, max_tokens: Optional[int] = None) -> str:
        """
        Retrieves recent and relevant coarse gists for session_id,
        ensuring the total context token count does not exceed max_tokens.
        """
        budget = max_tokens or self.max_tokens
        turns = self.store.get_session_turns(session_id, limit=30)
        if not turns:
            return ""

        # Extract keywords from new_query for simple lexical relevance weighting
        query_words = set(new_query.lower().split())

        scored_turns = []
        total_turns = len(turns)
        for i, turn in enumerate(turns):
            # Recency score normalized between 0.5 and 1.0
            recency_score = 0.5 + (0.5 * (i / max(1, total_turns - 1)))

            # Lexical overlap score
            gist_words = set(turn["coarse_gist"].lower().split())
            overlap = len(query_words.intersection(gist_words))
            overlap_score = min(1.0, overlap * 0.2)

            total_score = recency_score + overlap_score
            scored_turns.append((total_score, turn))

        # Sort turns by descending score
        scored_turns.sort(key=lambda x: x[0], reverse=True)

        selected_gists: List[str] = []
        current_tokens = 0

        for _, turn in scored_turns:
            role = turn["role"].capitalize()
            gist = turn["coarse_gist"]
            snippet = f"[{role}]: {gist}"
            tokens = self.estimate_tokens(snippet)

            if current_tokens + tokens <= budget:
                selected_gists.append(snippet)
                current_tokens += tokens
            else:
                break

        # Re-order chronologically for coherence
        return "\n".join(selected_gists)
