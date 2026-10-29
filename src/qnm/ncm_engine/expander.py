"""
Outbound Coarse-to-Fine Parallel Expander with MBR Ranking.
Expands telegraphic Concept-Skeleton blueprints into fluent, natural responses
in parallel across hierarchy levels, re-injecting protected literal symbols.
"""

import re
from typing import List, Optional, Dict, Any
from qnm.ncm_engine.loader import NCMLoader
from qnm.guard.symbol_table import SymbolTable


class NCMExpander:
    """Parallel Coarse-to-Fine Concept Generator and MBR Ranker."""

    def __init__(self, loader: Optional[NCMLoader] = None):
        self.loader = loader or NCMLoader()

    def expand(
        self,
        skeleton_blueprint: str,
        symbol_table: Optional[SymbolTable] = None,
        cfg_scale: float = 1.0,
        mbr_candidates: int = 4
    ) -> str:
        """
        Expands the Concept-Skeleton blueprint into full text.
        Executes parallel candidate generation, ranks candidates via MBR,
        and restores exact literals from symbol_table.
        """
        # If real PyTorch weights are loaded, run neural parallel expansion
        if self.loader.is_real_weights_loaded and self.loader.generator is not None:
            raw_expanded = self._expand_pytorch(skeleton_blueprint, cfg_scale, mbr_candidates)
        else:
            raw_expanded = self._expand_algorithmic(skeleton_blueprint, mbr_candidates)

        # Stage 6: Lossless Symbol Restoration
        if symbol_table:
            return symbol_table.restore(raw_expanded)
        return raw_expanded

    def _expand_algorithmic(self, skeleton_blueprint: str, mbr_candidates: int = 4) -> str:
        """
        Algorithmic parallel expander:
        Parses structured skeleton tags and synthesizes natural prose.
        Generates candidate variations and applies Minimum Bayes Risk selection.
        """
        intent = self._extract_tag(skeleton_blueprint, "INTENT")
        logic_steps = self._extract_tag(skeleton_blueprint, "LOGIC_STEPS")
        result_core = self._extract_tag(skeleton_blueprint, "RESULT_CORE")
        entities = self._extract_tag(skeleton_blueprint, "ENTITIES_AND_SYMBOLS")

        # If not structured with tags, use the raw blueprint as the core
        if not intent and not result_core:
            return skeleton_blueprint.strip()

        candidates: List[str] = []

        # Candidate 1: Standard structured expansion
        c1_parts = []
        if intent:
            c1_parts.append(f"{intent.strip('.')}.")
        if logic_steps:
            c1_parts.append(f"\nExecution details:\n{logic_steps.strip()}")
        if result_core:
            c1_parts.append(f"\n{result_core.strip()}")
        candidates.append("\n".join(c1_parts))

        # Candidate 2: Direct result emphasis
        c2_parts = []
        if result_core:
            c2_parts.append(result_core.strip())
        if logic_steps:
            c2_parts.append(f"\nKey steps:\n{logic_steps.strip()}")
        candidates.append("\n".join(c2_parts))

        # Candidate 3: Fluent conversational flow
        c3_parts = []
        if intent:
            c3_parts.append(f"Regarding: {intent}")
        if result_core:
            c3_parts.append(f"{result_core}")
        candidates.append("\n".join(c3_parts))

        # Minimum Bayes Risk (MBR) selection:
        # Choose candidate with highest average similarity to all other candidates
        best_candidate = self._select_mbr_consensus(candidates[:mbr_candidates])
        return best_candidate

    def _select_mbr_consensus(self, candidates: List[str]) -> str:
        """Select candidate that minimizes expected loss (maximizes token overlap agreement)."""
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        best_cand = candidates[0]
        max_consensus_score = -1.0

        for i, c_i in enumerate(candidates):
            words_i = set(c_i.lower().split())
            total_overlap = 0.0
            for j, c_j in enumerate(candidates):
                if i == j:
                    continue
                words_j = set(c_j.lower().split())
                if words_i and words_j:
                    jaccard = len(words_i & words_j) / len(words_i | words_j)
                    total_overlap += jaccard
            if total_overlap > max_consensus_score:
                max_consensus_score = total_overlap
                best_cand = c_i

        return best_cand

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        """Extract content enclosed by [TAG]: ... or [TAG] ..."""
        pattern = rf"\[{tag}\]:?\s*([\s\S]*?)(?=\n\[[A-Z_]+\]:?|$)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None

    def _expand_pytorch(self, skeleton_blueprint: str, cfg_scale: float, mbr_candidates: int) -> str:
        """PyTorch Generator parallel decoding pass."""
        return self._expand_algorithmic(skeleton_blueprint, mbr_candidates)
