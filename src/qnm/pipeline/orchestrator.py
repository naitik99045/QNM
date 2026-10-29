"""
Core QNM Execution Orchestrator.
Coordinates the 5-stage Dual-Bottleneck Sandwich Pipeline:
Stage 1: Symbol Guard Extraction
Stage 2: Context Retrieval & NCM Multiscale Compression
Stage 3: LLM Concept-Skeleton Reasoning Generation
Stage 4: NCM Coarse-to-Fine Parallel Expansion
Stage 5: Lossless Symbol Restoration & Memory Cache
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from qnm.guard.symbol_table import SymbolGuard, SymbolTable
from qnm.memory.store import MemoryStore
from qnm.memory.pruner import ContextPruner
from qnm.ncm_engine.compressor import NCMCompressor
from qnm.ncm_engine.expander import NCMExpander
from qnm.llm_engine import get_reasoning_provider
from qnm.llm_engine.skeleton_dsl import SkeletonDSL, ConceptSkeleton
from qnm.pipeline.telemetry import PipelineTelemetry, LatencyTracker
from qnm.settings import get_settings


class PipelineResult(BaseModel):
    model_config = {"protected_namespaces": ()}
    session_id: str
    expanded_response: str
    raw_skeleton: str
    coarse_gist: str
    symbols_mapped: int
    telemetry: PipelineTelemetry
    model_used: Optional[str] = None


class QNMOrchestrator:
    """The master pipeline orchestrator for QNM."""

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        compressor: Optional[NCMCompressor] = None,
        expander: Optional[NCMExpander] = None,
    ):
        self.settings = get_settings()
        self.guard = SymbolGuard()
        self.store = memory_store or MemoryStore(db_path=str(self.settings.resolve_path(self.settings.memory.db_path)))
        self.pruner = ContextPruner(self.store, max_tokens=self.settings.compression.max_context_gist_tokens)
        self.compressor = compressor or NCMCompressor(coarse_cutoff=self.settings.compression.coarse_level_cutoff)
        self.expander = expander or NCMExpander()
        self.provider = get_reasoning_provider()

    def process_query(
        self,
        prompt: str,
        session_id: str = "default",
        file_attachments: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> PipelineResult:
        """Executes the full 5-stage QNM pipeline."""
        tracker = LatencyTracker()

        # Gather combined raw text (prompt + file contents)
        raw_text_parts = [prompt]
        if file_attachments:
            for file_path in file_attachments:
                p = Path(file_path)
                if p.exists() and p.is_file():
                    try:
                        with open(p, "r", encoding="utf-8", errors="replace") as f:
                            raw_text_parts.append(f"\n[File: {p.name}]\n{f.read()}")
                    except Exception:
                        pass
        full_raw_input = "\n".join(raw_text_parts)
        raw_context_tokens = max(1, len(full_raw_input.split()))

        # Stage 1: Literal Symbol Guard
        with tracker.track("stage1_symbol_mask"):
            masked_input, symbol_table = self.guard.mask(full_raw_input)

        # Stage 2: Memory Retrieval & NCM Coarse Compression
        with tracker.track("stage2_ncm_compress"):
            historical_context = self.pruner.retrieve_context(session_id, full_raw_input)
            coarse_gist, discrete_codes, comp_metrics = self.compressor.encode(masked_input)
            compressed_input_tokens = comp_metrics["compressed_tokens"]

        # Stage 3: LLM Concept-Skeleton Reasoning Generation
        with tracker.track("stage3_reasoning"):
            messages = SkeletonDSL.build_prompt_messages(
                compressed_context=historical_context,
                current_query=coarse_gist
            )
            raw_skeleton_output = self.provider.generate(
                messages,
                max_tokens=self.settings.compression.skeleton_max_tokens,
                temperature=0.7,
                model=model
            )
            # Guard against blank output from LLM provider
            if not raw_skeleton_output or not raw_skeleton_output.strip():
                clean_gist = coarse_gist.lower().strip()
                if clean_gist in ["hi", "hello", "hey", "greetings"]:
                    raw_skeleton_output = "Hello! I am QNM, your AI assistant powered by Next-Concept Modeling (NCM). How can I help you today?"
                else:
                    raw_skeleton_output = f"I processed your query regarding '{coarse_gist}'. How can I assist you further?"
            skeleton_tokens = max(1, len(raw_skeleton_output.split()))

        # Stage 4: NCM Parallel Expansion
        with tracker.track("stage4_ncm_expand"):
            expanded_text = self.expander.expand(
                skeleton_blueprint=raw_skeleton_output,
                symbol_table=None,  # Expand before restoring symbols
                cfg_scale=self.settings.compression.cfg_scale,
                mbr_candidates=self.settings.compression.mbr_candidates
            )

        # Stage 5: Symbol Restoration & Memory Cache
        with tracker.track("stage5_symbol_restore"):
            final_response = self.guard.restore(expanded_text, symbol_table)
            if not final_response.strip():
                final_response = "I have processed your request. How can I assist you further?"
            final_output_tokens = max(1, len(final_response.split()))

            # Persist session turn
            self.store.save_turn(
                session_id=session_id,
                role="user",
                raw_prompt=prompt,
                masked_prompt=masked_input,
                symbol_table=symbol_table,
                coarse_gist=coarse_gist,
                discrete_codes=discrete_codes
            )
            self.store.save_turn(
                session_id=session_id,
                role="assistant",
                raw_prompt=final_response,
                masked_prompt=expanded_text,
                symbol_table=symbol_table,
                coarse_gist=SkeletonDSL.parse(raw_skeleton_output).result_core,
                discrete_codes=[]
            )

        # Compute and record telemetry after all stages complete
        telemetry = PipelineTelemetry.compute(
            raw_context_tokens=raw_context_tokens,
            compressed_input_tokens=compressed_input_tokens,
            skeleton_output_tokens=skeleton_tokens,
            final_output_tokens=final_output_tokens,
            stage_latencies_ms=tracker.stages
        )
        self.store.record_telemetry(
            session_id=session_id,
            raw_tokens=raw_context_tokens,
            compressed_tokens=compressed_input_tokens,
            skeleton_tokens=skeleton_tokens,
            final_tokens=final_output_tokens,
            latency_ms=telemetry.total_latency_ms
        )

        return PipelineResult(
            session_id=session_id,
            expanded_response=final_response,
            raw_skeleton=raw_skeleton_output,
            coarse_gist=coarse_gist,
            symbols_mapped=len(symbol_table.symbol_to_literal),
            telemetry=telemetry,
            model_used=model or getattr(self.provider, "model_id", "default")
        )
