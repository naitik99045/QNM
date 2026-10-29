"""
High-Precision Telemetry and Token Compression Metrics Engine.
Calculates token consumption across uncompressed vs. compressed flows,
measures per-stage latencies, and tracks computational efficiency.
"""

import time
from typing import Dict, Any
from pydantic import BaseModel, Field


class PipelineTelemetry(BaseModel):
    """Telemetry data capturing compression and generation efficiency."""
    raw_context_tokens: int = 0
    compressed_input_tokens: int = 0
    skeleton_output_tokens: int = 0
    final_output_tokens: int = 0
    context_saved_percent: float = 0.0
    generation_saved_percent: float = 0.0
    stage_latencies_ms: Dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0

    @classmethod
    def compute(
        cls,
        raw_context_tokens: int,
        compressed_input_tokens: int,
        skeleton_output_tokens: int,
        final_output_tokens: int,
        stage_latencies_ms: Dict[str, float]
    ) -> "PipelineTelemetry":
        ctx_saved = max(0, raw_context_tokens - compressed_input_tokens)
        ctx_pct = (ctx_saved / raw_context_tokens * 100.0) if raw_context_tokens > 0 else 0.0

        gen_saved = max(0, final_output_tokens - skeleton_output_tokens)
        gen_pct = (gen_saved / final_output_tokens * 100.0) if final_output_tokens > 0 else 0.0

        total_ms = sum(stage_latencies_ms.values())

        return cls(
            raw_context_tokens=raw_context_tokens,
            compressed_input_tokens=compressed_input_tokens,
            skeleton_output_tokens=skeleton_output_tokens,
            final_output_tokens=final_output_tokens,
            context_saved_percent=round(ctx_pct, 1),
            generation_saved_percent=round(gen_pct, 1),
            stage_latencies_ms={k: round(v, 2) for k, v in stage_latencies_ms.items()},
            total_latency_ms=round(total_ms, 2)
        )

    def summary_string(self) -> str:
        return (
            f"Context Saved: {self.context_saved_percent}% ({self.raw_context_tokens} -> {self.compressed_input_tokens} tok) | "
            f"Generation Saved: {self.generation_saved_percent}% ({self.final_output_tokens} -> {self.skeleton_output_tokens} tok) | "
            f"Latency: {self.total_latency_ms}ms"
        )


class LatencyTracker:
    """Timer context manager for measuring pipeline stage latency."""

    def __init__(self):
        self.stages: Dict[str, float] = {}

    def track(self, stage_name: str):
        tracker = self

        class StageContext:
            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration_ms = (time.perf_counter() - self.start) * 1000.0
                tracker.stages[stage_name] = duration_ms

        return StageContext()
