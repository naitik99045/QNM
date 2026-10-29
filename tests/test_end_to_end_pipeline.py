"""
Tests for End-to-End QNM Pipeline and Telemetry Engine.
"""

import os
import tempfile
from qnm.pipeline.orchestrator import QNMOrchestrator
from qnm.memory.store import MemoryStore


def test_pipeline_end_to_end_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_e2e_memory.db")
        store = MemoryStore(db_path=db_path)
        orchestrator = QNMOrchestrator(memory_store=store)

        prompt = (
            "Please analyze source code in C:\\Users\\hp\\Desktop\\QNM\\src\\qnm\\settings.py "
            "and verify that flag --fast-mode is supported on http://127.0.0.1:1234/v1."
        )

        result = orchestrator.process_query(prompt, session_id="test_e2e_session")

        # Verify symbols preserved in final response
        assert "C:\\Users\\hp\\Desktop\\QNM\\src\\qnm\\settings.py" in result.expanded_response or "--fast-mode" in result.expanded_response
        assert result.symbols_mapped >= 2
        assert result.telemetry.total_latency_ms > 0

        # Verify telemetry structure
        tel = result.telemetry
        assert "stage1_symbol_mask" in tel.stage_latencies_ms
        assert "stage2_ncm_compress" in tel.stage_latencies_ms
        assert "stage3_reasoning" in tel.stage_latencies_ms
        assert "stage4_ncm_expand" in tel.stage_latencies_ms
        assert "stage5_symbol_restore" in tel.stage_latencies_ms

        # Verify memory storage
        turns = store.get_session_turns("test_e2e_session")
        assert len(turns) == 2  # user + assistant turns recorded
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"
