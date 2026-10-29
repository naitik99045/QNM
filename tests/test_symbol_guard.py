"""
Tests for Literal Symbol Guard and Concept Memory Store.
"""

import os
import tempfile
import pytest
from qnm.guard.symbol_table import SymbolGuard, SymbolTable
from qnm.memory.store import MemoryStore
from qnm.memory.pruner import ContextPruner


def test_symbol_guard_mask_and_restore():
    guard = SymbolGuard()
    sample_text = (
        "Please inspect C:\\Users\\hp\\Desktop\\QNM\\src\\main.py and run "
        "`python -m pytest` with flag --verbose --timeout=30. "
        "Also check https://github.com/HazyResearch/ncm and function calculateTokenSavings."
    )

    masked_text, table = guard.mask(sample_text)

    # Verify placeholders exist
    assert "$SYM_" in masked_text
    assert len(table.symbol_to_literal) >= 4

    # Verify literal exact values were stored
    literals = list(table.symbol_to_literal.values())
    assert any("C:\\Users\\hp\\Desktop\\QNM\\src\\main.py" in lit for lit in literals)
    assert any("https://github.com/HazyResearch/ncm" in lit for lit in literals)
    assert any("--verbose" in lit for lit in literals)
    assert any("`python -m pytest`" in lit for lit in literals)

    # Verify 100% lossless restoration
    restored_text = guard.restore(masked_text, table)
    assert restored_text == sample_text


def test_symbol_guard_multiline_code_block():
    guard = SymbolGuard()
    code = "```python\ndef test_fn():\n    return 42\n```"
    text = f"Here is the implementation:\n{code}\nProceed to run."

    masked, table = guard.mask(text)
    assert code not in masked
    assert "$SYM_0" in masked

    restored = guard.restore(masked, table)
    assert restored == text


def test_memory_store_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_memory.db")
        store = MemoryStore(db_path=db_path)

        table = SymbolTable()
        sym = table.register("https://example.com/api")

        turn_id = store.save_turn(
            session_id="test_sess_1",
            role="user",
            raw_prompt="Connect to https://example.com/api",
            masked_prompt=f"Connect to {sym}",
            symbol_table=table,
            coarse_gist="User requests connection to remote endpoint",
            discrete_codes=[12, 45, 99]
        )
        assert turn_id is not None

        turns = store.get_session_turns("test_sess_1")
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        assert turns[0]["coarse_gist"] == "User requests connection to remote endpoint"
        assert turns[0]["discrete_codes"] == [12, 45, 99]

        # Test Telemetry logging
        store.record_telemetry(
            session_id="test_sess_1",
            raw_tokens=500,
            compressed_tokens=100,
            skeleton_tokens=40,
            final_tokens=180,
            latency_ms=45.2
        )
        stats = store.get_telemetry_stats()
        assert stats["total_requests"] == 1
        assert stats["context_saved_percent"] == 80.0
        assert stats["reasoning_saved_percent"] == pytest.approx(77.8, rel=1e-1)


def test_context_pruner_budget():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_pruner.db")
        store = MemoryStore(db_path=db_path)
        pruner = ContextPruner(store, max_tokens=50)

        # Insert multiple turns
        table = SymbolTable()
        for i in range(5):
            store.save_turn(
                session_id="sess_budget",
                role="user" if i % 2 == 0 else "assistant",
                raw_prompt=f"Step {i} query regarding system initialization and tokens",
                masked_prompt=f"Step {i} query",
                symbol_table=table,
                coarse_gist=f"Discussion of step {i} data processing",
                discrete_codes=[i, i + 1]
            )

        context = pruner.retrieve_context("sess_budget", "data processing", max_tokens=30)
        assert len(context) > 0
        assert ContextPruner.estimate_tokens(context) <= 30
