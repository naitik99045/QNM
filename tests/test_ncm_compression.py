"""
Tests for NCM Multiscale Compression and Parallel Expander Engine.
"""

import pytest
from qnm.ncm_engine.loader import NCMLoader
from qnm.ncm_engine.compressor import NCMCompressor
from qnm.ncm_engine.expander import NCMExpander
from qnm.guard.symbol_table import SymbolTable


def test_ncm_loader_fallback():
    loader = NCMLoader()
    assert loader.device in ["cuda", "mps", "cpu"]
    assert loader.config["model_type"] in ["ncm_fallback_multiscale", "ncm_pytorch"]


def test_ncm_compression_ratio_and_symbol_preservation():
    compressor = NCMCompressor(coarse_cutoff=2)

    # Construct a long text with multiple symbol tokens
    long_text = (
        "This is an extensive system specification document that details the architectural layers "
        "and runtime execution flow of the QNM engine. We must carefully verify that file $SYM_0 "
        "is correctly parsed, that configuration setting $SYM_1 is applied properly to the database, "
        "and that the network endpoint $SYM_2 is called without latency spikes or connection timeouts. "
        "Additionally, we want to ensure that all coarse concept codes are accurately captured in SQLite "
        "and that the resulting context window does not overflow or exhaust the available GPU memory. "
        "By enforcing a strict coarse level cutoff at level 2, fine syntactic filler is removed while "
        "preserving the core semantic essence and critical symbolic identifiers."
    )

    gist, codes, metrics = compressor.encode(long_text)

    # Verify all symbols are preserved
    assert "$SYM_0" in gist
    assert "$SYM_1" in gist
    assert "$SYM_2" in gist

    # Verify discrete codes were generated
    assert len(codes) > 0

    # Verify compression savings
    assert metrics["compressed_tokens"] < metrics["raw_tokens"]
    assert metrics["saved_percent"] >= 50.0  # Significant token reduction achieved
    assert metrics["compression_ratio"] > 1.5


def test_short_query_not_truncated():
    compressor = NCMCompressor(coarse_cutoff=2)

    # Short queries like greetings must not be truncated to empty string
    gist_hi, codes_hi, metrics_hi = compressor.encode("hi")
    assert gist_hi == "hi"
    assert metrics_hi["compressed_tokens"] == 1
    assert metrics_hi["saved_percent"] == 0.0

    # Short questions must remain intact
    gist_py, codes_py, metrics_py = compressor.encode("what is python?")
    assert gist_py == "what is python?"


def test_ncm_expander_with_mbr_and_symbol_restoration():
    expander = NCMExpander()
    table = SymbolTable()
    s0 = table.register("C:\\workspace\\config.yaml")
    s1 = table.register("--fast-mode")

    blueprint = (
        f"[INTENT]: Initialize system configuration\n"
        f"[ENTITIES_AND_SYMBOLS]: {s0}, {s1}\n"
        f"[LOGIC_STEPS]:\n"
        f"1. Read file {s0}\n"
        f"2. Apply flag {s1}\n"
        f"[RESULT_CORE]: System successfully configured using {s0} with {s1} enabled."
    )

    expanded = expander.expand(blueprint, symbol_table=table, mbr_candidates=3)

    # Verify symbols were restored
    assert "C:\\workspace\\config.yaml" in expanded
    assert "--fast-mode" in expanded
    assert "$SYM_" not in expanded
    assert "System successfully configured" in expanded
