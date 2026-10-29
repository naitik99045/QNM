"""
Tests for LM Studio Provider and Provider Factory.
"""

from qnm.llm_engine.lmstudio_provider import LMStudioProvider
from qnm.llm_engine import get_reasoning_provider


def test_lmstudio_provider_initialization():
    provider = LMStudioProvider(
        base_url="http://127.0.0.1:1234/v1",
        cli_path="C:\\Users\\hp\\.lmstudio\\bin\\lms.exe"
    )
    assert provider.base_url == "http://127.0.0.1:1234/v1"
    assert provider.cli_path == "C:\\Users\\hp\\.lmstudio\\bin\\lms.exe"


def test_lmstudio_fallback_generation_preserves_symbols():
    provider = LMStudioProvider()
    messages = [
        {"role": "user", "content": "Analyze file $SYM_0 with flag $SYM_1"}
    ]
    output = provider._generate_fallback(messages)
    assert "[INTENT]:" in output
    assert "$SYM_0" in output
    assert "$SYM_1" in output
    assert "[RESULT_CORE]:" in output


def test_provider_factory():
    provider = get_reasoning_provider()
    assert provider is not None
    assert hasattr(provider, "generate")
