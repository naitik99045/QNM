"""
Tests for Custom Agent Presets and Dynamic Model Selection.
"""

import pytest
from qnm.agents.custom_agent import (
    load_yaml_presets,
    get_available_agents,
    resolve_agent,
    CustomAgent
)
from qnm.agents.code_agent import CodeAgent
from qnm.llm_engine.lmstudio_provider import LMStudioProvider


def test_load_yaml_presets():
    presets = load_yaml_presets()
    assert "writer" in presets
    assert "debugger" in presets
    assert "analyst" in presets
    assert "tools" in presets["writer"]


def test_get_available_agents_catalog():
    catalog = get_available_agents()
    # Built-in
    assert "code" in catalog
    assert "research" in catalog
    assert "shell" in catalog
    assert catalog["code"]["type"] == "builtin"

    # Custom YAML presets
    assert "writer" in catalog
    assert "debugger" in catalog
    assert "analyst" in catalog
    assert catalog["writer"]["type"] == "custom"


def test_resolve_builtin_and_custom_agents():
    # Built-in agent
    code_ag = resolve_agent("code", model="qwen2.5-coder-3b-instruct")
    assert isinstance(code_ag, CodeAgent)
    assert code_ag.model == "qwen2.5-coder-3b-instruct"

    # Custom YAML agent
    writer_ag = resolve_agent("writer", model="llama-3.2-3b")
    assert isinstance(writer_ag, CustomAgent)
    assert writer_ag.name == "QNM-WriterAgent"
    assert writer_ag.model == "llama-3.2-3b"
    assert "read_file" in writer_ag.tools._tools
    assert "write_file" in writer_ag.tools._tools
    assert "search_web" in writer_ag.tools._tools

    # Unknown agent raises ValueError
    with pytest.raises(ValueError) as exc:
        resolve_agent("non_existent_specialist")
    assert "Unknown agent" in str(exc.value)


def test_lmstudio_dynamic_model_routing():
    provider = LMStudioProvider()
    messages = [{"role": "user", "content": "Explain $SYM_0"}]
    # Test fallback generation with explicit model
    fallback = provider._generate_fallback(messages)
    assert "[INTENT]:" in fallback
    assert "$SYM_0" in fallback
