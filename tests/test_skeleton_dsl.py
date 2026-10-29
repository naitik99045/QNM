"""
Tests for Concept-Skeleton DSL Formatter and Parser.
"""

from qnm.llm_engine.skeleton_dsl import SkeletonDSL, ConceptSkeleton


def test_skeleton_dsl_prompt_construction():
    messages = SkeletonDSL.build_prompt_messages(
        compressed_context="[User]: System overview",
        current_query="Run $SYM_0 now"
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Concept-Skeleton" in messages[0]["content"]
    assert "$SYM_0" in messages[-1]["content"]

    # With few-shot examples enabled
    messages_fs = SkeletonDSL.build_prompt_messages(
        compressed_context="[User]: System overview",
        current_query="Run $SYM_0 now",
        include_few_shot=True
    )
    assert len(messages_fs) == 4


def test_skeleton_dsl_parser_structured():
    sample_output = (
        "[INTENT]: Recompile dependencies and run tests\n"
        "[ENTITIES_AND_SYMBOLS]: $SYM_0, $SYM_1\n"
        "[LOGIC_STEPS]:\n"
        "1. Load configuration from $SYM_0\n"
        "2. Execute test command with $SYM_1\n"
        "[RESULT_CORE]: Tests completed successfully with zero failures."
    )

    skeleton = SkeletonDSL.parse(sample_output)
    assert skeleton.intent == "Recompile dependencies and run tests"
    assert "$SYM_0" in skeleton.entities_and_symbols
    assert "$SYM_1" in skeleton.entities_and_symbols
    assert len(skeleton.logic_steps) == 2
    assert skeleton.logic_steps[0] == "Load configuration from $SYM_0"
    assert skeleton.result_core == "Tests completed successfully with zero failures."


def test_skeleton_dsl_parser_unstructured_fallback():
    raw_text = "Here is an answer without tags."
    skeleton = SkeletonDSL.parse(raw_text)
    assert skeleton.result_core == raw_text
