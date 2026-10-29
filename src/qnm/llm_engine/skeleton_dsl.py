"""
Concept-Skeleton DSL Module.
Enforces compact, high-density reasoning blueprints ([INTENT], [ENTITIES_AND_SYMBOLS],
[LOGIC_STEPS], [RESULT_CORE]) capped at 80 to 120 tokens instead of verbose prose.
"""

import re
from typing import Optional, List
from pydantic import BaseModel, Field


class ConceptSkeleton(BaseModel):
    """Structured representation of the Concept-Skeleton DSL."""
    intent: str = Field(default="")
    entities_and_symbols: List[str] = Field(default_factory=list)
    logic_steps: List[str] = Field(default_factory=list)
    result_core: str = Field(default="")
    raw_text: str = Field(default="")

    def to_dsl_string(self) -> str:
        """Render the skeleton into the standard DSL representation."""
        steps_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(self.logic_steps))
        symbols_str = ", ".join(self.entities_and_symbols)
        return (
            f"[INTENT]: {self.intent}\n"
            f"[ENTITIES_AND_SYMBOLS]: {symbols_str}\n"
            f"[LOGIC_STEPS]:\n{steps_str}\n"
            f"[RESULT_CORE]: {self.result_core}"
        )


class SkeletonDSL:
    """Formatter and parser for the Concept-Skeleton DSL contract."""

    SYSTEM_PROMPT = (
        "You are QNM (Quantized-Concept Network Model), an advanced AI assistant augmented by "
        "Next-Concept Modeling (NCM) and concept-skeleton reasoning.\n"
        "Your mission is to provide high-quality, articulate, accurate, and comprehensive responses, "
        "combining the intelligence of a normal AI assistant with precise concept processing.\n\n"
        "GUIDELINES:\n"
        "1. Natural AI Behavior: Respond naturally, helpfully, and articulately. For greetings, respond warmly. "
        "For explanations, coding, and problem-solving, provide complete, clear, runnable solutions without artificial brevity.\n"
        "2. Literal Symbol Guard: Tokens formatted as $SYM_0, $SYM_1, etc. denote protected literal entities "
        "(file paths, URLs, code blocks, or parameters). ALWAYS retain these exact $SYM_n placeholders in your response "
        "wherever relevant so the NCM expander can restore them losslessly.\n"
        "3. Concept-Skeleton Blueprints: If structured step-by-step reasoning or blueprints are requested, "
        "you may format your response using [INTENT], [ENTITIES_AND_SYMBOLS], [LOGIC_STEPS], and [RESULT_CORE]."
    )

    FEW_SHOT_EXAMPLE_USER = "Compress context and run $SYM_0 with flag $SYM_1."
    FEW_SHOT_EXAMPLE_ASSISTANT = (
        "[INTENT]: Execute command with specified flag.\n"
        "[ENTITIES_AND_SYMBOLS]: $SYM_0, $SYM_1\n"
        "[LOGIC_STEPS]:\n"
        "1. Verify $SYM_0 path\n"
        "2. Invoke with $SYM_1\n"
        "[RESULT_CORE]: Command $SYM_0 successfully invoked with $SYM_1."
    )

    @classmethod
    def build_prompt_messages(
        cls,
        compressed_context: str,
        current_query: str,
        include_few_shot: bool = False
    ) -> List[dict]:
        """Constructs chat messages for AI reasoning with NCM conceptual context."""
        messages = [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
        ]
        if include_few_shot:
            messages.append({"role": "user", "content": cls.FEW_SHOT_EXAMPLE_USER})
            messages.append({"role": "assistant", "content": cls.FEW_SHOT_EXAMPLE_ASSISTANT})

        user_content = current_query
        if compressed_context.strip():
            user_content = f"Conceptual Context:\n{compressed_context.strip()}\n\nUser Query:\n{current_query}"

        messages.append({"role": "user", "content": user_content})
        return messages

    @classmethod
    def parse(cls, raw_output: str) -> ConceptSkeleton:
        """Parses raw LLM output into a ConceptSkeleton object."""
        intent = cls._extract_field(raw_output, "INTENT") or ""
        symbols_str = cls._extract_field(raw_output, "ENTITIES_AND_SYMBOLS") or ""
        steps_str = cls._extract_field(raw_output, "LOGIC_STEPS") or ""
        result_core = cls._extract_field(raw_output, "RESULT_CORE") or ""

        # Parse symbols
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()] if symbols_str else []

        # Parse logic steps
        steps = []
        if steps_str:
            for line in steps_str.split("\n"):
                cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
                if cleaned:
                    steps.append(cleaned)

        # Fallback if the model did not output exact tags
        if not result_core and not intent:
            result_core = raw_output.strip()

        return ConceptSkeleton(
            intent=intent,
            entities_and_symbols=symbols,
            logic_steps=steps,
            result_core=result_core,
            raw_text=raw_output
        )

    @staticmethod
    def _extract_field(text: str, field_name: str) -> Optional[str]:
        pattern = rf"\[{field_name}\]:?\s*([\s\S]*?)(?=\n\[[A-Z_]+\]:?|$)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None
