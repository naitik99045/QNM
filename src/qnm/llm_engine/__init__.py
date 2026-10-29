"""LLM Engine package: providers and Concept-Skeleton DSL."""
from qnm.llm_engine.skeleton_dsl import SkeletonDSL, ConceptSkeleton
from qnm.llm_engine.lmstudio_provider import LMStudioProvider
from qnm.llm_engine.gguf_provider import GGUFProvider
from qnm.llm_engine.api_provider import APIProvider
from qnm.settings import get_settings


def get_reasoning_provider():
    """Factory to return the active reasoning LLM provider based on settings."""
    settings = get_settings()
    mode = settings.execution_mode.lower()

    if mode == "lmstudio":
        return LMStudioProvider(
            base_url=settings.lmstudio.base_url,
            cli_path=settings.find_lms_cli(),
            model_id=settings.lmstudio.model_id,
            timeout_seconds=settings.lmstudio.timeout_seconds,
            auto_start=settings.lmstudio.auto_start
        )
    elif mode == "local_gguf":
        model_path = settings.resolve_path(f"{settings.artifacts.gguf_dir}/{settings.artifacts.default_gguf_name}")
        return GGUFProvider(model_path=str(model_path))
    elif mode == "api":
        return APIProvider()
    else:
        # Default to LM Studio
        return LMStudioProvider(
            base_url=settings.lmstudio.base_url,
            cli_path=settings.find_lms_cli(),
            model_id=settings.lmstudio.model_id
        )


__all__ = [
    "SkeletonDSL",
    "ConceptSkeleton",
    "LMStudioProvider",
    "GGUFProvider",
    "APIProvider",
    "get_reasoning_provider",
]
