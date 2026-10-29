"""
LM Studio Provider Module.
Interfaces with local LM Studio OpenAI-compatible endpoint (http://127.0.0.1:1234/v1),
manages lms.exe server lifecycle, and enforces compact Concept-Skeleton generation.
"""

import time
import subprocess
import logging
from typing import List, Dict, Any, Optional
import httpx
from qnm.llm_engine.skeleton_dsl import ConceptSkeleton, SkeletonDSL

logger = logging.getLogger("qnm.llm_engine.lmstudio")


class LMStudioProvider:
    """Connector for local LM Studio instance."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        cli_path: Optional[str] = None,
        model_id: str = "qwen2.5-coder-3b-instruct-q2_k",
        timeout_seconds: float = 60.0,
        auto_start: bool = True
    ):
        self.base_url = base_url.rstrip("/")
        self.cli_path = cli_path
        self.model_id = model_id
        self.timeout = timeout_seconds
        self.auto_start = auto_start
        self._attempted_start = False

    def is_available(self) -> bool:
        """Check if the LM Studio local server is responding."""
        try:
            with httpx.Client(timeout=0.3) as client:
                res = client.get(f"{self.base_url}/models")
                return res.status_code == 200
        except Exception:
            return False

    def start_server_if_needed(self) -> bool:
        """Attempt to boot LM Studio server via CLI if currently offline."""
        if self.is_available():
            return True

        if not self.auto_start or not self.cli_path or self._attempted_start:
            return False

        self._attempted_start = True
        try:
            logger.info(f"Attempting to start LM Studio server via: {self.cli_path} server start")
            subprocess.Popen(
                f'"{self.cli_path}" server start',
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )
            # Fast check
            for _ in range(3):
                time.sleep(0.3)
                if self.is_available():
                    logger.info("LM Studio server successfully initialized.")
                    return True
        except Exception as e:
            logger.warning(f"Could not auto-start LM Studio server: {e}")

        return False

    def list_models(self) -> List[str]:
        """Fetch list of loaded or available models from LM Studio."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/models")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("id", "") for m in data.get("data", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model: Optional[str] = None
    ) -> str:
        """Generate a response using LM Studio, falling back to deterministic mock if unavailable."""
        if not self.is_available():
            self.start_server_if_needed()

        if self.is_available():
            try:
                # Resolve active model ID
                active_models = self.list_models()
                target_model = model or self.model_id

                if active_models:
                    # Exact match
                    if target_model in active_models:
                        selected_model = target_model
                    else:
                        # Partial match (e.g. "qwen" matches "qwen2.5-coder-3b-instruct")
                        matched = next((m for m in active_models if target_model.lower() in m.lower()), None)
                        selected_model = matched if matched else active_models[0]
                else:
                    selected_model = target_model

                payload = {
                    "model": selected_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }

                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(f"{self.base_url}/chat/completions", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            return choices[0].get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.warning(f"LM Studio API request error: {e}. Utilizing fallback generator.")

        # Fallback generator if LM Studio server is offline
        return self._generate_fallback(messages)

    def _generate_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Deterministic fallback Concept-Skeleton generator when local server is offline."""
        user_prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_prompt = m.get("content", "")
                break

        # Extract symbols present in prompt
        import re
        symbols = re.findall(r"\$SYM_\d+", user_prompt)
        sym_str = ", ".join(symbols) if symbols else "none"

        # Check for conversational greeting without symbols
        clean_prompt = user_prompt.lower().strip()
        if not symbols and clean_prompt in ["hi", "hello", "hey", "greetings", "good morning", "good evening"]:
            return "Hello! I am QNM, your AI assistant with integrated Next-Concept Modeling (NCM). How can I assist you today?"

        # Generate a structured Concept-Skeleton
        return (
            f"[INTENT]: Process and answer the query accurately\n"
            f"[ENTITIES_AND_SYMBOLS]: {sym_str}\n"
            f"[LOGIC_STEPS]:\n"
            f"1. Analyze compressed concept context\n"
            f"2. Synthesize resolution referencing {sym_str}\n"
            f"[RESULT_CORE]: Successfully analyzed query with symbolic entities {sym_str} preserved."
        )
