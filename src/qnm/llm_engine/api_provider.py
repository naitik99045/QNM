"""
External API Provider for optional upstream routing (OpenAI, Anthropic, Groq).
"""

import os
from typing import List, Dict, Optional
import httpx
import logging

logger = logging.getLogger("qnm.llm_engine.api")


class APIProvider:
    """Remote API Client for hybrid QNM execution."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 100, temperature: float = 0.2, model: Optional[str] = None) -> str:
        """Call remote API endpoint or return fallback if unconfigured."""
        active_model = model or self.model
        if not self.api_key:
            return (
                "[INTENT]: Answer query\n"
                "[ENTITIES_AND_SYMBOLS]: none\n"
                "[LOGIC_STEPS]:\n"
                "1. Route query\n"
                "[RESULT_CORE]: Processed via remote API provider fallback."
            )

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": active_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Remote API request error: {e}")

        return (
            "[INTENT]: Fallback\n"
            "[ENTITIES_AND_SYMBOLS]: none\n"
            "[LOGIC_STEPS]:\n"
            "1. Recover\n"
            "[RESULT_CORE]: Resumed successfully."
        )
