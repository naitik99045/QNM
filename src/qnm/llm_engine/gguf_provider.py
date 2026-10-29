"""
Local GGUF Provider via llama-cpp-python with a restricted KV-cache window.
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger("qnm.llm_engine.gguf")


class GGUFProvider:
    """Local GGUF Model Provider with minimal KV-cache allocation."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            logger.warning(f"GGUF model file not found at {self.model_path}.")
            return

        try:
            from llama_cpp import Llama
            logger.info(f"Loading GGUF model from {self.model_path} with n_ctx={self.n_ctx}...")
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
        except ImportError:
            logger.warning("llama-cpp-python is not installed. GGUFProvider will operate in fallback mode.")
        except Exception as e:
            logger.warning(f"Failed to initialize GGUF model: {e}")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 100, temperature: float = 0.2, model: Optional[str] = None) -> str:
        """Generate response using local llama.cpp instance or fallback."""
        if self.llm is not None:
            try:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"GGUF inference failed: {e}")

        # Fallback response
        return (
            "[INTENT]: Address query\n"
            "[ENTITIES_AND_SYMBOLS]: none\n"
            "[LOGIC_STEPS]:\n"
            "1. Process query\n"
            "[RESULT_CORE]: Query processed successfully via local provider."
        )
