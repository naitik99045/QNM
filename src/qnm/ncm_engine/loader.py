"""
NCM Model Loader and Fallback Driver.
Loads tokenizer.pt, generator.pt, and config.yaml from artifacts/ncm/
or activates an algorithmic multiscale compression/expansion fallback.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("qnm.ncm_engine.loader")


class NCMLoader:
    """Manages loading of NCM checkpoints and fallback execution."""

    def __init__(self, ncm_dir: str = "artifacts/ncm", checkpoint_name: str = "xsum"):
        self.ncm_dir = Path(ncm_dir)
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = self.ncm_dir / checkpoint_name
        self.is_real_weights_loaded = False
        self.device = "cpu"
        self.config: Dict[str, Any] = {}
        self.tokenizer = None
        self.generator = None

        self._initialize()

    def _initialize(self):
        # Detect PyTorch and CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

        # Check if actual checkpoint files exist
        config_file = self.checkpoint_path / "config.yaml"
        tokenizer_file = self.checkpoint_path / "tokenizer.pt"
        generator_file = self.checkpoint_path / "generator.pt"

        if config_file.exists() and tokenizer_file.exists() and generator_file.exists():
            try:
                import torch
                import yaml
                with open(config_file, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Loading NCM PyTorch weights from {self.checkpoint_path} onto {self.device}...")
                self.tokenizer = torch.load(tokenizer_file, map_location=self.device)
                self.generator = torch.load(generator_file, map_location=self.device)
                self.is_real_weights_loaded = True
                return
            except Exception as e:
                logger.warning(f"Failed to load PyTorch checkpoints ({e}). Activating Algorithmic Fallback.")

        # Activate fallback mode
        self.is_real_weights_loaded = False
        self.config = {
            "model_type": "ncm_fallback_multiscale",
            "coarse_levels": 2,
            "total_levels": 4,
            "vocab_size": 4096,
            "device": self.device
        }
        logger.info(f"NCM initialized in Algorithmic Fallback mode (device: {self.device}).")
