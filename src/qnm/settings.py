"""
Configuration Loader and Validator for QNM Engine.
Loads YAML configuration, validates schemas with Pydantic v2, and provides runtime settings.
"""

from pathlib import Path
from typing import List, Optional
import os
import shutil
import yaml
from pydantic import BaseModel, Field

# Determine project root directory
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class LMStudioSettings(BaseModel):
    model_config = {"protected_namespaces": ()}
    base_url: str = Field(default="http://127.0.0.1:1234/v1")
    cli_path: Optional[str] = Field(default=None)
    model_id: str = Field(default="qwen2.5-coder-3b-instruct-q2_k")
    timeout_seconds: float = Field(default=60.0)
    auto_start: bool = Field(default=True)


class ServerSettings(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")


class CompressionSettings(BaseModel):
    coarse_level_cutoff: int = Field(default=2)
    max_context_gist_tokens: int = Field(default=512)
    skeleton_max_tokens: int = Field(default=1024)
    cfg_scale: float = Field(default=1.0)
    mbr_candidates: int = Field(default=4)


class ArtifactsSettings(BaseModel):
    gguf_dir: str = Field(default="artifacts/gguf")
    ncm_dir: str = Field(default="artifacts/ncm")
    default_gguf_name: str = Field(default="qwen2.5-coder-3b-instruct-q2_k.gguf")
    ncm_checkpoints: List[str] = Field(default_factory=lambda: ["xsum", "owt", "lm1b"])


class MemorySettings(BaseModel):
    db_path: str = Field(default="artifacts/concept_memory.db")
    max_history_turns: int = Field(default=20)


class Settings(BaseModel):
    execution_mode: str = Field(default="lmstudio")
    project_root: Path = Field(default=DEFAULT_PROJECT_ROOT)
    lmstudio: LMStudioSettings = Field(default_factory=LMStudioSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    compression: CompressionSettings = Field(default_factory=CompressionSettings)
    artifacts: ArtifactsSettings = Field(default_factory=ArtifactsSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path against the project root."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def find_lms_cli(self) -> Optional[str]:
        """Locate the LM Studio lms.exe executable on the host system."""
        # Check explicit path in settings
        if self.lmstudio.cli_path and os.path.exists(self.lmstudio.cli_path):
            return self.lmstudio.cli_path

        # Check typical Windows locations
        candidates = [
            Path.home() / ".lmstudio" / "bin" / "lms.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "LM Studio" / "resources" / "bin" / "lms.exe",
            Path.home() / ".cache" / "lm-studio" / "bin" / "lms.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        # Check system PATH
        which_lms = shutil.which("lms") or shutil.which("lms.exe")
        if which_lms:
            return which_lms

        return None


_settings_instance: Optional[Settings] = None


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Load configuration from YAML file or return defaults."""
    if config_path is None:
        env_config = os.getenv("QNM_CONFIG")
        if env_config:
            config_path = Path(env_config)
        else:
            config_path = DEFAULT_PROJECT_ROOT / "config" / "default_qnm.yaml"

    config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                config_data = raw

    settings = Settings(**config_data)

    # Auto-detect LM Studio CLI if not explicitly set
    detected_cli = settings.find_lms_cli()
    if detected_cli:
        settings.lmstudio.cli_path = detected_cli

    return settings


def get_settings() -> Settings:
    """Singleton getter for application settings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = load_settings()
    return _settings_instance
