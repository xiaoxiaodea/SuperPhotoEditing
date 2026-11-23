from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SUPERPHOTO_",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path = Path("./data")
    host: str = "127.0.0.1"
    port: int = 8765
    access_token: str = Field(default="", repr=False)
    max_upload_mb: int = 80
    working_edge: int = 2048
    preview_edge: int = 1600
    critique_edge: int = 1280
    # auto preserves Ollama -> cloud -> local; explicit cloud never probes Ollama.
    vision_provider: Literal["auto", "ollama", "openai", "xai", "local"] = "auto"
    openai_api_key: str = Field(
        default="", repr=False,
        validation_alias=AliasChoices("SUPERPHOTO_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    openai_protocol: Literal["chat/completions", "responses"] = "chat/completions"
    xai_protocol: Literal["chat/completions", "responses"] = "responses"
    vision_timeout: float = Field(default=90.0, gt=0, le=600)
    vision_max_retries: int = Field(default=1, ge=0, le=3)
    # auto: schema -> JSON object -> prompt, only on a format-specific rejection.
    vision_schema_mode: Literal["auto", "json_schema", "json_object", "prompt"] = "auto"
    # Client side: let other processes on this machine use the unlocked vault through a
    # loopback endpoint guarded by a random per-run token (data_dir/vision-bridge.token).
    # They get redacted answers and slot labels only, never keys, addresses or model ids.
    vision_bridge: bool = False
    # Caller side: the running client's URL and that token. Used in auto, before .env providers.
    vision_bridge_url: str = ""
    vision_bridge_token: str = Field(default="", repr=False)
    xai_api_key: str = Field(
        default="", repr=False,
        validation_alias=AliasChoices("XAI_API_KEY", "SUPERPHOTO_XAI_API_KEY"),
    )
    xai_model: str = Field(
        default="grok-4.6",
        validation_alias=AliasChoices("XAI_MODEL", "SUPERPHOTO_XAI_MODEL"),
    )
    xai_base_url: str = "https://api.x.ai/v1"
    ollama_enabled: bool = True
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3-vl:32b"
    ollama_timeout: float = 360.0
    semantic_enabled: bool = True
    semantic_model: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    semantic_min_confidence: float = 0.48
    lightroom_enabled: bool = True
    lightroom_mcp_server: Path | None = None
    # Executable lightcraft-cli. Empty uses PATH, then vendor/lightcraft/lightcraft-cli.
    lightcraft_bin: Path | None = None
    lightroom_export_edge: int = 2400
    lightroom_qa_min_score: float = 82.0
    lightroom_qa_max_refinements: int = 2
    lightroom_candidate_min_distance: float = 5.0
    # Legacy setting name: enables bounded diversity retries, never a quality failure.
    lightroom_require_distinct_candidates: bool = True
    visual_review_enabled: bool = True
    # Inspect an actual export for unfinished light/color work, then compare
    # the independent new render with the retained first version.
    # Remains experimental: the studied local critic missed obvious remaining
    # weaknesses and rejected a visually stronger reference-guided candidate.
    finishing_review_enabled: bool = False
    finishing_max_rounds: int = Field(default=2, ge=0, le=2)
    # auto: when a vision model and LightCraft are both available, the vision
    # model reads the baseline, writes native settings with linear/radial masks
    # and reviews the renders; otherwise the legacy slider pipeline runs.
    # directed: fail instead of falling back. legacy: never use the vision planner.
    pipeline_mode: Literal["auto", "directed", "legacy"] = "auto"
    # Review-and-revise rounds per candidate after its first render.
    directed_max_revisions: int = Field(default=2, ge=0, le=3)
    # Planning returns three complete versions; reasoning models need longer than a review.
    directed_vision_timeout: float = Field(default=240.0, gt=0, le=900)

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "superphoto.db"


def get_settings() -> Settings:
    return Settings()
