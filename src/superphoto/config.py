from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SUPERPHOTO_",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    host: str = "127.0.0.1"
    port: int = 8765
    max_upload_mb: int = 80
    working_edge: int = 2048
    preview_edge: int = 1600
    critique_edge: int = 1280
    xai_api_key: str = Field(
        default="",
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

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "superphoto.db"


def get_settings() -> Settings:
    return Settings()
