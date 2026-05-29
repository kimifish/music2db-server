from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "music2db_server"
    host: str = "0.0.0.0"
    port: int = 5005
    logging_config: Path | None = None


class EmbeddingsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://127.0.0.1:8098"
    model: str | None = "intfloat/multilingual-e5-small"
    normalize: bool = True
    timeout_seconds: float = 30


class ChromaDBSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str
    port: int = 8000
    collection_name: str = "music_collection"


class AdminSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clear_collection_enabled: bool = False


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppSettings = Field(default_factory=AppSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    chromadb: ChromaDBSettings
    admin: AdminSettings = Field(default_factory=AdminSettings)
