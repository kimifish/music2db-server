from __future__ import annotations

from typing import Any, Optional

import httpx
from pydantic import BaseModel

from .settings import Settings


class EmbeddingsServiceError(RuntimeError):
    """Raised when the external embeddings service fails."""


class EmbeddingsResponse(BaseModel):
    model: str
    input_type: str
    dimensions: int
    embeddings: list[list[float]]


def get_embeddings_settings(cfg: Settings, log: Any) -> tuple[str, Optional[str], bool, float]:
    embeddings_cfg = cfg.embeddings

    base_url = str(embeddings_cfg.base_url).rstrip("/")
    model_name = str(embeddings_cfg.model) if embeddings_cfg.model else None
    normalize = bool(getattr(embeddings_cfg, "normalize", True))
    timeout_seconds = float(getattr(embeddings_cfg, "timeout_seconds", 30))
    return base_url, model_name, normalize, timeout_seconds


def request_embeddings(cfg: Settings, log: Any, texts: list[str], input_type: str) -> EmbeddingsResponse:
    if not texts:
        raise ValueError("texts must not be empty")

    base_url, model_name, normalize, timeout_seconds = get_embeddings_settings(cfg, log)
    payload: dict[str, Any] = {
        "texts": texts,
        "input_type": input_type,
        "normalize": normalize,
    }
    if model_name:
        payload["model"] = model_name

    log.debug("`http` POST %s/v1/embeddings texts=%s input_type=%s", base_url, len(texts), input_type)

    try:
        with httpx.Client(timeout=timeout_seconds) as http_client:
            response = http_client.post(f"{base_url}/v1/embeddings", json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            error_payload = exc.response.json()
            detail = error_payload.get("error", {}).get("message", detail)
        except ValueError:
            pass
        raise EmbeddingsServiceError(
            f"Embeddings service returned {exc.response.status_code}: {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise EmbeddingsServiceError(f"Embeddings service request failed: {exc}") from exc

    return EmbeddingsResponse.model_validate(response.json())


def generate_embedding(cfg: Settings, log: Any, text: str, input_type: str) -> list[float]:
    return request_embeddings(cfg, log, [text], input_type).embeddings[0]


def generate_embeddings(cfg: Settings, log: Any, texts: list[str], input_type: str) -> list[list[float]]:
    return request_embeddings(cfg, log, texts, input_type).embeddings


def get_embeddings_health(cfg: Settings, log: Any) -> dict[str, Any]:
    base_url, model_name, _, timeout_seconds = get_embeddings_settings(cfg, log)

    try:
        log.debug("`http` GET %s/health", base_url)
        with httpx.Client(timeout=timeout_seconds) as http_client:
            response = http_client.get(f"{base_url}/health")
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise EmbeddingsServiceError(f"Embeddings health check failed: {exc}") from exc

    loaded_model = payload.get("model")
    if model_name and loaded_model and loaded_model != model_name:
        raise EmbeddingsServiceError(
            f"Configured embeddings model '{model_name}' does not match loaded model '{loaded_model}'"
        )

    return payload
