"""Ollama-only local inference client.

This module deliberately has no cloud-provider fallback. All model requests
are sent to the local Ollama HTTP server configured by OLLAMA_BASE_URL.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

logger = logging.getLogger("sellbird.local_ai.ollama")

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"


class OllamaError(RuntimeError):
    """Base error for local Ollama operations."""


class OllamaUnavailable(OllamaError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaModelNotFound(OllamaError):
    """Raised when the configured model has not been downloaded locally."""


@dataclass(frozen=True)
class OllamaSettings:
    """Connection settings for the local Ollama server."""

    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_OLLAMA_MODEL
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "OllamaSettings":
        # Load the project .env even when Brain is created by a CLI path that
        # reads only the brain section and does not load the full app config.
        load_dotenv(override=False)
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
        model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
        timeout_raw = os.getenv("OLLAMA_TIMEOUT_SECONDS", "120").strip()
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be a number") from exc
        if not base_url:
            raise ValueError("OLLAMA_BASE_URL cannot be empty")
        if not model:
            raise ValueError("OLLAMA_MODEL cannot be empty")
        if timeout <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than zero")
        return cls(base_url=base_url.rstrip("/"), model=model, timeout_seconds=timeout)


class OllamaClient:
    """Small async client for Ollama's local `/api` endpoints."""

    def __init__(
        self,
        settings: OllamaSettings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or OllamaSettings.from_env()
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=self.settings.base_url,
            timeout=httpx.Timeout(self.settings.timeout_seconds),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def is_available(self) -> bool:
        """Return whether the local Ollama server responds to its version API."""
        try:
            response = await self._http.get(f"{self.settings.base_url}/api/version")
            response.raise_for_status()
            return True
        except (httpx.HTTPError, OSError):
            return False

    async def list_models(self) -> list[str]:
        """Return names of models already downloaded into local Ollama."""
        try:
            response = await self._http.get(f"{self.settings.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            raise OllamaUnavailable(
                "Ollama is not running. Start Ollama and retry; no cloud fallback is available."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not read local Ollama models: {exc}") from exc
        except ValueError as exc:
            raise OllamaError("Ollama returned an invalid model-list response") from exc

        models = payload.get("models", []) if isinstance(payload, dict) else []
        return [str(item.get("name", "")) for item in models if isinstance(item, dict)
                and item.get("name")]

    async def require_model(self) -> None:
        """Fail clearly unless the configured model is present locally."""
        available = await self.list_models()
        if self.settings.model not in available:
            raise OllamaModelNotFound(
                f"Model '{self.settings.model}' is not installed in Ollama. "
                f"Run: ollama pull {self.settings.model}"
            )

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        expect_json: bool = False,
    ) -> str:
        """Generate text using the configured local model only."""
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if expect_json:
            payload["format"] = "json"

        logger.info(
            "LOCAL INFERENCE provider=ollama model=%s endpoint=%s",
            self.settings.model,
            self.settings.base_url,
        )
        try:
            response = await self._http.post(
                f"{self.settings.base_url}/api/chat", json=payload
            )
            if response.status_code == 404:
                detail = response.text.lower()
                if "model" in detail and "not found" in detail:
                    raise OllamaModelNotFound(
                        f"Model '{self.settings.model}' is not installed. "
                        f"Run: ollama pull {self.settings.model}"
                    )
            response.raise_for_status()
            result = response.json()
        except OllamaModelNotFound:
            raise
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            raise OllamaUnavailable(
                "Cannot connect to local Ollama. Start Ollama and retry; "
                "the application will not switch to a cloud model."
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Local Ollama inference failed: {exc}") from exc
        except ValueError as exc:
            raise OllamaError("Ollama returned an invalid inference response") from exc

        message = result.get("message", {}) if isinstance(result, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned an empty model response")
        return content.strip()
