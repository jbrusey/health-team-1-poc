"""Utility helpers for talking to different large language model APIs."""

from __future__ import annotations

import json
from typing import Any

import requests
from flask import current_app


class LLMError(RuntimeError):
    """Raised when an LLM provider cannot return a response."""


def generate_response(
    prompt: str, *, provider: str | None = None, model: str | None = None
) -> str:
    """Send ``prompt`` to the configured provider and return the generated text."""

    selected_provider = provider or current_app.config.get(
        "LLM_DEFAULT_PROVIDER", "ollama"
    )
    if selected_provider == "ollama":
        return _generate_with_ollama(prompt, model=model)
    if selected_provider == "openai":
        return _generate_with_openai(prompt, model=model)
    raise LLMError(f"Unsupported LLM provider '{selected_provider}'.")


def _generate_with_ollama(prompt: str, *, model: str | None = None) -> str:
    host = current_app.config.get("LLM_OLLAMA_HOST", "localhost")
    port = current_app.config.get("LLM_OLLAMA_PORT", 11434)
    scheme = current_app.config.get("LLM_OLLAMA_SCHEME", "http")
    default_model = current_app.config.get("LLM_OLLAMA_MODEL", "llama3.1:latest")
    timeout = current_app.config.get("LLM_REQUEST_TIMEOUT", 60)
    payload: dict[str, Any] = {
        "model": model or default_model,
        "prompt": prompt,
    }
    options = current_app.config.get("LLM_OLLAMA_OPTIONS")
    if isinstance(options, dict):
        payload["options"] = options

    url = f"{scheme}://{host}:{port}/api/generate"
    print(f"OLLAMA request to {url} with model {payload['model']}")  # Debug log
    response = requests.post(url, json=payload, timeout=timeout, stream=True)
    print(f"OLLAMA response status: {response.status_code}")  # Debug log
    response.raise_for_status()

    chunks: list[str] = []
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if data.get("done"):
            break
        chunk = data.get("response")
        if chunk:
            chunks.append(chunk)

    if not chunks:
        data = response.json()
        text = data.get("response")
        if text:
            chunks.append(text)

    return "".join(chunks).strip()


def _generate_with_openai(prompt: str, *, model: str | None = None) -> str:
    api_key = current_app.config.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not configured.")

    url = current_app.config.get(
        "LLM_OPENAI_URL", "https://api.openai.com/v1/chat/completions"
    )
    default_model = current_app.config.get("LLM_OPENAI_MODEL", "gpt-3.5-turbo")
    timeout = current_app.config.get("LLM_REQUEST_TIMEOUT", 60)

    payload = {
        "model": model or default_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    temperature = current_app.config.get("LLM_OPENAI_TEMPERATURE")
    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(
        url,
        json=payload,
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
        raise LLMError("Unexpected response from OpenAI API.") from exc


__all__ = ["generate_response", "LLMError"]
