"""Utility helpers for talking to different large language model APIs."""

from __future__ import annotations

import concurrent.futures
import json
from typing import Any

import requests
from flask import current_app


class LLMError(RuntimeError):
    """Raised when an LLM provider cannot return a response."""


def generate_response(
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Send ``prompt`` to the configured provider and return the generated text."""

    selected_provider = provider or current_app.config.get(
        "LLM_DEFAULT_PROVIDER", "ollama"
    )
    if selected_provider == "ollama":
        return _generate_with_ollama(prompt, model=model, system_prompt=system_prompt)
    if selected_provider == "openai":
        return _generate_with_openai(prompt, model=model, system_prompt=system_prompt)
    raise LLMError(f"Unsupported LLM provider '{selected_provider}'.")


def _generate_with_ollama(
    prompt: str, *, model: str | None = None, system_prompt: str | None = None
) -> str:
    host = current_app.config.get("LLM_OLLAMA_HOST", "localhost")
    port = current_app.config.get("LLM_OLLAMA_PORT", 11434)
    scheme = current_app.config.get("LLM_OLLAMA_SCHEME", "http")
    default_model = current_app.config.get("LLM_OLLAMA_MODEL", "llama3.1:latest")
    timeout = current_app.config.get("LLM_REQUEST_TIMEOUT", 60)
    options = current_app.config.get("LLM_OLLAMA_OPTIONS")

    return _send_ollama_request(
        host,
        port,
        scheme,
        prompt,
        model or default_model,
        system_prompt,
        options if isinstance(options, dict) else None,
        timeout,
    )


def generate_multi_agent_responses(
    prompt: str,
    *,
    model: str | None = None,
    ports: list[int] | None = None,
    system_prompt: str | None = None,
) -> dict[int, dict[str, str | None]]:
    """Send the prompt to multiple Ollama agents concurrently.

    Returns a mapping of agent ports to dictionaries containing ``response`` and
    ``error`` keys so the caller can render successful and failed agents.
    """

    host = current_app.config.get("LLM_OLLAMA_HOST", "localhost")
    scheme = current_app.config.get("LLM_OLLAMA_SCHEME", "http")
    timeout = current_app.config.get("LLM_REQUEST_TIMEOUT", 60)
    default_ports = current_app.config.get("MULTI_AGENT_OLLAMA_PORTS", [11434, 11435])
    agent_ports = ports or default_ports
    default_model = current_app.config.get("LLM_OLLAMA_MODEL", "llama3.1:latest")
    options = current_app.config.get("LLM_OLLAMA_OPTIONS")
    resolved_model = model or default_model
    payload_options = options if isinstance(options, dict) else None

    results: dict[int, dict[str, str | None]] = {}

    def _request_agent(target_port: int) -> str:
        return _send_ollama_request(
            host,
            target_port,
            scheme,
            prompt,
            resolved_model,
            system_prompt,
            payload_options,
            timeout,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(agent_ports)) as executor:
        future_map = {executor.submit(_request_agent, port): port for port in agent_ports}
        for future in concurrent.futures.as_completed(future_map):
            port = future_map[future]
            try:
                response_text = future.result()
                results[port] = {"response": response_text, "error": None}
            except Exception as exc:  # pragma: no cover - defensive
                results[port] = {"response": None, "error": str(exc)}

    return results


def _generate_with_openai(
    prompt: str, *, model: str | None = None, system_prompt: str | None = None
) -> str:
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
        "messages": [],
    }
    if system_prompt:
        payload["messages"].append({"role": "system", "content": system_prompt})
    payload["messages"].append({"role": "user", "content": prompt})
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


def _send_ollama_request(
    host: str,
    port: int,
    scheme: str,
    prompt: str,
    model: str,
    system_prompt: str | None,
    options: dict[str, Any] | None,
    timeout: int,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
    }
    if options:
        payload["options"] = options

    if system_prompt:
        payload["system"] = system_prompt

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


__all__ = ["generate_response", "generate_multi_agent_responses", "LLMError"]
