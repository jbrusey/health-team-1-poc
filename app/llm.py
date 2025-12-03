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
    query_agents: list[dict[str, object]] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str | int | dict | None]]:
    """Send the prompt to multiple Ollama agents concurrently.

    Returns a list of dictionaries containing metadata about each agent along
    with ``response`` and ``error`` keys so the caller can render successful and
    failed agents.
    """

    host = current_app.config.get("LLM_OLLAMA_HOST", "localhost")
    scheme = current_app.config.get("LLM_OLLAMA_SCHEME", "http")
    timeout = current_app.config.get("LLM_REQUEST_TIMEOUT", 60)
    configured_agents = query_agents or current_app.config.get("QUERY_AGENTS", [])
    default_model = current_app.config.get("LLM_OLLAMA_MODEL", "llama3.1:latest")
    options = current_app.config.get("LLM_OLLAMA_OPTIONS")
    payload_options = options if isinstance(options, dict) else None

    results: list[dict[str, str | int | dict | None]] = []

    def _request_agent(target_port: int, target_model: str, target_options: dict[str, Any] | None) -> str:
        return _send_ollama_request(
            host,
            target_port,
            scheme,
            prompt,
            target_model,
            system_prompt,
            target_options,
            timeout,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configured_agents) or 1) as executor:
        future_map: dict[concurrent.futures.Future[str], dict[str, object]] = {}
        for idx, agent in enumerate(configured_agents):
            try:
                port = int(agent.get("port"))
            except (TypeError, ValueError):
                results.append(
                    {
                        "port": None,
                        "model": None,
                        "options": None,
                        "response": None,
                        "error": "Agent configuration missing a valid port.",
                        "index": idx,
                    }
                )
                continue

            resolved_model = agent.get("model") or default_model
            resolved_options = _merge_options(agent, payload_options)
            future_map[
                executor.submit(
                    _request_agent, port, str(resolved_model), resolved_options
                )
            ] = {
                "port": port,
                "model": resolved_model,
                "options": resolved_options,
                "index": idx,
            }

        for future in concurrent.futures.as_completed(future_map):
            metadata = future_map[future]
            try:
                response_text = future.result()
                results.append(
                    {
                        "port": metadata["port"],
                        "model": metadata["model"],
                        "options": metadata["options"],
                        "response": response_text,
                        "error": None,
                        "index": metadata["index"],
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive
                results.append(
                    {
                        "port": metadata["port"],
                        "model": metadata["model"],
                        "options": metadata["options"],
                        "response": None,
                        "error": str(exc),
                        "index": metadata["index"],
                    }
                )

    return sorted(results, key=lambda item: item.get("index", 0))


def aggregate_agent_responses(
    prompt: str,
    agent_results: list[dict[str, object]],
    *,
    query_agents: list[dict[str, object]] | None = None,
    system_prompt: str | None = None,
) -> dict[str, str | int | dict | None]:
    """Summarise responses by forwarding them to a designated agent.

    The first configured agent is used as the aggregation endpoint. The
    provided ``system_prompt`` is applied to distinguish this request from the
    user-facing prompt, and agent numbers are preserved in the payload sent to
    the summariser.
    """

    host = current_app.config.get("LLM_OLLAMA_HOST", "localhost")
    scheme = current_app.config.get("LLM_OLLAMA_SCHEME", "http")
    timeout = current_app.config.get("LLM_REQUEST_TIMEOUT", 60)
    configured_agents = query_agents or current_app.config.get("QUERY_AGENTS", [])
    default_model = current_app.config.get("LLM_OLLAMA_MODEL", "llama3.1:latest")
    options = current_app.config.get("LLM_OLLAMA_OPTIONS")
    payload_options = options if isinstance(options, dict) else None

    if not configured_agents:
        return {
            "response": None,
            "error": "No aggregator agent is configured.",
            "model": None,
            "port": None,
            "agent_number": None,
            "options": None,
        }

    aggregator = configured_agents[0]
    try:
        port = int(aggregator.get("port"))
    except (TypeError, ValueError):
        return {
            "response": None,
            "error": "Aggregator agent is missing a valid port.",
            "model": None,
            "port": None,
            "agent_number": None,
            "options": None,
        }

    resolved_model = aggregator.get("model") or default_model
    resolved_options = _merge_options(aggregator, payload_options)

    aggregation_payload = {
        "prompt": prompt,
        "agent_responses": [
            {
                "agent_number": idx + 1,
                "port": result.get("port"),
                "model": result.get("model"),
                "response": result.get("response"),
                "error": result.get("error"),
            }
            for idx, result in enumerate(agent_results)
        ],
    }

    aggregation_prompt = (
        "Summarise the following multi-agent outputs. Maintain references to agent numbers.\n"
        f"{json.dumps(aggregation_payload, indent=2)}"
    )

    try:
        summary = _send_ollama_request(
            host,
            port,
            scheme,
            aggregation_prompt,
            str(resolved_model),
            system_prompt,
            resolved_options,
            timeout,
        )
        return {
            "response": summary,
            "error": None,
            "model": resolved_model,
            "port": port,
            "agent_number": 1,
            "options": resolved_options,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "response": None,
            "error": str(exc),
            "model": resolved_model,
            "port": port,
            "agent_number": 1,
            "options": resolved_options,
        }


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


def _merge_options(
    agent_options: dict[str, object], base_options: dict[str, Any] | None
) -> dict[str, Any] | None:
    merged: dict[str, Any] = dict(base_options or {})
    for key in ("temperature", "top_k", "top_p", "repeat_penalty"):
        value = agent_options.get(key)
        if value is not None:
            merged[key] = value
        elif key not in merged:
            merged[key] = None
    return merged or None


__all__ = [
    "generate_response",
    "generate_multi_agent_responses",
    "aggregate_agent_responses",
    "LLMError",
]
