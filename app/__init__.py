"""Flask application factory for the Health Team 1 proof of concept."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from flask import Flask
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix

DEFAULT_AGGREGATION_SYSTEM_PROMPT = (
    "Given the following JSON formatted responses to a medical query, "
    "summarise both the recommendation and the consistency between different "
    "agents. Highlight any marked differences. Similarly, summarise the "
    "explanation, citation, and assumptions."
)

DEFAULT_QUERY_AGENTS = [
    {
        "port": 11434,
        "model": "gemma3:27b",
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "repeat_penalty": 1.1,
    },
    {
        "port": 11435,
        "model": "phi4:latest",
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "repeat_penalty": 1.1,
    },
    {
        "port": 11435,
        "model": "dolphin-mixtral:latest",
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "repeat_penalty": 1.1,
    },
]


def create_app(test_config: dict | None = None) -> Flask:
    """Application factory used by Flask.

    Loads configuration from environment variables, instance config, and an
    optional ``test_config`` dict. Registers core blueprints and prepares
    folders used by the application.
    """

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    default_system_prompt = _load_default_system_prompt(
        project_root / "guidelines" / "default-prompt.txt"
    )

    app = Flask(__name__, instance_relative_config=True)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1, x_proto=1, x_host=1)

    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "dev-secret-key"),
        OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY"),
        LLM_DEFAULT_PROVIDER=os.environ.get("LLM_DEFAULT_PROVIDER", "ollama"),
        LLM_REQUEST_TIMEOUT=int(os.environ.get("LLM_REQUEST_TIMEOUT", 120)),
        LLM_OLLAMA_HOST=os.environ.get("LLM_OLLAMA_HOST", "localhost"),
        LLM_OLLAMA_PORT=int(os.environ.get("LLM_OLLAMA_PORT", 11434)),
        LLM_OLLAMA_SCHEME=os.environ.get("LLM_OLLAMA_SCHEME", "http"),
        LLM_OLLAMA_MODEL=os.environ.get("LLM_OLLAMA_MODEL", "gemma3:27b"),
        LLM_OLLAMA_OPTIONS=_safe_json(os.environ.get("LLM_OLLAMA_OPTIONS")),
        MULTI_AGENT_OLLAMA_PORTS=_safe_port_list(
            os.environ.get("LLM_MULTI_AGENT_PORTS"), [11434, 11435]
        ),
        QUERY_AGENTS=_safe_query_agents(
            os.environ.get("LLM_QUERY_AGENTS"),
            _default_query_agents,
        ),
        LLM_OPENAI_URL=os.environ.get(
            "LLM_OPENAI_URL", "https://api.openai.com/v1/chat/completions"
        ),
        LLM_OPENAI_MODEL=os.environ.get("LLM_OPENAI_MODEL", "gpt-3.5-turbo"),
        LLM_OPENAI_TEMPERATURE=_safe_float(os.environ.get("LLM_OPENAI_TEMPERATURE")),
        LLM_MULTI_AGENT_SUMMARY_PROMPT=os.environ.get(
            "LLM_MULTI_AGENT_SUMMARY_PROMPT", DEFAULT_AGGREGATION_SYSTEM_PROMPT
        ),
        LLM_SYSTEM_PROMPT=_system_prompt_from_env(
            os.environ.get("LLM_SYSTEM_PROMPT"), default_system_prompt
        ),
    )

    if test_config is not None:
        app.config.update(test_config)

    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        # In containerized deployments the instance folder may already exist.
        pass

    from .routes import bp as main_bp

    app.register_blueprint(main_bp)

    return app


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_json(value: str | None) -> dict | None:
    if not value:
        return None
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def _safe_query_agents(
    value: str | None, default_factory: Callable[[], list[dict]]
) -> list[dict]:
    if not value:
        return default_factory()

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return default_factory()

    if not isinstance(parsed, list):
        return default_factory()

    agents: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            port = int(item["port"])
        except (KeyError, TypeError, ValueError):
            continue

        agents.append(
            {
                "port": port,
                "model": str(item.get("model", "")) or None,
                "temperature": _safe_float_value(item.get("temperature")),
                "top_k": _safe_int_value(item.get("top_k")),
                "top_p": _safe_float_value(item.get("top_p")),
                "repeat_penalty": _safe_float_value(item.get("repeat_penalty")),
            }
        )

    return agents or default_factory()


def _default_query_agents() -> list[dict]:
    return [agent.copy() for agent in DEFAULT_QUERY_AGENTS]


def _load_default_system_prompt(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _system_prompt_from_env(value: str | None, default: str | None) -> str | None:
    if value is not None:
        return value
    return default


def _safe_float_value(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int_value(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_port_list(value: str | None, default: list[int] | None = None) -> list[int]:
    if not value:
        return list(default or [])

    ports: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            ports.append(int(stripped))
        except ValueError:
            continue

    return ports or list(default or [])


__all__ = ["create_app"]
