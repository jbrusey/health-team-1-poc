"""Flask application factory for the Health Team 1 proof of concept."""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask
from dotenv import load_dotenv


def create_app(test_config: dict | None = None) -> Flask:
    """Application factory used by Flask.

    Loads configuration from environment variables, instance config, and an
    optional ``test_config`` dict. Registers core blueprints and prepares
    folders used by the application.
    """

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "dev-secret-key"),
        OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY"),
        LLM_DEFAULT_PROVIDER=os.environ.get("LLM_DEFAULT_PROVIDER", "ollama"),
        LLM_REQUEST_TIMEOUT=int(os.environ.get("LLM_REQUEST_TIMEOUT", 60)),
        LLM_OLLAMA_HOST=os.environ.get("LLM_OLLAMA_HOST", "localhost"),
        LLM_OLLAMA_PORT=int(os.environ.get("LLM_OLLAMA_PORT", 11434)),
        LLM_OLLAMA_SCHEME=os.environ.get("LLM_OLLAMA_SCHEME", "http"),
        LLM_OLLAMA_MODEL=os.environ.get("LLM_OLLAMA_MODEL", "llama3"),
        LLM_OLLAMA_OPTIONS=_safe_json(os.environ.get("LLM_OLLAMA_OPTIONS")),
        LLM_OPENAI_URL=os.environ.get(
            "LLM_OPENAI_URL", "https://api.openai.com/v1/chat/completions"
        ),
        LLM_OPENAI_MODEL=os.environ.get("LLM_OPENAI_MODEL", "gpt-3.5-turbo"),
        LLM_OPENAI_TEMPERATURE=_safe_float(os.environ.get("LLM_OPENAI_TEMPERATURE")),
    )

    if test_config is not None:
        app.config.update(test_config)
    else:
        app.config.from_pyfile("config.py", silent=True)

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


__all__ = ["create_app"]
