"""Flask application factory for the Health Team 1 proof of concept."""
from __future__ import annotations

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


__all__ = ["create_app"]
