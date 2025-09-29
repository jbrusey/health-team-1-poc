"""Application configuration objects."""
from __future__ import annotations

import os
from pathlib import Path


class BaseConfig:
    """Base configuration shared across environments."""

    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    WTF_CSRF_ENABLED = True

    # Folders used by the prototype. Paths are resolved relative to project root.
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"
    VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class TestingConfig(BaseConfig):
    TESTING = True
    WTF_CSRF_ENABLED = False


class ProductionConfig(BaseConfig):
    DEBUG = False


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
