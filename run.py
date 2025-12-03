"""Entrypoint for running the Flask development server."""
from __future__ import annotations

from app import create_app


class PrefixMiddleware:
    """Middleware to adjust paths when the app is served under a prefix."""

    def __init__(self, app, prefix: str):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        if environ["PATH_INFO"].startswith(self.prefix):
            environ["SCRIPT_NAME"] = self.prefix
            environ["PATH_INFO"] = environ["PATH_INFO"][len(self.prefix) :]
        return self.app(environ, start_response)


app = create_app()
app = PrefixMiddleware(app, prefix="/tumourlite")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=app.config.get("DEBUG", False))
