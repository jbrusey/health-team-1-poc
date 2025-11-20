"""Application routes for the Health Team 1 proof of concept."""
from __future__ import annotations

import requests
from flask import Blueprint, current_app, flash, render_template
from markdown import markdown
from markupsafe import Markup

from .forms import CaseIntakeForm, LLMQueryForm
from .llm import LLMError, generate_response

bp = Blueprint("main", __name__)


@bp.route("/", methods=["GET", "POST"])
def index():
    """Render the landing page with the intake form."""
    form = CaseIntakeForm()

    if form.validate_on_submit():
        # Placeholder: logic to persist drafts or trigger LLM interactions will
        # be implemented in later steps.
        pass

    return render_template("index.html", form=form)


@bp.route("/llm", methods=["GET", "POST"])
def llm_prompt():
    """Render a tool that forwards prompts to an LLM provider."""

    form = LLMQueryForm()
    provider_choices = _provider_choices()
    form.provider.choices = provider_choices
    if form.provider.data is None:
        form.provider.data = current_app.config.get("LLM_DEFAULT_PROVIDER", "ollama")

    if not form.model.data:
        # Pre-populate with the configured default model to show what will be used.
        default_model = _default_model_for_provider(form.provider.data)
        if default_model:
            form.model.data = default_model

    response_text: str | None = None
    response_html: Markup | None = None
    if form.validate_on_submit():
        try:
            response_text = generate_response(
                form.prompt.data,
                provider=form.provider.data,
                model=form.model.data or None,
            )
            response_html = _render_markdown(response_text)
        except LLMError as exc:
            flash(str(exc), "danger")
        except requests.RequestException as exc:
            flash(f"Unable to contact LLM provider: {exc}", "danger")

    return render_template(
        "llm.html", form=form, response=response_text, response_html=response_html
    )


def _provider_choices() -> list[tuple[str, str]]:
    """Return the providers exposed to the user."""

    configured = current_app.config.get(
        "LLM_PROVIDER_CHOICES",
        {
            "ollama": "Ollama (local)",
            "openai": "OpenAI ChatGPT",
        },
    )
    return list(configured.items())


def _default_model_for_provider(provider: str | None) -> str | None:
    if provider == "ollama":
        return current_app.config.get("LLM_OLLAMA_MODEL")
    if provider == "openai":
        return current_app.config.get("LLM_OPENAI_MODEL")
    return None


def _render_markdown(content: str) -> Markup:
    """Render Markdown to HTML while preserving simple formatting."""

    return Markup(
        markdown(
            content,
            extensions=["extra", "nl2br"],
            output_format="html5",
        )
    )
