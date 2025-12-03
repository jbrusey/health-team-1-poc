"""Application routes for the Health Team 1 proof of concept."""
from __future__ import annotations

import json

import requests
from flask import Blueprint, current_app, flash, redirect, render_template, session, url_for
from markdown import markdown
from markupsafe import Markup

from .forms import CaseIntakeForm, LLMQueryForm, MultiAgentQueryForm, SettingsForm
from .llm import (
    LLMError,
    aggregate_agent_responses,
    generate_multi_agent_responses,
    generate_response,
)

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

    system_prompt = _current_system_prompt()
    system_prompt_preview = _system_prompt_preview(system_prompt)
    response_text: str | None = None
    response_html: Markup | None = None
    if form.validate_on_submit():
        try:
            response_text = generate_response(
                form.prompt.data,
                provider=form.provider.data,
                model=form.model.data or None,
                system_prompt=system_prompt,
            )
            response_html = _render_markdown(response_text)
        except LLMError as exc:
            flash(str(exc), "danger")
        except requests.RequestException as exc:
            flash(f"Unable to contact LLM provider: {exc}", "danger")

    return render_template(
        "llm.html",
        form=form,
        response=response_text,
        response_html=response_html,
        system_prompt=system_prompt,
        system_prompt_preview=system_prompt_preview,
    )


@bp.route("/llm/multi", methods=["GET", "POST"])
def multi_llm_prompt():
    """Render a tool that sends a prompt to multiple Ollama agents."""

    form = MultiAgentQueryForm()
    system_prompt = _current_system_prompt()
    system_prompt_preview = _system_prompt_preview(system_prompt)
    responses: list[dict[str, object]] = []
    aggregation_result: dict[str, object] | None = None
    aggregation_result_html: Markup | None = None
    aggregation_system_prompt = _aggregation_system_prompt()
    query_agents = _current_query_agents()

    if form.validate_on_submit():
        try:
            agent_results = generate_multi_agent_responses(
                form.prompt.data,
                query_agents=query_agents,
                system_prompt=system_prompt,
            )
            for result in agent_results:
                response_html = _render_markdown(result["response"]) if result.get("response") else None
                responses.append({**result, "response_html": response_html})

            aggregation_result = aggregate_agent_responses(
                form.prompt.data,
                agent_results,
                query_agents=query_agents,
                system_prompt=aggregation_system_prompt,
            )
            if aggregation_result.get("response"):
                aggregation_result_html = _render_markdown(
                    aggregation_result["response"]
                )
        except LLMError as exc:
            flash(str(exc), "danger")
        except requests.RequestException as exc:
            flash(f"Unable to contact LLM provider: {exc}", "danger")

    return render_template(
        "multi_llm.html",
        form=form,
        responses=responses,
        system_prompt=system_prompt,
        system_prompt_preview=system_prompt_preview,
        query_agents=query_agents,
        aggregation_result=aggregation_result,
        aggregation_result_html=aggregation_result_html,
        aggregation_system_prompt=aggregation_system_prompt,
    )


@bp.route("/settings", methods=["GET", "POST"])
def settings():
    """Allow runtime configuration of the system prompt."""

    form = SettingsForm()
    if not form.system_prompt.data:
        form.system_prompt.data = _current_system_prompt()
    if not form.query_agents.data:
        form.query_agents.data = json.dumps(_current_query_agents(), indent=2)

    if form.validate_on_submit():
        session["system_prompt"] = form.system_prompt.data or ""
        try:
            submitted_agents = json.loads(form.query_agents.data)
        except json.JSONDecodeError:
            flash("Query agents must be valid JSON.", "danger")
            return render_template("settings.html", form=form)

        normalized_agents = _normalize_query_agents(submitted_agents)
        if not normalized_agents:
            flash("Provide at least one query agent with a valid port.", "danger")
            return render_template("settings.html", form=form)

        session["query_agents"] = normalized_agents
        flash(
            "Settings saved. Updated system prompt and query agents will be used for LLM requests.",
            "info",
        )
        return redirect(url_for("main.settings"))

    return render_template("settings.html", form=form)


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


def _current_query_agents() -> list[dict]:
    agents = session.get("query_agents")
    if agents:
        return _normalize_query_agents(agents)
    return current_app.config.get("QUERY_AGENTS", [])


def _current_system_prompt() -> str | None:
    prompt = session.get("system_prompt")
    if prompt is not None:
        return prompt
    return current_app.config.get("LLM_SYSTEM_PROMPT")


def _system_prompt_preview(prompt: str | None, max_lines: int = 3) -> str | None:
    """Return the first few lines of the system prompt for display."""

    if not prompt:
        return None

    lines = prompt.splitlines()
    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)

    if len(lines) > max_lines:
        preview += "\n..."

    return preview


def _aggregation_system_prompt() -> str | None:
    return current_app.config.get("LLM_MULTI_AGENT_SUMMARY_PROMPT")


def _render_markdown(content: str) -> Markup:
    """Render Markdown to HTML while preserving simple formatting."""

    return Markup(
        markdown(
            content,
            extensions=["extra", "nl2br"],
            output_format="html5",
        )
    )


def _normalize_query_agents(data: object) -> list[dict]:
    if not isinstance(data, list):
        return []

    agents: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            port = int(item.get("port"))
        except (TypeError, ValueError):
            continue

        agent = {
            "port": port,
            "model": str(item.get("model", "")) or None,
            "temperature": _safe_float(item.get("temperature")),
            "top_k": _safe_int(item.get("top_k")),
            "top_p": _safe_float(item.get("top_p")),
            "repeat_penalty": _safe_float(item.get("repeat_penalty")),
        }
        agents.append(agent)

    return agents


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
