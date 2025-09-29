"""Application routes for the Health Team 1 proof of concept."""
from __future__ import annotations

from flask import Blueprint, render_template

from .forms import CaseIntakeForm

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
