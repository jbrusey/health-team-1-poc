"""WTForms definitions for case intake."""
from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import FieldList, FormField, SelectField, StringField, SubmitField, TextAreaField
from wtforms.validators import Optional


class DemographicsForm(FlaskForm):
    """Demographics and baseline patient information."""

    patient_id = StringField("Patient ID", validators=[Optional()])
    age = StringField("Age", validators=[Optional()])
    sex = SelectField(
        "Sex",
        choices=[("female", "Female"), ("male", "Male"), ("other", "Other")],
        validators=[Optional()],
    )
    clinical_notes = TextAreaField("Clinical Notes", validators=[Optional()])


class PathologyForm(FlaskForm):
    """Tumour pathology details."""

    tumour_type = StringField("Tumour Type", validators=[Optional()])
    stage = StringField("Stage", validators=[Optional()])
    receptor_status = TextAreaField("Receptor Status", validators=[Optional()])


class GenomicsForm(FlaskForm):
    """Genomic findings for the current case."""

    sequencing_platform = StringField("Sequencing Platform", validators=[Optional()])
    variants = TextAreaField("Variants", validators=[Optional()])


class CaseIntakeForm(FlaskForm):
    """Aggregate form that collects EHR, pathology, and genomic details."""

    demographics = FormField(DemographicsForm, label="EHR / Demographics")
    pathology = FormField(PathologyForm, label="Pathology")
    genomics = FormField(GenomicsForm, label="Genomics")
    retrieved_case_ids = FieldList(StringField("Retrieved Case ID"), min_entries=0)

    submit = SubmitField("Save Draft")
