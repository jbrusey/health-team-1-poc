# Proof of concept for Health Team 1

Proof-of-concept system to provide tumour advice to clinicians for the treatment of breast cancer.

## Project structure

```
app/
  __init__.py        # Flask application factory
  forms.py           # WTForms definitions for the intake workflow
  routes.py          # HTTP routes and view logic
  templates/         # Jinja templates for the UI
config.py            # Environment configuration objects
run.py               # Development server entrypoint
requirements.txt     # Python dependencies
```

Additional folders such as `data/` and `vector_store/` will be created in later steps to support retrieval-augmented generation.

## Getting started

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the environment template and populate secrets:
   ```bash
   cp .env.example .env
   ```
4. Run the development server:
   ```bash
   flask --app run:app --debug run
   ```

The default server listens on http://127.0.0.1:5000/ and exposes a preliminary form for capturing EHR, pathology, and genomic details. Subsequent steps will wire in prompt management, retrieval, and LLM orchestration.
