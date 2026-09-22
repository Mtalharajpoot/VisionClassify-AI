# Contributing

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Verify

```bash
pytest -q
```

## Guidelines

- Keep the core architecture aligned with the documented project.
- Add tests for deterministic helper functions.
- Do not commit generated model files or large datasets.
- Keep notebooks reproducible.
- Update documentation when changing the model architecture.
