# Repository Guidelines

## Project Structure & Module Organization
- `src/femr/` — Python package code. Key submodules: `models/`, `featurizers/`, `labelers/`, `transforms/`, `post_etl_pipelines/`.
- `tests/` — pytest suite organized by package area (e.g., `tests/models/`).
- `tools/` — utility scripts (e.g., Stanford OMOP helpers).
- `tutorials/` — notebooks and examples; excluded from some lint checks.
- CLI entry point: `femr_stanford_omop_fixer` (see README for usage).

## Build, Test, and Development Commands
- Dev install: `python -m venv .venv && source .venv/bin/activate`
  then `pip install -e '.[build]'`.
- Lint/format (via pre-commit): `pre-commit install && pre-commit run --all-files`.
- Standalone checks: `flake8`, `black --check .`, `isort --check-only .`, `mypy src/`.
- Auto-format: `black . && isort .`.
- Tests: `pytest -q` (use `-k <pattern>` to filter). Pytest is configured via `pyproject.toml`.
- Build distribution (optional): `python -m build`.

## Coding Style & Naming Conventions
- Python 3.10 target; 4-space indentation; max line length 120.
- Use type hints; `py.typed` is shipped and MyPy runs in stricter mode for `femr`.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Tools/config: Black (`line_length=120`), isort (profile in `pyproject.toml`), Flake8 (see `.flake8` ignores), MyPy (`.mypy.ini`).

## Testing Guidelines
- Framework: pytest. Place tests under `tests/` mirroring package paths.
- Naming: files `test_*.py`, functions `test_*`.
- Add unit tests for new logic and regression tests for fixed bugs.
- Prefer small, deterministic tests; use fixtures in `tests/femr_test_tools.py` when applicable.

## Commit & Pull Request Guidelines
- Commit messages: use imperative mood and concise scope, e.g., `feat(models): add RMSNorm`. Reference issues when relevant.
- Keep commits focused; include code, tests, and docs together.
- PRs must include: clear description, rationale, test coverage notes, and any CLI examples (e.g., data paths redacted). Add logs/screens if touching training/evaluation.
- Ensure `pre-commit` passes and CI is green before requesting review.

## Security & Configuration Tips
- Do not commit PHI, datasets, or large artifacts (wandb runs, logs). Use `.gitignore` and environment variables for paths/secrets.
- Reproducibility: record seeds, versions, and command lines in PR descriptions for modeling changes.
