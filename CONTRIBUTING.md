# Contributing to cuvis-ai-augment

We welcome contributions — bug reports, fixes, new transforms, docs.

## Pull requests

1. Fork the repo and create a branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've added a new transform, register it via `@register("Name")` and import the
   module in `cuvis_ai_augment/transforms/__init__.py`.
4. Make sure CI is green locally:

   ```bash
   uv run pytest tests/ -m "not slow"
   uv run ruff format --check cuvis_ai_augment tests
   uv run ruff check cuvis_ai_augment tests
   ```

5. Update `CHANGELOG.md` under `## Unreleased`.
6. Open a PR.

## Code style

- `ruff format` + `ruff check` (config in `pyproject.toml`).
- Type-annotate public surface; `mypy` is non-blocking in CI but should be clean.
- Keep transforms small (`__call__` only); no torch.nn ceremony unless needed.

## Issues

Use GitHub issues. Include cuvis-ai-core version, torch version, a minimal repro pipeline
YAML, and the failing tensor shape if applicable.

## License

By contributing you agree your contributions are licensed under Apache-2.0 (see
[LICENSE](LICENSE)).
