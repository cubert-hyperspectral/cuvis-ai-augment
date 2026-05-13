# Contributing to cuvis-ai-augment

We welcome contributions — bug reports, fixes, new transforms, docs.

## Workflow

1. Fork the repo and create a feature branch from `main` — never push directly to `main`.
2. If you've added code that should be tested, add tests.
3. If you've added a new transform, register it via `@register("Name")` and import the
   module in `cuvis_ai_augment/transforms/__init__.py`.
4. Make sure CI is green locally:

   ```bash
   uv run --no-sources --extra dev pytest tests/ -q
   uv run --no-sources --extra dev ruff format --check cuvis_ai_augment tests
   uv run --no-sources --extra dev ruff check cuvis_ai_augment tests
   uv run --no-sources --extra dev mypy cuvis_ai_augment/
   ```

5. Update `CHANGELOG.md` under `## Unreleased`.
6. Open a PR. Watch CI go green, then squash-merge.

## Code style

- `ruff format` + `ruff check` (config in `pyproject.toml`).
- Type-annotate public surface. `mypy cuvis_ai_augment/` is **non-blocking in CI**
  (`|| true`) — upstream stubs in `cuvis-ai-core` and `cuvis-ai-schemas` are incomplete,
  so a green mypy run typically reports a handful of `import-untyped` notes. Treat those
  as warnings, not failures; treat any other mypy error as a hard fail.
- Keep transforms small (`__call__` only); no `torch.nn` ceremony unless needed.

## Running tests

Tests live under `tests/`. The `pythonpath = ["."]` setting in `[tool.pytest.ini_options]`
(plus a `tests/__init__.py`) ensures that test fixtures under `tests/fixtures/` are
importable as `tests.fixtures.<module>` — this matters for the `extra_transform_modules`
test in `test_compose.py`. If you add new fixtures, mirror that import path.

## How the workflows work

The repo ships two GitHub Actions workflows. Together they form a tight, blameable feedback
loop: every push runs the full CI matrix, every `v*.*.*` tag cuts a release.

### `.github/workflows/ci.yml` — runs on every push and pull request

CI is split into five **independent** jobs that run on fresh runners in parallel. Splitting
gives clear failure attribution (ruff red ≠ tests red) and faster wall-clock feedback.

#### 1. `Focused Test Suite`

```yaml
- name: Run focused tests
  run: |
    uv run --no-sources --locked --extra dev \
      pytest tests/ --cov=cuvis_ai_augment --cov-report=xml
- uses: codecov/codecov-action@v4
  with:
    file: coverage.xml
```

- `--no-sources` ignores the local-editable `[tool.uv.sources]` overrides — CI installs
  pure-PyPI dependencies the way an external user would.
- `--locked` means the install must use *exactly* the resolved versions in `uv.lock`.
  Drift between local and CI ≡ a `uv.lock` update PR you forgot to make.
- Coverage XML is uploaded to Codecov; the badge in the README is wired to the result.

#### 2. `Focused Ruff Checks`

```yaml
- name: Check formatting
  run: uv run --no-sources --extra dev ruff format --check .
- name: Run ruff
  run: uv run --no-sources --extra dev ruff check .
```

`ruff format --check` reports diffs but doesn't rewrite — the build fails if anything is
mis-formatted. Run `ruff format .` locally to fix.

#### 3. `Type checking`

```yaml
- name: Run mypy
  run: uv run --no-sources --extra dev mypy cuvis_ai_augment/
```

Catches `None`-vs-`Tensor` mistakes, missing return annotations, and the kind of
silent-correctness bugs that hand-written unit tests miss.

#### 4. `Security scanning`

Three independent sub-steps in one job:

```yaml
- name: Run pip-audit
  run: uv run --no-sources --extra dev pip-audit
- name: Run detect-secrets
  run: uv run --no-sources --extra dev detect-secrets scan --baseline .secrets.baseline
- name: Run bandit
  run: uv run --no-sources --extra dev bandit -r cuvis_ai_augment
```

- `pip-audit` — CVE check against PyPI advisories for every pinned dep.
- `detect-secrets scan --baseline .secrets.baseline` — fails if any *new* high-entropy
  secret appears versus the committed baseline.
- `bandit -r` — Python AST security rules (use of `assert` for prod logic, hardcoded
  passwords, weak `random`, etc.).

#### 5. `Build And Validate`

```yaml
- name: Build wheel + sdist
  run: uv build --no-sources
- name: Validate metadata
  run: uv run --no-sources --extra dev twine check dist/*
```

Catches packaging mistakes — missing `MANIFEST.in` entries, malformed `pyproject.toml`,
non-PEP-508 dep specifiers — *before* a tag tries to publish them.

### `.github/workflows/release.yml` — runs only on `v*.*.*` tags

Same triggers, different shape. The release pipeline is more conservative because the
output is immutable: a GH Release at a fixed tag with attached wheel + sdist artifacts.
Four jobs, each gating the next:

#### 1. `Validate Release Candidate`

Asserts that the tag string matches `pyproject.toml [project] version` and that
`CHANGELOG.md` has a `## <version> - YYYY-MM-DD` section. Cheap pre-flight: a missing
changelog entry fails the release within ~10 seconds rather than after a 90s build.

#### 2. `Security Scanning`

Same triad as CI. Re-run here so a security regression that snuck onto `main` between the
last CI run and this tag still blocks the release.

#### 3. `Build And Validate Artifacts`

```yaml
- name: Build
  run: uv build --no-sources
- name: Validate
  run: uv run twine check dist/*
- uses: actions/upload-artifact@v4
  with:
    name: dist
    path: dist/*
```

Wheel + sdist are uploaded as workflow artifacts here, not yet attached to the GH Release.

#### 4. `Create GitHub Release`

```yaml
- uses: actions/download-artifact@v4
  with: { name: dist, path: dist }
- uses: softprops/action-gh-release@v2
  with:
    files: dist/*
    body_path: .release-body.md
```

Pulls the artifacts from job 3, writes the changelog section into the release body, and
publishes the Release at the tag. This is what makes
`uv add "git+https://github.com/...@v0.1.0"` work cleanly for downstream consumers.

### Why two workflows?

CI runs on **every push and PR** — fast, cheap, fail-loud feedback during development.
Release runs **only on `v*` tags** — heavier work, version-consistency checks, artifact
publishing. Same security and build checks in both for safety, different triggers and
outputs.

## Release process

Releases follow semver. See [the cuvis-ai-plugin release-checklist skill](https://github.com/cubert-hyperspectral/cuvis-ai-agentic-skills/blob/main/skills/cuvis-ai-plugin/release-checklist.md)
for the canonical procedure. Short version:

1. Bump `pyproject.toml [project] version`.
2. Move `## Unreleased` entries into a `## <version> - YYYY-MM-DD` section in `CHANGELOG.md`.
3. Open a PR titled `Release vX.Y.Z`. Merge once CI is green.
4. From `main`: `git tag -a vX.Y.Z -m "Release vX.Y.Z: <summary>"` then `git push origin vX.Y.Z`.
5. The release workflow runs; watch it. If it fails: delete the remote tag, fix on a
   branch, re-tag to vX.Y.(Z+1). **Tags are immutable** — don't repoint.
6. Verify the released tag clones + loads cleanly:

   ```bash
   mkdir -p /tmp/plugin-verify && cd /tmp/plugin-verify
   cat > registry.yaml <<'YAML'
   plugins:
     augment:
       repo: "https://github.com/cubert-hyperspectral/cuvis-ai-augment.git"
       tag:  "vX.Y.Z"
       provides:
         - cuvis_ai_augment.node.compose.AugmentationCompose
   YAML
   uv run python -c "from cuvis_ai_core.utils.node_registry import NodeRegistry; r=NodeRegistry(); r.load_plugins('registry.yaml'); print(r.list_plugins())"
   ```

   Expected output: `Loaded plugins: ['augment']`.

7. Open a PR against `cubert-hyperspectral/cuvis-ai` bumping the `tag:` in
   `configs/plugins/augment.yaml` (or adding the entry if the plugin is new).

## Issues

Use GitHub issues. Include cuvis-ai-core version, torch version, a minimal repro pipeline
YAML, and the failing tensor shape if applicable.

## License

By contributing you agree your contributions are licensed under Apache-2.0 (see
[LICENSE](LICENSE)).
