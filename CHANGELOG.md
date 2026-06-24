# Changelog

## [Unreleased]

## 0.3.1 - 2026-06-24

- Commented out the local `[tool.uv.sources]` `cuvis-ai-core` editable path so the released tag no longer carries a machine-specific source. The lock is unchanged (already generated `--no-sources`); the path stays as a commented dev-only override.
- Added a release gate that rejects any active local `[tool.uv.sources]` path entry, so a local override cannot ship in a tag again.

## 0.3.0 - 2026-06-24

- Migrated `examples/plugins.yaml` to the bare `name:` + `capabilities:` manifest shape required by cuvis-ai-schemas 0.7.0; dropped the retired nested `plugins:/provides:` wrapper.
- Adopted the cuvis-ai-core `register_plugin(path)` API (renamed from `load_plugins`) in the manifest-loading test.
- Require `cuvis-ai-core>=0.10.0` and `cuvis-ai-schemas>=0.7.0`, adopting the released framework versions; relocked against them with `--no-sources`, which drops the unused `cuvis` / `cuvis-il` SDK transitives the old core floor pulled.
- Switched to setuptools-scm tag-driven versioning; the package version now derives from the git tag, with `fallback_version` covering shallow CI checkouts.
- Added the `cuvis_ai_compat.yml` dependency-compatibility gate (audits the plugin's deps against the cuvis-ai-core lock on dependency PRs and a weekly cron).
- Repointed the `[tool.uv.sources]` cuvis-ai-core editable override at the canonical core checkout.

## 0.2.0 - 2026-05-15

- Added the spectral family (`GaussianBandNoise`, `RandomBandDropout`), photometric family (`MultiplicativeIlluminationScaling`), and mixing family (`Cutout`), drawn from the HSI augmentation literature (Nalepa 2019; Ahmad 2024; Roddan 2024; DeVries & Taylor 2017).
- Threaded an optional `wavelengths: list[float] | None = None` argument through `Transform.__call__` and the `AugmentationCompose` `wavelengths` hparam (backward-compatible default; reserved for future wavelength-aware transforms), and added the `Transform._validate_wavelengths` helper.
- Reorganised transforms by family on disk (`spatial.py` / `spectral.py` / `photometric.py` / `mixing.py`); no public symbol moves, all still resolve through `TRANSFORM_REGISTRY` by name.
- Migration: external `Transform` subclasses registered via `extra_transform_modules` must add `wavelengths: list[float] | None = None` to their `__call__` signature (wavelength-agnostic transforms can ignore it).
- Grew the test suite to 90 tests (~99.5% coverage on `cuvis_ai_augment/`).

## 0.1.1 - 2026-05-13

- Added the tutorial notebook `notebooks/use_cases/lentils_augmentation.ipynb` (HuggingFace lentils dataset loader, per-transform 3-band visualisations, Colab-runnable) and the `notebooks` optional-dependency group.
- Added coverage-gap tests, README badges, and the Tutorial / Node API / Compatibility / Acknowledgments sections.
- Corrected the `0.1.0` date and removed the superseded `docs/main-repo-manifest-snippet.yaml`.

## 0.1.0 - 2026-05-13

- Initial release: `AugmentationCompose` node plus four spatial transforms (`RandomHorizontalFlip`, `RandomVerticalFlip`, `Random90Rotate`, `RandomSpatialCrop`) operating on `(B, H, W, C)` float32 cubes and paired `(B, H, W)` masks.
- `execution_stages = {ExecutionStage.TRAIN}` so augmentation is a no-op at val/test/inference; `extra_transform_modules` lets external packages contribute transforms via the same `@register` decorator.
- Added the demo pipeline `configs/pipelines/demo_augment.yaml` and a visual-check notebook.
