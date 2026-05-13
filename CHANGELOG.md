# Changelog

All notable changes to this project will be documented in this file. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

## 0.1.0 - 2026-05-19

- Initial release.
- Added `AugmentationCompose` Node (`cuvis_ai_augment.node.compose`). Single Node entry-point; inner augmentations are plain Python `Transform` classes resolved against `TRANSFORM_REGISTRY`.
- Added four spatial Transforms (`cuvis_ai_augment.transforms.spatial`):
  - `RandomHorizontalFlip` — flip width axis with probability `prob` per sample.
  - `RandomVerticalFlip` — flip height axis with probability `prob` per sample.
  - `Random90Rotate` — rotate by random multiple of 90° (0/90/180/270) per sample.
  - `RandomSpatialCrop` — crop to fixed `(H_out, W_out)` at random offset per sample.
- All Transforms operate on `(B, H, W, C)` torch.float32 cubes and `(B, H, W)` masks; mask is paired (same per-sample decision tensors).
- `AugmentationCompose.available_transforms()` classmethod for introspection. Unknown `type:` raises `ValueError` listing all registered names.
- `extra_transform_modules: list[str]` hparam imports user modules before resolving names, so external packages can contribute transforms via the same `@register` decorator.
- `execution_stages={ExecutionStage.TRAIN}` so augmentations are a no-op at val/test/inference automatically.
- Demo pipeline at `configs/pipelines/demo_augment.yaml`.
- Visual-check notebook at `notebooks/visual_check.ipynb` (one section per Transform, plus full Compose).
