# cuvis-ai-augment

[![CI](https://github.com/cubert-hyperspectral/cuvis-ai-augment/actions/workflows/ci.yml/badge.svg)](https://github.com/cubert-hyperspectral/cuvis-ai-augment/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-augment/branch/main/graph/badge.svg)](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-augment)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Data-augmentation node for [cuvis.ai](https://www.cubert-hyperspectral.com/) hyperspectral
training pipelines.

This plugin provides a single `AugmentationCompose` node that applies a configurable list
of stochastic spatial transforms to hyperspectral cubes (and their paired masks) during
training. It is automatically a no-op at val/test/inference via `execution_stages`.

## Tutorial

The end-to-end walkthrough lives at
[`notebooks/use_cases/lentils_augmentation.ipynb`](notebooks/use_cases/lentils_augmentation.ipynb).
It downloads a real lentils hyperspectral cube from
[HuggingFace](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils),
shows each transform side-by-side at three spectral bands (low / mid / high) so you can
confirm the augmentation is channel-agnostic, and ends with a block of programmatic
sanity assertions that fail loudly on regression.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cubert-hyperspectral/cuvis-ai-augment/blob/main/notebooks/use_cases/lentils_augmentation.ipynb)

## Install

```bash
uv sync --extra dev                  # development install (tests, lint, type checks)
uv sync --extra dev --extra notebooks  # add the tutorial notebook extras
```

Or, from another cuvis-ai project, install from a tagged release:

```bash
uv add "git+https://github.com/cubert-hyperspectral/cuvis-ai-augment@v0.1.0"
```

## Usage (pipeline YAML)

```yaml
nodes:
  - name: Augment
    class_name: cuvis_ai_augment.node.compose.AugmentationCompose
    hparams:
      seed: 42
      transforms:
        - {type: RandomHorizontalFlip, prob: 0.5}
        - {type: RandomVerticalFlip,   prob: 0.5}
        - {type: Random90Rotate,       prob: 0.5}
        - {type: RandomSpatialCrop,    size: [256, 256]}
connections:
  - {source: Data.outputs.cube, target: Augment.inputs.cube}
  - {source: Data.outputs.mask, target: Augment.inputs.mask}
```

The mask port is optional. When connected, every transform applies the same per-sample
spatial decision to both the cube and the mask so the pair stays aligned.

## Node API

### `AugmentationCompose`

| Port / spec | Type | Shape | Notes |
|---|---|---|---|
| `inputs.cube` | `torch.float32` | `(B, H, W, C)` | Required. |
| `inputs.mask` | `torch.int32`   | `(B, H, W)`    | Optional; paired transforms apply the same per-sample decision. |
| `outputs.cube`| `torch.float32` | `(B, H, W, C)` | Same shape as input *except* under `RandomSpatialCrop` (H, W shrink to `size`). |
| `outputs.mask`| `torch.int32`   | `(B, H, W)`    | Present iff `mask` was connected. |

**hparams:**

| Name | Type | Default | Purpose |
|---|---|---|---|
| `transforms` | `list[dict]` | `[]` | Ordered list of `{type: <Name>, **kwargs}` specs from `TRANSFORM_REGISTRY`. |
| `seed` | `int | None` | `None` | Seeds the shared `torch.Generator`. `None` is non-deterministic. |
| `extra_transform_modules` | `list[str]` | `[]` | Import these module paths before resolving names — for external transform packages. |

**Execution:** `execution_stages = {ExecutionStage.TRAIN}` — automatic no-op outside training.

## Available transforms (v0.1.0)

| Name | Operation |
|---|---|
| `RandomHorizontalFlip` | Flip width axis with probability `prob` per sample |
| `RandomVerticalFlip`   | Flip height axis with probability `prob` per sample |
| `Random90Rotate`       | Rotate by random multiple of 90° (0/90/180/270) per sample. Requires `H == W` when any sample's k is odd. |
| `RandomSpatialCrop`    | Crop to fixed `(H_out, W_out)` at random offset per sample (centre crop when `prob<1`). |

Discover programmatically:

```python
from cuvis_ai_augment.node.compose import AugmentationCompose
AugmentationCompose.available_transforms()
# ['Random90Rotate', 'RandomHorizontalFlip', 'RandomSpatialCrop', 'RandomVerticalFlip']
```

## Extending with new transforms

### In this package
Add a class to `cuvis_ai_augment/transforms/` and decorate it:

```python
from cuvis_ai_augment.transforms.base import Transform, register

@register("MyTransform")
class MyTransform(Transform):
    def __call__(self, cube, mask, rng):
        ...
        return cube, mask
```

Then import the module in `cuvis_ai_augment/transforms/__init__.py` so the decorator runs
on plugin import.

### From an external package
Decorate your transforms the same way, then list your module path in
`extra_transform_modules` in the Compose hparams:

```yaml
hparams:
  extra_transform_modules:
    - my_pkg.my_transforms
  transforms:
    - {type: MyTransform, prob: 0.5}
```

The Compose node calls `importlib.import_module()` on each entry before resolving names,
so your `@register` decorators populate `TRANSFORM_REGISTRY` first.

## Plugin manifest

For local development (path relative to the manifest):

```yaml
plugins:
  augment:
    path: ".."
    provides:
      - cuvis_ai_augment.node.compose.AugmentationCompose
```

For releases, pin a git tag:

```yaml
plugins:
  augment:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-augment.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_augment.node.compose.AugmentationCompose
```

## Compatibility

| `cuvis-ai-augment` | `cuvis-ai-core` | `torch` | `numpy` |
|---|---|---|---|
| `0.1.0` | `>=0.1.0` (tested against 0.5.2) | `>=2.1` | `>=1.20.0` |

The tagged-manifest model is verified at release time by cloning the published tag fresh
and loading it via `NodeRegistry.load_plugins()` — see the release checklist in
[CONTRIBUTING.md](CONTRIBUTING.md#release-process).

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -q                                   # 48 tests
uv run pytest tests/ --cov=cuvis_ai_augment              # coverage report
uv run ruff format --check cuvis_ai_augment tests
uv run ruff check cuvis_ai_augment tests
uv run mypy cuvis_ai_augment/
```

To run the tutorial notebook end-to-end (also runs in CI on tag):

```bash
uv sync --extra dev --extra notebooks
uv run jupyter nbconvert --execute notebooks/use_cases/lentils_augmentation.ipynb \
    --to notebook --inplace --ExecutePreprocessor.timeout=900
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution workflow and a breakdown
of every CI / release job.

## Acknowledgments

- [cuvis-ai-core](https://github.com/cubert-hyperspectral/cuvis-ai) — the Node / pipeline
  framework this plugin extends.
- [PyTorch](https://pytorch.org/) — the underlying tensor and RNG primitives.
- [HuggingFace Hub](https://huggingface.co/) — hosts the public lentils dataset used by the
  tutorial notebook.

## License

Apache-2.0. See [LICENSE](LICENSE).
