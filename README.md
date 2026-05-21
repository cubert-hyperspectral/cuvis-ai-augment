# cuvis-ai-augment

[![CI](https://github.com/cubert-hyperspectral/cuvis-ai-augment/actions/workflows/ci.yml/badge.svg)](https://github.com/cubert-hyperspectral/cuvis-ai-augment/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-augment/branch/main/graph/badge.svg)](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-augment)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Data-augmentation node for [cuvis.ai](https://www.cubert-hyperspectral.com/) hyperspectral
training pipelines.

This plugin provides a single `AugmentationCompose` node that applies a configurable list
of stochastic transforms to hyperspectral cubes (and their paired masks) during training.
It is automatically a no-op at val/test/inference via `execution_stages`.

**v0.2.0 adds three new transform families** — spectral (`GaussianBandNoise`,
`RandomBandDropout`), photometric (`MultiplicativeIlluminationScaling`), and mixing
(`Cutout`) — drawn from the HSI augmentation literature (Nalepa et al. 2019;
Ahmad et al. 2024; Roddan et al. 2024). All transforms are pure-torch and HSI-aware,
preserving spectral signatures and paired masks by construction.

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
uv add "git+https://github.com/cubert-hyperspectral/cuvis-ai-augment@v0.3.0"
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
| `wavelengths` | `list[float] | None` | `None` | Optional per-band centre wavelengths in nm. Threaded through to each transform's `__call__`. Wavelength-agnostic transforms ignore it; reserved for v0.3.0+ wavelength-aware transforms. |

**Execution:** the node is registered as `execution_stages = {ALWAYS}` so cuvis-ai-core keeps it in the executable graph at every stage. TRAIN-only behavior is enforced *inside* `forward` via a `Context` stage check: at val/test/inference the node short-circuits to identity (cube and mask pass through unchanged), so downstream consumers always have a producer for the output ports without breaking eval determinism.

## Available transforms (v0.2.0)

### Spatial (`cuvis_ai_augment.transforms.spatial`)

| Name | Operation |
|---|---|
| `RandomHorizontalFlip` | Flip width axis with probability `prob` per sample |
| `RandomVerticalFlip`   | Flip height axis with probability `prob` per sample |
| `Random90Rotate`       | Rotate by random multiple of 90° (0/90/180/270) per sample. Requires `H == W` when any sample's k is odd. |
| `RandomSpatialCrop`    | Crop to fixed `(H_out, W_out)` at random offset per sample (centre crop when `prob<1`). |

### Spectral (`cuvis_ai_augment.transforms.spectral`) — new in v0.2.0

| Name | Operation |
|---|---|
| `GaussianBandNoise` | Add per-band Gaussian noise `N(0, σ²)`. Optional `per_band_scale` makes σ track each band's std. |
| `RandomBandDropout` | Zero out a random subset of bands per sample (`drop_fraction` of `C`). Spectral regulariser. |

### Photometric (`cuvis_ai_augment.transforms.photometric`) — new in v0.2.0

| Name | Operation |
|---|---|
| `MultiplicativeIlluminationScaling` | Multiply each band by a smooth random gain curve `g(λ) ∈ gain_range`. Models lamp drift / illumination variation without breaking spectral ratios. |

### Mixing (`cuvis_ai_augment.transforms.mixing`) — new in v0.2.0

| Name | Operation |
|---|---|
| `Cutout` | Zero a random rectangular spatial patch in both cube and mask. `mask_fill_value` (default `0`) lets you target your loss's ignore label. |

Discover programmatically:

```python
from cuvis_ai_augment.node.compose import AugmentationCompose
AugmentationCompose.available_transforms()
# ['Cutout', 'GaussianBandNoise', 'MultiplicativeIlluminationScaling',
#  'Random90Rotate', 'RandomBandDropout', 'RandomHorizontalFlip',
#  'RandomSpatialCrop', 'RandomVerticalFlip']
```

### References

The new families are drawn from the HSI augmentation literature:

- Nalepa, Myller, Kawulok — *Hyperspectral Data Augmentation*, IEEE TGRS (2019) — [arXiv:1903.05580](https://arxiv.org/abs/1903.05580)
- Ahmad et al. — *A Comprehensive Survey for HSI Classification* (2024) — [arXiv:2404.14955](https://arxiv.org/abs/2404.14955)
- Roddan et al. — *Calibration-Jitter*, Healthcare Technology Letters (2024) — [pmc.ncbi.nlm.nih.gov/articles/PMC11665780](https://pmc.ncbi.nlm.nih.gov/articles/PMC11665780/)
- DeVries & Taylor — *Improved Regularization of CNNs with Cutout* (2017) — [arXiv:1708.04552](https://arxiv.org/abs/1708.04552)

## Extending with new transforms

### In this package
Add a class to `cuvis_ai_augment/transforms/` and decorate it:

```python
from cuvis_ai_augment.transforms.base import Transform, register

@register("MyTransform")
class MyTransform(Transform):
    def __call__(self, cube, mask, rng, wavelengths=None):
        ...
        return cube, mask
```

Then import the module in `cuvis_ai_augment/transforms/__init__.py` so the decorator runs
on plugin import. The transforms are split by family (`spatial.py`, `spectral.py`,
`photometric.py`, `mixing.py`) — add your transform to the file that matches its operation,
or create a new family file.

The `wavelengths` argument is optional and defaults to `None`. Wavelength-agnostic
transforms can keep their signature simple (`del wavelengths` at the top); wavelength-aware
transforms (planned for v0.3.0+) read per-band centre wavelengths in nanometres.

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
name: augment
path: ".."
capabilities:
  - class_name: cuvis_ai_augment.node.compose.AugmentationCompose
```

For releases, pin a git tag:

```yaml
name: augment
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-augment.git"
tag: "v0.3.0"
package_name: cuvis-ai-augment
capabilities:
  - class_name: cuvis_ai_augment.node.compose.AugmentationCompose
```

## Compatibility

| `cuvis-ai-augment` | `cuvis-ai-core` | `cuvis-ai-schemas` | `torch` | `numpy` |
|---|---|---|---|---|
| `0.3.0` | `>=0.10.0` | `>=0.7.0` | `>=2.1` | `>=1.20.0` |
| `0.2.0` | `>=0.1.0` (tested against 0.5.2) | `>=0.4.0` | `>=2.1` | `>=1.20.0` |
| `0.1.x` | `>=0.1.0` (tested against 0.5.2) | `>=0.4.0` | `>=2.1` | `>=1.20.0` |

The tagged-manifest model is verified at release time by cloning the published tag fresh
and loading it via `NodeRegistry.register_plugin()` — see the release checklist in
[CONTRIBUTING.md](CONTRIBUTING.md#release-process).

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -q                                   # 90 tests
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
