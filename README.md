# cuvis-ai-augment

Data-augmentation node for [cuvis.ai](https://www.cubert-hyperspectral.com/) hyperspectral
training pipelines.

This plugin provides a single `AugmentationCompose` node that applies a configurable list
of stochastic spatial transforms to hyperspectral cubes (and their paired masks) during
training. It is automatically a no-op at val/test/inference via `execution_stages`.

## Install

```bash
uv sync --extra dev
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

## Available transforms (v0.1.0)

| Name | Operation |
|---|---|
| `RandomHorizontalFlip` | Flip width axis with probability `prob` per sample |
| `RandomVerticalFlip`   | Flip height axis with probability `prob` per sample |
| `Random90Rotate`       | Rotate by random multiple of 90° (0/90/180/270) per sample |
| `RandomSpatialCrop`    | Crop to fixed `(H_out, W_out)` at random offset per sample |

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

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -m "not slow"
uv run ruff format --check cuvis_ai_augment tests
uv run ruff check cuvis_ai_augment tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).
