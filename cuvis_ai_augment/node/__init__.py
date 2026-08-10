"""Node entry-points for cuvis-ai-augment."""

from cuvis_ai_augment.node.compose import AugmentationCompose
from cuvis_ai_augment.node.crop import Crop

__all__ = ["AugmentationCompose", "Crop"]
