"""wsi-patch-extractor: patch extraction from whole-slide images."""

from .extractor import PatchExtractor
from .tissue_mask import TissueMasker
from .metadata import PatchManifest

__version__ = "0.1.0"
__all__ = ["PatchExtractor", "TissueMasker", "PatchManifest"]
