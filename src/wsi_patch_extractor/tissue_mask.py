"""Tissue masking via HSV thresholding and Otsu binarization."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


class TissueMasker:
    """Generate a binary tissue mask from a WSI thumbnail.

    Two modes:
    - 'hsv'  : saturation threshold in HSV space (good for H&E)
    - 'otsu' : Otsu binarization on grayscale (fallback)
    """

    def __init__(
        self,
        mode: str = "hsv",
        hsv_sat_threshold: int = 20,
        kernel_size: int = 7,
    ) -> None:
        if mode not in ("hsv", "otsu"):
            raise ValueError(f"mode must be 'hsv' or 'otsu', got '{mode}'")
        self.mode = mode
        self.hsv_sat_threshold = hsv_sat_threshold
        self.kernel_size = kernel_size

    def get_mask(self, thumbnail: np.ndarray) -> np.ndarray:
        """Return a binary uint8 mask (255=tissue, 0=background).

        Args:
            thumbnail: RGB image as (H, W, 3) uint8 ndarray.

        Returns:
            Binary mask of shape (H, W) uint8.
        """
        if self.mode == "hsv":
            return self._hsv_mask(thumbnail)
        return self._otsu_mask(thumbnail)

    def tissue_fraction(self, thumbnail: np.ndarray) -> float:
        """Return fraction of pixels classified as tissue (0–1)."""
        mask = self.get_mask(thumbnail)
        return float(mask.sum()) / float(mask.size * 255)

    def _hsv_mask(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        _, binary = cv2.threshold(
            saturation, self.hsv_sat_threshold, 255, cv2.THRESH_BINARY
        )
        return self._morphological_clean(binary)

    def _otsu_mask(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return self._morphological_clean(binary)

    def _morphological_clean(self, binary: np.ndarray) -> np.ndarray:
        k = self.kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return opened


def patch_tissue_fraction(
    patch: np.ndarray,
    masker: TissueMasker | None = None,
) -> float:
    """Convenience wrapper: tissue fraction of a single patch."""
    if masker is None:
        masker = TissueMasker()
    return masker.tissue_fraction(patch)
