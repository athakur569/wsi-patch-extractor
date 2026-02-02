"""Tests for TissueMasker."""

import numpy as np
import pytest

from wsi_patch_extractor.tissue_mask import TissueMasker, patch_tissue_fraction


def _solid_rgb(r: int, g: int, b: int, size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = [r, g, b]
    return img


def test_tissue_masker_all_background():
    masker = TissueMasker(mode="hsv", hsv_sat_threshold=20)
    white = _solid_rgb(240, 240, 240)
    frac = masker.tissue_fraction(white)
    assert frac < 0.05, f"Expected near-zero tissue fraction, got {frac:.3f}"


def test_tissue_masker_all_tissue():
    masker = TissueMasker(mode="hsv", hsv_sat_threshold=20)
    pink = _solid_rgb(220, 150, 170)
    frac = masker.tissue_fraction(pink)
    assert frac > 0.80, f"Expected high tissue fraction, got {frac:.3f}"


def test_otsu_mode_smoke():
    masker = TissueMasker(mode="otsu")
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    frac = masker.tissue_fraction(img)
    assert 0.0 <= frac <= 1.0


def test_patch_tissue_fraction_convenience():
    pink = _solid_rgb(220, 150, 170)
    frac = patch_tissue_fraction(pink)
    assert 0.0 <= frac <= 1.0


def test_invalid_mode():
    with pytest.raises(ValueError, match="mode must be"):
        TissueMasker(mode="invalid")
