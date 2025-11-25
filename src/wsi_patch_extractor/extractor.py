"""Core patch extractor using OpenSlide."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

from .metadata import PatchManifest, PatchRecord
from .tissue_mask import TissueMasker

logger = logging.getLogger(__name__)


class PatchExtractor:
    """Extract fixed-size patches from a WSI at a target magnification.

    Patches with tissue fraction below ``tissue_threshold`` are discarded.

    Args:
        patch_size:        Edge length of square patches in pixels.
        magnification:     Target magnification (e.g. 20 for 20×).
        tissue_threshold:  Minimum tissue fraction to keep a patch (0–1).
        masker_mode:       Tissue masking strategy: 'hsv' or 'otsu'.
        overlap:           Overlap fraction between adjacent patches (0–1).
    """

    def __init__(
        self,
        patch_size: int = 256,
        magnification: float = 20.0,
        tissue_threshold: float = 0.70,
        masker_mode: str = "hsv",
        overlap: float = 0.0,
    ) -> None:
        self.patch_size = patch_size
        self.magnification = magnification
        self.tissue_threshold = tissue_threshold
        self.masker = TissueMasker(mode=masker_mode)
        self.stride = int(patch_size * (1.0 - overlap))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        slide_path: str | Path,
        output_dir: str | Path,
        slide_id: str | None = None,
    ) -> PatchManifest:
        """Extract patches from a single WSI.

        Args:
            slide_path:  Path to the .svs / .tif WSI file.
            output_dir:  Directory where patch images are saved.
            slide_id:    Optional identifier; defaults to the slide filename stem.

        Returns:
            PatchManifest with one entry per accepted patch.
        """
        slide_path = Path(slide_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        slide_id = slide_id or slide_path.stem

        slide = openslide.OpenSlide(str(slide_path))
        level, downsample = self._resolve_level(slide)
        w_lv, h_lv = slide.level_dimensions[level]

        step = int(self.stride * downsample)
        manifest = PatchManifest()

        xs = range(0, slide.dimensions[0] - self.patch_size, step)
        ys = range(0, slide.dimensions[1] - self.patch_size, step)
        total = len(xs) * len(ys)

        logger.info(
            "Extracting from %s — level %d (%.1f×), %d candidate patches",
            slide_path.name, level, self.magnification, total,
        )

        with tqdm(total=total, desc=slide_path.name, unit="patch") as pbar:
            for x in xs:
                for y in ys:
                    pbar.update(1)
                    patch = self._read_patch(slide, x, y, level, downsample)
                    arr = np.array(patch)
                    tf = self.masker.tissue_fraction(arr)
                    if tf < self.tissue_threshold:
                        continue
                    patch_id = f"{slide_id}_{x}_{y}"
                    patch_path = output_dir / f"{patch_id}.png"
                    patch.save(patch_path)
                    manifest.add(PatchRecord(
                        patch_id=patch_id,
                        slide_id=slide_id,
                        x=x,
                        y=y,
                        width=self.patch_size,
                        height=self.patch_size,
                        tissue_pct=round(tf, 4),
                        mag_level=level,
                        path=str(patch_path),
                    ))

        slide.close()
        logger.info("Kept %d / %d patches", len(manifest), total)
        return manifest

    def extract_batch(
        self,
        slide_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.svs",
        workers: int = 1,
    ) -> dict[str, PatchManifest]:
        """Extract patches from all slides matching ``pattern`` in a directory."""
        slide_dir = Path(slide_dir)
        slides = sorted(slide_dir.glob(pattern))
        if not slides:
            logger.warning("No slides matching '%s' in %s", pattern, slide_dir)
        manifests: dict[str, PatchManifest] = {}
        for slide_path in slides:
            out = Path(output_dir) / slide_path.stem
            manifests[slide_path.stem] = self.extract(slide_path, out)
        return manifests

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_level(
        self, slide: openslide.OpenSlide
    ) -> tuple[int, float]:
        """Find the level whose downsampled magnification is closest to target."""
        native_mag = float(
            slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40)
        )
        target_downsample = native_mag / self.magnification
        level = slide.get_best_level_for_downsample(target_downsample)
        actual_downsample = slide.level_downsamples[level]
        return level, actual_downsample

    def _read_patch(
        self,
        slide: openslide.OpenSlide,
        x: int,
        y: int,
        level: int,
        downsample: float,
    ) -> Image.Image:
        size = (self.patch_size, self.patch_size)
        region = slide.read_region((x, y), level, size)
        return region.convert("RGB")
