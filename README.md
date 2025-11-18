# wsi-patch-extractor

Efficient patch extraction from whole-slide images (WSIs) for computational pathology.

## Features
- Extract patches at any magnification level via OpenSlide
- HSV/Otsu tissue masking — skip background tiles automatically
- Tile-level metadata manifest (slide ID, coordinates, tissue %)
- Batch processing across cohorts
- CLI for quick extraction runs

## Installation
```bash
pip install wsi-patch-extractor
# or from source:
git clone https://github.com/athakur569/wsi-patch-extractor
cd wsi-patch-extractor && pip install -e .
```

## Quick start
```python
from wsi_patch_extractor import PatchExtractor

extractor = PatchExtractor(
    patch_size=256,
    magnification=20,
    tissue_threshold=0.70,
)
manifest = extractor.extract("slide.svs", output_dir="patches/")
print(f"Extracted {len(manifest)} patches")
```

## CLI
```bash
wsi-extract --slide slide.svs --output patches/ --size 256 --mag 20 --tissue 0.70
wsi-extract --cohort /path/to/slides/ --output patches/ --workers 8
```

## Manifest format
Each extraction produces a `manifest.csv`:
```
patch_id, slide_id, x, y, width, height, tissue_pct, mag_level
```

## Requirements
- openslide-python >= 1.3
- numpy, opencv-python, Pillow, pandas, tqdm

## License
MIT
