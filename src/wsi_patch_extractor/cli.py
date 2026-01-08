"""CLI entrypoint for wsi-patch-extractor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extractor import PatchExtractor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="wsi-extract",
        description="Extract tissue patches from whole-slide images.",
    )
    # Input — mutually exclusive: single slide or cohort directory
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--slide", type=Path, help="Path to a single WSI file")
    src.add_argument("--cohort", type=Path, help="Directory containing multiple WSIs")

    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=256, help="Patch edge length (default 256)")
    parser.add_argument("--mag", type=float, default=20.0, help="Target magnification (default 20)")
    parser.add_argument("--tissue", type=float, default=0.70, help="Min tissue fraction 0–1 (default 0.70)")
    parser.add_argument("--overlap", type=float, default=0.0, help="Patch overlap fraction 0–1 (default 0)")
    parser.add_argument("--mask-mode", choices=["hsv", "otsu"], default="hsv")
    parser.add_argument("--pattern", default="*.svs", help="Glob pattern for cohort mode")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (cohort mode)")

    args = parser.parse_args(argv)

    extractor = PatchExtractor(
        patch_size=args.size,
        magnification=args.mag,
        tissue_threshold=args.tissue,
        masker_mode=args.mask_mode,
        overlap=args.overlap,
    )

    if args.slide:
        manifest = extractor.extract(args.slide, args.output)
        manifest.save(args.output / "manifest.csv")
    else:
        manifests = extractor.extract_batch(
            args.cohort, args.output,
            pattern=args.pattern,
            workers=args.workers,
        )
        combined_path = args.output / "manifest_combined.csv"
        import pandas as pd
        frames = [m.to_dataframe() for m in manifests.values() if len(m) > 0]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(combined_path, index=False)
            print(f"Combined manifest: {combined_path}")

    print("Done.")


if __name__ == "__main__":
    main()
