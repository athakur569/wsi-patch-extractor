"""Patch manifest: per-tile metadata with CSV persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass
class PatchRecord:
    patch_id: str
    slide_id: str
    x: int
    y: int
    width: int
    height: int
    tissue_pct: float
    mag_level: int
    path: str


class PatchManifest:
    """Ordered collection of PatchRecords with CSV I/O."""

    def __init__(self) -> None:
        self._records: list[PatchRecord] = []

    def add(self, record: PatchRecord) -> None:
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[PatchRecord]:
        return iter(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self._records])

    def save(self, path: str | Path) -> None:
        self.to_dataframe().to_csv(path, index=False)
        print(f"Manifest saved: {path} ({len(self)} patches)")

    @classmethod
    def load(cls, path: str | Path) -> "PatchManifest":
        df = pd.read_csv(path)
        manifest = cls()
        for row in df.itertuples(index=False):
            manifest.add(PatchRecord(**row._asdict()))
        return manifest

    def filter_by_tissue(self, min_tissue: float) -> "PatchManifest":
        """Return a new manifest keeping only patches above tissue threshold."""
        filtered = PatchManifest()
        for r in self._records:
            if r.tissue_pct >= min_tissue:
                filtered.add(r)
        return filtered

    def slide_summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        return (
            df.groupby("slide_id")
            .agg(
                n_patches=("patch_id", "count"),
                mean_tissue=("tissue_pct", "mean"),
            )
            .reset_index()
        )
