"""Ingest the ReefNet point-annotation CSV into Parquet.

The CSV is large (about 925k rows), so it is read in chunks with every column as a string, then
Row and Column are cast explicitly. Rows whose Row or Column is not a whole number go to a rejects
CSV instead of silently becoming NaN, so that rows in = rows out + rejects always holds.

Run:  python -m reefwatch.data.ingest_reefnet [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

# Column names from the HF dataset card. Meanings of some columns are not documented there,
# so the audit notebook checks them on real data.
REQUIRED = ["CoralNet_source", "Image_name", "Row", "Column", "Sorted_ReefNet_label"]
OPTIONAL = [
    "CoralNet_label",
    "AphiaID",
    "Broad_category",
    "Train_or_Test",
    "Latitude",
    "Longitude",
    "Ecoregion",
    "Resolution",
]
INTS = ["Row", "Column"]
FLOATS = ["Latitude", "Longitude"]


@dataclass
class IngestStats:
    rows_in: int = 0
    rows_out: int = 0
    rows_rejected: int = 0


def _select_columns(csv_path: Path) -> list[str]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns {missing}; found {header}")
    return REQUIRED + [c for c in OPTIONAL if c in header]


def _cast(chunk: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Cast numeric columns; return the chunk and a mask of rows to reject."""
    bad = pd.Series(False, index=chunk.index)
    for c in INTS:
        num = pd.to_numeric(chunk[c], errors="coerce")
        # A pixel coordinate must be a non-negative whole number; "12.5" or "Bleached" is rejected.
        bad |= num.isna() | (num < 0) | (num % 1 != 0)
        chunk[c] = num.where(~bad).astype("Int32")
    for c in FLOATS:
        if c in chunk:
            # Coordinates are optional metadata: unparseable values become null, not rejects.
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("float64")
    return chunk, bad


def ingest(
    csv_path: str | Path,
    out_path: str | Path,
    rejects_path: str | Path,
    chunksize: int = 1_000_000,
) -> IngestStats:
    csv_path, out_path, rejects_path = Path(csv_path), Path(out_path), Path(rejects_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    # Start clean so reruns don't append to old rejects.
    rejects_path.unlink(missing_ok=True)

    cols = _select_columns(csv_path)
    stats = IngestStats()
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in pd.read_csv(csv_path, usecols=cols, dtype=str, chunksize=chunksize):
            chunk = chunk[cols]
            stats.rows_in += len(chunk)
            raw = chunk.copy()
            chunk, bad = _cast(chunk)
            if bad.any():
                raw[bad].to_csv(
                    rejects_path, mode="a", index=False, header=not rejects_path.exists()
                )
                stats.rows_rejected += int(bad.sum())
            good = chunk[~bad]
            table = pa.Table.from_pandas(good, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            stats.rows_out += len(good)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"{csv_path} has a header but no rows")
    return stats


def audit(points: pd.DataFrame) -> dict:
    """Summary numbers for the audit notebook."""
    out = {
        "rows": len(points),
        "images": points[["CoralNet_source", "Image_name"]].drop_duplicates().shape[0],
        "sources": points["CoralNet_source"].nunique(),
        "labels": points["Sorted_ReefNet_label"].nunique(),
    }
    for c in ["Sorted_ReefNet_label", "Broad_category", "Train_or_Test"]:
        if c in points:
            out[f"counts_{c}"] = points[c].value_counts(dropna=False)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    cfg = OmegaConf.load(args.config)
    root = Path(cfg.paths.root)
    interim = root / cfg.paths.interim
    stats = ingest(
        root / cfg.paths.reefnet_csv,
        interim / "reefnet_points.parquet",
        interim / "ingest_rejects.csv",
        chunksize=cfg.ingest.chunksize,
    )
    print(
        f"rows in {stats.rows_in:,} | written {stats.rows_out:,} | rejected {stats.rows_rejected:,}"
    )


if __name__ == "__main__":
    main()
