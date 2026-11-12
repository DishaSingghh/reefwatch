"""Ingest NOAA NCRMP benthic annotations from permanent sites and build before/after test pairs.

Source: "Benthic Cover Derived from Analysis of Annotated Benthic Images Collected During
Photoquadrat Surveys at Permanent Sites across the U.S. Pacific Islands" (InPort 78600, NCEI
0317416); column names from its data dictionary (InPort 78601). One row is one annotated point.

Two kinds of pairs, both between consecutive visits to the same SITE:
- site pairs: cover per functional group from all ~300 points of a visit. Reliable.
- photo pairs: same SITE, replicate letter and PHOTOID. Divers re-lay the tape from a permanent
  stake each visit, so the two photos show roughly, not exactly, the same patch.

Run:  python -m reefwatch.data.ingest_noaa [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from omegaconf import OmegaConf

DEFAULT_REQUIRED = [
    "SITE",
    "SITEVISITID",
    "OBS_YEAR",
    "IMAGE_NAME",
    "PHOTOID",
    "CATEGORY_NAME",
    "X_POS",
    "Y_POS",
]


@dataclass
class NoaaReport:
    rows_in: int = 0
    rows_not_permanent: int = 0
    rows_bad_position: int = 0
    rows_excluded_label: int = 0
    unmapped_labels: dict[str, int] = field(default_factory=dict)
    rows_out: int = 0


def load_l1_map(taxonomy_path: Path) -> dict[str, str | None]:
    tax = yaml.safe_load(taxonomy_path.read_text())
    return {k.lower(): v for k, v in tax["l1_map"]["noaa"].items()}


def _replicate(df: pd.DataFrame) -> pd.Series:
    """Replicate letter for the photo key. REP was used before 2018 and REPLICATE after. When
    REPLICATE holds a digit, the dictionary says it only encodes the photo number, so it is
    treated as "no letter" to keep photo keys comparable across years."""
    rep = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in ("REP", "REPLICATE"):
        if col in df:
            val = df[col].astype("string").str.strip().str.upper()
            letter = val.where(val.str.fullmatch(r"[A-Z]").fillna(False))
            rep = rep.fillna(letter)
    return rep


def ingest(
    csv_path: str | Path,
    l1_map: dict[str, str | None],
    required: list[str] | None = None,
    permanent_only: bool = True,
) -> tuple[pd.DataFrame, NoaaReport]:
    required = required or DEFAULT_REQUIRED
    df = pd.read_csv(csv_path, dtype=str)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}; found {list(df.columns)}")
    rep = NoaaReport(rows_in=len(df))

    if permanent_only and "PERM_SITE" in df:
        # The dictionary says PERM_SITE "may only be present for sites established prior to
        # 2023", so an empty value is kept; only an explicit non-permanent value is dropped.
        perm = pd.to_numeric(df["PERM_SITE"], errors="coerce")
        drop = perm.notna() & (perm != -1)
        rep.rows_not_permanent = int(drop.sum())
        df = df[~drop]

    for c in ("X_POS", "Y_POS"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[["X_POS", "Y_POS"]].isna().any(axis=1) | (df[["X_POS", "Y_POS"]] < 0).any(axis=1)
    rep.rows_bad_position = int(bad.sum())
    df = df[~bad].copy()

    label = df["CATEGORY_NAME"].astype("string").str.strip().str.lower()
    known = label.isin(list(l1_map))
    for name, n in label[~known].value_counts().items():
        rep.unmapped_labels[str(name)] = int(n)
    df["l1"] = label.map(l1_map)
    excluded = known & df["l1"].isna()
    rep.rows_excluded_label = int(excluded.sum())
    df = df[~excluded]

    df["survey_year"] = pd.to_numeric(df["OBS_YEAR"], errors="coerce").astype("Int16")
    df["PHOTOID"] = pd.to_numeric(df["PHOTOID"], errors="coerce").astype("Int16")
    df["replicate"] = _replicate(df)
    # Photo key: SITE / replicate letter (if any) / photo number. Never from file order.
    df["quadrat_id"] = (
        df["SITE"].str.strip()
        + "/"
        + df["replicate"].fillna("-")
        + "/"
        + df["PHOTOID"].astype("string")
    ).where(df["PHOTOID"].notna())
    if "DATE_" in df:
        # Before 2023 dates were d-mmm-yy, later YYYY-MM-DD (data dictionary change notes).
        iso = pd.to_datetime(df["DATE_"], format="%Y-%m-%d", errors="coerce")
        old = pd.to_datetime(df["DATE_"], format="%d-%b-%y", errors="coerce")
        df["survey_date"] = iso.fillna(old).dt.date.astype("string")
    else:
        df["survey_date"] = pd.NA
    out = df.rename(columns={"IMAGE_NAME": "Image_name", "X_POS": "x_pos", "Y_POS": "y_pos"})
    out["CoralNet_source"] = "NOAA_NCRMP"
    out["raw_label"] = out["CATEGORY_NAME"]
    keep = [
        "CoralNet_source",
        "SITE",
        "SITEVISITID",
        "survey_year",
        "survey_date",
        "Image_name",
        "replicate",
        "PHOTOID",
        "quadrat_id",
        "x_pos",
        "y_pos",
        "raw_label",
        "l1",
    ]
    keep += [c for c in ("SUBCATEGORY_NAME", "GENERA_NAME", "ISLAND", "REGION_NAME") if c in out]
    out = out[keep].reset_index(drop=True)
    rep.rows_out = len(out)
    return out, rep


def site_cover(points: pd.DataFrame) -> pd.DataFrame:
    """Share of points per L1 group for each site visit (rows: visits, columns: groups)."""
    pts = points.dropna(subset=["l1"])
    counts = pts.groupby(["SITE", "SITEVISITID", "survey_year", "l1"]).size().unstack(fill_value=0)
    cover = counts.div(counts.sum(axis=1), axis=0)
    cover["n_points"] = counts.sum(axis=1)
    return cover.reset_index()


def site_pairs(points: pd.DataFrame) -> pd.DataFrame:
    """Consecutive visits to the same SITE, with cover change (t1 - t0) per L1 group."""
    cov = site_cover(points).sort_values(["SITE", "survey_year", "SITEVISITID"])
    groups = [c for c in cov.columns if c not in ("SITE", "SITEVISITID", "survey_year", "n_points")]
    rows = []
    for site, g in cov.groupby("SITE"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if b.survey_year <= a.survey_year:
                continue  # two visits in one year: ambiguous order, skip
            row = {
                "SITE": site,
                "visit_t0": a.SITEVISITID,
                "visit_t1": b.SITEVISITID,
                "year_t0": int(a.survey_year),
                "year_t1": int(b.survey_year),
                "n_points_t0": int(a.n_points),
                "n_points_t1": int(b.n_points),
            }
            row.update({f"delta_{c}": float(b[c] - a[c]) for c in groups})
            rows.append(row)
    return pd.DataFrame(rows)


def photo_pairs(points: pd.DataFrame) -> pd.DataFrame:
    """Same SITE / replicate / PHOTOID in consecutive survey years. Approximate: the tape is
    re-laid from a permanent stake each visit, so frames are close but not identical."""
    from reefwatch.data.ingest_mlc import build_pairs

    return build_pairs(points)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    args = parser.parse_args(argv)
    cfg = OmegaConf.load(args.config)
    root = Path(cfg.paths.root)
    interim = root / cfg.paths.interim
    interim.mkdir(parents=True, exist_ok=True)

    points, rep = ingest(
        root / cfg.paths.noaa_csv,
        load_l1_map(Path(args.taxonomy)),
        list(cfg.noaa.required_columns),
        cfg.noaa.permanent_only,
    )
    print(f"rows in {rep.rows_in:,} | kept {rep.rows_out:,}")
    print(
        f"dropped: not permanent {rep.rows_not_permanent:,} | bad x/y {rep.rows_bad_position:,} "
        f"| gear/unclassified {rep.rows_excluded_label:,}"
    )
    if rep.unmapped_labels:
        print("labels with no L1 mapping (kept, l1 empty):", rep.unmapped_labels)
    points.to_parquet(interim / "noaa_points.parquet", index=False)
    sp = site_pairs(points)
    pp = photo_pairs(points)
    sp.to_parquet(interim / "noaa_site_pairs.parquet", index=False)
    pp.to_parquet(interim / "noaa_photo_pairs.parquet", index=False)
    years = sorted(int(y) for y in points["survey_year"].dropna().unique())
    print(
        f"sites {points['SITE'].nunique():,} | years {years} | site pairs {len(sp):,} "
        f"| photo pairs {len(pp):,}"
    )


if __name__ == "__main__":
    main()
