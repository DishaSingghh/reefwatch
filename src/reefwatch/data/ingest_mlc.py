"""Ingest Moorea Labeled Corals (MLC) point annotations and run the quadrat-link check.

MLC annotations are random points stored as (row, col, label), one annotation file per image.
Quadrat identity and survey year are parsed ONLY from explicit fields in the file name, using the
regex in configs/data.yaml (`mlc.filename_pattern`). A name that does not match gets null fields
and is reported; nothing is ever inferred from file order or image content.

Run:  python -m reefwatch.data.ingest_mlc [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from omegaconf import OmegaConf

KEY_PARTS = ["site", "habitat", "transect", "quadrat"]
SPLIT = re.compile(r"[;,\t ]+")


@dataclass
class NameInfo:
    site: str | None = None
    habitat: str | None = None
    transect: str | None = None
    quadrat: str | None = None
    year: int | None = None
    date: str | None = None

    @property
    def quadrat_id(self) -> str | None:
        if self.quadrat is None or self.year is None:
            return None
        return "/".join(p for p in (getattr(self, k) for k in KEY_PARTS) if p)


def compile_pattern(pattern: str) -> re.Pattern:
    rx = re.compile(pattern, re.IGNORECASE)
    missing = {"quadrat", "year"} - set(rx.groupindex)
    if missing:
        raise ValueError(f"mlc.filename_pattern needs named groups {sorted(missing)}")
    return rx


def parse_name(name: str, rx: re.Pattern) -> NameInfo:
    m = rx.search(Path(name).stem)
    if not m:
        return NameInfo()
    g = {k: (v.lower() if isinstance(v, str) else v) for k, v in m.groupdict().items()}
    return NameInfo(
        site=g.get("site"),
        habitat=g.get("habitat"),
        transect=g.get("transect"),
        quadrat=g.get("quadrat"),
        year=int(g["year"]) if g.get("year") else None,
        date=g.get("date"),
    )


@dataclass
class AnnotationResult:
    points: list[tuple[int, int, str]] = field(default_factory=list)
    bad_lines: int = 0


def read_annotation(path: Path) -> AnnotationResult:
    """Read `row, col, label` lines. Separators may be ; , tab or spaces. A header line or any
    line whose first two fields are not non-negative integers is counted as bad, not guessed."""
    res = AnnotationResult()
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = SPLIT.split(line, maxsplit=2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].strip():
            res.points.append((int(parts[0]), int(parts[1]), parts[2].strip()))
        else:
            res.bad_lines += 1
    return res


@dataclass
class MlcReport:
    images: int = 0
    names_matched: int = 0
    annotation_files: int = 0
    missing_annotations: list[str] = field(default_factory=list)
    unmatched_names: list[str] = field(default_factory=list)
    bad_lines: int = 0
    unmapped_labels: dict[str, int] = field(default_factory=dict)

    @property
    def match_rate(self) -> float:
        return self.names_matched / self.images if self.images else 0.0


def load_l1_map(taxonomy_path: Path) -> dict[str, str]:
    tax = yaml.safe_load(taxonomy_path.read_text())
    # Case-insensitive lookup: raw MLC labels may differ in case from the taxonomy file.
    return {k.lower(): v for k, v in tax["l1_map"]["mlc"].items()}


def ingest(
    mlc_dir: str | Path,
    pattern: str,
    l1_map: dict[str, str],
    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    annotation_ext: str = ".txt",
) -> tuple[pd.DataFrame, MlcReport]:
    rx = compile_pattern(pattern)
    report = MlcReport()
    rows: list[dict] = []
    images = sorted(
        p for p in Path(mlc_dir).rglob("*") if p.is_file() and p.suffix.lower() in image_exts
    )
    for img in images:
        report.images += 1
        info = parse_name(img.name, rx)
        if info.quadrat_id is None:
            report.unmatched_names.append(img.name)
        else:
            report.names_matched += 1
        ann_path = img.with_suffix(annotation_ext)
        if not ann_path.exists():
            report.missing_annotations.append(img.name)
            continue
        report.annotation_files += 1
        ann = read_annotation(ann_path)
        report.bad_lines += ann.bad_lines
        for r, c, label in ann.points:
            l1 = l1_map.get(label.lower())
            if l1 is None:
                report.unmapped_labels[label] = report.unmapped_labels.get(label, 0) + 1
            rows.append(
                {
                    "CoralNet_source": "MLC",
                    "Image_name": img.name,
                    "Row": r,
                    "Column": c,
                    "CoralNet_label": label,
                    "l1": l1,
                    "site": info.site,
                    "habitat": info.habitat,
                    "transect": info.transect,
                    "quadrat_id": info.quadrat_id,
                    "survey_year": info.year,
                    "survey_date": info.date,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Row"] = df["Row"].astype("Int32")
        df["Column"] = df["Column"].astype("Int32")
        df["survey_year"] = df["survey_year"].astype("Int16")
    return df, report


def quadrat_years(points: pd.DataFrame) -> pd.DataFrame:
    """One row per image with a parsed quadrat_id and year."""
    cols = ["quadrat_id", "survey_year", "survey_date", "Image_name"]
    return points.dropna(subset=["quadrat_id", "survey_year"])[cols].drop_duplicates()


def build_pairs(points: pd.DataFrame) -> pd.DataFrame:
    """Same-quadrat image pairs from consecutive survey years, matched by quadrat_id only.
    If a quadrat has several images in one year, every combination is listed and `n_in_year`
    flags it for review rather than picking one silently."""
    imgs = quadrat_years(points)
    years = imgs.groupby("quadrat_id")["survey_year"].apply(lambda s: sorted(set(s)))
    out = []
    for qid, ys in years.items():
        for y0, y1 in zip(ys, ys[1:], strict=False):
            a = imgs[(imgs.quadrat_id == qid) & (imgs.survey_year == y0)]
            b = imgs[(imgs.quadrat_id == qid) & (imgs.survey_year == y1)]
            for _, ra in a.iterrows():
                for _, rb in b.iterrows():
                    out.append(
                        {
                            "quadrat_id": qid,
                            "year_t0": int(y0),
                            "year_t1": int(y1),
                            "image_t0": ra.Image_name,
                            "image_t1": rb.Image_name,
                            "n_in_year": max(len(a), len(b)),
                        }
                    )
    return pd.DataFrame(
        out, columns=["quadrat_id", "year_t0", "year_t1", "image_t0", "image_t1", "n_in_year"]
    )


def link_check(points: pd.DataFrame, min_quadrats: int = 100) -> dict:
    imgs = quadrat_years(points)
    n_years = imgs.groupby("quadrat_id")["survey_year"].nunique()
    multi = int((n_years >= 2).sum())
    return {
        "quadrats": int(n_years.size),
        "quadrats_2plus_years": multi,
        "years_seen": sorted(int(y) for y in imgs["survey_year"].unique()),
        "min_quadrats": min_quadrats,
        "result": "PASS" if multi >= min_quadrats else "FAIL",
    }


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
        root / cfg.paths.mlc_dir,
        cfg.mlc.filename_pattern,
        load_l1_map(Path(args.taxonomy)),
        tuple(cfg.mlc.image_exts),
        cfg.mlc.annotation_ext,
    )
    print(f"images {rep.images:,} | names matched {rep.names_matched:,} ({rep.match_rate:.1%})")
    print(f"annotation files {rep.annotation_files:,} | missing {len(rep.missing_annotations):,}")
    print(f"points {len(points):,} | bad annotation lines {rep.bad_lines:,}")
    if rep.unmatched_names:
        print("unmatched names (first 5):", rep.unmatched_names[:5])
    if rep.unmapped_labels:
        top = sorted(rep.unmapped_labels.items(), key=lambda kv: -kv[1])[:10]
        print("labels with no L1 mapping (kept, l1 empty):", top)
    if rep.images and rep.match_rate < cfg.mlc.min_match_rate:
        print(
            f"WARNING: only {rep.match_rate:.1%} of names match mlc.filename_pattern. "
            "Fix the pattern before trusting the check below."
        )
    if points.empty:
        print("No points ingested.")
        return

    points.to_parquet(interim / "mlc_points.parquet", index=False)
    build_pairs(points).to_parquet(interim / "mlc_pairs.parquet", index=False)
    check = link_check(points, cfg.mlc.gate_min_quadrats)
    print(
        f"quadrats {check['quadrats']} | in 2+ years {check['quadrats_2plus_years']} | "
        f"years {check['years_seen']} | need {check['min_quadrats']} -> {check['result']}"
    )


if __name__ == "__main__":
    main()
