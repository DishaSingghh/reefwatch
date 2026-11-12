"""Tests use made-up rows in the NOAA NCRMP column layout (data dictionary InPort 78601)."""

from pathlib import Path

import pandas as pd
import pytest

from reefwatch.data.ingest_noaa import ingest, load_l1_map, photo_pairs, site_pairs

ROOT = Path(__file__).resolve().parents[1]
L1 = load_l1_map(ROOT / "configs" / "taxonomy.yaml")


def rows_for(site, visit, year, date, rep_col, rep, photo, labels, perm="-1"):
    out = []
    for i, lab in enumerate(labels):
        out.append(
            {
                "SITE": site,
                "SITEVISITID": visit,
                "OBS_YEAR": year,
                "DATE_": date,
                "IMAGE_NAME": f"{site}_{year}_{rep}_{photo:02d}",
                rep_col: rep,
                "PHOTOID": photo,
                "CATEGORY_NAME": lab,
                "X_POS": 100 + i,
                "Y_POS": 200 + i,
                "PERM_SITE": perm,
            }
        )
    return out


@pytest.fixture
def csv(tmp_path):
    r = []
    # Site TUT-903: 2015 (old date format, REP column), 2019 and 2023 (REPLICATE column).
    r += rows_for("TUT-903", 1, 2015, "12-Apr-15", "REP", "A", 1, ["Coral", "Coral", "Sediment"])
    r += rows_for("TUT-903", 2, 2019, "2019-05-01", "REPLICATE", "A", 1, ["Coral", "Turf alga"])
    r += rows_for("TUT-903", 3, 2023, "2023-06-02", "REPLICATE", "A", 1, ["Macroalga", "Sediment"])
    # Gear and unclassified points are dropped; an unknown label is kept with empty l1.
    r += rows_for("TUT-903", 3, 2023, "2023-06-02", "REPLICATE", "A", 2, ["Tape and wand"])
    r += rows_for("TUT-903", 3, 2023, "2023-06-02", "REPLICATE", "A", 3, ["Mystery thing"])
    # Non-permanent site is dropped; empty PERM_SITE is kept.
    r += rows_for("OCC-001", 9, 2019, "2019-01-01", "REPLICATE", "A", 1, ["Coral"], perm="0")
    r += rows_for("OFU-100", 4, 2019, "2019-01-01", "REPLICATE", "2", 4, ["Coral"], perm="")
    df = pd.DataFrame(r).astype(str)
    df.loc[0, "X_POS"] = "n/a"  # bad position -> dropped
    path = tmp_path / "noaa.csv"
    df.to_csv(path, index=False)
    return path


def test_report_counts(csv):
    points, rep = ingest(csv, L1)
    assert rep.rows_in == 11
    assert rep.rows_not_permanent == 1
    assert rep.rows_bad_position == 1
    assert rep.rows_excluded_label == 1
    assert rep.unmapped_labels == {"mystery thing": 1}
    assert rep.rows_out == len(points) == 8
    assert "OCC-001" not in set(points.SITE)
    assert points.loc[points.raw_label == "Mystery thing", "l1"].isna().all()


def test_dates_and_photo_keys(csv):
    points, _ = ingest(csv, L1)
    tut = points[points.SITE == "TUT-903"]
    assert set(tut.survey_date) == {"2015-04-12", "2019-05-01", "2023-06-02"}
    # REP (2015) and REPLICATE (2019+) give the same key, so the photo links across years.
    assert tut.loc[tut.PHOTOID == 1, "quadrat_id"].unique().tolist() == ["TUT-903/A/1"]
    # A digit in REPLICATE only encodes the photo number, so it is not used as a letter.
    assert points.loc[points.SITE == "OFU-100", "quadrat_id"].item() == "OFU-100/-/4"


def test_site_pairs_cover_change(csv):
    points, _ = ingest(csv, L1)
    sp = site_pairs(points).set_index("year_t0")
    assert list(sp.index) == [2015, 2019]
    # 2015: 1 coral of 2 valid points (one row dropped for bad x); 2019: 1 coral of 2.
    assert sp.loc[2015, "delta_hard_coral"] == pytest.approx(0.0)
    # 2019 -> 2023: hard coral 0.5 -> 0.0
    assert sp.loc[2019, "delta_hard_coral"] == pytest.approx(-0.5)


def test_photo_pairs(csv):
    points, _ = ingest(csv, L1)
    pp = photo_pairs(points)
    assert len(pp) == 2
    assert set(pp.quadrat_id) == {"TUT-903/A/1"}
    assert set(zip(pp.year_t0, pp.year_t1, strict=True)) == {(2015, 2019), (2019, 2023)}


def test_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"SITE": ["X"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        ingest(p, L1)


def test_cli(csv, tmp_path, capsys):
    from omegaconf import OmegaConf

    from reefwatch.data.ingest_noaa import main

    cfg = OmegaConf.load(ROOT / "configs" / "data.yaml")
    cfg.paths.root = str(tmp_path)
    cfg.paths.noaa_csv = str(csv)
    cfg.paths.interim = "interim"
    OmegaConf.save(cfg, tmp_path / "data.yaml")
    main(
        ["--config", str(tmp_path / "data.yaml"), "--taxonomy", str(ROOT / "configs/taxonomy.yaml")]
    )
    out = capsys.readouterr().out
    assert "site pairs 2" in out and "photo pairs 2" in out
    assert (tmp_path / "interim" / "noaa_site_pairs.parquet").exists()
