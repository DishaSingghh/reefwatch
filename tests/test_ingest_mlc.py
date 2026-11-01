"""Tests use synthetic file names and annotation files built in a temp folder (no real MLC data)."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from reefwatch.data.ingest_mlc import (
    build_pairs,
    compile_pattern,
    ingest,
    link_check,
    parse_name,
    read_annotation,
)

ROOT = Path(__file__).resolve().parents[1]
PATTERN = OmegaConf.load(ROOT / "configs" / "data.yaml").mlc.filename_pattern
L1 = {"acropora": "hard_coral", "porites": "hard_coral", "cca": "cca", "sand": "sand_rubble"}


def make_image(folder: Path, name: str, lines: list[str] | None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"")  # contents unused; only the name is parsed
    if lines is not None:
        (folder / name).with_suffix(".txt").write_text("\n".join(lines) + "\n")


@pytest.fixture
def mlc_tree(tmp_path):
    pts = ["10; 20; Acropora", "30; 40; CCA"]
    # Quadrat 1 and 2: three years each. Quadrat 3: one year only.
    for year in (2008, 2009, 2010):
        for q in (1, 2):
            make_image(
                tmp_path / str(year), f"mcr_lter1_fringingreef_pole1-2_qu{q}_{year}0415.jpg", pts
            )
    make_image(tmp_path / "2008", "mcr_lter2_outer10m_pole3-4_qu3_20080501.jpg", pts)
    # Unparseable name: kept, but never linked to a quadrat.
    make_image(tmp_path / "2009", "IMG_0001.jpg", pts)
    # Image without an annotation file.
    make_image(tmp_path / "2010", "mcr_lter1_fringingreef_pole1-2_qu9_20100415.jpg", None)
    # Header line, comma separators and a label with no L1 mapping.
    make_image(
        tmp_path / "2010",
        "mcr_lter2_outer10m_pole3-4_qu3_20100502.jpg",
        ["row,col,label", "5,6,Sand", "7,8,Goniopora"],
    )
    return tmp_path


def test_parse_valid_name():
    info = parse_name("MCR_LTER1_FringingReef_pole1-2_qu7_20090412.jpg", compile_pattern(PATTERN))
    assert info.year == 2009
    assert info.quadrat_id == "mcr_lter1/fringingreef/pole1-2/qu7"


@pytest.mark.parametrize(
    "name", ["IMG_0001.jpg", "mcr_lter1_fringingreef_pole1-2_20090412.jpg", "qu7_2009.jpg"]
)
def test_parse_invalid_names_give_nulls(name):
    info = parse_name(name, compile_pattern(PATTERN))
    assert info.quadrat_id is None and info.year is None


def test_pattern_must_name_quadrat_and_year():
    with pytest.raises(ValueError, match="quadrat"):
        compile_pattern(r"(?P<year>20\d\d)")


def test_read_annotation_separators_and_header(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("row col label\n1;2;Porites\n3,4,CCA\n5\t6\tSand\nbad line\n")
    res = read_annotation(f)
    assert res.points == [(1, 2, "Porites"), (3, 4, "CCA"), (5, 6, "Sand")]
    assert res.bad_lines == 2


def test_ingest_report(mlc_tree):
    points, rep = ingest(mlc_tree, PATTERN, L1)
    assert rep.images == 10
    assert rep.names_matched == 9
    assert rep.unmatched_names == ["IMG_0001.jpg"]
    assert len(rep.missing_annotations) == 1
    assert rep.bad_lines == 1
    assert rep.unmapped_labels == {"Goniopora": 1}
    assert str(points["Row"].dtype) == "Int32"
    assert points.loc[points.CoralNet_label == "Goniopora", "l1"].isna().all()
    assert points.loc[points.Image_name == "IMG_0001.jpg", "quadrat_id"].isna().all()


def test_link_check_and_pairs(mlc_tree):
    points, _ = ingest(mlc_tree, PATTERN, L1)
    check = link_check(points, min_quadrats=3)
    assert check["quadrats"] == 3
    assert check["quadrats_2plus_years"] == 3  # q1, q2 (3 years) and q3 (2008 + 2010)
    assert check["result"] == "PASS"
    assert link_check(points, min_quadrats=4)["result"] == "FAIL"

    pairs = build_pairs(points)
    # q1 and q2: 2008-09 and 2009-10; q3: 2008-10.
    assert len(pairs) == 5
    assert "IMG_0001.jpg" not in set(pairs.image_t0) | set(pairs.image_t1)
    assert (pairs.year_t1 > pairs.year_t0).all()


def test_pairs_ignore_file_order(tmp_path):
    # Same files created in reverse order must give identical pairs.
    names = [f"mcr_lter1_fringingreef_pole1-2_qu1_{y}0415.jpg" for y in (2008, 2009, 2010)]
    a, b = tmp_path / "a", tmp_path / "b"
    for n in names:
        make_image(a, n, ["1;1;Sand"])
    for n in reversed(names):
        make_image(b, n, ["1;1;Sand"])
    pa_ = build_pairs(ingest(a, PATTERN, L1)[0])
    pb_ = build_pairs(ingest(b, PATTERN, L1)[0])
    assert pa_.equals(pb_)


def test_cli(mlc_tree, tmp_path, capsys):
    from reefwatch.data.ingest_mlc import main

    cfg = OmegaConf.load(ROOT / "configs" / "data.yaml")
    cfg.paths.root = str(tmp_path)
    cfg.paths.mlc_dir = str(mlc_tree)
    cfg.paths.interim = "interim"
    cfg.mlc.gate_min_quadrats = 3
    cfg_path = tmp_path / "data.yaml"
    OmegaConf.save(cfg, cfg_path)
    main(["--config", str(cfg_path), "--taxonomy", str(ROOT / "configs" / "taxonomy.yaml")])
    out = capsys.readouterr().out
    assert "-> PASS" in out
    assert (tmp_path / "interim" / "mlc_pairs.parquet").exists()
