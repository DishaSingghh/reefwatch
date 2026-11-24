from pathlib import Path

import pandas as pd
import pytest

from reefwatch.data.taxonomy import load_taxonomy, map_l1

TAX = load_taxonomy(Path(__file__).resolve().parents[1] / "configs" / "taxonomy.yaml")


def test_reefnet_mapping_paths():
    labels = pd.Series(["Turf algae", "Shadow", "Favia", "Weird thing", "Corallinales"])
    broad = pd.Series(["Algae", "Abiotic background", "Hard corals", "Martian", "Algae"])
    l1, how = map_l1(labels, broad, TAX, "reefnet")
    assert l1.tolist() == ["other_algae", pd.NA, "hard_coral", pd.NA, "cca"]
    assert how.tolist() == ["label", "excluded", "fallback", "unmapped", "label"]


def test_unsorted_broad_is_excluded():
    l1, how = map_l1(pd.Series(["Something new"]), pd.Series(["Unsorted"]), TAX, "reefnet")
    assert l1.isna().all() and how.tolist() == ["excluded"]


def test_every_source_maps_to_known_groups():
    for source in TAX["l1_map"]:
        map_l1(pd.Series(["x"]), None, TAX, source)  # raises on an unknown group


def test_unknown_group_raises():
    tax = {"l1_groups": ["a"], "l1_map": {"s": {"x": "b"}}}
    with pytest.raises(ValueError, match="unknown groups"):
        map_l1(pd.Series(["x"]), None, tax, "s")


def test_water_is_excluded_not_substrate():
    # Water sits under "Abiotic background", whose fallback is sand_rubble; it must not get there.
    l1, how = map_l1(pd.Series(["Water"]), pd.Series(["Abiotic background"]), TAX, "reefnet")
    assert l1.isna().all() and how.tolist() == ["excluded"]


def _hc(label, source, n):
    return pd.DataFrame(
        {
            "Sorted_ReefNet_label": [label] * n,
            "CoralNet_source": [source] * n,
            "Broad_category": ["Hard corals"] * n,
        }
    )


def test_l2_candidates_rules():
    from reefwatch.data.taxonomy import l2_candidates

    pts = pd.concat(
        [
            # Porites: 3 sources with >= 10 each, 120 total -> keep
            _hc("Porites", "s1", 50),
            _hc("Porites", "s2", 40),
            _hc("Porites", "s3", 30),
            # Favia: 120 total but only 2 sources reach 10 -> drop
            _hc("Favia", "s1", 100),
            _hc("Favia", "s2", 15),
            _hc("Favia", "s3", 5),
            # Pavona: 3 good sources but only 30 total -> drop
            _hc("Pavona", "s1", 10),
            _hc("Pavona", "s2", 10),
            _hc("Pavona", "s3", 10),
            # Generic label is excluded even with many points
            _hc("Hard coral", "s1", 500),
        ]
    )
    out = l2_candidates(pts, exclude=["Hard coral"]).set_index("Sorted_ReefNet_label")
    assert "Hard coral" not in out.index
    assert out["keep"].to_dict() == {"Porites": True, "Favia": False, "Pavona": False}
    assert out.loc["Favia", "n_sources_ok"] == 2


def test_condition_map(tmp_path):
    from reefwatch.data.taxonomy import load_condition_map

    xlsx = tmp_path / "lm.xlsx"
    pd.DataFrame(
        {
            "Sorted ReefNet label": ["Acropora", "Acropora", "Porites"],
            "CoralNet label": ["Acropora", "Acropora Dead", "Porites bleached "],
            "Condition indication": [None, "Dead", "Bleached"],
        }
    ).to_excel(xlsx, sheet_name="All_labels", index=False)
    assert load_condition_map(xlsx) == {"Acropora Dead": "Dead", "Porites bleached": "Bleached"}


def test_dead_coral_is_not_live_coral():
    labels = pd.Series(["Acropora", "Acropora", "Hard coral"])
    broad = pd.Series(["Hard corals"] * 3)
    cond = pd.Series([pd.NA, "Dead", "Dead"])
    l1, how = map_l1(labels, broad, TAX, "reefnet", condition=cond)
    assert l1.tolist() == ["hard_coral", "sand_rubble", "sand_rubble"]
    assert how.tolist() == ["fallback", "condition", "condition"]


def test_l2_target_skips_dead_and_unknown():
    from reefwatch.data.taxonomy import l2_target

    classes = TAX["l2_classes"]
    labels = pd.Series(["Acropora", "Acropora", "Hard coral", "Porites"])
    cond = pd.Series([pd.NA, "Dead", pd.NA, "Bleached"])
    out = l2_target(labels, TAX, cond)
    assert out.tolist() == [classes.index("Acropora"), -1, -1, classes.index("Porites")]


def test_l2_classes_sorted_and_unique():
    classes = TAX["l2_classes"]
    assert classes == sorted(set(classes))
    assert "Hard coral" not in classes
