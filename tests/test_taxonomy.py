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
