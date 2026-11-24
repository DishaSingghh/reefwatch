"""Synthetic points: one very common class spread over many images, rarer classes in few."""

import pandas as pd
import pytest

from reefwatch.data.select_subset import coverage, select_images


def make_points(n_sources=5):
    rows = []
    for s in range(n_sources):
        for i in range(40):
            img = f"img{i}"
            # every image: 5 points of the common class
            rows += [(f"src{s}", img, "L1:other_algae")] * 5
            if i % 10 == 0:
                rows += [(f"src{s}", img, "L2:Porites")] * 2
    # a rare class in a single image of one source
    rows += [("src0", "img3", "L2:Rarus")]
    rows += [("src0", "img3", None)]  # ignored point
    df = pd.DataFrame(rows, columns=["CoralNet_source", "Image_name", "cls"])
    return df[["CoralNet_source", "Image_name"]], df["cls"].astype("string")


def test_every_class_in_pool_is_covered():
    pts, cls = make_points()
    chosen = select_images(pts, cls, budget=20, quota_cap=10, seed=0)
    cov = coverage(pts, cls, chosen)
    assert (cov["subset"] > 0).all()
    assert chosen.iloc[0]["reason"] == "L2:Rarus"  # rarest class is visited first


def test_budget_never_exceeded():
    pts, cls = make_points()
    for budget in (1, 7, 50):
        assert len(select_images(pts, cls, budget=budget, quota_cap=10_000, seed=0)) <= budget


def test_source_cap():
    pts, cls = make_points(n_sources=5)
    chosen = select_images(pts, cls, budget=40, quota_cap=10_000, seed=0, max_source_frac=0.25)
    assert chosen["CoralNet_source"].value_counts().max() <= 10


def test_same_seed_same_output_and_seed_matters():
    pts, cls = make_points()
    a = select_images(pts, cls, budget=30, quota_cap=10_000, seed=1)
    b = select_images(pts, cls, budget=30, quota_cap=10_000, seed=1)
    c = select_images(pts, cls, budget=30, quota_cap=10_000, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["Image_name"].equals(c["Image_name"]) or not a["CoralNet_source"].equals(
        c["CoralNet_source"]
    )


def test_quota_stops_a_class():
    pts, cls = make_points()
    chosen = select_images(pts, cls, budget=1000, quota_cap=10, seed=0)
    # once other_algae has >= 10 points no image is added for it
    assert (chosen["reason"] == "L1:other_algae").sum() <= 2


def test_point_classes_uses_l2_then_l1():
    from pathlib import Path

    from reefwatch.data.select_subset import point_classes
    from reefwatch.data.taxonomy import load_taxonomy

    tax = load_taxonomy(Path(__file__).resolve().parents[1] / "configs" / "taxonomy.yaml")
    pts = pd.DataFrame(
        {
            "Sorted_ReefNet_label": ["Porites", "Porites", "Hard coral", "Shadow", "Turf algae"],
            "Broad_category": ["Hard corals"] * 3 + ["Abiotic background", "Algae"],
        }
    )
    cond = pd.Series([pd.NA, "Dead", pd.NA, pd.NA, pd.NA])
    out = point_classes(pts, tax, cond)
    assert out.tolist() == [
        "L2:Porites",
        "L1:sand_rubble",
        "L1:hard_coral",
        pd.NA,
        "L1:other_algae",
    ]


@pytest.mark.parametrize("n_sources", [2, 3])
def test_no_source_cap_with_few_sources(n_sources):
    pts, cls = make_points(n_sources=n_sources)
    chosen = select_images(pts, cls, budget=40, quota_cap=10_000, seed=0)
    assert len(chosen) == 40
