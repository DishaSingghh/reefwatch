"""Pick a stratified subset of ReefNet images so every class is represented.

Macro recall weights every class equally, so rare classes must be guaranteed. Points come in
images, so whole images are selected:

1. Give every point a class: its coral class (L2) when it has one, otherwise its functional
   group (L1). Excluded points (shadow, unknown, ...) and dead coral never count.
2. quota[c] = min(points of c in the pool, quota_cap).
3. Visit classes from rarest to most common. For each class, shuffle its images (seeded) and take
   them round-robin across sources. Each chosen image adds all its points to every class tally.
4. Skip a class once its tally reaches its quota; stop at the image budget. A source may not exceed
   `max_source_frac` of the budget when there are at least 4 sources.

Run:  python -m reefwatch.data.select_subset [--config configs/data.yaml] [--tiers S M test]
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from reefwatch.data.taxonomy import l2_target, load_condition_map, load_taxonomy, map_l1

log = logging.getLogger(__name__)
KEY = ["CoralNet_source", "Image_name"]


def point_classes(points: pd.DataFrame, taxonomy: dict, condition: pd.Series | None) -> pd.Series:
    """Class name per point ("L2:<genus>" or "L1:<group>"), or NA when the point doesn't count."""
    l1, _ = map_l1(
        points["Sorted_ReefNet_label"], points["Broad_category"], taxonomy, "reefnet", condition
    )
    l2 = l2_target(points["Sorted_ReefNet_label"], taxonomy, condition)
    names = np.array(taxonomy["l2_classes"], dtype=object)
    out = pd.Series(pd.NA, index=points.index, dtype="string")
    has_l2 = l2 >= 0
    out[has_l2] = "L2:" + pd.Series(names[l2[has_l2].to_numpy()], index=l2[has_l2].index)
    use_l1 = ~has_l2 & l1.notna()
    out[use_l1] = "L1:" + l1[use_l1]
    return out


def select_images(
    points: pd.DataFrame,
    classes: pd.Series,
    budget: int,
    quota_cap: int,
    seed: int,
    max_source_frac: float = 0.25,
) -> pd.DataFrame:
    """Return the chosen images in selection order with the class that pulled each one in.

    `points` needs CoralNet_source and Image_name; `classes` is aligned to it (NA = ignore).
    Output columns: CoralNet_source, Image_name, reason, order, n_points.
    """
    rng = np.random.default_rng(seed)
    df = points[KEY].assign(cls=classes).dropna(subset=["cls"])
    img_id, uniques = pd.factorize(pd.MultiIndex.from_frame(df[KEY]))
    df = df.assign(img=img_id)
    img_source = np.asarray(uniques.get_level_values(0))

    # Per image: how many points of each class it holds.
    per_img = df.groupby(["img", "cls"]).size()
    img_classes: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for (i, c), n in per_img.items():
        img_classes[i].append((c, int(n)))
    n_points = {i: sum(n for _, n in v) for i, v in img_classes.items()}

    pool_counts = df["cls"].value_counts()
    quota = {c: min(int(n), quota_cap) for c, n in pool_counts.items()}
    # Rarest first; ties broken by name so the order never depends on hashing.
    order = sorted(pool_counts.index, key=lambda c: (pool_counts[c], c))

    n_sources = len(set(img_source))
    source_cap = int(max_source_frac * budget) if n_sources >= 4 else budget
    tally: dict[str, int] = defaultdict(int)
    per_source: dict[str, int] = defaultdict(int)
    chosen: dict[int, str] = {}

    images_of = per_img.reset_index().groupby("cls")["img"].apply(np.asarray)
    for c in order:
        if len(chosen) >= budget:
            break
        if tally[c] >= quota[c]:
            continue
        imgs = images_of[c].copy()
        rng.shuffle(imgs)
        by_source: dict[str, list[int]] = defaultdict(list)
        for i in imgs:
            by_source[img_source[i]].append(int(i))
        # Round-robin: one image from each source in turn (sources in a seeded order).
        sources = sorted(by_source)
        rng.shuffle(sources)
        for batch in zip_longest(*(by_source[s] for s in sources)):
            for i in batch:
                if i is None or i in chosen:
                    continue
                if tally[c] >= quota[c] or len(chosen) >= budget:
                    break
                s = img_source[i]
                if per_source[s] >= source_cap:
                    continue
                chosen[i] = c
                per_source[s] += 1
                for cc, n in img_classes[i]:
                    tally[cc] += n
            if tally[c] >= quota[c] or len(chosen) >= budget:
                break

    rows = [
        {
            "CoralNet_source": uniques[i][0],
            "Image_name": uniques[i][1],
            "reason": c,
            "order": k,
            "n_points": n_points[i],
        }
        for k, (i, c) in enumerate(chosen.items())
    ]
    return pd.DataFrame(rows, columns=[*KEY, "reason", "order", "n_points"])


def coverage(points: pd.DataFrame, classes: pd.Series, chosen: pd.DataFrame) -> pd.DataFrame:
    """Points per class in the pool and in the subset."""
    df = points[KEY].assign(cls=classes).dropna(subset=["cls"])
    keys = pd.MultiIndex.from_frame(chosen[KEY])
    in_subset = pd.MultiIndex.from_frame(df[KEY]).isin(keys)
    return (
        pd.DataFrame(
            {"pool": df["cls"].value_counts(), "subset": df.loc[in_subset, "cls"].value_counts()}
        )
        .fillna(0)
        .astype(int)
        .sort_values("pool")
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    parser.add_argument("--tiers", nargs="*", default=None)
    args = parser.parse_args(argv)
    cfg = OmegaConf.load(args.config)
    root = Path(cfg.paths.root)
    interim = root / cfg.paths.interim
    tax = load_taxonomy(args.taxonomy)

    cols = [*KEY, "Sorted_ReefNet_label", "Broad_category", "CoralNet_label", "Train_or_Test"]
    points = pd.read_parquet(interim / "reefnet_points.parquet", columns=cols)
    xlsx = root / cfg.subset.label_xlsx
    condition = None
    if xlsx.exists():
        condition = points["CoralNet_label"].str.strip().map(load_condition_map(xlsx))
    else:
        log.warning("%s not found: dead coral is not separated from live coral", xlsx)
    classes = point_classes(points, tax, condition)

    for name in args.tiers or list(cfg.subset.tiers):
        tier = cfg.subset.tiers[name]
        pool = points["Train_or_Test"].str.lower() == str(tier.pool).lower()
        chosen = select_images(
            points[pool],
            classes[pool],
            tier.images,
            tier.quota_cap,
            cfg.subset.seed,
            cfg.subset.max_source_frac,
        )
        chosen.to_parquet(interim / f"subset_{name}.parquet", index=False)
        cov = coverage(points[pool], classes[pool], chosen)
        missing = cov.index[(cov["pool"] > 0) & (cov["subset"] == 0)].tolist()
        log.info(
            "tier %s: %d images, %d points, %d/%d classes covered%s",
            name,
            len(chosen),
            int(chosen["n_points"].sum()),
            int((cov["subset"] > 0).sum()),
            len(cov),
            f" (missing: {missing})" if missing else "",
        )
        cov.to_csv(interim / f"subset_{name}_coverage.csv")


if __name__ == "__main__":
    main()
