"""Map raw labels to L1 functional groups using configs/taxonomy.yaml (no label lists in code)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_taxonomy(path: str | Path = "configs/taxonomy.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def map_l1(
    labels: pd.Series,
    broad: pd.Series | None,
    taxonomy: dict,
    source: str,
) -> tuple[pd.Series, pd.Series]:
    """Return (l1, how) per row.

    `how` is "label" (exact label in l1_map), "fallback" (via l1_fallback on the broad category),
    "excluded" (mapped to null on purpose) or "unmapped" (nothing matched; l1 is null and the
    label needs a line in taxonomy.yaml).
    """
    label_map = taxonomy["l1_map"].get(source) or {}
    fallback = (taxonomy.get("l1_fallback") or {}).get(source) or {}
    groups = set(taxonomy["l1_groups"])
    bad = {v for v in [*label_map.values(), *fallback.values()] if v is not None} - groups
    if bad:
        raise ValueError(f"taxonomy maps {source} labels to unknown groups {sorted(bad)}")

    in_label = labels.isin(list(label_map))
    l1 = labels.map(label_map)
    how = pd.Series("unmapped", index=labels.index, dtype="string")
    how[in_label] = "label"
    if broad is not None and fallback:
        use_fb = ~in_label & broad.isin(list(fallback))
        l1 = l1.where(~use_fb, broad.map(fallback))
        how[use_fb] = "fallback"
    how[(how != "unmapped") & l1.isna()] = "excluded"
    return l1.astype("string"), how
