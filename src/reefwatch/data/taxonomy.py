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
    condition: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return (l1, how) per row.

    `how` is "label" (exact label in l1_map), "fallback" (via l1_fallback on the broad category),
    "condition" (overridden by `condition_l1`, e.g. dead coral), "excluded" (mapped to null on
    purpose) or "unmapped" (nothing matched; l1 is null and the label needs a line in
    taxonomy.yaml). `condition` is the per-point condition from `load_condition_map`, if known.
    """
    label_map = taxonomy["l1_map"].get(source) or {}
    fallback = (taxonomy.get("l1_fallback") or {}).get(source) or {}
    groups = set(taxonomy["l1_groups"])
    targets = [
        *label_map.values(),
        *fallback.values(),
        *(taxonomy.get("condition_l1") or {}).values(),
    ]
    bad = {v for v in targets if v is not None} - groups
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
    cond_map = taxonomy.get("condition_l1") or {}
    if condition is not None and cond_map:
        use_cond = condition.isin(list(cond_map))
        l1 = l1.where(~use_cond, condition.map(cond_map))
        how[use_cond] = "condition"
    how[(how != "unmapped") & l1.isna()] = "excluded"
    return l1.astype("string"), how


def l2_candidates(
    points: pd.DataFrame,
    exclude: list[str],
    min_total: int = 100,
    min_sources: int = 3,
    min_per_source: int = 10,
    label_col: str = "Sorted_ReefNet_label",
    broad_col: str = "Broad_category",
    source_col: str = "CoralNet_source",
) -> pd.DataFrame:
    """Hard-coral labels that pass the ReefNet paper's stated count rules (Sec 3.2, App B.1):
    at least `min_total` points, in at least `min_sources` sources with `min_per_source` points
    each. The paper also dropped labels after expert review, which cannot be redone here, so the
    result can contain more classes than the official 39.

    Returns one row per label with `n`, `n_sources_ok` and `keep`.
    """
    hc = points[(points[broad_col] == "Hard corals") & ~points[label_col].isin(exclude)]
    per_source = hc.groupby([label_col, source_col]).size()
    n_sources_ok = (per_source >= min_per_source).groupby(level=0).sum()
    out = pd.DataFrame({"n": hc.groupby(label_col).size(), "n_sources_ok": n_sources_ok})
    out["keep"] = (out["n"] >= min_total) & (out["n_sources_ok"] >= min_sources)
    return out.sort_values("n", ascending=False).reset_index(names=label_col)


def load_condition_map(xlsx_path: str | Path) -> dict[str, str]:
    """CoralNet label -> condition ("Dead", "Bleached", "Trematodiasis") from the
    All_labels sheet of ReefNet_labelmapping.xlsx. Labels with no condition are left out."""
    sheet = pd.read_excel(xlsx_path, sheet_name="All_labels", dtype=str)
    sheet = sheet.dropna(subset=["CoralNet label", "Condition indication"])
    return dict(
        zip(sheet["CoralNet label"].str.strip(), sheet["Condition indication"], strict=True)
    )


def l2_target(labels: pd.Series, taxonomy: dict, condition: pd.Series | None = None) -> pd.Series:
    """Index into `l2_classes` per point, or -1 when the point is not a coral-class example
    (not in the list, or a condition such as Dead in `condition_exclude_l2`)."""
    index = {name: i for i, name in enumerate(taxonomy.get("l2_classes") or [])}
    out = labels.map(index).fillna(-1).astype("int16")
    excl = taxonomy.get("condition_exclude_l2") or []
    if condition is not None and excl:
        out[condition.isin(excl)] = -1
    return out
