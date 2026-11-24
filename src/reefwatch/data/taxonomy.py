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
