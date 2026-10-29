from pathlib import Path

import yaml

import reefwatch

ROOT = Path(__file__).resolve().parents[1]


def test_version():
    assert reefwatch.__version__ == "0.1.0"


def test_taxonomy_l1_groups():
    tax = yaml.safe_load((ROOT / "configs" / "taxonomy.yaml").read_text())
    assert len(tax["l1_groups"]) == 6
    assert len(set(tax["l1_groups"])) == 6


def test_mlc_labels_map_to_known_groups():
    tax = yaml.safe_load((ROOT / "configs" / "taxonomy.yaml").read_text())
    mlc = tax["l1_map"]["mlc"]
    assert len(mlc) == 9
    assert set(mlc.values()) <= set(tax["l1_groups"])


def test_data_config_loads():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ROOT / "configs" / "data.yaml")
    assert cfg.patches.crop_px > cfg.patches.store_px
    assert set(cfg.subset.tiers) == {"S", "M"}
