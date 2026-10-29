from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from reefwatch.data.ingest_reefnet import audit, ingest

FIXTURE = Path(__file__).parent / "fixtures" / "reefnet_sample.csv"


@pytest.fixture
def ingested(tmp_path):
    out, rejects = tmp_path / "points.parquet", tmp_path / "rejects.csv"
    # chunksize 6 forces several chunks, so chunk handling is tested too.
    stats = ingest(FIXTURE, out, rejects, chunksize=6)
    return stats, out, rejects


def test_counts_balance(ingested):
    stats, out, rejects = ingested
    assert stats.rows_in == 20
    assert stats.rows_in == stats.rows_out + stats.rows_rejected
    assert pq.read_metadata(out).num_rows == stats.rows_out
    assert len(pd.read_csv(rejects, dtype=str)) == stats.rows_rejected


def test_bad_rows_go_to_rejects(ingested):
    stats, _, rejects = ingested
    rej = pd.read_csv(rejects, dtype=str)
    assert stats.rows_rejected == 2
    assert "Bleached" in rej["Row"].tolist()
    assert "12.5" in rej["Column"].tolist()


def test_output_dtypes(ingested):
    _, out, _ = ingested
    df = pd.read_parquet(out)
    assert str(df["Row"].dtype) == "Int32"
    assert str(df["Column"].dtype) == "Int32"
    assert df["Latitude"].dtype == "float64"
    assert df["Latitude"].isna().sum() == 1  # "n/a" kept as null, not rejected


def test_rerun_does_not_append(tmp_path):
    out, rejects = tmp_path / "p.parquet", tmp_path / "r.csv"
    ingest(FIXTURE, out, rejects)
    ingest(FIXTURE, out, rejects)
    assert len(pd.read_csv(rejects)) == 2


def test_missing_required_column(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.read_csv(FIXTURE, dtype=str).drop(columns=["Row"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        ingest(bad, tmp_path / "p.parquet", tmp_path / "r.csv")


def test_audit_numbers(ingested):
    _, out, _ = ingested
    summary = audit(pd.read_parquet(out))
    assert summary["rows"] == 18
    assert summary["sources"] == 3
    assert summary["counts_Train_or_Test"].sum() == 18


def test_cli_reads_config(tmp_path, capsys):
    from reefwatch.data.ingest_reefnet import main

    cfg = tmp_path / "data.yaml"
    cfg.write_text(
        f"paths: {{root: {tmp_path}, reefnet_csv: {FIXTURE}, interim: interim}}\n"
        "ingest: {chunksize: 1000}\n"
    )
    main(["--config", str(cfg)])
    assert "rejected 2" in capsys.readouterr().out
    assert (tmp_path / "interim" / "reefnet_points.parquet").exists()
