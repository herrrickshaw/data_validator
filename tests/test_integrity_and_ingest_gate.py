"""Tests for integrity.py (SHA-256 manifest) and ingest_gate.py (dedup + dtypes).
Everything runs inside tmp_path; the repo's own manifest is never written."""

import pandas as pd
import pytest

import ingest_gate as ig
import integrity


@pytest.fixture
def repo(tmp_path, monkeypatch):
    files = {"a.txt": b"alpha", "cache_seed/d.csv": b"x,y\n1,2\n"}
    for rel, data in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    tracked = sorted(files)
    monkeypatch.setattr(integrity, "HERE", tmp_path)
    monkeypatch.setattr(integrity, "MANIFEST", tmp_path / "cache_seed" / "CHECKSUMS.sha256")
    monkeypatch.setattr(integrity, "_tracked_files",
                        lambda data_only=False: [f for f in tracked
                                                 if not data_only or f.startswith(integrity.DATA_PREFIXES)])
    return tmp_path, tracked


def test_verify_without_manifest_fails(repo):
    assert integrity.verify(verbose=False) == 1


def test_generate_then_verify_is_clean(repo):
    assert integrity.generate(verbose=False) == 2
    line = (repo[0] / "cache_seed" / "CHECKSUMS.sha256").read_text().splitlines()[0]
    assert line == "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8  a.txt"
    assert integrity.verify(verbose=False) == 0


def test_verify_detects_modified_missing_and_new(repo, capsys):
    root, tracked = repo
    integrity.generate(verbose=False)
    (root / "a.txt").write_bytes(b"tampered")
    assert integrity.verify() == 1
    assert "MISMATCH  a.txt" in capsys.readouterr().out

    integrity.generate(verbose=False)
    (root / "cache_seed" / "d.csv").unlink()
    assert integrity.verify() == 1
    assert "MISSING   cache_seed/d.csv" in capsys.readouterr().out

    (root / "cache_seed" / "d.csv").write_bytes(b"x,y\n1,2\n")
    tracked.append("new.py")
    (root / "new.py").write_text("")
    assert integrity.verify() == 1
    assert "UNTRACKED-BY-MANIFEST  new.py" in capsys.readouterr().out


def test_data_only_ignores_code_files(repo):
    root, _ = repo
    integrity.generate(verbose=False)
    (root / "a.txt").write_bytes(b"code change")
    assert integrity.verify(data_only=True, verbose=False) == 0


# ---- ingest_gate --------------------------------------------------------------------------

def test_dedup_keeps_last_symbol_date():
    df = pd.DataFrame({"Symbol": ["A", "A", "B"], "Date": ["d1", "d1", "d1"], "Close": [1, 2, 3]})
    clean, dropped = ig.dedup(df)
    assert dropped == 1 and clean["Close"].tolist() == [2, 3]


def test_dedup_passthrough_without_keys():
    df = pd.DataFrame({"x": [1, 1]})
    assert ig.dedup(df) == (df, 0)
    assert ig.dedup(None) == (None, 0)


def test_enforce_dtypes_compacts_and_coerces():
    df = pd.DataFrame({"Open": ["1.5", "bad"], "Close": [2.0, 3.0], "Volume": [10, None]})
    out = ig.enforce_dtypes(df)
    assert str(out["Open"].dtype) == "float32" and pd.isna(out["Open"][1])
    assert str(out["Volume"].dtype) == "int32" and out["Volume"].tolist() == [10, 0]
    assert df["Open"].tolist() == ["1.5", "bad"]       # input not mutated


def test_enforce_dtypes_large_volume_stays_int64():
    out = ig.enforce_dtypes(pd.DataFrame({"Volume": [3_000_000_000]}))
    assert str(out["Volume"].dtype) == "int64" and out["Volume"][0] == 3_000_000_000


def test_gate_frame_dedups_and_enforces(tmp_path, capsys):
    df = pd.DataFrame({"Symbol": ["A", "A"], "Date": ["d", "d"], "Close": [1, 2], "Volume": [5, 6]})
    out = ig.gate_frame("IN", df, tmp_path / "IN.parquet")
    assert len(out) == 1 and str(out["Close"].dtype) == "float32"
    assert "dropped 1 dup rows" in capsys.readouterr().out
