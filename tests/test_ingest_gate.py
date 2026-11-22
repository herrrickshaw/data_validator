"""Tests for ingest_gate.py: de-duplication, compact dtypes, the pre-write gate report, the
post-write size/audit report and the self-check CLI.

The VCRUD audit talks to postgres in production, so ``db_handler`` and ``vcrud_manager`` are always
faked (via sys.modules) and ``HERE`` is redirected into tmp_path - nothing touches a database.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import ingest_gate as ig


def _long(rows):
    return pd.DataFrame(rows, columns=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])


# ---- dedup ------------------------------------------------------------------------------


def test_dedup_keeps_the_newest_row_for_each_symbol_and_date():
    df = _long(
        [
            ("A", "2025-01-02", 1, 2, 0.5, 1.5, 10),
            ("A", "2025-01-02", 1, 2, 0.5, 1.8, 11),  # a later correction of the same bar
            ("A", "2025-01-03", 1, 2, 0.5, 1.6, 12),
            ("B", "2025-01-02", 5, 6, 4, 5.5, 20),
        ]
    )
    clean, dropped = ig.dedup(df)
    assert dropped == 1 and len(clean) == 3
    assert clean[(clean.Symbol == "A") & (clean.Date == "2025-01-02")]["Close"].item() == 1.8
    assert len(df) == 4  # the input is not modified


def test_dedup_is_a_noop_without_duplicates_or_keys():
    df = _long([("A", "2025-01-02", 1, 2, 0.5, 1.5, 10)])
    clean, dropped = ig.dedup(df)
    assert dropped == 0 and clean.equals(df)
    assert ig.dedup(None) == (None, 0)
    empty = df.iloc[0:0]
    assert ig.dedup(empty)[1] == 0
    nokeys = pd.DataFrame({"Symbol": ["A", "A"], "Close": [1, 1]})  # no Date column
    out, n = ig.dedup(nokeys)
    assert n == 0 and out is nokeys


# ---- enforce_dtypes ---------------------------------------------------------------------


def test_enforce_dtypes_uses_float32_prices_and_int32_volume():
    df = _long([("A", "d", 1.0, 2.0, 0.5, 1.5, 1000)])
    out = ig.enforce_dtypes(df)
    assert all(out[c].dtype == np.float32 for c in ("Open", "High", "Low", "Close"))
    assert out["Volume"].dtype == np.int32
    assert out["Symbol"].dtype == df["Symbol"].dtype  # non-price columns untouched


def test_enforce_dtypes_widens_volume_only_when_int32_cannot_hold_it():
    edge = ig.enforce_dtypes(_long([("A", "d", 1, 1, 1, 1, 2_147_483_647)]))
    assert edge["Volume"].dtype == np.int32 and edge["Volume"].iloc[0] == 2_147_483_647
    big = ig.enforce_dtypes(_long([("A", "d", 1, 1, 1, 1, 2_147_483_648)]))
    assert big["Volume"].dtype == np.int64 and big["Volume"].iloc[0] == 2_147_483_648
    neg = ig.enforce_dtypes(_long([("A", "d", 1, 1, 1, 1, -2_147_483_649)]))
    assert neg["Volume"].dtype == np.int64  # the magnitude test uses abs()


def test_enforce_dtypes_coerces_junk_prices_to_nan_and_junk_volume_to_zero():
    df = _long([("A", "d", "x", 2, 1, "1.5", None), ("B", "d", 1, 2, 1, 1.5, "7")])
    out = ig.enforce_dtypes(df)
    assert np.isnan(out.loc[0, "Open"]) and out.loc[0, "Close"] == np.float32(1.5)
    assert list(out["Volume"]) == [0, 7]


def test_enforce_dtypes_copies_and_tolerates_missing_columns_and_empties():
    df = pd.DataFrame({"Symbol": ["A"], "Close": [1.25]})
    out = ig.enforce_dtypes(df)
    assert out is not df and df["Close"].dtype == np.float64  # caller's frame is left alone
    assert out["Close"].dtype == np.float32 and "Volume" not in out
    assert ig.enforce_dtypes(None) is None
    empty = df.iloc[0:0]
    assert ig.enforce_dtypes(empty) is empty


# ---- gate_frame -------------------------------------------------------------------------


def test_gate_frame_dedups_and_enforces_dtypes(tmp_path):
    df = _long(
        [
            ("A", "2025-01-02", 1, 2, 0.5, 1.5, 10),
            ("A", "2025-01-02", 1, 2, 0.5, 1.9, 10),
            ("B", "2025-01-02", 5, 6, 4, 5.5, 20),
        ]
    )
    out = ig.gate_frame("US", df, tmp_path / "US.parquet", verbose=False)
    assert len(out) == 2 and out["Close"].dtype == np.float32 and out["Volume"].dtype == np.int32


def test_gate_frame_reports_rows_symbols_duplicates_and_previous_size(tmp_path, capsys):
    path = tmp_path / "US.parquet"
    path.write_bytes(b"x" * 3000)  # the previous on-disk file
    df = _long(
        [
            ("A", "2025-01-02", 1, 2, 0.5, 1.5, 10),
            ("A", "2025-01-02", 1, 2, 0.5, 1.9, 10),
            ("B", "2025-01-02", 5, 6, 4, 5.5, 20),
        ]
    )
    ig.gate_frame("US", df, path)
    out = capsys.readouterr().out
    assert "ingest-gate[US]: 2 rows / 2 symbols" in out
    assert "dropped 1 dup rows" in out and "prev 0.0MB (~1500.0 B/row)" in out


def test_gate_frame_counts_distinct_symbols_not_rows(tmp_path, capsys):
    df = _long(
        [
            ("A", "2025-01-02", 1, 2, 0.5, 1.5, 10),
            ("A", "2025-01-03", 1, 2, 0.5, 1.6, 10),
            ("B", "2025-01-02", 5, 6, 4, 5.5, 20),
        ]
    )
    ig.gate_frame("US", df, tmp_path / "x.parquet")
    assert "3 rows / 2 symbols" in capsys.readouterr().out


def test_gate_frame_says_zero_dups_and_is_quiet_when_not_verbose(tmp_path, capsys):
    df = _long([("A", "2025-01-02", 1, 2, 0.5, 1.5, 10)])
    ig.gate_frame("US", df, tmp_path / "missing.parquet")
    out = capsys.readouterr().out
    assert "0 dups" in out and "prev 0.0MB (~0.0 B/row)" in out
    ig.gate_frame("US", df, tmp_path / "missing.parquet", verbose=False)
    assert capsys.readouterr().out == ""


def test_gate_frame_handles_an_empty_frame(tmp_path, capsys):
    out = ig.gate_frame("US", pd.DataFrame(), tmp_path / "x.parquet")
    assert out.empty and "0 rows / 0 symbols" in capsys.readouterr().out


# ---- audit / report_after ---------------------------------------------------------------


@pytest.fixture
def vcrud(monkeypatch, tmp_path):
    """Fake VCRUD modules, HERE redirected into tmp_path, sys.path restored afterwards."""
    monkeypatch.setattr(ig, "HERE", tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    log = {"db": [], "scan": [], "created": []}

    class DatabaseHandler:
        def __init__(self, url):
            log["db"].append(url)

        def create_file(self, rec):
            log["created"].append(rec)

    class LocalFileIndexer:
        def __init__(self, root):
            log["root"] = root

        def scan_directory(self, name, patterns):
            log["scan"].append((name, patterns))
            return ["rec1", "rec2"]

    monkeypatch.setitem(
        sys.modules, "db_handler", types.SimpleNamespace(DatabaseHandler=DatabaseHandler)
    )
    monkeypatch.setitem(
        sys.modules, "vcrud_manager", types.SimpleNamespace(LocalFileIndexer=LocalFileIndexer)
    )
    return log


def test_audit_registers_the_written_file_with_vcrud(vcrud, tmp_path, monkeypatch):
    monkeypatch.setenv("VCRUD_DB", "postgresql://example.test/vcrud")
    path = tmp_path / "cache_seed" / "US.parquet"
    assert ig._audit("US", path, {}) == "audited"
    assert vcrud["db"] == ["postgresql://example.test/vcrud"]  # honours VCRUD_DB
    assert vcrud["root"] == str(tmp_path)
    assert vcrud["scan"] == [("ingest-US", ["cache_seed/US.parquet"])]  # path relative to the repo
    assert vcrud["created"] == ["rec1", "rec2"]  # every scanned record is registered
    assert str(tmp_path / ".vcrud_bin") in sys.path  # the tools dir is made importable


def test_audit_falls_back_to_a_local_postgres_when_no_env_is_set(vcrud, tmp_path, monkeypatch):
    monkeypatch.delenv("VCRUD_DB", raising=False)
    assert ig._audit("US", tmp_path / "US.parquet", {}) == "audited"
    assert vcrud["db"][0].startswith("postgresql://") and vcrud["db"][0].endswith("/vcrud")


def test_audit_adds_the_tools_dir_to_sys_path_only_once(vcrud, tmp_path):
    ig._audit("US", tmp_path / "a.parquet", {})
    ig._audit("US", tmp_path / "b.parquet", {})
    assert sys.path.count(str(tmp_path / ".vcrud_bin")) == 1


def test_audit_is_never_fatal(vcrud, tmp_path):
    outside = tmp_path.parent / "elsewhere.parquet"  # relative_to(HERE) raises
    assert ig._audit("US", outside, {}) == "audit-skipped"

    def boom(*a, **k):
        raise RuntimeError("postgres is down")

    sys.modules["db_handler"].DatabaseHandler = boom
    assert ig._audit("US", tmp_path / "US.parquet", {}) == "audit-skipped"


def test_report_after_reports_the_size_delta_and_audit_status(vcrud, tmp_path, capsys):
    path = tmp_path / "US.parquet"
    path.write_bytes(b"x" * 3_000_000)
    rep = ig.report_after("US", path, before_bytes=1_000_000)
    assert rep == {
        "market": "US",
        "before_mb": 1.0,
        "after_mb": 3.0,
        "delta_mb": 2.0,
        "audit": "audited",
    }
    out = capsys.readouterr().out
    assert "ingest-gate[US]: on-disk 3.0MB (+2.0MB) · audited" in out


def test_report_after_shows_a_shrinking_file_and_a_missing_one(vcrud, tmp_path, capsys):
    path = tmp_path / "US.parquet"
    path.write_bytes(b"x" * 1_000_000)
    assert ig.report_after("US", path, before_bytes=4_000_000)["delta_mb"] == -3.0
    assert "(-3.0MB)" in capsys.readouterr().out
    gone = ig.report_after("US", tmp_path / "gone.parquet", before_bytes=2_000_000, verbose=False)
    assert gone["after_mb"] == 0.0 and gone["delta_mb"] == -2.0
    assert capsys.readouterr().out == ""


# ---- main -------------------------------------------------------------------------------


def test_main_self_checks_the_ltm_store(monkeypatch, capsys, tmp_path):
    df = _long(
        [
            ("A", "2025-01-02", 1, 2, 0.5, 1.5, 10),
            ("A", "2025-01-02", 1, 2, 0.5, 1.9, 10),
        ]
    )
    seen = []
    fake = types.SimpleNamespace(
        _ltm_path=lambda m: seen.append(m) or tmp_path / f"{m}.parquet",
        _read_long=lambda p: df,
    )
    monkeypatch.setitem(sys.modules, "market_memory", fake)
    monkeypatch.setattr(sys, "argv", ["ingest_gate.py", "--market", "JP"])
    assert ig.main() == 0
    out = capsys.readouterr().out
    assert seen == ["JP"] and "JP: 2 rows, dupes dropped=1" in out
    assert "'Close': 'float32'" in out and "'Volume': 'int32'" in out


def test_main_defaults_to_the_india_market(monkeypatch, tmp_path, capsys):
    seen = []
    fake = types.SimpleNamespace(
        _ltm_path=lambda m: seen.append(m) or tmp_path / "x.parquet",
        _read_long=lambda p: _long([("A", "d", 1, 1, 1, 1, 1)]),
    )
    monkeypatch.setitem(sys.modules, "market_memory", fake)
    monkeypatch.setattr(sys, "argv", ["ingest_gate.py"])
    assert ig.main() == 0 and seen == ["IN"] and capsys.readouterr().out.startswith("IN: 1 rows")
