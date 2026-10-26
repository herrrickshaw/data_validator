"""Tests for data_validator.py on small synthetic CSV/parquet fixtures in tmp_path.
No network, no Kaggle API, no real data tree."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

import data_validator as dv


def _prices(symbols, dates, close, sym_col="symbol"):
    rows = [{sym_col: s, "date": d, "close": close(s, i)} for s in symbols for i, d in enumerate(dates)]
    return pd.DataFrame(rows)


RECENT = [(datetime.now() - timedelta(days=k)).strftime("%Y-%m-%d") for k in (3, 2, 1)]


def _validator(tmp_path, frames):
    """Write each {name: df} as a CSV under tmp_path/data and load it like the
    validator does (inventory -> validate)."""
    (tmp_path / "data").mkdir(exist_ok=True)
    for name, df in frames.items():
        df.to_csv(tmp_path / "data" / f"{name}.csv", index=False)
    v = dv.DataValidator(base_path=str(tmp_path))
    return v


def _load(v, sources):
    for s in sources:
        v._validate_single_source(s)
    return sources


# ---- static helpers ------------------------------------------------------------

def test_extract_symbols_and_date_range():
    df = pd.DataFrame({"Ticker": ["A", "B", "A"], "Date": ["2024-01-03", "2024-01-01", "2024-02-01"]})
    assert dv.DataValidator._extract_symbols(df) == {"A", "B"}
    assert dv.DataValidator._extract_date_range(df) == ("2024-01-01", "2024-02-01")
    assert dv.DataValidator._extract_date_range(pd.DataFrame({"x": [1]})) == ("unknown", "unknown")


def test_compute_file_hash(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert dv.DataValidator._compute_file_hash(p) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


# ---- single-source checks -----------------------------------------------------------

def _meta(path, kind="csv"):
    return dv.DataSourceMetadata(source_name=path.stem, source_type=kind, path=str(path),
                                 created_at="", last_modified="", file_size_mb=0, record_count=0,
                                 symbols_count=0, date_range=("", ""))


def test_clean_source_scores_full_quality(tmp_path):
    p = tmp_path / "ok.csv"
    pd.DataFrame({"date": RECENT, "open": [1, 2, 3], "high": [2, 3, 4],
                  "low": [0.5, 1, 2], "close": [1.5, 2.5, 3.5]}).to_csv(p, index=False)
    m = _meta(p)
    dv.DataValidator(str(tmp_path))._validate_single_source(m)
    assert m.validation_errors == [] and m.data_quality_score == 1.0


def test_quality_checks_flag_nulls_dupes_types_and_staleness(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2020-01-01", "2020-01-01", "2020-01-02"],
                  "close": ["x", "x", None], "high": [1, 1, 1], "low": [2, 2, 2],
                  "open": [1, 1, 1]}).to_csv(p, index=False)
    m = _meta(p)
    dv.DataValidator(str(tmp_path))._validate_single_source(m)
    joined = " | ".join(m.validation_errors)
    for frag in ("High nulls", "duplicate rows", "should be numeric", "high < low", "days stale"):
        assert frag in joined
    assert m.data_quality_score == pytest.approx(0.5)


def test_ohlc_sanity_applies_to_capitalised_columns(tmp_path):
    """Check 3 (numeric dtypes) already matches columns case-insensitively; the
    OHLC sanity check must too, or 'High'/'Low' data (yfinance style) is never checked."""
    p = tmp_path / "caps.csv"
    pd.DataFrame({"Date": RECENT, "Open": [1, 1, 1], "High": [1, 1, 1],
                  "Low": [5, 5, 5], "Close": [1, 1, 1]}).to_csv(p, index=False)
    m = _meta(p)
    dv.DataValidator(str(tmp_path))._validate_single_source(m)
    assert any("high < low" in e for e in m.validation_errors)


def test_unreadable_source_scores_zero(tmp_path):
    m = _meta(tmp_path / "missing.csv")
    dv.DataValidator(str(tmp_path))._validate_single_source(m)
    assert m.data_quality_score == 0.0 and m.validation_errors[0].startswith("Failed to validate")


# ---- inventory ----------------------------------------------------------------------

def test_inventory_lists_each_csv_once(tmp_path):
    """'**/*.csv' and 'data/**/*.csv' both match data/x.csv; the same file must not
    become two 'sources' (which inflates counts and makes it 'agree with itself')."""
    v = _validator(tmp_path, {"prices": pd.DataFrame({"symbol": ["A"], "date": RECENT[:1], "close": [1]})})
    csv_sources = [s for s in v._inventory_data_sources() if s.source_type == "csv"]
    assert [s.path for s in csv_sources] == [str(tmp_path / "data" / "prices.csv")]


def test_kaggle_placeholders_do_not_dilute_quality(tmp_path):
    """No Kaggle API is called; the four hardcoded dataset names are not data this
    validator inventoried, so they must not count as sources with quality 0."""
    p = tmp_path / "cache_seed"
    p.mkdir()
    pd.DataFrame({"date": RECENT, "close": [1.0, 2.0, 3.0]}).to_parquet(p / "clean.parquet")
    v = dv.DataValidator(str(tmp_path))
    report = v.run_full_validation()
    assert report.total_sources == 1
    assert report.avg_quality_score == pytest.approx(1.0)


# ---- cross-source consistency ---------------------------------------------------------

def _two_sources(tmp_path, close_a, close_b, symbols=("AAA",), sym_col="symbol"):
    v = _validator(tmp_path, {
        "src_a": _prices(symbols, RECENT, close_a, sym_col),
        "src_b": _prices(symbols, RECENT, close_b, sym_col),
    })
    sources = [s for s in v._inventory_data_sources() if s.source_type == "csv"]
    _load(v, sources)
    return v, sources


def test_cross_source_reports_a_large_price_discrepancy(tmp_path):
    """Proves the review lead: with the same symbol in two sources and a 50% close-price
    disagreement, the check must report it. On the original code every comparison
    raised inside a bare `except: pass`, so it always printed 'Found 0 ... inconsistencies'."""
    v, sources = _two_sources(tmp_path, lambda s, i: 100.0, lambda s, i: 150.0)
    issues = v._check_cross_source_consistency(sources)
    assert len(issues) == len(RECENT)
    assert {i.field for i in issues} == {"close"}
    i = issues[0]
    # discrepancy is relative to whichever source is listed first (glob order)
    expected = 0.5 if (i.source_1, i.value_1) == ("src_a", 100.0) else 50 / 150
    assert i.discrepancy_pct == pytest.approx(expected)


def test_cross_source_agreeing_sources_have_no_issues(tmp_path):
    v, sources = _two_sources(tmp_path, lambda s, i: 100.0 + i, lambda s, i: 100.0 + i)
    assert v._check_cross_source_consistency(sources) == []
    assert v.cross_source_errors == []


def test_cross_source_filters_by_ticker_column(tmp_path):
    """Symbols found via a 'ticker' column must also be filtered on that column;
    otherwise rows of different symbols are compared with each other."""
    a = _prices(["AAA", "BBB"], RECENT, lambda s, i: 10.0 if s == "AAA" else 500.0, "ticker")
    b = a.iloc[::-1].reset_index(drop=True)          # same data, different row order
    v = _validator(tmp_path, {"src_a": a, "src_b": b})
    sources = [s for s in v._inventory_data_sources() if s.source_type == "csv"]
    _load(v, sources)
    assert v._check_cross_source_consistency(sources) == []


def test_cross_source_errors_are_surfaced_not_swallowed(tmp_path):
    """A comparison that cannot run (here: no 'date' column in one source) must be
    reported as a failed check, so '0 discrepancies' is not claimed for it."""
    a = _prices(["AAA"], RECENT, lambda s, i: 1.0)
    b = a.rename(columns={"date": "when"})
    v = _validator(tmp_path, {"src_a": a, "src_b": b})
    sources = [s for s in v._inventory_data_sources() if s.source_type == "csv"]
    _load(v, sources)
    issues = v._check_cross_source_consistency(sources)
    assert issues == []
    assert len(v.cross_source_errors) == 1 and v.cross_source_errors[0]["symbol"] == "AAA"


def test_cross_source_checks_every_common_symbol(tmp_path):
    """Module docstring: 'Exhaustive validation (all data, not samples)'. A
    discrepancy in the 21st common symbol must be found."""
    syms = [f"S{i:02d}" for i in range(25)]
    v, sources = _two_sources(tmp_path, lambda s, i: 100.0,
                              lambda s, i: 200.0 if s == "S24" else 100.0, symbols=syms)
    issues = v._check_cross_source_consistency(sources)
    assert {i.symbol for i in issues} == {"S24"}


# ---- leakage scan --------------------------------------------------------------------------

def test_leakage_scan_flags_patterns(tmp_path):
    (tmp_path / "a.py").write_text("y = df.close.shift(-1)\nok = df.close.shift(1)\n")
    (tmp_path / "b.py").write_text("train = all_data\n")
    issues = dv.DataValidator(str(tmp_path))._detect_data_leakage()
    assert sorted((i.issue_type, i.line_number) for i in issues) == [("future_data", 1), ("test_in_train", 1)]


def test_leakage_scan_flags_unseeded_sample_with_frac(tmp_path):
    """The 'random sampling without seed' rule was written as `sample\\(\\).*frac`,
    which can never match `df.sample(frac=0.2)` (frac is inside the parentheses)."""
    (tmp_path / "s.py").write_text("t = df.sample(frac=0.2)\nu = df.sample(frac=0.2, random_state=1)\n")
    issues = dv.DataValidator(str(tmp_path))._detect_data_leakage()
    assert [(i.issue_type, i.line_number) for i in issues] == [("unfair_split", 1)]


def test_leakage_scan_covers_all_scripts_it_reports(tmp_path):
    """It prints 'Scanned N scripts'; it must actually scan all N, not the first 50."""
    for k in range(55):
        (tmp_path / f"m{k:02d}.py").write_text("x = s.shift(-2)\n")
    issues = dv.DataValidator(str(tmp_path))._detect_data_leakage()
    assert len(issues) == 55


# ---- lineage + report -------------------------------------------------------------------------

def test_lineage_chain():
    t = dv.DataLineageTracker()
    t.track_load("file.csv", "raw", pd.DataFrame({"a": [1, 2]}))
    t.track_transform("raw", "clean", "dropna")
    t.track_transform("clean", "feat", "rsi")
    chain = t.get_lineage("feat")
    assert [c.get("operation") for c in chain] == ["rsi", "dropna", None]
    assert chain[-1]["records"] == 2


def test_report_roundtrips_to_json(tmp_path):
    p = tmp_path / "cache_seed"
    p.mkdir()
    pd.DataFrame({"symbol": ["A"] * 3, "date": RECENT, "close": [1.0, 2, 3]}).to_parquet(p / "x.parquet")
    report = dv.DataValidator(str(tmp_path)).run_full_validation()
    out = tmp_path / "r.json"
    report.save_report(str(out))
    import json
    d = json.loads(out.read_text())
    assert d["total_records"] == 3 and d["data_sources"][0]["source_name"] == "x"


def test_compute_file_hash_honours_algorithm(tmp_path):
    # the `algorithm` parameter must select the digest, not be silently ignored
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    assert dv.DataValidator._compute_file_hash(p, algorithm="md5") == (
        "5eb63bbbe01eeed093cb22bb8f5acdc3"  # md5("hello world"), hand-computed
    )
    assert dv.DataValidator._compute_file_hash(p, algorithm="sha256") == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )
