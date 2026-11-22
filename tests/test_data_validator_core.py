"""Edge/boundary tests for data_validator.py: quality-check thresholds, inventory
limits and failure isolation, repo mapping, summary counts, lineage, report printing
and the ``__main__`` entry point. All on tmp_path fixtures; no network, no real data tree."""

import builtins
import json
import runpy
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import data_validator as dv
from data_validator import DataValidator


def _day(days_ago):
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _meta(path, kind="csv", name=None):
    return dv.DataSourceMetadata(
        source_name=name or path.stem,
        source_type=kind,
        path=str(path),
        created_at="",
        last_modified="",
        file_size_mb=0,
        record_count=0,
        symbols_count=0,
        date_range=("", ""),
    )


def _validate(tmp_path, df, name="s"):
    p = tmp_path / f"{name}.csv"
    df.to_csv(p, index=False)
    m = _meta(p)
    DataValidator(str(tmp_path))._validate_single_source(m)
    return m


# ---- _validate_single_source: thresholds -----------------------------------------------


def test_null_fraction_exactly_ten_percent_is_not_flagged(tmp_path):
    vals = [None] + [float(i) for i in range(1, 10)]  # 1 null / 10 rows = 0.1
    m = _validate(tmp_path, pd.DataFrame({"x": vals}))
    assert m.validation_errors == []


def test_null_fraction_above_ten_percent_names_the_column(tmp_path):
    vals = [None, None] + [float(i) for i in range(1, 9)]  # 0.2
    m = _validate(tmp_path, pd.DataFrame({"x": vals, "y": list(range(10))}))
    assert m.validation_errors == ["High nulls in ['x']"]
    assert m.data_quality_score == pytest.approx(0.9)


def test_duplicate_row_count_is_reported(tmp_path):
    m = _validate(tmp_path, pd.DataFrame({"x": [1, 1, 1, 2]}))
    assert m.validation_errors == ["2 duplicate rows found"]


def test_non_numeric_ohlcv_columns_are_flagged_case_insensitively(tmp_path):
    m = _validate(tmp_path, pd.DataFrame({"VOLUME": ["a", "b"], "name": ["p", "q"]}))
    assert len(m.validation_errors) == 1
    assert m.validation_errors[0].startswith("Column 'VOLUME' should be numeric")


def test_ohlc_check_needs_all_four_columns(tmp_path):
    # high < low but no 'open'/'close': the OHLC sanity check must not run
    m = _validate(tmp_path, pd.DataFrame({"high": [1, 1], "low": [5, 6]}))
    assert m.validation_errors == []


def test_ohlc_equal_high_low_is_valid_and_invalid_rows_are_counted(tmp_path):
    ok = _validate(
        tmp_path,
        pd.DataFrame({"open": [1, 2], "high": [3, 3], "low": [3, 1], "close": [2, 2]}),
        name="ok",
    )
    assert ok.validation_errors == []
    bad = _validate(
        tmp_path,
        pd.DataFrame({"open": [1, 2, 3], "high": [1, 1, 9], "low": [2, 2, 1], "close": [1, 1, 1]}),
        name="bad",
    )
    assert bad.validation_errors == ["2 rows with high < low (invalid OHLC)"]


def _dated(n, step_days):
    """n dates `step_days` apart, ending today (so freshness never triggers)."""
    return [_day(step_days * (n - 1 - i)) for i in range(n)]


def test_date_gap_boundary_five_days_is_not_a_gap(tmp_path):
    m = _validate(tmp_path, pd.DataFrame({"date": _dated(30, 5), "v": range(30)}))
    assert m.validation_errors == []


def test_date_gaps_need_more_than_ten_to_be_flagged(tmp_path):
    ten = _validate(tmp_path, pd.DataFrame({"date": _dated(11, 6), "v": range(11)}), name="ten")
    assert ten.validation_errors == []  # 10 gaps > 5 days: not "many"
    eleven = _validate(
        tmp_path, pd.DataFrame({"date": _dated(12, 6), "v": range(12)}), name="eleven"
    )
    assert eleven.validation_errors == ["Many date gaps detected (11 gaps > 5 days)"]


def test_unsorted_dates_do_not_count_as_gaps_backwards(tmp_path):
    # a negative diff is never > 5 days
    dates = list(reversed(_dated(5, 1)))
    m = _validate(tmp_path, pd.DataFrame({"date": dates, "v": range(5)}))
    assert m.validation_errors == []


def test_staleness_boundary_30_days_fresh_31_stale(tmp_path):
    fresh = _validate(tmp_path, pd.DataFrame({"date": [_day(31), _day(30)], "v": [1, 2]}), "f")
    assert fresh.validation_errors == []
    stale = _validate(tmp_path, pd.DataFrame({"date": [_day(32), _day(31)], "v": [1, 2]}), "s")
    assert len(stale.validation_errors) == 1
    assert stale.validation_errors[0].startswith("Data is 31 days stale")
    assert _day(31) in stale.validation_errors[0]


def test_lowercase_date_column_is_preferred_over_capitalised(tmp_path):
    df = pd.DataFrame({"date": [_day(1), _day(0)], "Date": ["2001-01-01", "2001-01-02"]})
    m = _validate(tmp_path, df)
    assert m.validation_errors == []  # stale 'Date' column was not the one used


def test_quality_score_drops_by_point_one_per_error(tmp_path):
    df = pd.DataFrame({"date": ["2020-01-01"] * 2, "x": [1, 1]})  # duplicates + stale
    m = _validate(tmp_path, df)
    assert len(m.validation_errors) == 2
    assert m.data_quality_score == pytest.approx(0.8)


def test_quality_score_is_floored_at_zero(tmp_path):
    cols = {c: ["a", "a"] for c in ("open", "high", "low", "close", "volume")}
    cols["date"] = ["2020-01-01", "2020-01-01"]
    cols["x"] = [None, None]
    m = _validate(tmp_path, pd.DataFrame(cols))
    # nulls, dupes, 5 non-numeric, gaps none, stale -> at least 8 errors; score never negative
    assert len(m.validation_errors) >= 8
    assert 0.0 <= m.data_quality_score <= 0.2 + 1e-9


def test_status_marker_in_output(tmp_path, capsys):
    _validate(tmp_path, pd.DataFrame({"x": [1, 2]}), name="clean")
    out = capsys.readouterr().out
    assert "✓ clean: 2 rows, quality=100.00%" in out
    _validate(tmp_path, pd.DataFrame({"x": [1, 1]}), name="dup")
    out = capsys.readouterr().out
    assert "⚠️ dup: 2 rows, quality=90.00%" in out


def test_non_file_source_types_are_left_untouched(tmp_path):
    m = _meta(tmp_path / "nothing", kind="kaggle")
    m.validation_errors = ["Not yet downloaded"]
    v = DataValidator(str(tmp_path))
    v._validate_single_source(m)
    assert m.validation_errors == ["Not yet downloaded"]
    assert m.data_quality_score == 0.0
    assert v._loaded_data == {}


def test_validated_source_is_cached_by_name(tmp_path):
    p = tmp_path / "c.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(p, index=False)
    v = DataValidator(str(tmp_path))
    v._validate_single_source(_meta(p))
    assert list(v._loaded_data) == ["c"] and len(v._loaded_data["c"]) == 3


def test_parquet_source_is_loaded_with_parquet_reader(tmp_path):
    p = tmp_path / "p.parquet"
    pd.DataFrame({"x": [1.0, 2.0]}).to_parquet(p)
    m = _meta(p, kind="parquet")
    DataValidator(str(tmp_path))._validate_single_source(m)
    assert m.validation_errors == [] and m.data_quality_score == 1.0


def test_header_only_csv_with_date_column_fails_cleanly(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("date,close\n")
    m = _meta(p)
    DataValidator(str(tmp_path))._validate_single_source(m)
    assert m.validation_errors[0].startswith("Failed to validate")
    assert m.data_quality_score == 0.0


# ---- inventory ------------------------------------------------------------------------------


def test_corrupt_parquet_is_skipped_and_reported(tmp_path, capsys):
    cs = tmp_path / "cache_seed"
    cs.mkdir()
    (cs / "bad.parquet").write_bytes(b"not a parquet file")
    pd.DataFrame({"symbol": ["A", "B", "A"], "date": [_day(1)] * 3}).to_parquet(cs / "good.parquet")
    sources = DataValidator(str(tmp_path))._scan_parquet_files()
    assert [s.source_name for s in sources] == ["good"]
    assert "Error reading bad.parquet" in capsys.readouterr().out


def test_parquet_metadata_fields(tmp_path):
    p = tmp_path / "m.parquet"
    pd.DataFrame(
        {"ticker": ["A", "B", "A"], "date": ["2024-03-02", "2024-03-01", "2024-03-09"]}
    ).to_parquet(p)
    m = DataValidator(str(tmp_path))._extract_parquet_metadata(pd.read_parquet(p), p)
    assert (m.source_name, m.source_type, m.path) == ("m", "parquet", str(p))
    assert m.record_count == 3 and m.symbols_count == 2
    assert m.date_range == ("2024-03-01", "2024-03-09")
    assert m.columns == ["ticker", "date"]
    assert m.file_size_mb == pytest.approx(p.stat().st_size / 1048576)
    assert len(m.checksums["sha256"]) == 64


def test_unreadable_csv_is_skipped_silently(tmp_path):
    (tmp_path / "empty.csv").write_text("")
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "ok.csv", index=False)
    sources = DataValidator(str(tmp_path))._scan_csv_files()
    assert [s.source_name for s in sources] == ["ok"]


def test_csv_scan_is_capped_at_ten_files_per_pattern(tmp_path):
    for k in range(12):
        pd.DataFrame({"x": [k]}).to_csv(tmp_path / f"f{k:02d}.csv", index=False)
    assert len(DataValidator(str(tmp_path))._scan_csv_files()) == 10


def test_csv_metadata_samples_at_most_1000_rows(tmp_path):
    pd.DataFrame({"symbol": [f"S{i}" for i in range(1500)], "date": ["2024-01-01"] * 1500}).to_csv(
        tmp_path / "big.csv", index=False
    )
    (m,) = DataValidator(str(tmp_path))._scan_csv_files()
    assert m.record_count == 1000 and m.symbols_count == 1000
    assert m.source_type == "csv" and m.date_range == ("2024-01-01", "2024-01-01")


def test_kaggle_candidates_are_hardcoded_placeholders(tmp_path):
    cands = DataValidator(str(tmp_path))._scan_kaggle_datasets()
    assert [c.source_name for c in cands] == [
        "stock-market-data",
        "stock-market-data",
        "nifty50-stock-market-data",
        "stock-market-dataset",
    ]
    assert all(c.source_type == "kaggle" and c.record_count == 0 for c in cands)
    assert cands[0].path == "kaggle://sudalairajkumar/stock-market-data"
    assert all(c.validation_errors == ["Not yet downloaded"] for c in cands)


def test_inventory_keeps_kaggle_candidates_out_of_sources(tmp_path):
    v = DataValidator(str(tmp_path))
    assert v._inventory_data_sources() == []
    assert len(v.kaggle_candidates) == 4


# ---- symbol / date helpers --------------------------------------------------------------------


@pytest.mark.parametrize("col", ["symbol", "Symbol", "ticker", "Ticker", "SYMBOL"])
def test_symbol_column_aliases(col):
    df = pd.DataFrame({col: ["A", "B", "A"]})
    assert DataValidator._symbol_column(df) == col
    assert DataValidator._extract_symbols(df) == {"A", "B"}


def test_symbol_column_priority_and_absence():
    df = pd.DataFrame({"ticker": ["T"], "symbol": ["S"]})
    assert DataValidator._symbol_column(df) == "symbol"
    assert DataValidator._symbol_column(pd.DataFrame({"x": [1]})) is None
    assert DataValidator._extract_symbols(pd.DataFrame({"x": [1]})) == set()


@pytest.mark.filterwarnings("ignore")
def test_date_range_falls_through_unparseable_column():
    df = pd.DataFrame({"date": ["zzz"], "Date": ["2024-05-01"]})
    assert DataValidator._extract_date_range(df) == ("2024-05-01", "2024-05-01")


@pytest.mark.filterwarnings("ignore")
def test_date_range_all_unparseable_is_unknown():
    df = pd.DataFrame({"date": ["zzz", "yyy"]})
    assert DataValidator._extract_date_range(df) == ("unknown", "unknown")


def test_date_range_prefers_earlier_alias():
    df = pd.DataFrame({"Date": ["2001-01-01"], "date": ["2024-04-04"], "DATE": ["1999-09-09"]})
    assert DataValidator._extract_date_range(df) == ("2024-04-04", "2024-04-04")


@pytest.mark.parametrize("col", ["date", "Date", "datetime", "Datetime", "DATE"])
def test_date_range_column_aliases(col):
    df = pd.DataFrame({col: ["2024-02-02", "2024-01-01"]})
    assert DataValidator._extract_date_range(df) == ("2024-01-01", "2024-02-02")


def test_file_hash_covers_files_larger_than_one_chunk(tmp_path):
    import hashlib

    data = bytes(range(256)) * 100  # 25,600 bytes: many 4096-byte chunks
    p = tmp_path / "big.bin"
    p.write_bytes(data)
    assert DataValidator._compute_file_hash(p) == hashlib.sha256(data).hexdigest()


# ---- cross-source: thresholds and pairing ------------------------------------------------------


def _xs(frames):
    """Validator with pre-loaded frames, in the given order (no filesystem)."""
    v = DataValidator("/nonexistent-never-read")
    v._loaded_data = dict(frames)
    return v, [_src(n) for n in frames]


def _px(closes, symbol="AAA", dates=None):
    dates = dates or [f"2024-01-{d:02d}" for d in range(1, len(closes) + 1)]
    return pd.DataFrame({"symbol": symbol, "date": dates, "close": closes})


def test_cross_source_two_percent_is_the_threshold():
    v, srcs = _xs({"a": _px([100.0, 100.0]), "b": _px([102.0, 102.5])})
    (issue,) = v._check_cross_source_consistency(srcs)
    assert issue.date.startswith("2024-01-02")
    assert (issue.value_1, issue.value_2) == (100.0, 102.5)
    assert issue.discrepancy_pct == pytest.approx(0.025)
    assert (issue.source_1, issue.source_2, issue.symbol, issue.field) == ("a", "b", "AAA", "close")


def test_cross_source_ignores_non_positive_reference_prices():
    v, srcs = _xs({"a": _px([0.0, -5.0, 10.0]), "b": _px([50.0, 50.0, 10.0])})
    assert v._check_cross_source_consistency(srcs) == []


def test_cross_source_compares_only_shared_dates():
    a = _px([100.0, 100.0], dates=["2024-01-01", "2024-01-02"])
    b = _px([500.0, 100.0], dates=["2023-12-31", "2024-01-02"])
    v, srcs = _xs({"a": a, "b": b})
    assert v._check_cross_source_consistency(srcs) == []


def test_cross_source_duplicate_dates_use_first_row_only():
    a = _px([100.0, 999.0], dates=["2024-01-01", "2024-01-01"])
    b = _px([100.0, 1.0], dates=["2024-01-01", "2024-01-01"])
    v, srcs = _xs({"a": a, "b": b})
    assert v._check_cross_source_consistency(srcs) == []


def test_cross_source_three_sources_are_compared_pairwise():
    v, srcs = _xs({"a": _px([100.0]), "b": _px([100.0]), "c": _px([200.0])})
    issues = v._check_cross_source_consistency(srcs)
    assert sorted((i.source_1, i.source_2) for i in issues) == [("a", "c"), ("b", "c")]


def test_cross_source_symbol_in_one_source_only_is_not_compared():
    v, srcs = _xs({"a": _px([100.0], symbol="AAA"), "b": _px([900.0], symbol="BBB")})
    assert v._check_cross_source_consistency(srcs) == []
    assert v.cross_source_errors == []


def test_cross_source_source_without_close_column_is_skipped_quietly():
    b = _px([1.0]).rename(columns={"close": "adj"})
    v, srcs = _xs({"a": _px([100.0]), "b": b})
    assert v._check_cross_source_consistency(srcs) == []
    assert v.cross_source_errors == []


def test_cross_source_unloaded_sources_are_ignored():
    v, srcs = _xs({"a": _px([100.0]), "b": _px([900.0])})
    assert v._check_cross_source_consistency(srcs[:1] + [_src("ghost")]) == []


def test_cross_source_failure_is_recorded_per_symbol_and_others_still_checked():
    bad = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-01"], "close": ["oops"]})
    good = _px([100.0], symbol="BBB")
    a = pd.concat([bad, good], ignore_index=True)
    b = pd.concat([_px([1.0], symbol="AAA"), _px([300.0], symbol="BBB")], ignore_index=True)
    v, srcs = _xs({"a": a, "b": b})
    issues = v._check_cross_source_consistency(srcs)
    assert [i.symbol for i in issues] == ["BBB"]
    assert [e["symbol"] for e in v.cross_source_errors] == ["AAA"]
    assert v.cross_source_errors[0]["sources"] == ["a", "b"]
    assert v.cross_source_errors[0]["error"]


def test_cross_source_counts_only_symbols_shared_by_two_sources(capsys):
    v, srcs = _xs({"a": _px([100.0], symbol="AAA"), "b": _px([100.0], symbol="BBB")})
    v._check_cross_source_consistency(srcs)
    assert "Checking 0 symbols across sources" in capsys.readouterr().out


def test_single_invalid_ohlc_row_is_flagged(tmp_path):
    m = _validate(
        tmp_path,
        pd.DataFrame({"open": [1, 2], "high": [1, 5], "low": [2, 1], "close": [1, 2]}),
    )
    assert m.validation_errors == ["1 rows with high < low (invalid OHLC)"]


def test_cross_source_prints_counts(capsys):
    v, srcs = _xs({"a": _px([100.0]), "b": _px([200.0])})
    v._check_cross_source_consistency(srcs)
    out = capsys.readouterr().out
    assert "Checking 1 symbols across sources" in out
    assert "Found 1 cross-source inconsistencies" in out
    assert "could not be run" not in out


# ---- leakage scan -----------------------------------------------------------------------------


def test_leakage_scan_all_patterns_and_severity(tmp_path):
    (tmp_path / "m.py").write_text(
        "a = x.rolling(3).mean()[-1]\n"
        "test = d[:len(d)]\n"
        "split(random_state=None)\n"
        "b = s.shift(-12)\n"
        "c = s.shift(+1)\n"
    )
    issues = DataValidator(str(tmp_path))._detect_data_leakage()
    got = sorted((i.issue_type, i.line_number) for i in issues)
    assert got == [
        ("future_data", 1),
        ("future_data", 4),
        ("test_in_train", 2),
        ("unfair_split", 3),
    ]
    assert {i.severity for i in issues} == {"low"}
    assert all(i.file_path == str(tmp_path / "m.py") for i in issues)


def test_leakage_scan_skips_undecodable_files_but_keeps_going(tmp_path, capsys):
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe\x00 s.shift(-1)\n\x80\x81")
    (tmp_path / "good.py").write_text("s.shift(-1)\n")
    issues = DataValidator(str(tmp_path))._detect_data_leakage()
    assert [(Path(i.file_path).name, i.line_number) for i in issues] == [("good.py", 1)]
    assert "Scanned 2 scripts, found 1 potential leakage flags" in capsys.readouterr().out


def test_leakage_scan_recurses_into_subdirectories(tmp_path):
    (tmp_path / "pkg" / "deep").mkdir(parents=True)
    (tmp_path / "pkg" / "deep" / "x.py").write_text("\ny = s.shift(-3)\n")
    (issue,) = DataValidator(str(tmp_path))._detect_data_leakage()
    assert issue.line_number == 2 and issue.file_path.endswith("x.py")


# ---- repo mapping -----------------------------------------------------------------------------


def _src(name):
    return dv.DataSourceMetadata(name, "csv", name, "", "", 0, 0, 0, ("", ""))


def test_repo_mapping_lists_only_scripts_that_mention_a_source(tmp_path):
    (tmp_path / "uses.py").write_text("df = read('alpha'); g = read('beta')\n")
    (tmp_path / "one.py").write_text("alpha\n")
    (tmp_path / "none.py").write_text("print('nothing')\n")
    mapping = DataValidator(str(tmp_path))._map_repos_to_data([_src("alpha"), _src("beta")])
    assert set(mapping) == {str(tmp_path / "uses.py"), str(tmp_path / "one.py")}
    assert sorted(mapping[str(tmp_path / "uses.py")]) == ["alpha", "beta"]
    assert mapping[str(tmp_path / "one.py")] == ["alpha"]


def test_repo_mapping_is_a_sample_of_thirty_scripts(tmp_path):
    for k in range(35):
        (tmp_path / f"s{k:02d}.py").write_text("alpha\n")
    mapping = DataValidator(str(tmp_path))._map_repos_to_data([_src("alpha")])
    assert len(mapping) == 30


def test_repo_mapping_skips_unreadable_scripts(tmp_path):
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe\x00alpha\x80")
    (tmp_path / "good.py").write_text("alpha\n")
    mapping = DataValidator(str(tmp_path))._map_repos_to_data([_src("alpha")])
    assert list(mapping) == [str(tmp_path / "good.py")]


def test_repo_mapping_with_no_sources_is_empty(tmp_path):
    (tmp_path / "a.py").write_text("anything\n")
    assert DataValidator(str(tmp_path))._map_repos_to_data([]) == {}


# ---- summary + report compile -----------------------------------------------------------------


def _issue(field="close"):
    return dv.CrossSourceIssue("A", "2024-01-01", "s1", "s2", field, 1.0, 2.0, 1.0)


def _leak(kind):
    return dv.DataLeakageIssue("f.py", 1, kind)


def test_summary_counts_each_category(tmp_path):
    s_hi = _src("hi")
    s_hi.data_quality_score = 0.81
    s_edge = _src("edge")
    s_edge.data_quality_score = 0.8  # not > 0.8
    s_mid = _src("mid")
    s_mid.data_quality_score = 0.5  # neither > 0.8 nor < 0.5
    s_lo = _src("lo")
    s_lo.data_quality_score = 0.49
    v = DataValidator(str(tmp_path))
    v.cross_source_errors = [{"symbol": "A"}]
    text = v._generate_summary(
        [s_hi, s_edge, s_mid, s_lo],
        [_issue("close"), _issue("close"), _issue("date")],
        [
            _leak("future_data"),
            _leak("test_in_train"),
            _leak("test_in_train"),
            _leak("unfair_split"),
        ],
        {"a.py": ["hi", "lo"], "b.py": ["hi"], "c.py": ["mid"]},
    )
    assert "Sources with quality > 80%: 1/4" in text
    assert "Sources with quality < 50%: 1/4" in text
    assert "Price discrepancies (>2%): 2" in text
    assert "Date range mismatches: 1" in text
    assert "Comparisons that failed to run: 1" in text
    assert "Future-looking operations: 1" in text
    assert "Train-test contamination: 2" in text
    assert "Scripts using multiple sources: 1" in text
    assert "Data sources used by multiple scripts: 3" in text


def test_compile_report_aggregates_sources(tmp_path):
    a, b = _src("a"), _src("b")
    a.record_count, a.symbols_count, a.data_quality_score = 10, 2, 1.0
    b.record_count, b.symbols_count, b.data_quality_score = 5, 3, 0.5
    v = DataValidator(str(tmp_path))
    rep = v._compile_report([a, b], [_issue()], [_leak("future_data")], {"x.py": ["a"]})
    assert (rep.total_sources, rep.total_records, rep.total_symbols) == (2, 15, 5)
    assert rep.avg_quality_score == pytest.approx(0.75)
    assert rep.data_sources == [a, b] and len(rep.cross_source_issues) == 1
    assert rep.repo_data_mapping == {"x.py": ["a"]}
    assert "DATA QUALITY BREAKDOWN" in rep.summary
    datetime.fromisoformat(rep.generated_at)


def test_run_full_validation_stores_report_and_reports_leakage(tmp_path, capsys):
    cs = tmp_path / "cache_seed"
    cs.mkdir()
    pd.DataFrame(
        {"symbol": ["A", "B"], "date": [_day(1), _day(0)], "close": [1.0, 2.0]}
    ).to_parquet(cs / "prices.parquet")
    (tmp_path / "script.py").write_text("x = s.shift(-1)  # reads prices\n")
    v = DataValidator(str(tmp_path))
    report = v.run_full_validation()
    assert v.report is report
    assert report.total_sources == 1 and report.total_symbols == 2
    assert [i.issue_type for i in report.leakage_issues] == ["future_data"]
    assert report.repo_data_mapping == {str(tmp_path / "script.py"): ["prices"]}
    out = capsys.readouterr().out
    for step in ("[1/5]", "[2/5]", "[3/5]", "[4/5]", "[5/5]"):
        assert step in out


# ---- ValidationReport -------------------------------------------------------------------------


def _report(**kw):
    base = dict(
        generated_at="2024-01-01T00:00:00",
        total_sources=3,
        total_records=1234567,
        total_symbols=8900,
        avg_quality_score=0.8567,
        summary="SUMMARY-TEXT",
    )
    base.update(kw)
    return dv.ValidationReport(**base)


def test_print_summary_formats_numbers(capsys):
    r = _report(cross_source_issues=[_issue()], leakage_issues=[_leak("x"), _leak("y")])
    r.print_summary()
    out = capsys.readouterr().out
    assert "=" * 60 in out
    assert "DATA VALIDATION REPORT SUMMARY" in out
    assert "Generated: 2024-01-01T00:00:00" in out
    assert "Total Data Sources: 3" in out
    assert "Total Records: 1,234,567" in out
    assert "Unique Symbols: 8,900" in out
    assert "Average Quality Score: 85.67%" in out
    assert "Cross-Source Inconsistencies: 1" in out
    assert "Data Leakage Issues: 2" in out
    assert "SUMMARY-TEXT" in out


def test_report_to_dict_is_recursive_and_defaults_are_empty():
    r = _report(cross_source_issues=[_issue()])
    d = r.to_dict()
    assert d["cross_source_issues"][0]["field"] == "close"
    assert d["data_sources"] == [] and d["repo_data_mapping"] == {}
    assert d["leakage_issues"] == [] and d["data_freshness_check"] == {}


def test_save_report_writes_indented_json_with_str_fallback(tmp_path, capsys):
    r = _report(data_freshness_check={"when": datetime(2024, 1, 2, 3, 4, 5)})
    out = tmp_path / "r.json"
    r.save_report(str(out))
    text = out.read_text()
    assert json.loads(text)["data_freshness_check"] == {"when": "2024-01-02 03:04:05"}
    assert text.startswith('{\n  "generated_at"')  # indent=2
    assert f"Report saved to {out}" in capsys.readouterr().out


def test_leakage_issue_defaults():
    i = dv.DataLeakageIssue("f.py", 3, "future_data")
    assert i.symbol is None and i.date_range is None and i.severity == "medium"


# ---- Kaggle integrator --------------------------------------------------------------------------


def test_kaggle_discovery_returns_fixed_catalogue_regardless_of_arguments():
    default = dv.KaggleDataIntegrator.discover_relevant_datasets()
    other = dv.KaggleDataIntegrator.discover_relevant_datasets(markets=["X"], categories=["y"])
    assert default == other
    assert [d["records"] for d in default] == [1_500_000, 8_000_000, 50_000_000]
    assert default[0]["coverage"] == "India (NSE/BSE)"
    assert all(set(d) == {"title", "url", "records", "last_updated", "coverage"} for d in default)


# ---- lineage ----------------------------------------------------------------------------------


def test_track_load_records_shape():
    t = dv.DataLineageTracker()
    t.track_load("a.csv", "raw", pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    e = t.lineage["raw"]
    assert (e["source"], e["records"], e["columns"]) == ("a.csv", 3, ["a", "b"])
    datetime.fromisoformat(e["timestamp"])


def test_track_transform_of_untracked_source_registers_empty_parent():
    t = dv.DataLineageTracker()
    t.track_transform("ghost", "child", "op")
    assert t.lineage["ghost"] == {}
    assert t.lineage["child"]["parent"] == "ghost"
    assert t.lineage["child"]["operation"] == "op"
    chain = t.get_lineage("child")
    assert [c.get("operation") for c in chain] == ["op", None]


def test_track_transform_keeps_existing_parent_entry():
    t = dv.DataLineageTracker()
    t.track_load("f", "raw", pd.DataFrame({"a": [1]}))
    t.track_transform("raw", "c", "op")
    assert t.lineage["raw"]["source"] == "f"


def test_get_lineage_unknown_name_is_empty():
    assert dv.DataLineageTracker().get_lineage("nope") == []


# ---- __main__ entry point --------------------------------------------------------------------------


def test_main_block_runs_validation_and_writes_report(tmp_path, monkeypatch, capsys):
    """Run the module as a script with the data tree and output path redirected into tmp_path."""
    parquet = tmp_path / "seed.parquet"
    pd.DataFrame({"symbol": ["A"], "date": [_day(1)], "close": [1.0]}).to_parquet(parquet)
    real_glob = Path.glob

    def fake_glob(self, pattern, *a, **k):
        if pattern == "*.parquet":
            return iter([parquet])
        return iter([])

    target = tmp_path / "report.json"
    real_open = builtins.open

    def fake_open(file, *a, **k):
        if str(file) == "/tmp/data_validation_report.json":
            file = target
        return real_open(file, *a, **k)

    monkeypatch.setattr(Path, "glob", fake_glob)
    monkeypatch.setattr(builtins, "open", fake_open)
    runpy.run_path(dv.__file__, run_name="__main__")
    monkeypatch.setattr(Path, "glob", real_glob)
    out = capsys.readouterr().out
    assert "DATA VALIDATION REPORT SUMMARY" in out
    assert "Discovering Kaggle datasets" in out
    assert "NSE/BSE Historical Stock Data" in out
    assert "Coverage: US (S&P 500), Records: 8,000,000" in out
    saved = json.loads(target.read_text())
    assert saved["total_sources"] == 1
