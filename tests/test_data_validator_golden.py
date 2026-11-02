"""Golden tests: verdicts recorded from the pre-refactor DataValidator on synthetic inputs."""
from __future__ import annotations

import re

import pytest

from data_validator import DataValidator
from leakage_scan import LEAKAGE_PATTERNS, scan_lines
from quality_rules import RULES, quality_score, run_rules
from tests.golden_inputs import golden_frames, write_base

STALE = "Data is <N> days stale (last date: {d})"

# (errors with the day-count masked, quality score) as produced by the original code
GOLDEN = {
    "clean_stale": ([STALE.format(d="2020-01-30")], 0.9),
    "dup": (["2 duplicate rows found", STALE.format(d="2020-01-30")], 0.8),
    "nulls": (["High nulls in ['close']", STALE.format(d="2020-01-30")], 0.8),
    "badohlc": (["1 rows with high < low (invalid OHLC)", STALE.format(d="2020-01-30")], 0.8),
    "badtype": (["Column 'Volume' should be numeric but is str", STALE.format(d="2020-01-30")], 0.8),
    "gaps_capital_date": (["Many date gaps detected (13 gaps > 5 days)",
                           STALE.format(d="2020-04-27")], 0.8),
    "nodate": ([], 1.0),
}


def _mask(msg: str) -> str:
    return re.sub(r"\d+ days stale", "<N> days stale", msg)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_rule_verdicts_match_original(name):
    df = golden_frames()[name]
    errors = run_rules(df.copy())
    want_errors, want_score = GOLDEN[name]
    assert [_mask(e) for e in errors] == want_errors
    assert quality_score(len(errors)) == pytest.approx(want_score)


def test_rule_order_is_stable():
    assert [r.__name__ for r in RULES] == [
        "check_missing", "check_duplicates", "check_numeric_types",
        "check_ohlc_order", "check_date_gaps", "check_freshness",
    ]


def test_quality_score_floors_at_zero():
    assert quality_score(0) == 1.0 and quality_score(25) == 0


def test_leakage_scan_lines():
    lines = ["x = df.shift(-1)", "train = all_data", "r = df.sample().frac", "ok = 1"]
    assert scan_lines(lines) == [("future_data", 1), ("test_in_train", 2), ("unfair_split", 3)]
    assert set(LEAKAGE_PATTERNS) == {"future_data", "test_in_train", "unfair_split"}


def test_full_validation_end_to_end(tmp_path):
    base = write_base(tmp_path)
    report = DataValidator(str(base)).run_full_validation()
    assert report.total_sources == 11 and report.total_records == 171
    by_name = {s.source_name: s for s in report.data_sources if s.source_type == "csv"}
    for name, (want_errors, want_score) in GOLDEN.items():
        s = by_name[name]
        assert [_mask(e) for e in s.validation_errors] == want_errors
        assert s.data_quality_score == pytest.approx(want_score)
    assert report.avg_quality_score == pytest.approx(0.5363636363636364)
    assert sorted((pathname.rsplit("/", 1)[-1], i.issue_type, i.line_number)
                  for i in report.leakage_issues for pathname in [i.file_path]) == [
        ("model.py", "future_data", 1), ("model.py", "test_in_train", 2),
        ("model.py", "unfair_split", 3)]
    assert report.repo_data_mapping == {}
