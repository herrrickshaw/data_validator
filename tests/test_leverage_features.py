"""Tests for leverage_features.py: the debt-trend helper, the per-symbol frame built from cached
fundamentals, the left-merge onto a feature frame and the CLI.

Offline: the fundamentals cache directory is redirected into tmp_path.
"""

import sys

import numpy as np
import pandas as pd
import pytest

import leverage_features as lf


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(lf, "CACHE", tmp_path)
    return tmp_path


def _write(cache, market="US", **cols):
    pd.DataFrame(cols).to_parquet(cache / f"{market}.parquet", index=False)


# ---- _debt_trend ------------------------------------------------------------------------


def test_debt_trend_is_last_over_first_minus_one():
    assert lf._debt_trend([10, 15]) == pytest.approx(0.5)
    assert lf._debt_trend([20, 18, 10]) == pytest.approx(-0.5)  # only the ends matter
    assert lf._debt_trend((4.0, 4.0)) == 0.0
    assert lf._debt_trend(np.array([100.0, 150.0, 250.0])) == pytest.approx(1.5)  # parquet arrays


def test_debt_trend_only_a_zero_start_is_undefined():
    assert lf._debt_trend([1, 2]) == pytest.approx(1.0)  # a start of 1 is a perfectly good base
    assert lf._debt_trend([0.5, 0.25]) == pytest.approx(-0.5)


def test_debt_trend_scales_by_the_absolute_starting_level():
    assert lf._debt_trend([-10, -5]) == pytest.approx(0.5)  # |first| in the denominator


def test_debt_trend_ignores_non_finite_points():
    assert lf._debt_trend([10, np.nan, np.inf, 20]) == pytest.approx(1.0)
    assert lf._debt_trend([10, 20, np.inf]) == pytest.approx(1.0)  # a trailing inf is dropped
    assert np.isnan(lf._debt_trend([np.nan, 5, np.nan]))  # only one usable point left


@pytest.mark.parametrize(
    "hist",
    [None, "abc", 5, [], [7], [0, 5], ["a", "b"], [None, 1], {"a": 1}],
)
def test_debt_trend_is_nan_when_it_cannot_be_computed(hist):
    assert np.isnan(lf._debt_trend(hist))


# ---- leverage_frame ---------------------------------------------------------------------


def _empty():
    return pd.DataFrame(columns=["Symbol", *lf.LEV_FEATURES])


def test_leverage_frame_is_empty_without_a_cache_file(cache):
    out = lf.leverage_frame("US")
    assert out.empty and list(out.columns) == ["Symbol", *lf.LEV_FEATURES]


def test_leverage_frame_is_empty_for_a_corrupt_empty_or_symbol_less_file(cache):
    (cache / "US.parquet").write_bytes(b"not parquet")
    assert lf.leverage_frame("US").empty
    pd.DataFrame({"Symbol": []}).to_parquet(cache / "JP.parquet")
    assert lf.leverage_frame("JP").empty
    pd.DataFrame({"debt_to_equity": [1.0]}).to_parquet(cache / "UK.parquet")
    out = lf.leverage_frame("UK")
    assert out.empty and list(out.columns) == ["Symbol", *lf.LEV_FEATURES]


def test_leverage_frame_maps_each_fundamental_to_its_feature(cache):
    _write(
        cache,
        Symbol=["A", "B"],
        debt_to_equity=[0.5, 2.0],
        debt_to_assets=[0.30, 0.10],
        debt_to_assets_prev=[0.25, 0.20],
        debt_history=[[10.0, 20.0], [8.0, 4.0]],
    )
    out = lf.leverage_frame("US").set_index("Symbol")
    assert list(lf.leverage_frame("US").columns) == ["Symbol", *lf.LEV_FEATURES]
    assert out.loc["A", "lev_de"] == 0.5 and out.loc["B", "lev_de"] == 2.0
    assert out.loc["A", "lev_da"] == pytest.approx(0.30)
    assert out.loc["A", "lev_da_delta"] == pytest.approx(0.05)  # rising leverage is positive
    assert out.loc["B", "lev_da_delta"] == pytest.approx(-0.10)  # falling leverage is negative
    assert out.loc["A", "lev_debt_trend"] == pytest.approx(1.0)
    assert out.loc["B", "lev_debt_trend"] == pytest.approx(-0.5)


def test_leverage_frame_coerces_junk_to_nan_and_handles_a_missing_history(cache):
    _write(
        cache,
        Symbol=["A", "B"],
        debt_to_equity=["n/a", "1.5"],
        debt_to_assets=[0.2, None],
        debt_to_assets_prev=[0.1, 0.1],
    )
    out = lf.leverage_frame("US").set_index("Symbol")
    assert (
        np.isnan(out.loc["A", "lev_de"]) and out.loc["B", "lev_de"] == 1.5
    )  # numeric strings parse
    assert np.isnan(out.loc["B", "lev_da"]) and np.isnan(out.loc["B", "lev_da_delta"])
    assert out["lev_debt_trend"].isna().all()  # no debt_history column at all


def test_leverage_frame_keeps_the_last_row_per_symbol(cache):
    _write(
        cache,
        Symbol=["A", "B", "A"],
        debt_to_equity=[1.0, 2.0, 3.0],
        debt_to_assets=[0.1, 0.2, 0.3],
        debt_to_assets_prev=[0.1, 0.1, 0.1],
    )
    out = lf.leverage_frame("US")
    assert list(out["Symbol"]) == ["B", "A"]  # duplicate dropped, index reset
    assert list(out.index) == [0, 1]
    assert out.set_index("Symbol").loc["A", "lev_de"] == 3.0


def test_leverage_frame_tolerates_a_cache_without_a_debt_column(cache):
    _write(cache, Symbol=["A"], debt_to_assets=[0.2], debt_to_assets_prev=[0.1])
    out = lf.leverage_frame("US")
    assert np.isnan(out.loc[0, "lev_de"])  # the missing column becomes NaN, not an error
    assert out.loc[0, "lev_da_delta"] == pytest.approx(0.1)


# ---- merge_leverage ---------------------------------------------------------------------


def test_merge_leverage_left_joins_on_symbol(cache):
    _write(
        cache,
        Symbol=["A"],
        debt_to_equity=[0.5],
        debt_to_assets=[0.3],
        debt_to_assets_prev=[0.2],
    )
    frame = pd.DataFrame({"Symbol": ["B", "A"], "x": [1, 2]})
    out = lf.merge_leverage(frame, "US")
    assert list(out["Symbol"]) == ["B", "A"] and list(out["x"]) == [1, 2]  # order and rows kept
    assert np.isnan(out.loc[0, "lev_de"]) and out.loc[1, "lev_de"] == 0.5


def test_merge_leverage_adds_nan_columns_when_nothing_is_cached(cache):
    frame = pd.DataFrame({"Symbol": ["A"], "lev_de": [9.0]})
    out = lf.merge_leverage(frame, "US")
    assert out is frame  # filled in place
    assert out.loc[0, "lev_de"] == 9.0  # an existing column is not overwritten
    assert out[["lev_da", "lev_da_delta", "lev_debt_trend"]].isna().all().all()


def test_merge_leverage_without_a_symbol_column_only_adds_the_features(cache):
    _write(
        cache, Symbol=["A"], debt_to_equity=[0.5], debt_to_assets=[0.3], debt_to_assets_prev=[0.2]
    )
    out = lf.merge_leverage(pd.DataFrame({"x": [1, 2]}), "US")
    assert list(out.columns) == ["x", *lf.LEV_FEATURES] and out[lf.LEV_FEATURES].isna().all().all()


# ---- main -------------------------------------------------------------------------------


def test_main_reports_the_symbol_count_and_a_preview(cache, monkeypatch, capsys):
    _write(
        cache,
        "IN",
        Symbol=["A", "B"],
        debt_to_equity=[0.5, 1.0],
        debt_to_assets=[0.3, 0.2],
        debt_to_assets_prev=[0.2, 0.2],
    )
    monkeypatch.setattr(sys, "argv", ["leverage_features.py", "--market", "IN"])
    assert lf.main() == 0
    out = capsys.readouterr().out
    assert "IN: 2 symbols with leverage features" in out and "lev_da_delta" in out


def test_main_previews_at_most_twenty_rows(cache, monkeypatch, capsys):
    n = 25
    _write(
        cache,
        "IN",
        Symbol=[f"S{i:02d}" for i in range(n)],
        debt_to_equity=[1.0] * n,
        debt_to_assets=[0.2] * n,
        debt_to_assets_prev=[0.2] * n,
    )
    monkeypatch.setattr(sys, "argv", ["leverage_features.py"])
    lf.main()
    out = capsys.readouterr().out
    assert "IN: 25 symbols" in out and "S19" in out and "S20" not in out


def test_main_prints_only_the_count_when_empty(cache, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["leverage_features.py"])
    assert lf.main() == 0
    out = capsys.readouterr().out
    assert out.strip() == "IN: 0 symbols with leverage features"
