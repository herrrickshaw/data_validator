"""Tests for the acquisition / orchestration side of fundamental_metrics.py: source routing,
the yfinance mapping, the on-disk cache, the screen runner and the CLI.

Offline: every data source (SEC, Trendlyne, screener.in, yfinance, the serving layer) is a
fake module injected into sys.modules, and the cache directory is redirected to tmp_path.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import fundamental_metrics as fm


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


@pytest.fixture
def fake_modules(monkeypatch):
    def install(name, **attrs):
        mod = _module(name, **attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    return install


# ---- _fetch_one routing -----------------------------------------------------------------


def test_fetch_one_us_uses_sec_and_never_returns_none(fake_modules):
    fake_modules("sec_fundamentals", fundamentals=lambda sym: {"pe": 20.0, "sym": sym})
    assert fm._fetch_one("US", "AAPL") == {"pe": 20.0, "sym": "AAPL"}
    fake_modules("sec_fundamentals", fundamentals=lambda sym: None)
    assert fm._fetch_one("US", "AAPL") == {}


def test_fetch_one_india_takes_first_non_empty_source(monkeypatch):
    calls = []

    def src(name, result):
        def f(sym):
            calls.append(name)
            if isinstance(result, Exception):
                raise result
            return result

        return f

    monkeypatch.setattr(fm, "_yf_fundamentals", lambda m, s: calls.append("yf") or {"source": "yf"})

    monkeypatch.setattr(fm, "_in_trendlyne", src("tl", {"pe": 10}))
    monkeypatch.setattr(fm, "_in_screener", src("scr", {"pe": 99}))
    assert fm._fetch_one("IN", "TCS") == {"pe": 10} and calls == ["tl"]  # Trendlyne wins

    calls.clear()
    monkeypatch.setattr(fm, "_in_trendlyne", src("tl", {}))
    assert fm._fetch_one("IN", "TCS") == {"pe": 99} and calls == ["tl", "scr"]

    calls.clear()
    monkeypatch.setattr(fm, "_in_trendlyne", src("tl", RuntimeError("login failed")))
    assert fm._fetch_one("IN", "TCS") == {"pe": 99}  # an exception just moves to the next source

    calls.clear()
    monkeypatch.setattr(fm, "_in_screener", src("scr", {}))
    assert fm._fetch_one("IN", "TCS") == {"source": "yf"} and calls[-1] == "yf"


def test_fetch_one_other_markets_go_straight_to_yfinance(monkeypatch):
    seen = []
    monkeypatch.setattr(fm, "_yf_fundamentals", lambda m, s: seen.append((m, s)) or {"ok": 1})
    assert fm._fetch_one("JP", "7203") == {"ok": 1}
    assert seen == [("JP", "7203")]


def test_trendlyne_session_is_created_once_and_reused(monkeypatch, fake_modules):
    made = []
    fake_modules(
        "trendlyne_auth",
        session=lambda: made.append(1) or "SESSION",
        fundamentals=lambda sym, sess: {"sym": sym, "sess": sess},
    )
    monkeypatch.setattr(fm, "_TL_SESSION", None)
    assert fm._in_trendlyne("A") == {"sym": "A", "sess": "SESSION"}
    assert fm._in_trendlyne("B") == {"sym": "B", "sess": "SESSION"}
    assert len(made) == 1  # login happens once, not per symbol


def test_in_screener_and_trendlyne_return_empty_dict_for_no_data(monkeypatch, fake_modules):
    fake_modules("screener_in_auth", company_financials=lambda sym: None)
    assert fm._in_screener("X") == {}
    fake_modules("trendlyne_auth", session=lambda: "S", fundamentals=lambda sym, s: None)
    monkeypatch.setattr(fm, "_TL_SESSION", None)
    assert fm._in_trendlyne("X") == {}


# ---- _yf_fundamentals -------------------------------------------------------------------


def _yf(monkeypatch, info=None, boom=None, seen=None):
    class T:
        def __init__(self, symbol):
            if seen is not None:
                seen.append(symbol)
            if boom:
                raise boom
            self.info = info

    monkeypatch.setitem(sys.modules, "yfinance", _module("yfinance", Ticker=T))


def test_yf_fundamentals_maps_and_scales_fields(monkeypatch):
    seen = []
    _yf(
        monkeypatch,
        seen=seen,
        info={
            "trailingPE": 18.5,
            "forwardPE": 15.0,
            "returnOnEquity": 0.22,
            "debtToEquity": 45.0,
            "netIncomeToCommon": 1e9,
            "totalRevenue": 5e9,
            "trailingEps": 3.2,
            "marketCap": 2e10,
            "grossMargins": 0.4,
            "earningsGrowth": 0.1,
        },
    )
    out = fm._yf_fundamentals("JP", "7203")
    assert seen == ["7203.T"]  # market suffix appended
    assert out["source"] == "yfinance"
    assert out["pe"] == 18.5  # trailing preferred over forward
    assert out["roe"] == pytest.approx(22.0)  # fraction -> percent
    assert out["debt_to_equity"] == pytest.approx(0.45)  # yfinance percent style -> ratio
    assert out["gross_margin"] == pytest.approx(40.0)
    assert out["profit_growth"] == pytest.approx(10.0)
    assert out["market_cap"] == 2e10 and out["eps_ttm"] == 3.2


def test_yf_fundamentals_fallbacks_and_missing_values(monkeypatch):
    seen = []
    _yf(
        monkeypatch,
        seen=seen,
        info={
            "forwardPE": 12.0,
            "debtToEquity": 0.8,
            "returnOnEquity": float("nan"),
            "marketCap": "",
        },
    )
    out = fm._yf_fundamentals("XX", "ABC")  # unknown market -> no suffix
    assert seen == ["ABC"]
    assert out["pe"] == 12.0  # forwardPE when no trailingPE
    assert out["debt_to_equity"] == 0.8  # small values are already ratios
    assert "roe" not in out and "market_cap" not in out  # NaN / empty strings are dropped
    assert "gross_margin" not in out and "net_income" not in out


def test_yf_fundamentals_empty_or_failing_lookup_gives_empty_dict(monkeypatch):
    _yf(monkeypatch, info={})
    assert fm._yf_fundamentals("JP", "X") == {}
    _yf(monkeypatch, info=None)
    assert fm._yf_fundamentals("JP", "X") == {}
    _yf(monkeypatch, boom=RuntimeError("network down"))
    assert fm._yf_fundamentals("JP", "X") == {}


# ---- serving-layer helpers --------------------------------------------------------------


def test_liquid_symbols_ranks_by_turnover_and_handles_empty(fake_modules):
    df = pd.DataFrame(
        {"Symbol": ["A", "B", "C"], "Turnover_USD": [5.0, 50.0, 20.0], "Close": [1.0, 2.0, 3.0]}
    )
    fake_modules("serving_layer", serving=lambda market: df)
    assert fm._liquid_symbols("US", 2) == ["B", "C"]
    fake_modules("serving_layer", serving=lambda market: df.drop(columns="Turnover_USD"))
    assert fm._liquid_symbols("US", 2) == ["A", "B"]  # no turnover column: keep order
    fake_modules("serving_layer", serving=lambda market: pd.DataFrame())
    assert fm._liquid_symbols("US", 5) == []


def test_prices_maps_symbol_to_close(fake_modules):
    df = pd.DataFrame({"Symbol": ["A", "B"], "Close": [10.0, 20.0]})
    fake_modules("serving_layer", serving=lambda market: df)
    assert fm._prices("US") == {"A": 10.0, "B": 20.0}
    fake_modules("serving_layer", serving=lambda market: pd.DataFrame())
    assert fm._prices("US") == {}


# ---- load_fundamentals (cache) ----------------------------------------------------------


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, "CACHE", tmp_path)
    monkeypatch.setattr(fm, "_prices", lambda market: {})
    monkeypatch.setattr(fm.time, "sleep", lambda s: None)
    return tmp_path


def test_load_fundamentals_fetches_only_uncached_symbols_and_merges(cache, monkeypatch):
    pd.DataFrame([{"Symbol": "OLD", "pe": 5.0}]).to_parquet(cache / "IN.parquet", index=False)
    fetched = []

    def fetch(market, sym):
        fetched.append(sym)
        return {"pe": 10.0} if sym != "EMPTY" else {}

    monkeypatch.setattr(fm, "_fetch_one", fetch)
    out = fm.load_fundamentals(
        "IN", limit=3, symbols=["OLD", "NEW1", "EMPTY", "NEW2", "NEW3"], verbose=False
    )
    assert fetched == ["NEW1", "EMPTY", "NEW2"]  # OLD is cached; limit=3 stops the rest
    assert sorted(out["Symbol"]) == ["NEW1", "NEW2", "OLD"]  # EMPTY returned nothing
    assert (out.set_index("Symbol").loc["OLD", "pe"]) == 5.0  # cached row untouched
    assert pd.read_parquet(cache / "IN.parquet")["Symbol"].nunique() == 3  # persisted


def test_load_fundamentals_checkpoints_every_25_and_dedupes(cache, monkeypatch):
    monkeypatch.setattr(fm, "_fetch_one", lambda m, s: {"pe": float(len(s))})
    syms = [f"S{i:03d}" for i in range(30)]
    out = fm.load_fundamentals("IN", limit=30, symbols=syms, verbose=False)
    assert len(out) == 30 and out["Symbol"].is_unique
    again = fm.load_fundamentals("IN", limit=30, symbols=syms, verbose=False)
    assert len(again) == 30  # nothing new to fetch; the cache is the answer


def test_load_fundamentals_derives_market_cap_from_price_and_shares(cache, monkeypatch):
    monkeypatch.setattr(fm, "_fetch_one", lambda m, s: {"shares": 100.0})
    monkeypatch.setattr(fm, "_prices", lambda market: {"AAA": 2.5})
    out = fm.load_fundamentals("IN", symbols=["AAA", "BBB"], verbose=False)
    row = out.set_index("Symbol")
    assert row.loc["AAA", "price"] == 2.5 and row.loc["AAA", "market_cap"] == 250.0
    assert np.isnan(row.loc["BBB", "market_cap"])  # no price -> no market cap


def test_load_fundamentals_uses_liquid_universe_when_no_symbols(cache, monkeypatch):
    seen = {}
    monkeypatch.setattr(fm, "_liquid_symbols", lambda m, n: seen.setdefault("n", n) and ["X", "Y"])
    monkeypatch.setattr(fm, "_fetch_one", lambda m, s: {"pe": 1.0})
    out = fm.load_fundamentals("IN", limit=7, verbose=False)
    assert seen["n"] == 21  # asks for limit * 3 candidates
    assert sorted(out["Symbol"]) == ["X", "Y"]


def test_load_fundamentals_us_is_rate_limited_and_verbose(cache, monkeypatch, capsys):
    sleeps = []
    monkeypatch.setattr(fm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(fm, "_fetch_one", lambda m, s: {"pe": 1.0})
    fm.load_fundamentals("US", symbols=["A", "B"], limit=2)
    assert sleeps == [0.12, 0.12]  # SEC politeness: pause after each symbol
    assert "fundamentals[US]: +2 fetched" in capsys.readouterr().out


def test_load_fundamentals_with_nothing_to_do_returns_empty_frame(cache, monkeypatch):
    monkeypatch.setattr(fm, "_fetch_one", lambda m, s: {})
    assert fm.load_fundamentals("IN", symbols=["A"], verbose=False).empty


# ---- formulas ---------------------------------------------------------------------------


def test_roce_uses_ebit_over_assets_else_falls_back_to_roa():
    assert fm._roce({"ebit": 30.0, "total_assets": 200.0}) == pytest.approx(15.0)
    assert fm._roce({"ebit": 30.0, "assets": 100.0}) == pytest.approx(30.0)  # alias key
    assert fm._roce({"roa": 7.0}) == 7.0  # assets absent: ROA as a lower-bound proxy
    assert fm._roce({"ebit": 30.0, "roa": 7.0}) == 7.0  # no capital figure
    assert fm._roce({}) is None


def test_enrich_adds_all_derived_columns_without_mutating_input():
    df = pd.DataFrame(
        [
            {
                "Symbol": "A",
                "ebit": 20.0,
                "market_cap": 100.0,
                "total_assets": 80.0,
                "cfo": 15.0,
                "capex_history": [5.0],
                "dividend_history": [-3.0],
                "net_income": 10.0,
                "roa": 12.0,
            }
        ]
    )
    cols_before = list(df.columns)
    out = fm._enrich(df)
    assert list(df.columns) == cols_before  # input untouched
    r = out.iloc[0]
    assert r["roce"] == pytest.approx(25.0) and r["earnings_yield"] == pytest.approx(20.0)
    assert r["fcf_yield"] == pytest.approx(10.0) and r["div_yield"] == pytest.approx(3.0)
    assert r["piotroski"] == 4  # net income > 0, ROA > 0, CFO > 0, and CFO > net income (accruals)


def test_fcf_and_dividend_yield_guard_against_missing_or_zero_market_cap():
    base = {"cfo": 15.0, "capex_history": [5.0], "dividend_history": [-3.0]}
    for mc in (None, 0, float("nan")):
        r = {**base, "market_cap": mc}
        assert fm._fcf_yield(r) is None and fm._dividend_yield(r) is None
    assert fm._fcf_yield({"cfo": None, "market_cap": 100.0}) is None
    assert fm._fcf_yield({"cfo": 15.0, "market_cap": 100.0}) == pytest.approx(15.0)  # no capex data
    assert fm._dividend_yield({"market_cap": 100.0, "dividend_history": []}) is None


# ---- screens: run_all / run_screen / main -----------------------------------------------


def _frame():
    return pd.DataFrame(
        [
            {
                "Symbol": "AAA",
                "roe": 25.0,
                "debt_to_equity": 0.05,
                "ebit": 40.0,
                "market_cap": 100.0,
                "total_assets": 100.0,
                "cfo": 30.0,
                "dividend_history": [-4.0],
                "net_income": 20.0,
                "roa": 20.0,
            },
            {
                "Symbol": "BBB",
                "roe": 10.0,
                "debt_to_equity": 0.9,
                "ebit": 5.0,
                "market_cap": 100.0,
                "total_assets": 100.0,
                "cfo": 1.0,
                "dividend_history": [-1.0],
                "net_income": 2.0,
                "roa": 2.0,
            },
            {
                "Symbol": "CCC",
                "roe": 30.0,
                "debt_to_equity": 0.2,
                "ebit": 20.0,
                "market_cap": 100.0,
                "total_assets": 100.0,
                "cfo": 8.0,
                "dividend_history": [-5.0],
                "net_income": 9.0,
                "roa": 9.0,
            },
        ]
    )


def test_run_all_counts_every_screen_and_prints_examples(monkeypatch, capsys):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit=40, verbose=True: _frame())
    counts = fm.run_all("IN")
    assert set(counts) == set(fm.FUND_SCREENS)
    assert counts["high_roe"] == 2  # AAA (25) and CCC (30) beat 20
    assert counts["debt_free"] == 1  # only AAA has D/E < 0.1
    assert counts["high_roce"] == 1  # AAA: 40 / 100 -> 40%
    assert counts["high_dividend"] == 2  # AAA 4%, CCC 5% (BBB 1%)
    out = capsys.readouterr().out
    assert "fundamental screens on IN (3 names" in out and "AAA" in out


def test_run_all_with_no_data_returns_empty_dict(monkeypatch, capsys):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit=40, verbose=True: pd.DataFrame())
    assert fm.run_all("IN") == {}
    assert "no fundamentals for IN" in capsys.readouterr().out


def test_run_all_treats_a_failing_screen_as_zero_matches(monkeypatch):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit=40, verbose=True: _frame())
    monkeypatch.setitem(fm.FUND_SCREENS, "broken", lambda d: d["no_such_column"] > 1)
    assert fm.run_all("IN", verbose=False)["broken"] == 0


def test_run_screen_filters_sorts_and_truncates(monkeypatch):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit=40: _frame())
    out = fm.run_screen("high_roe", "IN")
    assert set(out["Symbol"]) == {"AAA", "CCC"}
    assert set(out.columns) <= {
        "Symbol",
        "piotroski",
        "roce",
        "roe",
        "earnings_yield",
        "fcf_yield",
        "div_yield",
        "debt_to_equity",
        "eps_growth",
    }
    assert out["piotroski"].is_monotonic_decreasing  # sorted best-first
    assert len(fm.run_screen("high_roe", "IN", top=1)) == 1
    assert len(fm.run_screen("high_roe", "IN", top=None)) == 2


def test_run_screen_sorts_roce_screens_by_roce(monkeypatch):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit=40: _frame())
    out = fm.run_screen("magic_formula", "IN")
    assert list(out["Symbol"]) == ["AAA", "CCC"]  # both clear EY > 8 and ROCE > 15 ...
    assert list(out["roce"]) == pytest.approx([40.0, 20.0])  # ... listed by ROCE, best first
    # Where ROCE order and Piotroski order disagree, the ROCE screens must follow ROCE:
    # HIGHQ has the higher ROCE (40%) but the lower F-score (3); STEADY has ROCE 30% and F-score 4.
    disagree = pd.DataFrame(
        [
            {
                "Symbol": "STEADY",
                "ebit": 30.0,
                "total_assets": 100.0,
                "market_cap": 100.0,
                "net_income": 5.0,
                "roa": 5.0,
                "cfo": 9.0,
            },
            {
                "Symbol": "HIGHQ",
                "ebit": 40.0,
                "total_assets": 100.0,
                "market_cap": 100.0,
                "net_income": 5.0,
                "roa": 5.0,
                "cfo": 1.0,
            },
        ]
    )
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit=40: disagree)
    out = fm.run_screen("high_roce", "IN")
    assert list(out["Symbol"]) == ["HIGHQ", "STEADY"]
    assert list(out["piotroski"]) == [3, 4]  # i.e. NOT ordered by F-score


def test_run_screen_unknown_name_or_no_data_is_empty(monkeypatch):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit=40: _frame())
    assert fm.run_screen("no_such_screen", "IN").empty
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit=40: pd.DataFrame())
    assert fm.run_screen("high_roe", "IN").empty


def test_main_runs_a_named_screen_or_all_screens(monkeypatch, capsys):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit=40, verbose=True: _frame())
    monkeypatch.setattr(
        sys, "argv", ["fundamental_metrics.py", "--market", "IN", "--screen", "high_roe"]
    )
    assert fm.main() == 0
    out = capsys.readouterr().out
    assert "high_roe on IN" in out and "names:" in out

    monkeypatch.setattr(
        sys, "argv", ["fundamental_metrics.py", "--market", "IN", "--screen", "nonesuch"]
    )
    assert fm.main() == 0
    assert "none / no fundamentals" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["fundamental_metrics.py", "--market", "IN"])
    assert fm.main() == 0
    assert "fundamental screens on IN" in capsys.readouterr().out


# ---- regression: a screen whose column is missing must match nothing, not crash -----------


def _without_eps_growth():
    # yfinance-sourced markets (JP, KR, UK, ...) provide no eps_growth field.
    return _frame().drop(columns=[c for c in ("eps_growth",) if c in _frame().columns])


def test_run_all_survives_a_screen_whose_column_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(
        fm, "load_fundamentals", lambda m, limit=40, verbose=True: _without_eps_growth()
    )
    counts = fm.run_all("JP")  # verbose: used to raise while printing example names
    assert counts["eps_growth"] == 0 and counts["high_roe"] == 2
    assert "eps_growth" in capsys.readouterr().out


def test_run_screen_with_a_missing_column_is_empty_not_an_error(monkeypatch):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit=40: _without_eps_growth())
    assert fm.run_screen("eps_growth", "JP").empty
    assert len(fm.run_screen("high_roe", "JP")) == 2  # other screens unaffected


def test_screen_mask_is_aligned_boolean_for_present_and_missing_columns():
    df = _frame()
    mask = fm._screen_mask("high_roe", df)
    assert (
        mask.dtype == bool
        and list(mask.index) == list(df.index)
        and mask.tolist() == [True, False, True]
    )
    missing = fm._screen_mask("eps_growth", df)
    assert missing.dtype == bool and not missing.any() and len(missing) == len(df)
