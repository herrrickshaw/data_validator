"""Tests for screen_metrics.py beyond the basics: every computed metric against an independent
NumPy calculation, each price-screen predicate at its boundaries, the metrics frame / liquidity
plumbing, run_screen / run_all and the CLI.

Offline and sandboxed: the data layer is a fake ``datalink.load_market`` and the liquidity
annotator is a fake module (the real one reads the repo's own turnover index).
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import screen_metrics as sm


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _bars(close, volume=1000.0):
    close = np.asarray(close, dtype=float)
    vol = (
        np.full(len(close), float(volume))
        if np.isscalar(volume)
        else np.asarray(volume, dtype=float)
    )
    return pd.DataFrame(
        {"Close": close, "Volume": vol}, index=pd.bdate_range("2023-01-02", periods=len(close))
    )


def _wiggly(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0.2, 1.5, n))


# ---- _metrics: guards -------------------------------------------------------------------


def test_metrics_need_60_bars_and_a_positive_last_price():
    assert sm._metrics(None) is None
    assert sm._metrics(_bars(np.linspace(100, 110, 59))) is None
    assert sm._metrics(_bars(np.linspace(100, 110, 60))) is not None
    bad = np.linspace(100, 110, 100)
    bad[-1] = 0.0
    assert sm._metrics(_bars(bad)) is None
    bad[-1] = -3.0
    assert sm._metrics(_bars(bad)) is None


# ---- _metrics: values against independent calculations ----------------------------------


def test_metrics_match_independent_calculations():
    c = _wiggly(300)
    rng = np.random.default_rng(1)
    v = rng.integers(500, 5000, 300).astype(float)
    m = sm._metrics(_bars(c, v))
    assert m["Close"] == c[-1] and m["Vol"] == v[-1]
    assert m["DMA50"] == pytest.approx(c[-50:].mean())
    assert m["DMA200"] == pytest.approx(c[-200:].mean())
    assert m["DMA50_prev"] == pytest.approx(c[-60:-10].mean())  # ten sessions earlier
    assert m["DMA200_prev"] == pytest.approx(c[-210:-10].mean())
    assert m["High52"] == c[-252:].max() and m["Low52"] == c[-252:].min()
    assert m["PctFromHigh"] == pytest.approx((c[-1] / c[-252:].max() - 1) * 100)
    assert m["PctFromLow"] == pytest.approx((c[-1] / c[-252:].min() - 1) * 100)
    assert m["Turnover"] == pytest.approx((c[-20:] * v[-20:]).mean())
    prior = [
        v[-5 * (i + 2) : -5 * (i + 1)].sum() for i in range(8)
    ]  # the eight weeks before this one
    assert m["WeekVol"] == v[-5:].sum() and m["WeekVolAvg"] == pytest.approx(np.mean(prior))
    assert m["VolSpike"] == pytest.approx(v[-5:].sum() / np.mean(prior))


def test_metrics_52_week_window_is_252_bars():
    c = np.full(300, 100.0)
    c[300 - 252] = 500.0  # the oldest bar inside the window
    assert sm._metrics(_bars(c))["High52"] == 500.0
    c[300 - 252] = 100.0
    c[300 - 253] = 500.0  # one bar too old
    assert sm._metrics(_bars(c))["High52"] == 100.0


def test_metrics_long_windows_are_nan_until_there_is_enough_history():
    m = sm._metrics(_bars(np.linspace(100, 130, 100)))
    assert np.isnan(m["DMA200"]) and np.isnan(m["DMA200_prev"])
    assert not np.isnan(m["DMA50"]) and not np.isnan(
        m["DMA50_prev"]
    )  # 90 bars cover a 50-day average
    assert not np.isnan(m["Ret63"]) and np.isnan(m["Ret126"]) and np.isnan(m["Ret252"])
    mid = sm._metrics(_bars(np.linspace(100, 130, 205)))
    assert not np.isnan(mid["DMA200"]) and np.isnan(
        mid["DMA200_prev"]
    )  # 195 < 200 bars ten sessions ago


def test_metrics_rsi_extremes_and_balance():
    up = _bars(np.linspace(100, 200, 100))
    assert sm._metrics(up)["RSI"] == 100.0  # no losses
    assert (
        sm._metrics(_bars(np.full(100, 50.0)))["RSI"] == 100.0
    )  # no change at all: no losses either
    assert sm._metrics(_bars(np.linspace(200, 100, 100)))["RSI"] == pytest.approx(0.0)
    steps = np.array([2.0, -2.0] * 7)  # 14 changes: equal gains and losses
    c = np.concatenate([np.full(86, 100.0), 100.0 + np.cumsum(steps)])
    assert sm._metrics(_bars(c))["RSI"] == pytest.approx(50.0)


def test_metrics_return_over_k_bars_needs_more_than_k_bars_of_history():
    # exactly 252 bars: c[-252] exists, but the guard is n > k, so Ret252 stays NaN (Ret126 is fine)
    m = sm._metrics(_bars(np.linspace(100, 130, 252)))
    assert np.isnan(m["Ret252"]) and not np.isnan(m["Ret126"])
    assert not np.isnan(sm._metrics(_bars(np.linspace(100, 130, 253)))["Ret252"])


def test_metrics_rsi_averages_exactly_the_last_14_changes():
    # a +5 jump just BEFORE the 14-change window must not leak in; inside it: 10 gains of 1 and 4 losses of 1
    steps = np.array([1.0] * 5 + [-1.0] * 2 + [1.0] * 5 + [-1.0] * 2)
    c = np.concatenate([np.full(84, 100.0), [105.0], 105.0 + np.cumsum(steps)])
    assert len(c) == 99 and len(steps) == 14
    rs = (10 / 14) / (4 / 14)  # average gain over average loss across the 14 changes
    assert sm._metrics(_bars(c))["RSI"] == pytest.approx(100 - 100 / (1 + rs))  # 71.43


def test_metrics_return_signs():
    assert sm._metrics(_bars(np.linspace(100, 300, 300)))["Ret252"] > 0
    assert sm._metrics(_bars(np.linspace(300, 100, 300)))["Ret252"] < 0


def test_metrics_volume_spike_is_this_week_over_the_prior_weekly_average():
    assert sm._metrics(_bars(np.linspace(100, 110, 100), 1000.0))["VolSpike"] == pytest.approx(1.0)
    v = np.full(100, 1000.0)
    v[-5:] = 5000.0
    m = sm._metrics(_bars(np.linspace(100, 110, 100), v))
    assert (
        m["WeekVol"] == 25000.0
        and m["WeekVolAvg"] == 5000.0
        and m["VolSpike"] == pytest.approx(5.0)
    )
    quiet = np.zeros(100)
    quiet[-5:] = 1000.0  # no volume in the prior eight weeks: the ratio is undefined, not infinite
    assert np.isnan(sm._metrics(_bars(np.linspace(100, 110, 100), quiet))["VolSpike"])


# ---- PRICE_SCREENS predicates at their boundaries ---------------------------------------


def _mf(**cols):
    base = {
        "Close": 100.0,
        "DMA50": 100.0,
        "DMA200": 100.0,
        "DMA50_prev": 100.0,
        "DMA200_prev": 100.0,
        "RSI": 50.0,
        "High52": 200.0,
        "Low52": 50.0,
        "VolSpike": 1.0,
        "Ret252": 0.0,
    }
    base.update(cols)
    return pd.DataFrame({k: (v if isinstance(v, list) else [v]) for k, v in base.items()})


def _hit(name, **cols):
    return bool(sm.PRICE_SCREENS[name](_mf(**cols)).fillna(False).iloc[0])


@pytest.mark.parametrize(
    "d50,d200,p50,p200,expected",
    [
        (101, 100, 100, 100, True),  # now above, was level: crossed up (the previous check is <=)
        (101, 100, 99, 100, True),
        (101, 100, 101, 100, False),  # already above ten sessions ago
        (100, 100, 99, 100, False),  # not above now
        (np.nan, 100, 99, 100, False),  # undefined average: never a hit
    ],
)
def test_golden_crossover(d50, d200, p50, p200, expected):
    assert (
        _hit("golden_crossover", DMA50=d50, DMA200=d200, DMA50_prev=p50, DMA200_prev=p200)
        is expected
    )


@pytest.mark.parametrize(
    "d50,d200,p50,p200,expected",
    [
        (99, 100, 100, 100, True),
        (99, 100, 101, 100, True),
        (99, 100, 99, 100, False),
        (100, 100, 101, 100, False),
    ],
)
def test_bearish_crossover(d50, d200, p50, p200, expected):
    assert (
        _hit("bearish_crossover", DMA50=d50, DMA200=d200, DMA50_prev=p50, DMA200_prev=p200)
        is expected
    )


def test_golden_state_is_strictly_above():
    assert _hit("golden_state", DMA50=100.01, DMA200=100.0) and not _hit(
        "golden_state", DMA50=100.0, DMA200=100.0
    )


def test_rsi_screens_are_strict():
    assert _hit("rsi_oversold", RSI=29.99) and not _hit("rsi_oversold", RSI=30.0)
    assert _hit("rsi_overbought", RSI=70.01) and not _hit("rsi_overbought", RSI=70.0)


def test_new_high_screens_use_their_own_tolerances():
    # within 2% of the 52-week high: 98 of 100 qualifies, 97.99 does not
    assert _hit("companies_creating_new_high", Close=98.0, High52=100.0)
    assert not _hit("companies_creating_new_high", Close=97.99, High52=100.0)
    # at the high: within 0.1%
    assert _hit("at_52w_high", Close=99.95, High52=100.0) and not _hit(
        "at_52w_high", Close=99.8, High52=100.0
    )
    # darvas: within 3% of the high AND the 50-day above the 200-day
    assert _hit("darvas_scan", Close=97.0, High52=100.0, DMA50=11, DMA200=10)
    assert not _hit("darvas_scan", Close=96.99, High52=100.0, DMA50=11, DMA200=10)
    assert not _hit("darvas_scan", Close=100.0, High52=100.0, DMA50=10, DMA200=11)


def test_near_low_and_near_200dma():
    assert _hit("near_52w_low", Close=105.0, Low52=100.0) and not _hit(
        "near_52w_low", Close=105.01, Low52=100.0
    )
    assert _hit("near_200dma", Close=102.9, DMA200=100.0) and _hit(
        "near_200dma", Close=97.5, DMA200=100.0
    )
    assert not _hit("near_200dma", Close=103.5, DMA200=100.0)  # more than 3% away, on either side


def test_volume_screens():
    assert _hit("price_volume_action", VolSpike=5.0) and not _hit(
        "price_volume_action", VolSpike=4.99
    )
    assert _hit(
        "breakout_stocks", Close=98.0, High52=100.0, VolSpike=2.0
    )  # new high + volume expansion
    assert not _hit("breakout_stocks", Close=98.0, High52=100.0, VolSpike=1.99)
    assert not _hit("breakout_stocks", Close=90.0, High52=100.0, VolSpike=9.0)


def test_multibagger_needs_more_than_100_percent_and_an_uptrend():
    assert _hit("multibagger_momentum", Ret252=100.01, DMA50=11, DMA200=10)
    assert not _hit("multibagger_momentum", Ret252=100.0, DMA50=11, DMA200=10)  # strict
    assert not _hit("multibagger_momentum", Ret252=300.0, DMA50=10, DMA200=11)


def test_price_and_fundamental_screen_names_do_not_overlap():
    assert not set(sm.PRICE_SCREENS) & set(sm.FUNDAMENTAL_ONLY)


# ---- _metrics_frame ---------------------------------------------------------------------


@pytest.fixture
def market(monkeypatch):
    data = {
        "UP": _bars(np.linspace(100, 300, 300)),  # strong riser: golden state, Ret252 ~ +127%
        "SLOW": _bars(np.linspace(100, 140, 300)),  # mild riser, Ret252 ~ +32%
        "DOWN": _bars(np.linspace(300, 100, 300)),  # faller
        "SHORT": _bars(np.linspace(100, 110, 59)),  # too little history
        "NONE": None,
    }
    calls = []

    def load(market, tier="stm", **kw):
        calls.append((market, tier))
        return data if tier == "ltm" else {}

    monkeypatch.setattr(sm.datalink, "load_market", load)
    monkeypatch.setitem(sys.modules, "liquidity", None)  # default: no liquidity annotator available
    return types.SimpleNamespace(data=data, calls=calls)


def _annotator(monkeypatch, turnover_usd):
    def annotate(df):
        return df.assign(Turnover_USD=df["Symbol"].map(turnover_usd), Liquidity="High")

    monkeypatch.setitem(sys.modules, "liquidity", _module("liquidity", annotate=annotate))


def test_metrics_frame_has_one_row_per_usable_symbol(market):
    mf = sm._metrics_frame("US")
    assert sorted(mf["Symbol"]) == ["DOWN", "SLOW", "UP"]  # SHORT and NONE are skipped
    assert {"Close", "DMA50", "RSI", "VolSpike", "Ret252", "Turnover"} <= set(mf.columns)


def test_metrics_frame_falls_back_to_the_stm_layer_and_handles_no_data(monkeypatch, market):
    market.data.clear()
    assert sm._metrics_frame("US").empty
    assert market.calls == [("US", "ltm"), ("US", "stm")]


def test_metrics_frame_annotates_liquidity_when_available(monkeypatch, market):
    seen = {}

    def annotate(df):
        seen["market_col"] = list(df["Market"])
        return df.assign(Turnover_USD=1.0, Liquidity="High")

    monkeypatch.setitem(sys.modules, "liquidity", _module("liquidity", annotate=annotate))
    mf = sm._metrics_frame("JP")
    assert (
        set(seen["market_col"]) == {"JP"} and "Liquidity" in mf.columns
    )  # the annotator sees the market


def test_metrics_frame_without_an_annotator_has_no_liquidity_columns(market):
    mf = sm._metrics_frame("US")
    assert "Turnover_USD" not in mf.columns and "Market" not in mf.columns


def test_metrics_frame_survives_a_failing_annotator(monkeypatch, market):
    def boom(df):
        raise RuntimeError("no index")

    monkeypatch.setitem(sys.modules, "liquidity", _module("liquidity", annotate=boom))
    assert len(sm._metrics_frame("US")) == 3


# ---- run_screen -------------------------------------------------------------------------


def test_run_screen_rejects_fundamental_and_unknown_names(market):
    for name in ("piotroski_9", "no_such_screen"):
        with pytest.raises(ValueError, match="is not price-computable"):
            sm.run_screen(name, "US")


def test_run_screen_returns_matches_best_return_first(market):
    out = sm.run_screen("golden_state", "US")
    assert out["Symbol"].tolist() == [
        "UP",
        "SLOW",
    ]  # DOWN is not in a golden state; UP has the higher Ret252
    assert out.index.tolist() == [0, 1]
    assert sm.run_screen("golden_state", "US", top=1)["Symbol"].tolist() == ["UP"]
    assert sm.run_screen("multibagger_momentum", "US")["Symbol"].tolist() == [
        "UP"
    ]  # +127%, above 100%


def test_run_screen_with_no_data_is_empty(market):
    market.data.clear()
    assert sm.run_screen("golden_state", "US").empty


def test_run_screen_applies_the_turnover_floor_when_known(monkeypatch, market):
    _annotator(monkeypatch, {"UP": 5e6, "SLOW": 5e5, "DOWN": 2e6})
    assert sm.run_screen("golden_state", "US")["Symbol"].tolist() == ["UP"]  # SLOW is below $1M
    assert sm.run_screen("golden_state", "US", min_turnover_usd=0)["Symbol"].tolist() == [
        "UP",
        "SLOW",
    ]
    assert sm.run_screen("golden_state", "US", min_turnover_usd=6e6).empty


def test_run_screen_sorts_unknown_returns_last(monkeypatch, market):
    market.data["YOUNG"] = _bars(np.linspace(100, 200, 120))  # 120 bars: no Ret252 yet
    out = sm.run_screen("rsi_overbought", "US")  # a screen that does not need the 200-day average
    assert out["Symbol"].tolist() == ["UP", "SLOW", "YOUNG"]
    assert np.isnan(out["Ret252"].iloc[-1])  # unknown returns sort after known ones


# ---- run_all ----------------------------------------------------------------------------


def test_run_all_counts_every_price_screen(market):
    counts = sm.run_all("US", verbose=False)
    assert set(counts) == set(sm.PRICE_SCREENS)
    assert counts["golden_state"] == 2 and counts["multibagger_momentum"] == 1
    assert counts["bearish_crossover"] == 0


def test_run_all_with_no_data_says_so(market, capsys):
    market.data.clear()
    assert sm.run_all("US") == {}
    assert "no data for US" in capsys.readouterr().out


def test_run_all_prints_a_ranked_summary(market, capsys):
    sm.run_all("US")
    out = capsys.readouterr().out
    assert "price-metric screens on US (3 liquid stocks in DB)" in out
    assert "golden_state" in out and "UP, SLOW" in out  # examples: best return first
    assert "fundamental screens (need financials" in out and "piotroski_9" in out
    assert out.index("golden_state") < out.index("bearish_crossover")  # most hits first
    sm.run_all("US", verbose=False)
    assert capsys.readouterr().out == ""


def test_run_all_applies_the_turnover_floor(monkeypatch, market, capsys):
    _annotator(monkeypatch, {"UP": 5e6, "SLOW": 5e5, "DOWN": 2e6})
    assert sm.run_all("US", verbose=False)["golden_state"] == 1
    sm.run_all("US")
    assert "(2 liquid stocks in DB)" in capsys.readouterr().out


# ---- main -------------------------------------------------------------------------------


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["screen_metrics.py", *args])


def test_main_lists_one_screen(monkeypatch, market, capsys):
    _argv(monkeypatch, "--market", "US", "--screen", "golden_state", "--top", "1")
    assert sm.main() == 0
    out = capsys.readouterr().out
    assert "golden_state on US — 1 stocks:" in out and "UP" in out and "SLOW" not in out
    _argv(monkeypatch, "--screen", "bearish_crossover")
    assert sm.main() == 0
    assert "none" in capsys.readouterr().out


def test_main_summarises_every_market_with_the_top_screens(monkeypatch, market, capsys):
    monkeypatch.setattr(sm.datalink, "MARKETS", ["US", "JP"])
    _argv(monkeypatch, "--all")
    assert sm.main() == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 2 and lines[0].startswith("US") and "golden_state=2" in lines[0]
    assert lines[0].count("=") == 6  # the six biggest counts per market


def test_main_defaults_to_the_full_run_for_the_market(monkeypatch, market, capsys):
    _argv(monkeypatch)
    assert sm.main() == 0 and "price-metric screens on IN" in capsys.readouterr().out
