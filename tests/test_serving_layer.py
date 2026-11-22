"""Tests for serving_layer.py: per-symbol feature engineering, building the materialised view,
the CDC delta log, the cached reader, the vectorised screen, refresh orchestration and the CLI.

Offline and sandboxed: SERVING_DIR / CDC_DIR point into tmp_path, the process cache is reset, and
the data layer (datalink) and optional liquidity annotator are replaced where the module calls out.
"""

import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

import serving_layer as sl


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path):
    serving, cdc = tmp_path / "serving", tmp_path / "cdc"
    serving.mkdir()
    cdc.mkdir()
    monkeypatch.setattr(sl, "SERVING_DIR", serving)
    monkeypatch.setattr(sl, "CDC_DIR", cdc)
    monkeypatch.setattr(sl, "_SERVE_CACHE", {})
    return tmp_path


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _bars(closes, volume=1000.0, start="2024-01-01"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes))
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": float(volume)},
        index=idx,
    )


def _zigzag(n, base=100.0):
    """Deterministic up/down series with both gains and losses (so RSI is defined)."""
    steps = np.where(np.arange(n) % 3 == 0, -1.0, 1.5)
    return base + np.cumsum(steps)


# ---- _features --------------------------------------------------------------------------


def test_features_need_at_least_20_bars():
    assert sl._features("X", "US", None) is None
    assert sl._features("X", "US", _bars(_zigzag(19))) is None
    assert sl._features("X", "US", _bars(_zigzag(20))) is not None


def test_features_basic_fields_match_hand_calculations():
    closes = _zigzag(300)
    d = _bars(closes, volume=500.0)
    f = sl._features("ABC", "IN", d)
    assert f["Symbol"] == "ABC" and f["Market"] == "IN" and f["Bars"] == 300
    assert f["Close"] == closes[-1]
    assert f["SMA20"] == pytest.approx(closes[-20:].mean())
    assert f["SMA50"] == pytest.approx(closes[-50:].mean())
    assert f["SMA200"] == pytest.approx(closes[-200:].mean())
    win = closes[-252:]  # 52-week window is the last 252 bars
    assert f["High252"] == win.max() and f["Low252"] == win.min()
    assert f["PctFromHigh"] == pytest.approx((closes[-1] / win.max() - 1) * 100)
    assert f["PctFromLow"] == pytest.approx((closes[-1] / win.min() - 1) * 100)
    assert f["TurnoverLocal"] == pytest.approx((closes[-20:] * 500.0).mean())  # 20-day average
    assert f["LastDate"] == str(d.index[-1])[:10]


def test_features_rsi_matches_an_independent_wilder_recursion():
    closes = _zigzag(60)
    diffs = np.diff(closes)
    up, dn = np.clip(diffs, 0, None), np.clip(-diffs, 0, None)
    alpha, s_up, s_dn = 1 / 14, up[0], dn[0]
    for u, v in zip(up[1:], dn[1:]):
        s_up, s_dn = (1 - alpha) * s_up + alpha * u, (1 - alpha) * s_dn + alpha * v
    expected = 100 - 100 / (1 + s_up / s_dn)
    assert sl._features("X", "US", _bars(closes))["RSI14"] == pytest.approx(expected)
    assert 0 < expected < 100


def test_features_windows_are_nan_until_enough_history():
    f = sl._features("X", "US", _bars(_zigzag(30)))
    assert not np.isnan(f["SMA20"])
    assert np.isnan(f["SMA50"]) and np.isnan(f["SMA200"])
    assert f["Above200DMA"] is False and f["GoldenCross"] is False  # no 200-day average yet
    # return windows need k+1 bars: 22 bars is enough for Ret21 but not for Ret63
    short = sl._features("X", "US", _bars(_zigzag(22)))
    assert not np.isnan(short["Ret21"]) and np.isnan(short["Ret63"])
    assert np.isnan(sl._features("X", "US", _bars(_zigzag(21)))["Ret21"])


def test_features_trend_flags_follow_the_moving_averages():
    up = sl._features("UP", "US", _bars(np.linspace(100, 300, 260)))
    assert up["Above200DMA"] is True and up["GoldenCross"] is True
    assert up["PctFromHigh"] == pytest.approx(0.0)  # closing at the 52-week high
    down = sl._features("DN", "US", _bars(np.linspace(300, 100, 260)))
    assert down["Above200DMA"] is False and down["GoldenCross"] is False
    assert down["PctFromLow"] == pytest.approx(0.0)


def test_features_zero_volume_gives_zero_turnover():
    assert sl._features("X", "US", _bars(_zigzag(40), volume=0.0))["TurnoverLocal"] == 0.0


# ---- _source / build_serving ------------------------------------------------------------


@pytest.fixture
def datalink_stub(monkeypatch, sandbox):
    stub = {"ltm": {}, "calls": []}

    def load_market(market, tier="stm", **kw):
        stub["calls"].append((market, tier))
        return stub["ltm"] if tier == "ltm" else {}

    monkeypatch.setattr(sl.datalink, "load_market", load_market)
    monkeypatch.setattr(sl.datalink, "SEED_DIR", sandbox / "seed")
    (sandbox / "seed").mkdir()
    return stub


def test_source_prefers_the_ltm_layer(datalink_stub):
    datalink_stub["ltm"] = {"AAA": _bars(_zigzag(30))}
    assert list(sl._source("US")) == ["AAA"]
    assert datalink_stub["calls"] == [("US", "ltm")]


def test_source_falls_back_to_the_committed_seed(datalink_stub, sandbox):
    long = _bars(_zigzag(25)).assign(Symbol="SEEDED").rename_axis("Date").reset_index()
    long.to_parquet(sandbox / "seed" / "cleaned_long_US.parquet", index=False)
    out = sl._source("US")  # no LTM: reads the 1y seed
    assert list(out) == ["SEEDED"] and len(out["SEEDED"]) == 25
    assert sl._source("JP") == {}  # no seed either


def test_build_serving_writes_a_view_of_usable_symbols(monkeypatch, sandbox, capsys):
    monkeypatch.setattr(
        sl, "_source", lambda m: {"GOOD": _bars(_zigzag(40)), "SHORT": _bars(_zigzag(10))}
    )
    assert sl.build_serving("US") == 1
    df = pd.read_parquet(sl.SERVING_DIR / "US.parquet")
    assert df["Symbol"].tolist() == ["GOOD"] and df["Market"].iloc[0] == "US"
    assert {"Close", "RSI14", "SMA20", "PctFromHigh", "LastDate"} <= set(df.columns)
    assert "serving[US]: 1 symbols" in capsys.readouterr().out


def test_build_serving_with_no_usable_symbols_writes_nothing(monkeypatch):
    monkeypatch.setattr(sl, "_source", lambda m: {"SHORT": _bars(_zigzag(10))})
    assert sl.build_serving("US", verbose=False) == 0
    assert not (sl.SERVING_DIR / "US.parquet").exists()
    monkeypatch.setattr(sl, "_source", lambda m: {})
    assert sl.build_serving("US", verbose=False) == 0


def test_build_serving_annotates_liquidity_when_available_and_survives_without(monkeypatch):
    monkeypatch.setattr(sl, "_source", lambda m: {"A": _bars(_zigzag(40))})
    monkeypatch.setitem(
        sys.modules,
        "liquidity",
        _module("liquidity", annotate=lambda df: df.assign(Liquidity="High")),
    )
    sl.build_serving("US", verbose=False)
    assert pd.read_parquet(sl.SERVING_DIR / "US.parquet")["Liquidity"].tolist() == ["High"]

    def boom(df):
        raise RuntimeError("no liquidity index")

    monkeypatch.setitem(sys.modules, "liquidity", _module("liquidity", annotate=boom))
    assert sl.build_serving("JP", verbose=False) == 1  # annotation is best-effort
    assert "Liquidity" not in pd.read_parquet(sl.SERVING_DIR / "JP.parquet").columns


# ---- capture_deltas (CDC) ---------------------------------------------------------------


def _view(market, closes, date="2024-02-01"):
    pd.DataFrame(
        {"Symbol": list(closes), "Close": list(closes.values()), "LastDate": date}
    ).to_parquet(sl.SERVING_DIR / f"{market}.parquet", index=False)


def _log(market):
    return pd.read_parquet(sl.CDC_DIR / f"{market}.parquet")


def test_capture_deltas_without_a_view_is_zero():
    assert sl.capture_deltas("US") == 0


def test_capture_deltas_logs_inserts_then_only_real_changes(capsys):
    _view("US", {"AAA": 10.0, "BBB": 20.0})
    assert sl.capture_deltas("US") == 2
    log = _log("US")
    assert sorted(log["key"]) == ["AAA", "BBB"] and set(log["op"]) == {"INSERT"}
    assert set(log.columns) == {"op", "key", "value", "asof", "ts"}

    assert sl.capture_deltas("US") == 0  # nothing moved
    assert "no changes" in capsys.readouterr().out
    assert len(_log("US")) == 2  # and nothing was appended

    _view("US", {"AAA": 10.0, "BBB": 21.5, "CCC": 5.0}, date="2024-02-02")
    assert sl.capture_deltas("US") == 2  # BBB updated, CCC inserted; AAA unchanged
    new = _log("US").iloc[2:].set_index("key")
    assert new.loc["BBB", "op"] == "UPDATE" and new.loc["BBB", "value"] == 21.5
    assert new.loc["CCC", "op"] == "INSERT" and new.loc["BBB", "asof"] == "2024-02-02"
    assert "AAA" not in new.index and len(_log("US")) == 4  # the log only ever grows


def test_capture_deltas_compares_against_the_latest_logged_value_per_symbol():
    _view("US", {"AAA": 10.0})
    sl.capture_deltas("US")
    _view("US", {"AAA": 11.0})
    sl.capture_deltas("US")
    _view("US", {"AAA": 10.0})  # back to the original close: still a change vs the latest (11)
    assert sl.capture_deltas("US") == 1
    assert _log("US")["value"].tolist() == [10.0, 11.0, 10.0]


# ---- serving() reader -------------------------------------------------------------------


def test_serving_returns_empty_for_a_missing_view():
    assert sl.serving("US").empty


def test_serving_reads_and_caches_by_file_signature():
    _view("US", {"AAA": 10.0})
    first = sl.serving("US")
    assert first["Symbol"].tolist() == ["AAA"]
    assert sl.serving("US") is first  # unchanged file: served from memory

    p = sl.SERVING_DIR / "US.parquet"
    _view("US", {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0})  # different size -> new signature
    refreshed = sl.serving("US")
    assert refreshed is not first and sorted(refreshed["Symbol"]) == ["AAA", "BBB", "CCC"]
    assert p.exists()


# ---- screen_fast ------------------------------------------------------------------------


@pytest.fixture
def view():
    pd.DataFrame(
        {
            "Symbol": ["A", "B", "C", "D"],
            "RSI14": [30.0, 55.0, 80.0, 62.0],
            "Above200DMA": [True, True, False, True],
            "PctFromHigh": [-2.0, -15.0, -1.0, -6.0],
            "Ret252": [10.0, 40.0, 25.0, 5.0],
        }
    ).to_parquet(sl.SERVING_DIR / "IN.parquet", index=False)


def test_screen_fast_applies_every_criterion_and_sorts_by_return(view):
    out = sl.screen_fast(
        {"Above200DMA": ("==", True), "RSI14": ("<", 65), "PctFromHigh": (">", -10)}, "IN"
    )
    assert out["Symbol"].tolist() == [
        "A",
        "D",
    ]  # B is too far from its high; C is below the 200 DMA
    assert out["Ret252"].tolist() == [10.0, 5.0]  # sorted by Ret252, best first
    assert out.index.tolist() == [0, 1]  # clean index


@pytest.mark.parametrize(
    "op,val,expected",
    [
        ("<", 55.0, {"A"}),
        ("<=", 55.0, {"A", "B"}),
        (">", 62.0, {"C"}),
        (">=", 62.0, {"C", "D"}),
        ("==", 80.0, {"C"}),
        ("!=", 80.0, {"A", "B", "D"}),
    ],
)
def test_screen_fast_supports_all_comparison_operators(view, op, val, expected):
    assert set(sl.screen_fast({"RSI14": (op, val)}, "IN", top=None)["Symbol"]) == expected


def test_screen_fast_top_sort_and_empty_inputs(view):
    assert sl.screen_fast({}, "IN", top=2)["Symbol"].tolist() == ["B", "C"]  # top 2 by Ret252
    assert len(sl.screen_fast({}, "IN", top=None)) == 4
    assert sl.screen_fast({}, "IN", top=1, sort="RSI14")["Symbol"].tolist() == ["C"]  # custom sort
    assert sl.screen_fast({}, "IN", sort="NoSuchColumn")["Symbol"].tolist() == ["A", "B", "C", "D"]
    assert sl.screen_fast({}, "US").empty  # no view for that market


# ---- refresh ----------------------------------------------------------------------------


@pytest.fixture
def refresh_env(monkeypatch):
    env = {"built": [], "deltas": [], "changed": []}
    monkeypatch.setattr(sl, "MARKETS", ["AA", "BB", "CC"])
    monkeypatch.setattr(sl.datalink, "changed_markets", lambda: list(env["changed"]))

    def build(m, verbose=True):
        env["built"].append(m)
        return 5 if m != "CC" else 0  # CC has no usable symbols

    monkeypatch.setattr(sl, "build_serving", build)
    monkeypatch.setattr(sl, "capture_deltas", lambda m, verbose=True: env["deltas"].append(m) or 2)
    return env


def test_refresh_builds_only_changed_or_missing_markets_by_default(refresh_env, capsys):
    (sl.SERVING_DIR / "AA.parquet").write_bytes(b"x")  # AA already has a current view
    (sl.SERVING_DIR / "BB.parquet").write_bytes(b"x")
    refresh_env["changed"] = ["BB"]  # only BB's seed moved
    out = sl.refresh()
    assert refresh_env["built"] == ["BB", "CC"]  # AA skipped; CC has no view yet
    assert out == {"views": 1, "deltas": 2}  # CC built nothing, so no delta capture for it
    assert refresh_env["deltas"] == ["BB"]
    assert "serving refresh: 1 views, 2 deltas" in capsys.readouterr().out


def test_refresh_with_explicit_markets_rebuilds_them_all(refresh_env):
    for m in ("AA", "BB"):
        (sl.SERVING_DIR / f"{m}.parquet").write_bytes(b"x")
    out = sl.refresh(["AA", "BB"], verbose=False)
    assert refresh_env["built"] == ["AA", "BB"]  # asked for by name: no "unchanged" shortcut
    assert out == {"views": 2, "deltas": 4}


# ---- CLI --------------------------------------------------------------------------------


def test_main_screen_prints_matches_or_a_hint(view, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["serving_layer.py", "--screen", "IN"])
    assert sl.main() == 0
    assert "Symbol" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["serving_layer.py", "--screen", "US"])
    assert sl.main() == 0
    assert "no serving view" in capsys.readouterr().out


def test_main_refresh_passes_the_market_filter(monkeypatch):
    seen = []
    monkeypatch.setattr(
        sl, "refresh", lambda markets=None, verbose=True: seen.append(markets) or {}
    )
    monkeypatch.setattr(sys, "argv", ["serving_layer.py", "--refresh", "--market", "IN", "US"])
    assert sl.main() == 0
    monkeypatch.setattr(sys, "argv", ["serving_layer.py", "--refresh"])
    assert sl.main() == 0
    assert seen == [["IN", "US"], None]


def test_capture_deltas_stamps_utc_times_without_deprecation_warnings():
    # Timestamp.utcnow() is deprecated (Pandas4Warning) and slated for removal; this module hides
    # warnings globally, so turn them into errors here to keep the stamp on a supported API.
    _view("US", {"AAA": 10.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sl.capture_deltas("US", verbose=False) == 1
    stamp = pd.Timestamp(_log("US")["ts"].iloc[0])
    assert stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
    assert abs((pd.Timestamp.now("UTC") - stamp).total_seconds()) < 60
