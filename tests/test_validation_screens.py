"""Tests for validation.py beyond the basics in test_validation.py: every popular-screen predicate
at its boundaries, liquidity filtering, failure isolation, the cached public-screen overlay, the
grounded/novel maths, the report and the CLI.

Offline: ``serving_layer.serving`` and the optional ``public_screens`` / ``auto_screener`` modules
are faked.
"""

import sys
import types

import pandas as pd
import pytest

import validation as v

BASE = {
    "Close": 100.0,
    "SMA200": 200.0,
    "RSI14": 50.0,
    "PctFromHigh": -50.0,
    "PctFromLow": 50.0,
    "Ret126": 0.0,
    "Ret252": 0.0,
    "Above200DMA": False,
    "GoldenCross": False,
}


def _frame(*rows):
    return pd.DataFrame([{**BASE, **r} for r in rows])


# (screen, rows that must pass, rows that must fail) - each fail row is one step past the edge
CASES = [
    ("golden_crossover", [{"GoldenCross": True}], [{"GoldenCross": False}]),
    ("bearish_crossover", [{"GoldenCross": False}], [{"GoldenCross": True}]),
    (
        "companies_creating_new_high",
        [{"PctFromHigh": -1}, {"PctFromHigh": 3}],
        [{"PctFromHigh": -1.01}],
    ),
    (
        "darvas_scan",
        [{"PctFromHigh": -3, "Above200DMA": True}],
        [{"PctFromHigh": -3.01, "Above200DMA": True}, {"PctFromHigh": 0, "Above200DMA": False}],
    ),
    ("rsi_oversold", [{"RSI14": 29.9}], [{"RSI14": 30}]),
    (
        "stocks_near_200dma",
        [{"Close": 102, "SMA200": 100}, {"Close": 98, "SMA200": 100}],
        [{"Close": 103.5, "SMA200": 100}, {"Close": 96.5, "SMA200": 100}],
    ),
    ("52w_high_breakout", [{"PctFromHigh": 0}], [{"PctFromHigh": -0.01}]),
    ("all_time_high", [{"PctFromHigh": 0}], [{"PctFromHigh": -0.01}]),
    ("52w_low_contrarian", [{"PctFromLow": 5}], [{"PctFromLow": 5.01}]),
    (
        "midcap_momentum",
        [{"Ret126": 20.1, "Above200DMA": True}],
        [{"Ret126": 20, "Above200DMA": True}, {"Ret126": 30, "Above200DMA": False}],
    ),
    (
        "value_plus_momentum",
        [{"Ret126": 15.1, "PctFromHigh": -9.9}],
        [{"Ret126": 15, "PctFromHigh": -9.9}, {"Ret126": 20, "PctFromHigh": -10}],
    ),
    (
        "quality_plus_momentum",
        [{"Above200DMA": True, "GoldenCross": True, "Ret252": 15.1}],
        [
            {"Above200DMA": True, "GoldenCross": True, "Ret252": 15},
            {"Above200DMA": True, "GoldenCross": False, "Ret252": 50},
            {"Above200DMA": False, "GoldenCross": True, "Ret252": 50},
        ],
    ),
    ("multibagger_momentum", [{"Ret252": 100.1}], [{"Ret252": 100}]),
    (
        "steady_uptrend",
        [{"Above200DMA": True, "RSI14": 45}, {"Above200DMA": True, "RSI14": 70}],
        [
            {"Above200DMA": True, "RSI14": 44.9},
            {"Above200DMA": True, "RSI14": 70.1},
            {"Above200DMA": False, "RSI14": 55},
        ],
    ),
]


def test_cases_cover_every_local_screen():
    assert {c[0] for c in CASES} == set(v.LOCAL_SCREENS)


@pytest.mark.parametrize("name,passing,failing", CASES, ids=[c[0] for c in CASES])
def test_each_screen_at_its_boundaries(name, passing, failing):
    pred = v.LOCAL_SCREENS[name]
    assert list(pred(_frame(*passing))) == [True] * len(passing)
    assert list(pred(_frame(*failing))) == [False] * len(failing)


def test_fundamental_registry_is_disjoint_from_the_local_screens_and_described():
    assert not set(v.FUNDAMENTAL_SCREENS) & set(v.LOCAL_SCREENS)
    assert all(isinstance(d, str) and d for d in v.FUNDAMENTAL_SCREENS.values())


# ---- local_screens ----------------------------------------------------------------------


@pytest.fixture
def serving(monkeypatch):
    holder = {"df": pd.DataFrame()}
    monkeypatch.setattr(v.sl, "serving", lambda m: holder["df"])
    return holder


def _universe():
    return pd.DataFrame(
        [
            {**BASE, "Symbol": "HI", "GoldenCross": True, "Liquidity": "High"},
            {**BASE, "Symbol": "MD", "GoldenCross": True, "Liquidity": "Medium"},
            {**BASE, "Symbol": "LO", "GoldenCross": True, "Liquidity": "Low"},
            {**BASE, "Symbol": "NA", "GoldenCross": True, "Liquidity": None},
        ]
    )


def test_local_screens_only_consider_liquid_names(serving):
    serving["df"] = _universe()
    assert v.local_screens("US")["golden_crossover"] == {"HI", "MD"}  # Low and unknown dropped
    assert set(v.local_screens("US")) == set(v.LOCAL_SCREENS)


def test_local_screens_keep_everything_when_there_is_no_liquidity_column(serving):
    serving["df"] = _universe().drop(columns="Liquidity")
    assert v.local_screens("US")["golden_crossover"] == {"HI", "MD", "LO", "NA"}


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # ~ on object-dtype bools
@pytest.mark.filterwarnings("ignore::FutureWarning")  # fillna downcast on object dtype
def test_local_screens_treat_missing_values_as_not_matching(serving):
    df = _universe().drop(columns="Liquidity")
    df["GoldenCross"] = pd.Series([True, None, False, True], dtype=object)  # None = unknown
    serving["df"] = df
    assert v.local_screens("US")["golden_crossover"] == {"HI", "NA"}


def test_a_screen_that_cannot_be_evaluated_becomes_an_empty_set(serving):
    serving["df"] = _universe().drop(columns=["Liquidity", "RSI14"])  # rsi screens need RSI14
    out = v.local_screens("US")
    assert out["rsi_oversold"] == set() and out["steady_uptrend"] == set()
    assert out["golden_crossover"]  # the others are unaffected


def test_local_screens_is_empty_when_serving_has_no_data(serving):
    assert v.local_screens("US") == {}


# ---- the cached public overlay (India only) -----------------------------------------------


def _fake_public(monkeypatch, cached=None, boom=False):
    def cached_screens():
        if boom:
            raise RuntimeError("cache unreadable")
        return cached

    monkeypatch.setitem(
        sys.modules, "public_screens", types.SimpleNamespace(cached_screens=cached_screens)
    )


def test_public_fundamental_screens_are_prefixed_and_india_only(monkeypatch):
    _fake_public(monkeypatch, {"piotroski_9": {"X", "Y"}})
    assert v._public_fundamental("IN") == {"pub::piotroski_9": {"X", "Y"}}
    assert v._public_fundamental("US") == {}  # screener.in screens are Indian companies


def test_public_fundamental_failures_are_swallowed(monkeypatch):
    _fake_public(monkeypatch, boom=True)
    assert v._public_fundamental("IN") == {}
    monkeypatch.setitem(sys.modules, "public_screens", None)  # not importable at all
    assert v._public_fundamental("IN") == {}


def test_local_screens_add_public_screens_even_without_serving_data(serving, monkeypatch):
    _fake_public(monkeypatch, {"magic_formula": {"M"}})
    assert v.local_screens("IN") == {"pub::magic_formula": {"M"}}
    serving["df"] = _universe()
    assert "pub::magic_formula" in v.local_screens("IN") and "golden_crossover" in v.local_screens(
        "IN"
    )
    assert "pub::magic_formula" not in v.local_screens("US")


def test_known_universe_unions_local_and_public_names(serving, monkeypatch):
    _fake_public(monkeypatch, {"magic_formula": {"M"}})
    serving["df"] = _universe()
    assert v.known_universe("IN") == {"HI", "MD", "M"}


def test_known_universe_is_empty_without_any_screens(serving):
    assert v.known_universe("US") == set()


# ---- validate ---------------------------------------------------------------------------


def test_validate_with_no_picks_returns_zeros(serving):
    assert v.validate([], "US") == {"n": 0, "grounded_pct": 0.0, "novel_pct": 0.0, "per_screen": {}}


def test_validate_reports_overlap_per_screen_sorted_by_share(monkeypatch, serving):
    monkeypatch.setattr(
        v,
        "local_screens",
        lambda m: {
            "s_big": {"A", "B", "C"},
            "s_small": {"A"},
            "s_none": {"Z"},
            "s_mid": {"A", "B"},
        },
    )
    out = v.validate(["A", "B", "C", "Q"], "US")
    assert out["n"] == 4 and out["grounded_pct"] == 0.75 and out["novel_pct"] == 0.25
    assert out["per_screen"] == {
        "s_big": 0.75,
        "s_mid": 0.5,
        "s_small": 0.25,
    }  # no zero-overlap screens
    assert list(out["per_screen"]) == ["s_big", "s_mid", "s_small"]


def test_validate_deduplicates_picks_and_rounds(monkeypatch):
    monkeypatch.setattr(v, "local_screens", lambda m: {"s": {"A"}})
    out = v.validate(["A", "A", "B", "C"], "US")
    assert out["n"] == 3  # duplicates counted once
    assert out["grounded_pct"] == 0.333 and out["novel_pct"] == 0.667


def test_validate_with_no_screens_marks_everything_novel(monkeypatch):
    monkeypatch.setattr(v, "local_screens", lambda m: {})
    assert v.validate(["A", "B"], "US") == {
        "n": 2,
        "grounded_pct": 0.0,
        "novel_pct": 1.0,
        "per_screen": {},
    }


# ---- report / main ----------------------------------------------------------------------


def _recommend(monkeypatch, picks=None, boom=None):
    def recommend(market, verbose=True):
        assert verbose is False  # the report asks for a quiet recommendation
        if boom:
            raise boom
        return {"picks": picks}

    monkeypatch.setitem(sys.modules, "auto_screener", types.SimpleNamespace(recommend=recommend))


def test_report_validates_the_auto_screener_recommendation(monkeypatch, capsys):
    monkeypatch.setattr(v, "local_screens", lambda m: {"s_big": {"A", "B"}, "s_small": {"A"}})
    _recommend(monkeypatch, picks=["A", "Q"])
    out = v.report("US")
    assert out == {
        "local_screens": {"s_big": 2, "s_small": 1},
        "validation": {
            "n": 2,
            "grounded_pct": 0.5,
            "novel_pct": 0.5,
            "per_screen": {"s_big": 0.5, "s_small": 0.5},
        },
    }
    text = capsys.readouterr().out
    assert "validation vs Screener.in popular screens — US" in text
    assert f"local popular screens computed: 2 (+{len(v.FUNDAMENTAL_SCREENS)} fundamental" in text
    assert text.index("s_big") < text.index("s_small")  # biggest screen listed first
    assert "auto-screener recommendation (2 picks)" in text
    assert "grounded in popular screens: 50%   novel (new pattern): 50%" in text
    assert "overlaps: s_big 50%, s_small 50%" in text


def test_report_shows_only_the_top_six_overlaps(monkeypatch, capsys):
    monkeypatch.setattr(v, "local_screens", lambda m: {f"s{i}": {"A"} for i in range(8)})
    _recommend(monkeypatch, picks=["A"])
    v.report("US")
    line = [l for l in capsys.readouterr().out.splitlines() if "overlaps:" in l][0]
    assert line.count("%") == 6


def test_report_survives_a_failing_recommender(monkeypatch, capsys):
    monkeypatch.setattr(v, "local_screens", lambda m: {"s": {"A"}})
    _recommend(monkeypatch, boom=RuntimeError("no model"))
    out = v.report("US")
    assert out["validation"] == {"error": "no model"} and out["local_screens"] == {"s": 1}
    text = capsys.readouterr().out
    assert "auto-screener recommendation" not in text and "local popular screens computed" in text


def test_report_survives_a_failing_validation_of_the_picks(monkeypatch, capsys):
    monkeypatch.setattr(v, "local_screens", lambda m: {"s": {"A"}})
    _recommend(monkeypatch, picks=["A"])

    def boom(picks, market):
        raise ValueError("bad picks")

    monkeypatch.setattr(v, "validate", boom)
    assert v.report("US")["validation"] == {"error": "bad picks"}
    assert "auto-screener recommendation" not in capsys.readouterr().out


def test_report_is_quiet_when_not_verbose(monkeypatch, capsys):
    monkeypatch.setattr(v, "local_screens", lambda m: {})
    _recommend(monkeypatch, picks=[])
    assert v.report("US", verbose=False)["validation"]["n"] == 0
    assert capsys.readouterr().out == ""


def test_main_runs_the_report_for_the_requested_market(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(v, "report", lambda m: seen.append(m))
    monkeypatch.setattr(sys, "argv", ["validation.py", "--market", "JP"])
    assert v.main() == 0 and seen == ["JP"]
    monkeypatch.setattr(sys, "argv", ["validation.py"])
    v.main()
    assert seen == ["JP", "IN"]
