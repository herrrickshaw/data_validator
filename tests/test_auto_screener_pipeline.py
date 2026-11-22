"""Tests for auto_screener.py: the supervised anchor (known_good), market regime, the feature matrix,
unsupervised discovery (clustering -> rule), rule application, reward evaluation, RL threshold
refinement and the recommend() orchestration.

Offline and dependency-free: scikit-learn's KMeans / StandardScaler are small deterministic fakes,
the serving layer is a frame returned by a patched ``sl.serving``, and every optional collaborator
(ml_supervised, validation, fundamental_metrics' cache, sector_map, datalink's CCC map) is a fake
so nothing reads real models, caches or the network. Tests for the boom-sector gating only run
where that feature exists (``boom_sector_members``).
"""

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import auto_screener as a

HAS_BOOM = hasattr(a, "boom_sector_members")
REAL_FUNDAMENTAL_ANCHOR = a.fundamental_anchor
REAL_BOOM_SECTOR_MEMBERS = getattr(a, "boom_sector_members", None)
needs_boom = pytest.mark.skipif(
    not HAS_BOOM, reason="sector-boom gating is not in this repo's auto_screener"
)


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _serve(monkeypatch, df):
    monkeypatch.setattr(a.sl, "serving", lambda market: df)


def _view(n=6, **overrides):
    """A serving-style frame; every column can be overridden with a list or scalar."""
    base = {
        "Symbol": [f"S{i}" for i in range(n)],
        "RSI14": 55.0,
        "PctFromHigh": -10.0,
        "PctFromLow": 40.0,
        "Ret21": 2.0,
        "Ret63": 5.0,
        "Ret126": 10.0,
        "Ret252": 20.0,
        "Above200DMA": True,
        "GoldenCross": False,
        "Liquidity": "High",
    }
    base.update(overrides)
    return pd.DataFrame({k: (v if isinstance(v, list) else [v] * n) for k, v in base.items()})


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(a, "OUT_DIR", tmp_path)
    # default collaborators: nothing extra is known unless a test installs it
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module("ml_supervised", known_good_supervised=lambda m: set()),
    )
    monkeypatch.setitem(
        sys.modules, "validation", _module("validation", known_universe=lambda m: set())
    )
    monkeypatch.setattr(a, "fundamental_anchor", lambda m: set())
    if HAS_BOOM:
        monkeypatch.setattr(a, "boom_sector_members", lambda m: set())
    return tmp_path


# ---- known_good -------------------------------------------------------------------------


def test_known_good_is_empty_without_a_serving_view(monkeypatch):
    _serve(monkeypatch, pd.DataFrame())
    assert a.known_good("US") == set()


def test_known_good_applies_the_three_price_gates(monkeypatch):
    df = _view(
        8,
        Above200DMA=[True, True, True, True, False, True, True, False],
        PctFromHigh=[-3.0, -5.0, -3.0, -3.0, -1.0, -50.0, -50.0, -1.0],  # darvas: > -5 (strict)
        Ret126=[5.0, 5.0, -1.0, 5.0, 5.0, 0.0, 0.0, 5.0],  # darvas: > 0
        GoldenCross=[False, False, False, False, True, True, False, True],
        Ret252=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0001, 99.0],  # quality: > 15 and above the 200 DMA
    )
    _serve(monkeypatch, df)
    # S0 darvas | S1 PctFromHigh == -5 is NOT > -5 | S2 Ret126 < 0 | S3 passes darvas
    # S4 golden cross but below the 200 DMA | S5 golden cross + above | S6 quality | S7 below the 200 DMA
    assert a.known_good("US") == {"S0", "S3", "S5", "S6"}


def test_known_good_quality_gate_boundary_is_strict(monkeypatch):
    _serve(monkeypatch, _view(2, PctFromHigh=-50.0, Ret126=-5.0, Ret252=[15.0, 15.01]))
    assert a.known_good("US") == {"S1"}


def test_known_good_restricts_to_liquid_names_when_liquidity_is_known(monkeypatch):
    _serve(
        monkeypatch, _view(4, Liquidity=["High", "Medium", "Low", "Unknown"])
    )  # all pass the price gates
    assert a.known_good("US") == {"S0", "S1"}
    no_liq = _view(3).drop(columns="Liquidity")
    _serve(monkeypatch, no_liq)
    assert a.known_good("US") == {"S0", "S1", "S2"}  # no liquidity column: no liquidity filter


def test_known_good_folds_in_every_optional_label_source(monkeypatch):
    _serve(
        monkeypatch, _view(1, Above200DMA=False, GoldenCross=False, Ret252=0.0)
    )  # S0 fails every gate
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module("ml_supervised", known_good_supervised=lambda m: {"SUP"}),
    )
    monkeypatch.setitem(
        sys.modules, "validation", _module("validation", known_universe=lambda m: {"VAL"})
    )
    monkeypatch.setattr(a, "fundamental_anchor", lambda m: {"FUND"})
    if HAS_BOOM:
        monkeypatch.setattr(a, "boom_sector_members", lambda m: {"BOOM"})
    expected = {"SUP", "VAL", "FUND"} | ({"BOOM"} if HAS_BOOM else set())
    assert a.known_good("US") == expected


def test_known_good_survives_failing_label_sources(monkeypatch):
    _serve(monkeypatch, _view(2))

    def boom(*args, **kw):
        raise RuntimeError("source unavailable")

    monkeypatch.setitem(
        sys.modules, "ml_supervised", _module("ml_supervised", known_good_supervised=boom)
    )
    monkeypatch.setitem(sys.modules, "validation", _module("validation", known_universe=boom))
    monkeypatch.setattr(a, "fundamental_anchor", boom)
    if HAS_BOOM:
        monkeypatch.setattr(a, "boom_sector_members", boom)
    assert a.known_good("US") == {"S0", "S1"}  # the price gates still stand on their own


def test_known_good_adds_the_ccc_universe_for_india_only(monkeypatch):
    import datalink

    _serve(monkeypatch, _view(1, Above200DMA=False, GoldenCross=False, Ret252=0.0))
    monkeypatch.setattr(datalink, "ccc_map_cached", lambda: {"CCC1": 12.0, "CCC2": -3.0})
    assert a.known_good("IN") == {"CCC1", "CCC2"}
    assert a.known_good("US") == set()  # not applied to other markets

    def boom():
        raise RuntimeError("no parquet")

    monkeypatch.setattr(datalink, "ccc_map_cached", boom)
    assert a.known_good("IN") == set()  # best-effort


# ---- fundamental_anchor -----------------------------------------------------------------


def _fund_frame():
    return pd.DataFrame(
        [
            {
                "Symbol": "QUALITY",
                "roe": 25.0,
                "debt_to_equity": 0.05,
                "ebit": 40.0,
                "market_cap": 100.0,
                "total_assets": 100.0,
                "cfo": 30.0,
                "net_income": 20.0,
                "roa": 20.0,
            },
            {
                "Symbol": "WEAK",
                "roe": 5.0,
                "debt_to_equity": 2.0,
                "ebit": 1.0,
                "market_cap": 100.0,
                "total_assets": 100.0,
                "cfo": -1.0,
                "net_income": -2.0,
                "roa": -2.0,
            },
        ]
    )


def test_fundamental_anchor_collects_names_passing_any_fundamental_screen(monkeypatch, tmp_path):
    import fundamental_metrics as fmet

    monkeypatch.setattr(a, "fundamental_anchor", REAL_FUNDAMENTAL_ANCHOR)
    monkeypatch.setattr(fmet, "CACHE", tmp_path)
    _fund_frame().to_parquet(tmp_path / "US.parquet", index=False)
    assert a.fundamental_anchor("US") == {
        "QUALITY"
    }  # high ROE / ROCE / magic formula, none for WEAK


def test_fundamental_anchor_is_empty_without_a_cache_or_with_an_empty_one(monkeypatch, tmp_path):
    import fundamental_metrics as fmet

    monkeypatch.setattr(a, "fundamental_anchor", REAL_FUNDAMENTAL_ANCHOR)
    monkeypatch.setattr(fmet, "CACHE", tmp_path)
    assert a.fundamental_anchor("US") == set()  # nothing fetched yet
    _fund_frame().iloc[0:0].to_parquet(tmp_path / "US.parquet", index=False)
    assert a.fundamental_anchor("US") == set()


def test_fundamental_anchor_skips_a_screen_that_cannot_run(monkeypatch, tmp_path):
    import fundamental_metrics as fmet

    monkeypatch.setattr(a, "fundamental_anchor", REAL_FUNDAMENTAL_ANCHOR)
    monkeypatch.setattr(fmet, "CACHE", tmp_path)
    _fund_frame().to_parquet(tmp_path / "US.parquet", index=False)

    def broken(df):
        raise KeyError("column missing")

    monkeypatch.setitem(
        fmet.FUND_SCREENS, "high_roe", broken
    )  # one screen fails, the others still count
    assert a.fundamental_anchor("US") == {"QUALITY"}


# ---- boom-sector gating (only where the feature exists) ---------------------------------


@needs_boom
def test_boom_sectors_are_those_in_a_boom_phase(monkeypatch):
    rot = pd.DataFrame(
        {
            "Sector": ["Tech", "Banks", "Energy", "Metals"],
            "Phase": ["Early Boom", "Bust", "Peak Boom", None],
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "sector_map",
        _module(
            "sector_map", sector_map=lambda m: {"A": "Tech"}, rotation=lambda serving, smap: rot
        ),
    )
    monkeypatch.setattr(a.sl, "serving", lambda m: pd.DataFrame({"Symbol": ["A"]}))
    assert a.boom_sectors("US") == {"Tech", "Energy"}


@needs_boom
def test_boom_sectors_degrade_to_empty(monkeypatch):
    monkeypatch.setattr(a.sl, "serving", lambda m: pd.DataFrame({"Symbol": ["A"]}))
    monkeypatch.setitem(
        sys.modules,
        "sector_map",
        _module("sector_map", sector_map=lambda m: {}, rotation=lambda s, m: pd.DataFrame()),
    )
    assert a.boom_sectors("US") == set()  # no sector map fetched yet
    monkeypatch.setitem(
        sys.modules,
        "sector_map",
        _module(
            "sector_map", sector_map=lambda m: {"A": "Tech"}, rotation=lambda s, m: pd.DataFrame()
        ),
    )
    assert a.boom_sectors("US") == set()  # empty rotation table

    def boom(*args):
        raise RuntimeError("down")

    monkeypatch.setitem(
        sys.modules, "sector_map", _module("sector_map", sector_map=boom, rotation=boom)
    )
    assert a.boom_sectors("US") == set()


@needs_boom
def test_boom_sector_members_are_the_symbols_in_booming_sectors(monkeypatch):
    monkeypatch.setattr(a, "boom_sector_members", REAL_BOOM_SECTOR_MEMBERS)
    smap = {"A": "Tech", "B": "Banks", "C": "Tech", "D": "Energy"}
    monkeypatch.setitem(sys.modules, "sector_map", _module("sector_map", sector_map=lambda m: smap))
    monkeypatch.setattr(a, "boom_sectors", lambda m: {"Tech", "Energy"})
    assert a.boom_sector_members("US") == {"A", "C", "D"}
    monkeypatch.setattr(a, "boom_sectors", lambda m: set())
    assert a.boom_sector_members("US") == set()


@needs_boom
def test_boom_sector_members_degrade_to_empty_when_the_sector_map_fails(monkeypatch):
    monkeypatch.setattr(a, "boom_sector_members", REAL_BOOM_SECTOR_MEMBERS)
    monkeypatch.setattr(a, "boom_sectors", lambda m: {"Tech"})

    def boom(market):
        raise RuntimeError("sector map unavailable")

    monkeypatch.setitem(sys.modules, "sector_map", _module("sector_map", sector_map=boom))
    assert a.boom_sector_members("US") == set()


@needs_boom
def test_known_good_includes_boom_sector_members(monkeypatch):
    _serve(monkeypatch, _view(1, Above200DMA=False, GoldenCross=False, Ret252=0.0))
    monkeypatch.setattr(a, "boom_sector_members", lambda m: {"HOT1", "HOT2"})
    assert a.known_good("US") == {"HOT1", "HOT2"}


# ---- _ret / regime ----------------------------------------------------------------------


def test_ret_prefers_the_252_day_return_and_falls_back_to_126():
    df = pd.DataFrame({"Ret252": [10.0, np.nan, np.nan], "Ret126": [1.0, 2.0, np.nan]})
    assert a._ret(df).tolist()[:2] == [10.0, 2.0] and np.isnan(a._ret(df).iloc[2])
    assert a._ret(df.drop(columns="Ret126")).iloc[0] == 10.0  # young seeds without a 126-day return


@pytest.mark.parametrize(
    "above,ret,expected",
    [
        ([True] * 6 + [False] * 4, 5.0, "Bull"),  # breadth 0.6 > 0.55 and positive trailing return
        ([True] * 55 + [False] * 45, 5.0, "Neutral"),  # breadth exactly 0.55 is not > 0.55
        ([True] * 6 + [False] * 4, 0.0, "Neutral"),  # breadth fine but median return not positive
        ([True] * 3 + [False] * 7, 5.0, "Bear"),  # breadth 0.3 < 0.35
        ([True] * 5 + [False] * 5, -5.01, "Bear"),  # deeply negative median return
        ([True] * 5 + [False] * 5, -5.0, "Neutral"),  # exactly -5 is not "< -5"
        ([True] * 35 + [False] * 65, 5.0, "Neutral"),  # breadth exactly 0.35 is not < 0.35
    ],
)
def test_regime_classifies_the_tape(above, ret, expected):
    n = len(above)
    df = _view(n, Above200DMA=above, Ret252=ret)
    assert a.regime(df)["tape"] == expected


def test_regime_reports_breadth_rsi_return_and_liquid_share():
    df = _view(
        4,
        Above200DMA=[True, True, False, False],
        RSI14=[40.0, 50.0, 60.0, 70.0],
        Ret252=[1.0, 2.0, 3.0, 4.0],
        Liquidity=["High", "Medium", "Low", "Low"],
    )
    r = a.regime(df)
    assert (
        r["breadth"] == 0.5
        and r["med_rsi"] == 55.0
        and r["med_ret252"] == 2.5
        and r["liquid_share"] == 0.5
    )


def test_regime_defaults_when_columns_are_missing():
    r = a.regime(_view(4).drop(columns=["Above200DMA", "Liquidity"]))
    assert (
        r["breadth"] == 0.0 and r["liquid_share"] == 0.0 and r["tape"] == "Bear"
    )  # no breadth data -> Bear


# ---- _matrix ----------------------------------------------------------------------------


def test_matrix_keeps_rows_with_the_core_features_and_imputes_the_rest():
    df = _view(
        5,
        RSI14=[50.0, np.nan, 60.0, 70.0, 80.0],  # row 1 has no RSI -> dropped
        PctFromHigh=[
            -1.0,
            -2.0,
            -3.0,
            np.inf,
            -5.0,
        ],  # inf becomes NaN, then is dropped (a core feature)
        Ret63=[1.0, 2.0, 3.0, 4.0, np.nan],  # row 4 has no Ret63 -> dropped
        Ret252=[10.0, 20.0, np.nan, 40.0, 50.0],  # NaN here is imputed, not dropped
    )
    X, kept = a._matrix(df)
    assert list(kept["Symbol"]) == ["S0", "S2"] and X.shape == (2, len(a.FEATS) + len(a.BOOLS))
    ret252_col = a.FEATS.index("Ret252")
    assert (
        X[0, ret252_col] == 10.0 and X[1, ret252_col] == 10.0
    )  # S2's missing Ret252 -> column median (10.0)
    assert not np.isnan(X).any() and np.isfinite(X).all()


def test_matrix_encodes_booleans_as_floats():
    X, _ = a._matrix(_view(2, Above200DMA=[True, False], GoldenCross=[False, True]))
    assert X[:, len(a.FEATS)].tolist() == [1.0, 0.0] and X[:, len(a.FEATS) + 1].tolist() == [
        0.0,
        1.0,
    ]


# ---- discover (fake sklearn) ------------------------------------------------------------


class FakeScaler:
    def fit_transform(self, X):
        sd = X.std(axis=0)
        return (X - X.mean(axis=0)) / np.where(sd == 0, 1.0, sd)


class FakeKMeans:
    """Two clusters split on the first (standardised) feature, RSI14."""

    def __init__(self, n_clusters, n_init, random_state):
        self.k = n_clusters

    def fit_predict(self, Z):
        return (Z[:, 0] > 0).astype(int)


@pytest.fixture
def fake_sklearn(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", _module("sklearn"))
    monkeypatch.setitem(
        sys.modules, "sklearn.cluster", _module("sklearn.cluster", KMeans=FakeKMeans)
    )
    monkeypatch.setitem(
        sys.modules,
        "sklearn.preprocessing",
        _module("sklearn.preprocessing", StandardScaler=FakeScaler),
    )


def _two_groups():
    """12 'trending' names (RSI 60..71) and 12 'weak' names (RSI 30..41), plus 3 illiquid ones."""
    n = 27
    rsi = [60.0 + i for i in range(12)] + [30.0 + i for i in range(12)] + [50.0, 50.0, 50.0]
    return _view(
        n,
        Symbol=[f"T{i}" for i in range(12)] + [f"W{i}" for i in range(12)] + ["L0", "L1", "L2"],
        RSI14=rsi,
        PctFromHigh=[-2.0] * 12 + [-30.0] * 12 + [-1.0] * 3,
        Ret126=[20.0] * 12 + [-10.0] * 12 + [0.0] * 3,
        Above200DMA=[True] * 12 + [False] * 15,
        Liquidity=["High"] * 24 + ["Low"] * 3,
    )


def test_discover_picks_the_cluster_most_enriched_in_known_good_names(monkeypatch, fake_sklearn):
    _serve(monkeypatch, _two_groups())
    monkeypatch.setattr(a, "known_good", lambda m: {f"T{i}" for i in range(12)})
    out = a.discover("US", k=2)
    assert out["market"] == "US" and out["cluster_size"] == 12 and out["cluster_overlap"] == 1.0
    assert out["known_good_n"] == 12
    # the rule is the trending cluster's 10th-90th percentile bounds: RSI 60..71 -> 61.1 and 69.9
    assert out["rule"]["RSI14"] == [(">=", 61.1), ("<=", 69.9)]
    assert out["rule"]["PctFromHigh"] == [(">=", -2.0), ("<=", -2.0)]
    assert set(out["regime"]) == {"tape", "breadth", "med_rsi", "med_ret252", "liquid_share"}


def test_discover_ignores_illiquid_names_but_the_regime_uses_everything(monkeypatch, fake_sklearn):
    df = _two_groups()
    _serve(monkeypatch, df)
    monkeypatch.setattr(a, "known_good", lambda m: set())
    out = a.discover("US", k=2)
    assert (
        out["cluster_size"] == 12
    )  # 24 liquid names split 12/12; the 3 illiquid ones are never clustered
    assert out["regime"]["breadth"] == pytest.approx(
        12 / 27
    )  # but they count in the market breadth
    assert out["regime"]["liquid_share"] == pytest.approx(24 / 27)


def test_discover_refuses_without_a_view_or_enough_names(monkeypatch, fake_sklearn):
    _serve(monkeypatch, pd.DataFrame())
    with pytest.raises(RuntimeError, match="no serving view for US"):
        a.discover("US")
    _serve(monkeypatch, _view(9))  # fewer than k * 5 = 10 liquid names
    monkeypatch.setattr(a, "known_good", lambda m: set())
    with pytest.raises(RuntimeError, match=r"too few liquid names in US \(9\)"):
        a.discover("US", k=2)


def test_discover_works_without_a_liquidity_column(monkeypatch, fake_sklearn):
    df = _two_groups().drop(columns="Liquidity")
    _serve(monkeypatch, df)
    monkeypatch.setattr(a, "known_good", lambda m: set())
    assert a.discover("US", k=2)["cluster_size"] > 0


# ---- _rule_from_cluster -----------------------------------------------------------------


def _cluster():
    return pd.DataFrame(
        {
            "RSI14": np.arange(0.0, 11.0),
            "PctFromHigh": np.arange(-10.0, 1.0),
            "Ret126": np.arange(100.0, 111.0),
        }
    )


def test_rule_from_cluster_uses_10th_to_90th_percentile_bounds():
    rule = a._rule_from_cluster(_cluster(), {"tape": "Neutral"})
    assert rule == {
        "RSI14": [(">=", 1.0), ("<=", 9.0)],
        "PctFromHigh": [(">=", -9.0), ("<=", -1.0)],
        "Ret126": [(">=", 101.0), ("<=", 109.0)],
    }


def test_rule_from_cluster_adds_regime_guards():
    bull = a._rule_from_cluster(_cluster(), {"tape": "Bull"})
    assert bull["GoldenCross"] == [("==", True)] and "Above200DMA" not in bull
    bear = a._rule_from_cluster(_cluster(), {"tape": "Bear"})
    assert bear["Above200DMA"] == [("==", True)]
    assert bear["PctFromHigh"] == [
        (">=", -20.0)
    ]  # the falling-knife guard replaces the cluster bounds


# ---- apply_rule -------------------------------------------------------------------------


def test_apply_rule_returns_the_view_unchanged_when_empty(monkeypatch):
    empty = pd.DataFrame()
    _serve(monkeypatch, empty)
    assert a.apply_rule({"RSI14": [(">=", 50)]}, "US") is empty


def test_apply_rule_filters_liquid_names_and_sorts_by_return(monkeypatch):
    df = _view(
        5,
        RSI14=[60.0, 70.0, 65.0, 40.0, 80.0],
        Ret252=[10.0, 30.0, 20.0, 99.0, 50.0],
        Liquidity=["High", "Medium", "High", "High", "Low"],
    )
    _serve(monkeypatch, df)
    out = a.apply_rule({"RSI14": [(">=", 55.0), ("<=", 75.0)]}, "US")
    assert out["Symbol"].tolist() == [
        "S1",
        "S2",
        "S0",
    ]  # S3 fails RSI, S4 is illiquid; best Ret252 first
    assert out.index.tolist() == [0, 1, 2]
    assert a.apply_rule({"RSI14": [(">=", 55.0)]}, "US", top=1)["Symbol"].tolist() == ["S1"]


def test_apply_rule_ignores_unknown_columns_and_operators(monkeypatch):
    _serve(monkeypatch, _view(3))
    assert len(a.apply_rule({"NoSuchColumn": [(">", 1)], "RSI14": [("~=", 50)]}, "US")) == 3


# ---- evaluate ---------------------------------------------------------------------------


def _sel(monkeypatch, symbols, liquidity="High", ret=20.0):
    n = len(symbols)
    _serve(monkeypatch, _view(n, Symbol=symbols, Liquidity=liquidity, Ret252=ret))


def test_evaluate_scores_overlap_liquidity_and_return(monkeypatch):
    _sel(monkeypatch, ["A", "B", "C", "D"])
    out = a.evaluate({}, "US", good={"A", "B", "X"}, target=(1, 10))
    # overlap 2/4 = 0.5; liq 1.0; ret median 20; size inside the target -> no size deviation
    # deviation = 0.5 * 0.6 = 0.3;  reward = 2*0.5 + 0.5*1.0 + 0.01*20 - 1.5*0.3 = 1.25
    assert out == {
        "reward": 1.25,
        "n": 4,
        "overlap": 0.5,
        "liq": 1.0,
        "ret": 20.0,
        "deviation": 0.3,
    }


def test_evaluate_penalises_a_selection_of_the_wrong_size(monkeypatch):
    _sel(monkeypatch, ["A", "B"])
    out = a.evaluate({}, "US", good={"A", "B"}, target=(10, 20))
    # n=2 is below the band: size_dev = min(|2-10|, |2-20|) / 20 = 0.4 -> deviation = 0.4 * 0.2 = 0.08
    assert out["deviation"] == 0.08
    assert out["reward"] == pytest.approx(
        2 * 1.0 + 0.5 * 1.0 + 0.01 * 20 - 1.5 * 0.08, abs=1e-3
    )  # 2.58
    over = a.evaluate({}, "US", good={"A", "B"}, target=(0, 1))
    assert over["deviation"] == pytest.approx(
        0.2 * min((2 - 1) / 1, 1.0), abs=1e-3
    )  # capped at 1.0 -> 0.2


def test_evaluate_only_ever_sees_liquid_names_because_the_selection_is_pre_filtered(monkeypatch):
    _serve(
        monkeypatch,
        _view(4, Symbol=["A", "B", "C", "D"], Liquidity=["High", "Medium", "Low", "Unknown"]),
    )
    out = a.evaluate({}, "US", {"A", "B"}, target=(1, 10))
    assert (
        out["n"] == 2 and out["liq"] == 1.0
    )  # Low/Unknown names never enter the selection (Medium counts)


@pytest.mark.xfail(
    strict=True,
    reason="known bug: evaluate() reads sel['Liquidity'] unguarded, although apply_rule() and discover() "
    "handle a view without that column (e.g. where the liquidity annotator is absent), so recommend() "
    "raises KeyError there",
)
def test_evaluate_works_on_a_view_without_a_liquidity_column(monkeypatch):
    _serve(monkeypatch, _view(4).drop(columns="Liquidity"))
    assert a.evaluate({}, "US", {"S0"}, target=(1, 10))["n"] == 4


def test_evaluate_an_empty_selection_is_the_worst_case(monkeypatch):
    _serve(monkeypatch, _view(3, RSI14=10.0))
    out = a.evaluate({"RSI14": [(">=", 99.0)]}, "US", {"S0"})
    assert out == {"reward": -3.0, "n": 0, "overlap": 0.0, "liq": 0.0, "ret": 0.0, "deviation": 1.0}


# ---- rl_refine --------------------------------------------------------------------------


def test_rl_refine_never_returns_a_worse_rule_and_leaves_the_input_alone(monkeypatch):
    rule = {"RSI14": [(">=", 20.0), ("<=", 40.0)], "Above200DMA": [("==", True)]}
    original = json.loads(json.dumps(rule))
    target = 60.0  # the best lower bound for this fake world

    def evaluate(r, market, good, target_=(15, 60)):
        return {"reward": -abs(r["RSI14"][0][1] - target)}

    monkeypatch.setattr(a, "evaluate", evaluate)
    best_rule, best = a.rl_refine(rule, "US", set(), iters=60, verbose=False)
    assert (
        best["reward"] >= evaluate(rule, "US", set())["reward"]
    )  # hill-climbing only accepts improvements
    assert best["reward"] > -40.0  # it actually moved towards the optimum (started at -40)
    assert best_rule["Above200DMA"] == [["==", True]] or best_rule["Above200DMA"] == [
        ("==", True)
    ]  # untouched
    assert json.loads(json.dumps(rule)) == original  # the caller's rule was not mutated
    assert best_rule["RSI14"][1][0] == "<="  # operators are preserved


def test_rl_refine_with_no_iterations_returns_the_original_rule(monkeypatch):
    rule = {"RSI14": [(">=", 20.0)]}
    monkeypatch.setattr(a, "evaluate", lambda r, m, g, target=(15, 60): {"reward": 1.0})
    best_rule, best = a.rl_refine(rule, "US", set(), iters=0, verbose=False)
    assert best_rule == json.loads(json.dumps(rule)) and best == {"reward": 1.0}


def test_rl_refine_only_perturbs_numeric_thresholds(monkeypatch):
    seen = []

    def evaluate(r, m, g, target=(15, 60)):
        seen.append(json.loads(json.dumps(r)))
        return {"reward": 0.0}

    monkeypatch.setattr(a, "evaluate", evaluate)
    rule = {"GoldenCross": [("==", True)], "RSI14": [(">=", 50.0)]}
    a.rl_refine(rule, "US", set(), iters=3, verbose=False)
    assert all(c["GoldenCross"] == [["==", True]] for c in seen)  # booleans are never nudged
    assert any(c["RSI14"][0][1] != 50.0 for c in seen[1:])  # numbers are


def test_rl_refine_reports_the_reward_change(monkeypatch, capsys):
    monkeypatch.setattr(a, "evaluate", lambda r, m, g, target=(15, 60): {"reward": 0.5})
    a.rl_refine({"RSI14": [(">=", 50.0)]}, "US", set(), iters=2)
    assert "RL correction: reward 0.5 → 0.5" in capsys.readouterr().out


# ---- recommend --------------------------------------------------------------------------


@pytest.fixture
def pipeline(monkeypatch):
    calls = {"rl": 0}
    disc = {
        "market": "US",
        "regime": {
            "tape": "Bull",
            "breadth": 0.6,
            "med_rsi": 55.0,
            "med_ret252": 12.0,
            "liquid_share": 0.9,
        },
        "cluster_size": 30,
        "cluster_overlap": 0.4,
        "known_good_n": 50,
        "rule": {"RSI14": [(">=", 50.0)]},
    }
    state = {
        "eval": {"reward": 1.5, "n": 20, "overlap": 0.6, "liq": 1.0, "ret": 15.0, "deviation": 0.2}
    }
    monkeypatch.setattr(a, "discover", lambda market: disc)
    monkeypatch.setattr(a, "known_good", lambda market: {"A"})
    monkeypatch.setattr(a, "evaluate", lambda rule, market, good: state["eval"])

    def rl(rule, market, good, verbose=True):
        calls["rl"] += 1
        return {"RSI14": [(">=", 42.0)]}, {
            "reward": 2.0,
            "n": 25,
            "overlap": 0.7,
            "liq": 1.0,
            "ret": 18.0,
            "deviation": 0.1,
        }

    monkeypatch.setattr(a, "rl_refine", rl)
    monkeypatch.setattr(
        a,
        "apply_rule",
        lambda rule, market, top=None: pd.DataFrame({"Symbol": ["P1", "P2"], "Close": [1.0, 2.0]}),
    )
    return types.SimpleNamespace(calls=calls, state=state, disc=disc)


def test_recommend_keeps_a_good_discovery_without_rl(pipeline, isolate):
    rec = a.recommend("US", verbose=False)
    assert rec["rl_correction_applied"] is False and pipeline.calls["rl"] == 0
    assert rec["final_rule"] == rec["discovered_rule"] == {"RSI14": [(">=", 50.0)]}
    assert (
        rec["metrics"]["reward"] == 1.5 and rec["picks"] == ["P1", "P2"] and rec["market"] == "US"
    )
    saved = json.loads((isolate / "US.json").read_text())
    assert saved["picks"] == ["P1", "P2"] and saved["rl_correction_applied"] is False


@pytest.mark.parametrize(
    "reward,deviation,triggers",
    [(0.59, 0.2, True), (0.6, 0.2, False), (1.5, 0.51, True), (1.5, 0.5, False)],
)
def test_recommend_triggers_rl_below_the_reward_floor_or_above_the_deviation_cap(
    pipeline, reward, deviation, triggers
):
    pipeline.state["eval"] = {**pipeline.state["eval"], "reward": reward, "deviation": deviation}
    rec = a.recommend("US", verbose=False)
    assert rec["rl_correction_applied"] is triggers and (pipeline.calls["rl"] == 1) is triggers
    if triggers:
        assert rec["final_rule"] == {"RSI14": [(">=", 42.0)]} and rec["metrics"]["reward"] == 2.0
        assert rec["discovered_rule"] == {"RSI14": [(">=", 50.0)]}  # the original is kept alongside


def test_recommend_min_reward_is_configurable(pipeline):
    assert a.recommend("US", min_reward=2.0, verbose=False)["rl_correction_applied"] is True


def test_recommend_passes_top_to_the_rule_application(pipeline, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        a, "apply_rule", lambda rule, market, top=None: seen.update(top=top) or pd.DataFrame()
    )
    rec = a.recommend("US", top=7, verbose=False)
    assert seen == {"top": 7} and rec["picks"] == []  # no picks -> an empty list, not an error


def test_recommend_prints_the_summary(pipeline, capsys):
    a.recommend("US")
    out = capsys.readouterr().out
    assert "auto-screener recommendation: US" in out and "regime: Bull" in out
    assert "discovered from cluster (30 names, 40% overlap w/ 50 known-good)" in out
    assert "RL correction applied: False" in out and "top picks" in out and "P1" in out


# ---- main -------------------------------------------------------------------------------


def test_main_passes_the_options_to_recommend(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        a,
        "recommend",
        lambda market, top=15, min_reward=0.6: seen.update(m=market, t=top, r=min_reward),
    )
    monkeypatch.setattr(
        sys, "argv", ["auto_screener.py", "--market", "US", "--top", "5", "--min-reward", "0.9"]
    )
    assert a.main() == 0 and seen == {"m": "US", "t": 5, "r": 0.9}
    monkeypatch.setattr(sys, "argv", ["auto_screener.py"])
    assert a.main() == 0 and seen == {"m": "IN", "t": 15, "r": 0.6}
