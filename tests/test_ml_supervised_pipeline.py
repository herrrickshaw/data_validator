"""Tests for ml_supervised.py beyond the label/feature basics: point-in-time features, the
walk-forward dataset, model construction, training orchestration and prediction.

Offline and dependency-free: scikit-learn, xgboost and joblib are small fakes injected into
sys.modules (so these run even where those libraries are not installed), the data layer is a
fake ``datalink.load_market`` and the model directory is redirected to tmp_path.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import ml_supervised as ms


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# ---- fakes ------------------------------------------------------------------------------


class FakeJoblib:
    """In-memory joblib: dump() also touches the file so existence checks behave."""

    def __init__(self):
        self.store = {}

    def dump(self, obj, path):
        self.store[str(path)] = obj
        open(path, "wb").close()

    def load(self, path):
        return self.store[str(path)]


class FakeModel:
    """Majority-class classifier with feature importances (enough for the training loop)."""

    def __init__(self):
        self.fitted_on = None

    def fit(self, X, y):
        self.fitted_on = (len(X), X.shape[1])
        self.majority = int(np.bincount(np.asarray(y, dtype=int)).argmax())

    def predict(self, X):
        return np.full(len(X), self.majority)

    @property
    def feature_importances_(self):
        n = self.fitted_on[1]
        return np.arange(1, n + 1, dtype=float) / np.arange(1, n + 1).sum()


class FakeTimeSeriesSplit:
    """sklearn's expanding-window split: the test blocks are the last n_splits equal chunks."""

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X):
        n = len(X)
        size = n // (self.n_splits + 1)
        for start in range(n - self.n_splits * size, n, size):
            yield np.arange(start), np.arange(start, start + size)


@pytest.fixture
def env(monkeypatch, tmp_path):
    jl = FakeJoblib()
    monkeypatch.setattr(ms, "MODEL_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "joblib", jl)
    monkeypatch.setitem(sys.modules, "sklearn", _module("sklearn"))
    monkeypatch.setitem(
        sys.modules,
        "sklearn.metrics",
        _module(
            "sklearn.metrics",
            accuracy_score=lambda y, p: float(np.mean(np.asarray(y) == np.asarray(p))),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sklearn.model_selection",
        _module("sklearn.model_selection", TimeSeriesSplit=FakeTimeSeriesSplit),
    )
    monkeypatch.setattr(ms, "LEV_FEATURES", [])
    monkeypatch.setattr(ms, "merge_leverage", lambda frame, market: frame)
    return types.SimpleNamespace(joblib=jl, dir=tmp_path)


def _series(n=300, growth=0.002, start=100.0, vol=1000.0):
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = start * (1 + growth) ** np.arange(n)
    return pd.DataFrame({"Close": close, "Volume": float(vol)}, index=idx)


def _load_market(monkeypatch, ltm=None, stm=None, calls=None):
    def load(market, tier="stm", **kw):
        if calls is not None:
            calls.append((market, tier))
        return (ltm if tier == "ltm" else stm) or {}

    monkeypatch.setattr(ms.datalink, "load_market", load)


# ---- _feature_list ----------------------------------------------------------------------


def test_feature_list_appends_only_leverage_features_that_are_present(monkeypatch):
    monkeypatch.setattr(ms, "LEV_FEATURES", ["lev_a", "lev_b"])
    assert ms._feature_list(["rsi14", "lev_b"]) == ms.FEATURES + ["lev_b"]
    assert ms._feature_list([]) == ms.FEATURES
    monkeypatch.setattr(ms, "LEV_FEATURES", [])
    assert ms._feature_list(["lev_a"]) == ms.FEATURES


# ---- _feat_at ---------------------------------------------------------------------------


def _arr(n=300, f=lambda i: 100.0 + i * 0.5, vol=1000.0):
    close = np.array([f(i) for i in range(n)], dtype=float)
    return close, np.full(n, float(vol))


def test_feat_at_needs_200_bars_and_a_positive_price():
    close, vol = _arr()
    assert ms._feat_at(close, vol, 199) is None
    assert ms._feat_at(close, vol, 200) is not None
    bad = close.copy()
    bad[250] = 0.0
    assert ms._feat_at(bad, vol, 250) is None
    bad[250] = -1.0
    assert ms._feat_at(bad, vol, 250) is None


def test_feat_at_returns_every_model_feature():
    close, vol = _arr()
    assert set(ms._feat_at(close, vol, 280)) == set(ms.FEATURES)


def test_feat_at_rsi_extremes_and_balance():
    up_close, vol = _arr(f=lambda i: 100.0 + i)  # only gains
    assert ms._feat_at(up_close, vol, 299)["rsi14"] == 100.0
    flat_close, _ = _arr(f=lambda i: 50.0)  # no losses at all
    assert ms._feat_at(flat_close, vol, 299)["rsi14"] == 100.0
    down_close, _ = _arr(f=lambda i: 500.0 - i)  # only losses
    assert ms._feat_at(down_close, vol, 299)["rsi14"] == pytest.approx(0.0)
    # 29 steps of +2 and 29 of -2 (plus one flat step) in the window: gains equal losses -> 50
    steps = np.array(([2.0, -2.0] * 29) + [0.0])
    close = np.concatenate([np.full(240, 100.0), 100.0 + np.cumsum(steps)])
    assert ms._feat_at(close, np.full(len(close), 1000.0), len(close) - 1)[
        "rsi14"
    ] == pytest.approx(50.0)


def test_feat_at_trend_flags_follow_the_moving_averages():
    vol = np.full(300, 1000.0)
    up = ms._feat_at(np.linspace(100, 300, 300), vol, 299)
    assert up["above_200dma"] == 1.0 and up["golden_cross"] == 1.0
    down = ms._feat_at(np.linspace(300, 100, 300), vol, 299)
    assert down["above_200dma"] == 0.0 and down["golden_cross"] == 0.0
    assert up["pct_from_high"] == pytest.approx(0.0) and down["pct_from_low"] == pytest.approx(0.0)
    assert down["pct_from_high"] < 0 < up["pct_from_low"]


def test_feat_at_volume_ratio_is_20_day_over_60_day_average():
    close = np.full(300, 100.0)
    vol = np.full(300, 1000.0)
    assert ms._feat_at(close, vol, 299)["vol_ratio"] == pytest.approx(1.0)
    vol[-20:] = 2000.0  # last 20 days double: v20 = 2000, v60 = (40*1000 + 20*2000) / 60
    assert ms._feat_at(close, vol, 299)["vol_ratio"] == pytest.approx(2000.0 / (80000.0 / 60.0))
    assert ms._feat_at(close, np.zeros(300), 299)["vol_ratio"] == 1.0  # no volume at all


def test_feat_at_log_turnover_and_return_signs():
    close = np.linspace(100, 300, 300)
    vol = np.full(300, 500.0)
    f = ms._feat_at(close, vol, 299)
    assert f["log_turnover"] == pytest.approx(np.log1p((close[-20:] * 500.0).mean()))
    assert f["ret21"] > 0 and f["ret63"] > 0 and f["ret126"] > 0  # a steady riser
    falling = ms._feat_at(np.linspace(300, 100, 300), vol, 299)
    assert falling["ret21"] < 0 and falling["ret63"] < 0 and falling["ret126"] < 0


def test_feat_at_uses_only_data_up_to_t():
    close, vol = _arr()
    base = ms._feat_at(close, vol, 250)
    changed = close.copy()
    changed[251:] = 1e9  # the future moves wildly
    assert ms._feat_at(changed, vol, 250) == base


# ---- build_dataset ----------------------------------------------------------------------


def test_build_dataset_labels_follow_the_forward_return(monkeypatch):
    # steady growth g per bar => the 63-bar forward return is (1+g)**63 - 1
    data = {
        "STRONG": _series(growth=0.002),  # +13.4%  >= 12  -> 3
        "BUY": _series(growth=0.001),  # +6.5%   >= 6   -> 2
        "HOLD": _series(growth=0.0),  # 0%     >= 0   -> 1
        "AVOID": _series(growth=-0.002),  # -11.9% <  0   -> 0
    }
    _load_market(monkeypatch, ltm=data)
    ds = ms.build_dataset("US", horizon=63)
    by_sym = ds.groupby("Symbol")["label"].agg(lambda s: set(s))
    assert by_sym["STRONG"] == {3} and by_sym["BUY"] == {2}
    assert by_sym["HOLD"] == {1} and by_sym["AVOID"] == {0}
    assert 1.002**63 - 1 > 0.12 > 1.001**63 - 1 > 0.06  # the thresholds the labels rely on


def test_build_dataset_looks_exactly_one_horizon_ahead(monkeypatch):
    # growth chosen so 63 bars just clears +12% (1.12 ** (63/62.5)) while 62 bars does not
    g = 1.12 ** (1 / 62.5) - 1
    assert (1 + g) ** 63 - 1 > 0.12 > (1 + g) ** 62 - 1
    _load_market(monkeypatch, ltm={"A": _series(growth=g)})
    assert set(ms.build_dataset("US", horizon=63, strong=12.0, buy=6.0)["label"]) == {3}
    assert set(ms.build_dataset("US", horizon=62, strong=12.0, buy=6.0)["label"]) == {
        2
    }  # 62 bars: below +12%


def test_build_dataset_thresholds_are_parameters(monkeypatch):
    _load_market(monkeypatch, ltm={"A": _series(growth=0.002)})  # +13.4%
    assert set(ms.build_dataset("US", horizon=63, strong=20.0, buy=10.0)["label"]) == {2}
    assert set(ms.build_dataset("US", horizon=63, strong=20.0, buy=15.0)["label"]) == {1}


def test_build_dataset_samples_asof_points_that_leave_room_for_the_horizon(monkeypatch):
    s = _series(n=300)
    _load_market(monkeypatch, ltm={"A": s})
    ds = ms.build_dataset("US", horizon=63, n_asof=8)
    assert len(ds) == 8 and ds["Symbol"].eq("A").all()
    assert ds["asof"].min() == s.index[200]  # earliest usable as-of
    assert ds["asof"].max() == s.index[300 - 63 - 1]  # latest with a full forward window
    assert ds["asof"].is_monotonic_increasing  # sorted by time (walk-forward order)
    assert set(ms.FEATURES) | {"Symbol", "asof", "label"} <= set(ds.columns)


def test_build_dataset_skips_symbols_without_enough_history(monkeypatch):
    # needs more than 200 + horizon bars, and a last as-of strictly after bar 200
    _load_market(
        monkeypatch,
        ltm={
            "OK": _series(n=300),
            "TOOSHORT": _series(n=250),
            "EDGE": _series(n=264),
            "NONE": None,
        },
    )
    assert set(ms.build_dataset("US", horizon=63)["Symbol"]) == {"OK"}


def test_build_dataset_drops_an_asof_point_with_a_non_positive_price(monkeypatch):
    s = _series(n=300)
    s.iloc[200, s.columns.get_loc("Close")] = 0.0  # the very first as-of bar is a bad print
    _load_market(monkeypatch, ltm={"A": s})
    ds = ms.build_dataset("US", horizon=63, n_asof=8)
    assert len(ds) == 7 and s.index[200] not in set(ds["asof"])


def test_build_dataset_empty_result_has_the_expected_columns(monkeypatch):
    _load_market(monkeypatch, ltm={}, stm={})
    ds = ms.build_dataset("US")
    assert ds.empty and list(ds.columns) == ms.FEATURES + ["Symbol", "asof", "label"]


def test_build_dataset_falls_back_to_the_stm_layer(monkeypatch):
    calls = []
    _load_market(monkeypatch, ltm={}, stm={"A": _series()}, calls=calls)
    assert set(ms.build_dataset("IN")["Symbol"]) == {"A"}
    assert calls == [("IN", "ltm"), ("IN", "stm")]


def test_build_dataset_merges_leverage_features(monkeypatch):
    seen = {}

    def merge(frame, market):
        seen["market"], seen["rows"] = market, len(frame)
        return frame.assign(lev_x=1.5)

    monkeypatch.setattr(ms, "merge_leverage", merge)
    _load_market(monkeypatch, ltm={"A": _series()})
    ds = ms.build_dataset("JP")
    assert seen["market"] == "JP" and seen["rows"] == len(ds) and (ds["lev_x"] == 1.5).all()


# ---- _make_model / _importances ---------------------------------------------------------


def test_make_model_prefers_xgboost_with_the_documented_settings(monkeypatch):
    class XGB:
        def __init__(self, **kw):
            self.kw = kw

    monkeypatch.setitem(sys.modules, "xgboost", _module("xgboost", XGBClassifier=XGB))
    m = ms._make_model()
    assert isinstance(m, XGB)
    assert m.kw["n_estimators"] == 300 and m.kw["max_depth"] == 5 and m.kw["learning_rate"] == 0.05
    assert (
        m.kw["subsample"] == 0.8
        and m.kw["colsample_bytree"] == 0.8
        and m.kw["eval_metric"] == "mlogloss"
    )


def test_make_model_falls_back_to_sklearn_when_xgboost_is_missing(monkeypatch):
    class HGB:
        def __init__(self, **kw):
            self.kw = kw

    monkeypatch.setitem(sys.modules, "xgboost", None)  # makes `import xgboost` raise ImportError
    monkeypatch.setitem(sys.modules, "sklearn", _module("sklearn"))
    monkeypatch.setitem(
        sys.modules,
        "sklearn.ensemble",
        _module("sklearn.ensemble", HistGradientBoostingClassifier=HGB),
    )
    m = ms._make_model()
    assert isinstance(m, HGB) and m.kw == {"max_depth": 5, "learning_rate": 0.05, "max_iter": 300}


def test_importances_use_the_model_attribute_or_fall_back_to_permutation(monkeypatch):
    X, y = np.zeros((4, 3)), np.zeros(4)

    class WithAttr:
        feature_importances_ = [0.2, 0.3, 0.5]

    assert ms._importances(WithAttr(), X, y).tolist() == [0.2, 0.3, 0.5]

    seen = {}

    def permutation_importance(model, X_, y_, n_repeats, random_state):
        seen.update(n_repeats=n_repeats, random_state=random_state)
        return types.SimpleNamespace(importances_mean=np.array([0.1, 0.0, 0.9]))

    monkeypatch.setitem(sys.modules, "sklearn", _module("sklearn"))
    monkeypatch.setitem(
        sys.modules,
        "sklearn.inspection",
        _module("sklearn.inspection", permutation_importance=permutation_importance),
    )
    assert ms._importances(object(), X, y).tolist() == [0.1, 0.0, 0.9]
    assert seen == {"n_repeats": 3, "random_state": 0}  # deterministic and cheap


# ---- train ------------------------------------------------------------------------------


def _dataset(n, labels, extra=None):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, len(ms.FEATURES))), columns=ms.FEATURES)
    df["Symbol"] = "S"
    df["asof"] = pd.date_range("2023-01-01", periods=n)
    df["label"] = labels
    for k, v in (extra or {}).items():
        df[k] = v
    return df


@pytest.fixture
def trainable(monkeypatch, env):
    calls = []

    def build(market, horizon=63, n_asof=8, strong=12.0, buy=6.0):
        calls.append((horizon, round(strong, 4), round(buy, 4)))
        return env.datasets.get(horizon, _dataset(0, []))

    env.datasets, env.calls = {}, calls
    monkeypatch.setattr(ms, "build_dataset", build)
    monkeypatch.setattr(ms, "_make_model", FakeModel)
    return env


def test_train_cross_validates_walk_forward_and_saves_the_bundle(trainable, capsys):
    # 240 samples: first 120 labelled 0, the rest 1. Expanding-window folds (test blocks of 40):
    # folds 1-2 predict 0 correctly, folds 3-5 miss => cv accuracy (1 + 1 + 0 + 0 + 0) / 5 = 0.4
    trainable.datasets[252] = _dataset(240, [0] * 120 + [1] * 120)
    out = ms.train("US")
    assert out["market"] == "US" and out["samples"] == 240
    assert out["cv_acc"] == pytest.approx(0.4)
    assert out["features"] == ms.FEATURES and out["leverage_used"] == []
    saved = trainable.joblib.store[str(trainable.dir / "US.pkl")]
    assert set(saved) == {"model", "features", "horizon"}
    assert saved["features"] == ms.FEATURES and saved["horizon"] == 252
    assert saved["model"].fitted_on == (240, len(ms.FEATURES))  # the final model sees all the data
    printed = capsys.readouterr().out
    assert "effective horizon: 252d" in printed and "trained US: 240 samples" in printed
    assert (
        "0.400" in printed and "top features: log_turnover" in printed
    )  # largest importance first
    assert "label mix: Avoid=120, Hold=120" in printed


def test_train_shortens_the_horizon_until_there_are_enough_samples(trainable, capsys):
    trainable.datasets[126] = _dataset(240, [1] * 240)  # 252 and 189 return nothing usable
    out = ms.train("US", verbose=False)
    assert trainable.calls == [
        (252, 30.0, 15.0),
        (189, 22.5, 11.25),
        (126, 15.0, 7.5),
    ]  # annual bars pro-rated
    saved = trainable.joblib.store[str(trainable.dir / "US.pkl")]
    assert saved["horizon"] == 126 and out["samples"] == 240 and out["cv_acc"] == 1.0


def test_train_never_tries_a_horizon_longer_than_requested(trainable):
    trainable.datasets[63] = _dataset(200, [2] * 200)  # exactly the 200-sample minimum
    ms.train("US", horizon=126, verbose=False)
    tried = [c[0] for c in trainable.calls]
    assert (
        max(tried) == 126 and tried[-1] == 63
    )  # nothing longer than asked for; fell through to 63
    assert 252 not in tried and 189 not in tried
    saved = trainable.joblib.store[str(trainable.dir / "US.pkl")]
    assert (
        saved["horizon"] == 63
    )  # 200 samples is enough, so the shorter horizon is the one adopted


def test_train_refuses_to_fit_on_too_few_samples(trainable):
    trainable.datasets[63] = _dataset(199, [1] * 199)
    with pytest.raises(RuntimeError, match=r"too few training samples for US \(199\)"):
        ms.train("US", verbose=False)
    assert trainable.joblib.store == {}  # nothing was saved


def test_train_reports_the_leverage_features_it_used(trainable, monkeypatch, capsys):
    monkeypatch.setattr(ms, "LEV_FEATURES", ["lev_a", "lev_b"])
    trainable.datasets[252] = _dataset(
        210, [1] * 210, extra={"lev_a": 0.5}
    )  # lev_b not in the frame
    out = ms.train("IN")
    assert out["leverage_used"] == ["lev_a"] and out["features"] == ms.FEATURES + ["lev_a"]
    assert "leverage importances: lev_a=" in capsys.readouterr().out


# ---- predict_current / known_good_supervised --------------------------------------------


class FakeProbModel:
    """Class probabilities driven by the first feature column; records what it was given."""

    def __init__(self):
        self.seen = None

    def predict_proba(self, X):
        self.seen = X
        p_buy = np.clip(X[:, 0] / 100.0, 0.0, 0.6)
        p_strong = np.where(X[:, 0] > 90, 0.3, 0.0)
        rest = (1.0 - p_buy - p_strong) / 2
        return np.column_stack([rest, rest, p_buy, p_strong])


def _bundle(model, features=None):
    return {"model": model, "features": features or ms.FEATURES, "horizon": 252}


def _save(env, market, bundle):
    env.joblib.dump(bundle, env.dir / f"{market}.pkl")


def test_predict_current_without_a_saved_model_is_empty(env, monkeypatch):
    _load_market(monkeypatch, ltm={"A": _series()})
    assert ms._load("US") is None
    assert ms.predict_current("US").empty


def test_predict_current_scores_ranks_and_names_each_symbol(env, monkeypatch):
    model = FakeProbModel()
    _save(env, "US", _bundle(model))
    _load_market(monkeypatch, ltm={"HI": _series(growth=0.004), "LO": _series(growth=-0.001)})
    out = ms.predict_current("US")
    assert set(out["Symbol"]) == {"HI", "LO"}
    assert out["score"].is_monotonic_decreasing  # best first
    for _, r in out.iterrows():  # score = P(Buy) + P(Strong Buy)
        assert r["score"] == pytest.approx(
            model.predict_proba(out[ms.FEATURES].to_numpy(float))[
                out.index[out["Symbol"] == r["Symbol"]][0], 2:
            ].sum()
        )
        assert r["pred_name"] == ms.CLASS_NAMES[int(r["pred_class"])]
    assert model.seen.dtype == np.float64 and model.seen.shape == (2, len(ms.FEATURES))


def test_predict_current_skips_symbols_with_under_200_bars_and_falls_back_to_stm(env, monkeypatch):
    _save(env, "US", _bundle(FakeProbModel()))
    calls = []
    _load_market(
        monkeypatch,
        ltm={},
        stm={"OK": _series(n=250), "SHORT": _series(n=199), "NONE": None},
        calls=calls,
    )
    assert list(ms.predict_current("US")["Symbol"]) == ["OK"]
    assert calls == [("US", "ltm"), ("US", "stm")]
    _load_market(monkeypatch, ltm={"SHORT": _series(n=100)})
    assert ms.predict_current("US").empty  # nothing usable


def test_predict_current_fills_features_the_frame_lacks_with_nan(env, monkeypatch):
    model = FakeProbModel()
    _save(env, "US", _bundle(model, features=ms.FEATURES + ["lev_missing"]))
    _load_market(monkeypatch, ltm={"A": _series()})
    out = ms.predict_current("US")
    assert model.seen.shape == (1, len(ms.FEATURES) + 1) and np.isnan(model.seen[0, -1])
    assert np.isnan(out["lev_missing"]).all()


def test_predict_current_defaults_to_the_technical_features_for_old_bundles(env, monkeypatch):
    model = FakeProbModel()
    _save(env, "US", {"model": model})  # no "features" key
    _load_market(monkeypatch, ltm={"A": _series()})
    ms.predict_current("US")
    assert model.seen.shape[1] == len(ms.FEATURES)


def test_known_good_supervised_keeps_names_at_or_above_the_score_cut(env, monkeypatch):
    class Fixed:
        def predict_proba(self, X):
            return np.tile([0.2, 0.2, 0.3, 0.3], (len(X), 1))  # score exactly 0.6 for everyone

    _save(env, "US", _bundle(Fixed()))
    _load_market(monkeypatch, ltm={"A": _series(), "B": _series(growth=0.001)})
    assert ms.known_good_supervised("US", min_score=0.6) == {"A", "B"}  # inclusive
    assert ms.known_good_supervised("US", min_score=0.61) == set()
    assert ms.known_good_supervised("JP") == set()  # no model for that market


# ---- main -------------------------------------------------------------------------------


def test_main_train_flag_trains_with_the_requested_horizon(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        ms, "train", lambda market, horizon=252, verbose=True: seen.update(market=market, h=horizon)
    )
    monkeypatch.setattr(
        sys, "argv", ["ml_supervised.py", "--market", "US", "--train", "--horizon", "126"]
    )
    assert ms.main() == 0 and seen == {"market": "US", "h": 126}


def test_main_prints_the_top_buys_or_a_hint(monkeypatch, capsys):
    monkeypatch.setattr(ms, "predict_current", lambda market: pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["ml_supervised.py"])
    assert ms.main() == 0 and "no model" in capsys.readouterr().out

    cur = pd.DataFrame(
        {
            "Symbol": ["GOOD", "MEH", "BEST"],
            "pred_name": ["Buy", "Hold", "Strong Buy"],
            "score": [0.7, 0.3, 0.9],
            "rsi14": [55.0, 50.0, 60.0],
            "ret63": [10.0, 1.0, 20.0],
            "pct_from_high": [-3.0, -9.0, -1.0],
        }
    )
    monkeypatch.setattr(ms, "predict_current", lambda market: cur)
    monkeypatch.setattr(sys, "argv", ["ml_supervised.py", "--market", "US", "--top", "1"])
    assert ms.main() == 0
    out = capsys.readouterr().out
    assert "Top 1 supervised Buy/Strong-Buy" in out and "GOOD" in out
    assert (
        "MEH" not in out and "BEST" not in out
    )  # score < 0.5 dropped; --top 1 keeps the first row only
