"""Tests for rl_trader.py beyond the basics: reward boundaries, state encoding, the trading
environment, the Q-learning agent, regime detection, training and decisions.

Offline and dependency-free: joblib is an in-memory fake, the data layer is a fake
``datalink.load_market``, optional collaborators (ml_supervised, auto_screener) are fake modules,
random draws are pinned where an exact Q-value is asserted, and the agent directory is tmp_path.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import rl_trader as rl


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class FakeJoblib:
    def __init__(self):
        self.store = {}

    def dump(self, obj, path):
        self.store[str(path)] = obj
        open(path, "wb").close()

    def load(self, path):
        return self.store[str(path)]


@pytest.fixture
def env(monkeypatch, tmp_path):
    jl = FakeJoblib()
    monkeypatch.setattr(rl, "AGENT_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "joblib", jl)
    np.random.seed(0)
    return types.SimpleNamespace(joblib=jl, dir=tmp_path)


def _state(**kw):
    s = {
        "score": 0.5,
        "regime": "Neutral",
        "days_held": 0,
        "pnl": 0.0,
        "exposure": 0.0,
        "drawdown_from_peak": 0.0,
        "in_position": 0,
        "sector_boom_phase": "mid",
    }
    s.update(kw)
    return s


# ---- reward_function: every branch and boundary -----------------------------------------


@pytest.mark.parametrize("action", ["HOLD", "ADD", "PARTIAL_EXIT", "FULL_EXIT", "STOP_LOSS"])
def test_position_actions_without_a_position_are_penalised(action):
    assert rl.reward_function(_state(in_position=0), action, {"pnl_pct": 0.5}) == -0.2


@pytest.mark.parametrize("action", ["ENTER_SMALL", "ENTER_FULL"])
def test_entering_while_already_holding_is_penalised(action):
    assert rl.reward_function(_state(in_position=1), action, {"pnl_pct": 0.5}) == -0.2


@pytest.mark.parametrize(
    "pnl,expected",
    [
        (0.25, 3.0),
        (0.2001, 3.0),
        (0.20, 1.5),  # thresholds are strict: exactly 20% is only the middle tier
        (0.15, 1.5),
        (0.10, 0.5),
        (0.05, 0.5),
        (0.0, -0.5),
        (-0.05, -0.5),
        (-0.10, -0.5),  # exactly -10% is not "below -10%"
        (-0.1001, -2.0),
        (-0.40, -2.0),
    ],
)
def test_entry_reward_tiers(pnl, expected):
    assert rl.reward_function(_state(), "ENTER_FULL", {"pnl_pct": pnl}) == expected
    assert rl.reward_function(_state(), "ENTER_SMALL", {"pnl_pct": pnl}) == expected


@pytest.mark.parametrize(
    "pnl,expected", [(0.05, 1.0), (-0.05, 1.0), (-0.0799, 1.0), (-0.08, -0.5), (-0.30, -0.5)]
)
def test_stop_loss_is_rewarded_only_when_it_cuts_a_small_loss_or_better(pnl, expected):
    assert rl.reward_function(_state(in_position=1), "STOP_LOSS", {"pnl_pct": pnl}) == expected


def test_partial_exit_is_rewarded_after_a_deep_drawdown():
    s = _state(in_position=1)
    assert (
        rl.reward_function({**s, "drawdown_from_peak": 0.16}, "PARTIAL_EXIT", {"pnl_pct": 0.0})
        == 1.5
    )
    assert (
        rl.reward_function({**s, "drawdown_from_peak": 0.15}, "PARTIAL_EXIT", {"pnl_pct": 0.0})
        == 0.0
    )
    assert (
        rl.reward_function({**s, "drawdown_from_peak": 0.30}, "HOLD", {"pnl_pct": 0.0}) == 0.0
    )  # only for partial exits


def test_early_boom_bonus_needs_a_gain_over_15_percent():
    s = _state(in_position=1, sector_boom_phase="early")
    assert rl.reward_function(s, "HOLD", {"pnl_pct": 0.16}) == 1.0
    assert rl.reward_function(s, "HOLD", {"pnl_pct": 0.15}) == 0.0
    assert rl.reward_function({**s, "sector_boom_phase": "mid"}, "HOLD", {"pnl_pct": 0.16}) == 0.0


def test_skipping_a_big_winner_is_penalised():
    assert rl.reward_function(_state(), "SKIP", {"pnl_pct": 0.0, "fwd_return": 0.21}) == -1.0
    assert rl.reward_function(_state(), "SKIP", {"pnl_pct": 0.0, "fwd_return": 0.20}) == 0.0
    assert rl.reward_function(_state(), "SKIP", {"pnl_pct": 0.0}) == 0.0  # fwd_return optional


def test_concentration_penalty_applies_above_70_percent_portfolio_exposure():
    # reward_function reads the key "portfolio_exposure" (see the xfail test below for the env)
    held = _state(in_position=1, portfolio_exposure=0.71)
    assert rl.reward_function(held, "ADD", {"pnl_pct": 0.0}) == -1.5
    assert rl.reward_function({**held, "portfolio_exposure": 0.70}, "ADD", {"pnl_pct": 0.0}) == 0.0
    assert rl.reward_function(held, "HOLD", {"pnl_pct": 0.0}) == 0.0  # only for adding risk
    crowded_entry = _state(portfolio_exposure=0.8)  # strong entry (+3.0) minus the penalty
    assert rl.reward_function(crowded_entry, "ENTER_FULL", {"pnl_pct": 0.25}) == pytest.approx(1.5)


@pytest.mark.xfail(
    strict=True,
    reason="known bug: the environment's state carries `exposure`, but reward_function reads "
    "`portfolio_exposure`, so the concentration penalty never fires in a real episode",
)
def test_concentration_penalty_reaches_the_agent_inside_an_episode():
    e = rl.StockEnv(np.array([100.0, 101.0, 102.0, 103.0]), 0, 0.5, "Neutral", horizon=3)
    e.reset()
    e.step("ENTER_SMALL")
    e.exposure = 0.95  # heavily concentrated
    _, reward, _ = e.step("ADD")  # flat price: no other reward terms apply
    assert reward == -1.5


# ---- state encoding ---------------------------------------------------------------------


def test_bucket_edges_are_left_inclusive_of_the_next_bucket():
    assert [rl._bucket(v, [0.33, 0.66]) for v in (0.0, 0.32, 0.33, 0.65, 0.66, 1.0)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert [rl._bucket(v, [1, 5, 20]) for v in (0, 1, 4, 5, 19, 20, 99)] == [0, 1, 1, 2, 2, 3, 3]


def test_encode_state_maps_each_component_to_its_bucket():
    s = _state(
        score=0.7,
        regime="Bull",
        days_held=6,
        pnl=0.12,
        exposure=0.5,
        drawdown_from_peak=0.09,
        in_position=1,
    )
    assert rl.encode_state(s) == (2, 2, 2, 3, 1, 1, 1)
    assert (
        rl.encode_state(_state(days_held=19))[2] == 2
        and rl.encode_state(_state(days_held=20))[2] == 3
    )
    low = _state(
        score=0.1,
        regime="Bear",
        days_held=0,
        pnl=-0.2,
        exposure=0.0,
        drawdown_from_peak=0.0,
        in_position=0,
    )
    assert rl.encode_state(low) == (0, 0, 0, 0, 0, 0, 0)
    assert rl.encode_state(_state(regime="Sideways"))[1] == 1  # unknown regime is Neutral
    assert (
        rl.encode_state({k: v for k, v in _state().items() if k != "in_position"})[-1] == 0
    )  # optional


def test_encoded_states_always_fit_the_declared_table_shape():
    extremes = [
        _state(
            score=s, regime=r, days_held=d, pnl=p, exposure=e, drawdown_from_peak=dd, in_position=i
        )
        for s in (0.0, 1.0)
        for r in ("Bear", "Neutral", "Bull")
        for d in (0, 1000)
        for p in (-1.0, 1.0)
        for e in (0.0, 1.0)
        for dd in (0.0, 1.0)
        for i in (0, 1)
    ]
    for st in extremes:
        key = rl.encode_state(st)
        assert len(key) == len(rl.N_STATE_DIMS)
        assert all(0 <= k < n for k, n in zip(key, rl.N_STATE_DIMS))


# ---- StockEnv ---------------------------------------------------------------------------


def test_env_reset_starts_flat_at_t0():
    e = rl.StockEnv(
        np.array([100.0, 101.0, 102.0, 103.0]), t0=1, score=0.8, regime="Bull", horizon=2
    )
    s = e.reset()
    assert e.t == 1 and e.entry is None
    assert s["score"] == 0.8 and s["regime"] == "Bull" and s["in_position"] == 0
    assert s["sector_boom_phase"] == "early"  # only a Bull regime counts as an early boom
    assert rl.StockEnv(np.ones(4), 0, 0.5, "Neutral").reset()["sector_boom_phase"] == "mid"


@pytest.mark.parametrize(
    "action,exposure", [("ENTER_SMALL", 0.35), ("ENTER_FULL", 0.7), ("ADD", 0.7)]
)
def test_env_entering_sets_exposure_and_tracks_pnl_and_holding_time(action, exposure):
    close = np.array([100.0, 110.0, 121.0, 100.0, 100.0, 100.0])
    e = rl.StockEnv(close, t0=0, score=0.5, regime="Neutral", horizon=5)
    e.reset()
    s, r, done = e.step(action)  # buy at 100, step to t=1 (110)
    assert e.entry == 100.0 and e.exposure == exposure and e.days_held == 1 and not done
    assert s["pnl"] == pytest.approx(0.10) and s["in_position"] == 1
    s, r, done = e.step("HOLD")  # t=2 (121)
    assert s["pnl"] == pytest.approx(0.21) and e.days_held == 2


def test_env_a_second_entry_does_not_reset_the_position():
    e = rl.StockEnv(np.array([100.0, 120.0, 130.0, 130.0]), 0, 0.5, "Neutral", horizon=3)
    e.reset()
    e.step("ENTER_SMALL")
    entry = e.entry
    e.step("ENTER_FULL")  # already in: no-op (and penalised by the reward function)
    assert e.entry == entry == 100.0 and e.exposure == 0.35


def test_env_tracks_drawdown_from_the_peak_gain():
    close = np.array([100.0, 120.0, 130.0, 104.0, 104.0, 104.0])
    e = rl.StockEnv(close, 0, 0.5, "Neutral", horizon=5)
    e.reset()
    e.step("ENTER_FULL")  # t=1: +20%
    e.step("HOLD")  # t=2: +30% (the peak)
    s, _, _ = e.step("HOLD")  # t=3: +4%
    assert s["pnl"] == pytest.approx(0.04)
    assert s["drawdown_from_peak"] == pytest.approx(0.30 - 0.04)


@pytest.mark.parametrize("exit_action", ["FULL_EXIT", "STOP_LOSS"])
def test_env_ends_the_episode_on_a_full_exit_or_stop(exit_action):
    e = rl.StockEnv(np.array([100.0, 101.0, 102.0, 103.0, 104.0]), 0, 0.5, "Neutral", horizon=4)
    e.reset()
    e.step("ENTER_FULL")
    _, _, done = e.step(exit_action)
    assert done is True


def test_env_never_walks_past_the_horizon_or_the_data_and_then_reports_done():
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    e = rl.StockEnv(close, t0=1, score=0.5, regime="Neutral", horizon=2)  # may run until t=3
    e.reset()
    done, steps = False, 0
    while not done and steps < 20:
        _, _, done = e.step("SKIP")
        steps += 1
        assert e.t <= 3
    assert done and e.t == 3
    short = rl.StockEnv(
        np.array([100.0, 101.0]), 0, 0.5, "Neutral", horizon=63
    )  # data shorter than horizon
    short.reset()
    while True:
        _, _, d = short.step("SKIP")
        assert short.t <= 1
        if d:
            break


def test_env_step_rewards_use_the_pre_action_state_and_the_realised_outcome():
    close = np.array([100.0, 130.0, 130.0, 130.0])
    e = rl.StockEnv(close, 0, 0.5, "Neutral", horizon=3)
    e.reset()
    _, r, _ = e.step("ENTER_FULL")  # +30% on the next bar -> top entry tier
    assert r == 3.0
    e2 = rl.StockEnv(close, 0, 0.5, "Neutral", horizon=3)
    e2.reset()
    _, r2, _ = e2.step("STOP_LOSS")  # no position -> the guardrail penalty, not a free reward
    assert r2 == -0.2


# ---- QAgent -----------------------------------------------------------------------------


def test_q_table_rows_are_created_on_demand_as_zeros():
    agent = rl.QAgent()
    row = agent.q((1, 2, 3))
    assert row.tolist() == [0.0] * len(rl.ACTIONS) and agent.q((1, 2, 3)) is row


def test_act_exploits_the_best_known_action_unless_exploring(monkeypatch):
    agent = rl.QAgent(eps=0.0)
    st = _state()
    agent.q(rl.encode_state(st))[rl.A["ENTER_FULL"]] = 2.0
    assert agent.act(st) == "ENTER_FULL"
    assert (
        agent.act(_state(score=0.9, regime="Bull")) == "SKIP"
    )  # unseen state: all zeros -> first action

    explorer = rl.QAgent(eps=1.0)
    monkeypatch.setattr(np.random, "random", lambda: 0.0)  # always below eps -> explore
    monkeypatch.setattr(np.random, "randint", lambda n: 3)
    assert explorer.act(st) == rl.ACTIONS[3]
    assert explorer.act(st, explore=False) == "SKIP"  # exploration can be switched off


def test_remember_stores_encoded_states_and_the_action_index():
    agent = rl.QAgent()
    s, s2 = _state(score=0.9), _state(score=0.1, in_position=1)
    agent.remember(s, "ENTER_FULL", 1.25, s2, True)
    assert agent.buffer == [
        (rl.encode_state(s), rl.A["ENTER_FULL"], 1.25, rl.encode_state(s2), True)
    ]


def test_replay_applies_the_q_learning_update(monkeypatch):
    agent = rl.QAgent(alpha=0.3, gamma=0.95)
    s, s2 = _state(), _state(in_position=1)
    k, k2 = rl.encode_state(s), rl.encode_state(s2)
    monkeypatch.setattr(np.random, "randint", lambda lo, hi, size: np.zeros(size, dtype=int))
    assert agent.replay() is None  # empty buffer: nothing to do

    agent.q(k2)[rl.A["HOLD"]] = 2.0  # a valuable next state, so ignoring `done` would show
    agent.remember(s, "ENTER_FULL", 1.0, s2, True)  # terminal: target = reward only
    agent.replay()
    assert agent.q(k)[rl.A["ENTER_FULL"]] == pytest.approx(0.3 * 1.0)

    agent.buffer.clear()
    agent.remember(s, "ENTER_SMALL", 1.0, s2, False)  # non-terminal: target = 1 + 0.95 * 2 = 2.9
    agent.replay()
    assert agent.q(k)[rl.A["ENTER_SMALL"]] == pytest.approx(0.3 * 2.9)


def test_replay_samples_at_most_a_batch(monkeypatch):
    agent = rl.QAgent()
    sizes = []
    monkeypatch.setattr(
        np.random,
        "randint",
        lambda lo, hi, size: sizes.append((hi, size)) or np.zeros(size, dtype=int),
    )
    for _ in range(5):
        agent.remember(_state(), "SKIP", 0.0, _state(), True)
    agent.replay(batch=3)
    agent.replay(batch=256)
    assert sizes == [(5, 3), (5, 5)]  # min(batch, buffer length)


# ---- _regime_at -------------------------------------------------------------------------


def test_regime_is_neutral_until_there_are_200_bars():
    assert rl._regime_at(np.linspace(100, 300, 400), 199) == "Neutral"


def test_regime_reads_trend_against_the_200_day_average_and_63_day_return():
    flat = np.full(300, 100.0)
    bull, bear, meh = flat.copy(), flat.copy(), flat.copy()
    bull[-1], bear[-1], meh[-1] = 120.0, 80.0, 102.0
    assert rl._regime_at(bull, 299) == "Bull"  # above the 200-day mean and +20% over 63 bars
    assert rl._regime_at(bear, 299) == "Bear"  # below it and -20%
    assert rl._regime_at(meh, 299) == "Neutral"  # above the mean but only +2%
    below_but_flat = flat.copy()
    below_but_flat[-1] = 98.0
    assert rl._regime_at(below_but_flat, 299) == "Neutral"  # below the mean but only -2%


# ---- train ------------------------------------------------------------------------------


def _data(n_symbols=12, n=300, start=100.0):
    idx = pd.bdate_range("2023-01-02", periods=n)
    out = {}
    for i in range(n_symbols):
        close = start * (1 + 0.001 * (i % 3 - 1)) ** np.arange(n) + np.sin(np.arange(n) / 7.0)
        out[f"S{i:02d}"] = pd.DataFrame({"Close": close, "Volume": 1000.0}, index=idx)
    return out


def _load(monkeypatch, ltm=None, stm=None, calls=None):
    def load(market, tier="stm", **kw):
        if calls is not None:
            calls.append((market, tier))
        return (ltm if tier == "ltm" else stm) or {}

    monkeypatch.setattr(rl.datalink, "load_market", load)


def test_train_learns_a_q_table_and_saves_it(env, monkeypatch, capsys):
    _load(monkeypatch, ltm=_data())
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module("ml_supervised", predict_current=lambda m: pd.DataFrame()),
    )
    out = rl.train("US", episodes=40, horizon=20)
    assert out["market"] == "US" and out["episodes"] == 40 and out["states"] > 0
    saved = env.joblib.store[str(env.dir / "US_rl.pkl")]
    assert saved["actions"] == rl.ACTIONS and len(saved["Q"]) == out["states"]
    assert all(len(v) == len(rl.ACTIONS) for v in saved["Q"].values())
    printed = capsys.readouterr().out
    assert "trained RL US: 40 episodes" in printed and "avg reward/ep" in printed


def test_train_refuses_without_enough_symbols_with_enough_history(env, monkeypatch):
    # a symbol needs MORE than 202 + horizon bars: 222 bars is not enough for horizon 20
    data = _data(n_symbols=9, n=300)
    data.update({"X1": _data(1, n=222)["S00"], "X2": None})
    _load(monkeypatch, ltm=data)
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module("ml_supervised", predict_current=lambda m: pd.DataFrame()),
    )
    with pytest.raises(RuntimeError, match="too little history to train RL for US"):
        rl.train("US", episodes=5, horizon=20)
    assert env.joblib.store == {}


def test_train_falls_back_to_the_stm_layer(env, monkeypatch):
    calls = []
    _load(monkeypatch, ltm={}, stm=_data(), calls=calls)
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module("ml_supervised", predict_current=lambda m: pd.DataFrame()),
    )
    assert rl.train("IN", episodes=5, horizon=20, verbose=False)["states"] > 0
    assert calls[:2] == [("IN", "ltm"), ("IN", "stm")]


def test_train_anchors_each_episode_on_the_supervised_score_when_available(env, monkeypatch):
    data = _data()
    _load(monkeypatch, ltm=data)
    scores = pd.DataFrame({"Symbol": ["S00", "S01"], "score": [0.91, 0.12]})
    monkeypatch.setitem(
        sys.modules, "ml_supervised", _module("ml_supervised", predict_current=lambda m: scores)
    )
    seen = {}

    class Recording(rl.StockEnv):
        def __init__(self, close, t0, score, regime, horizon=63, stop=-0.08):
            seen.setdefault("scores", []).append(score)
            super().__init__(close, t0, score, regime, horizon, stop)

    monkeypatch.setattr(rl, "StockEnv", Recording)
    rl.train("US", episodes=60, horizon=20, verbose=False)
    assert set(seen["scores"]) <= {
        0.91,
        0.12,
        0.5,
    }  # scored names use their score, the rest default to 0.5
    assert 0.5 in seen["scores"]


def test_train_survives_a_failing_supervised_model(env, monkeypatch):
    def boom(market):
        raise RuntimeError("no model")

    _load(monkeypatch, ltm=_data())
    monkeypatch.setitem(
        sys.modules, "ml_supervised", _module("ml_supervised", predict_current=boom)
    )
    assert rl.train("US", episodes=5, horizon=20, verbose=False)["states"] > 0


def test_train_decays_exploration_but_never_below_the_floor(env, monkeypatch):
    _load(monkeypatch, ltm=_data())
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module("ml_supervised", predict_current=lambda m: pd.DataFrame()),
    )
    made = []
    real = rl.QAgent

    class Spy(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(rl, "QAgent", Spy)
    rl.train("US", episodes=30, horizon=20, verbose=False)
    assert made[0].eps == pytest.approx(max(0.02, 0.2 * 0.9995**30))  # decays 0.05% per episode
    assert made[0].eps < 0.2


# ---- decide -----------------------------------------------------------------------------


def _agent_with(env, market, preferred):
    """Saves a Q table that makes `preferred` the greedy action for the (flat, unheld) states."""
    Q = {}
    for regime in ("Bear", "Neutral", "Bull"):
        for score in (0.1, 0.5, 0.9):
            key = rl.encode_state(_state(score=score, regime=regime))
            Q[key] = np.zeros(len(rl.ACTIONS))
            Q[key][rl.A[preferred]] = 1.0
    env.joblib.dump({"Q": Q, "actions": rl.ACTIONS}, env.dir / f"{market}_rl.pkl")


def test_decide_without_a_trained_agent_is_empty_and_says_so(env, capsys):
    assert rl.decide("US", shortlist=["A"]).empty
    assert "no RL agent" in capsys.readouterr().out
    assert rl.decide("US", shortlist=["A"], verbose=False).empty


def test_decide_acts_greedily_on_each_shortlisted_name(env, monkeypatch, capsys):
    _agent_with(env, "US", "ENTER_FULL")
    _load(monkeypatch, ltm=_data())
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module(
            "ml_supervised",
            predict_current=lambda m: pd.DataFrame({"Symbol": ["S00"], "score": [0.9123456]}),
        ),
    )
    out = rl.decide("US", shortlist=["S00", "S01"])
    assert out["Symbol"].tolist() == ["S00", "S01"] and set(out["Action"]) == {"ENTER_FULL"}
    assert out.set_index("Symbol").loc["S00", "Score"] == 0.912  # rounded to 3 places
    assert out.set_index("Symbol").loc["S01", "Score"] == 0.5  # no supervised score -> default
    assert set(out["Regime"]) <= {"Bear", "Neutral", "Bull"}
    assert "RL trade decisions" in capsys.readouterr().out


def test_decide_skips_names_without_enough_history(env, monkeypatch):
    _agent_with(env, "US", "ENTER_SMALL")
    data = _data(2)
    data["SHORT"] = _data(1, n=200)["S00"]  # needs at least 201 bars
    _load(monkeypatch, ltm=data)
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module("ml_supervised", predict_current=lambda m: pd.DataFrame()),
    )
    out = rl.decide("US", shortlist=["S00", "SHORT", "MISSING"], verbose=False)
    assert out["Symbol"].tolist() == ["S00"]


def test_decide_unseen_states_default_to_skip(env, monkeypatch):
    env.joblib.dump(
        {"Q": {}, "actions": rl.ACTIONS}, env.dir / "US_rl.pkl"
    )  # an agent that learned nothing
    _load(monkeypatch, ltm=_data())
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module("ml_supervised", predict_current=lambda m: pd.DataFrame()),
    )
    assert rl.decide("US", shortlist=["S00"], verbose=False)["Action"].tolist() == ["SKIP"]


def test_decide_uses_the_auto_screener_shortlist_when_none_is_given(env, monkeypatch):
    _agent_with(env, "US", "HOLD")
    _load(monkeypatch, ltm=_data())
    seen = {}

    def recommend(market, verbose=True):
        seen.update(market=market, verbose=verbose)
        return {"picks": ["S02"]}

    monkeypatch.setitem(sys.modules, "auto_screener", _module("auto_screener", recommend=recommend))
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module("ml_supervised", predict_current=lambda m: pd.DataFrame()),
    )
    assert rl.decide("US", verbose=False)["Symbol"].tolist() == ["S02"]
    assert seen == {"market": "US", "verbose": False}


def test_decide_with_no_shortlist_and_a_failing_screener_is_empty(env, monkeypatch):
    _agent_with(env, "US", "HOLD")
    _load(monkeypatch, ltm=_data())

    def boom(market, verbose=True):
        raise RuntimeError("screener down")

    monkeypatch.setitem(sys.modules, "auto_screener", _module("auto_screener", recommend=boom))
    monkeypatch.setitem(
        sys.modules,
        "ml_supervised",
        _module("ml_supervised", predict_current=lambda m: pd.DataFrame()),
    )
    assert rl.decide("US", verbose=False).empty


# ---- main -------------------------------------------------------------------------------


def test_main_trains_or_decides_and_explains_the_ppo_flag(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        rl, "train", lambda market, episodes=4000: calls.append(("train", market, episodes))
    )
    monkeypatch.setattr(rl, "decide", lambda market: calls.append(("decide", market)))
    monkeypatch.setattr(
        sys, "argv", ["rl_trader.py", "--market", "US", "--train", "--episodes", "10"]
    )
    assert rl.main() == 0 and calls == [("train", "US", 10)]
    monkeypatch.setattr(sys, "argv", ["rl_trader.py", "--market", "JP"])
    assert rl.main() == 0 and calls[-1] == ("decide", "JP")
    monkeypatch.setattr(sys, "argv", ["rl_trader.py", "--ppo"])
    assert rl.main() == 0
    assert "stable_baselines3" in capsys.readouterr().out and calls[-1] == ("decide", "IN")
