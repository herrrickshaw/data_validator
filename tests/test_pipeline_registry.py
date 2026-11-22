"""Tests for pipeline.py beyond the basic promote/demote/sync cases: registry file handling, the
filter set that drives promotion, LTM trimming on demotion, sync bookkeeping, the refresh step,
status and the CLI.

Offline and sandboxed: the registry file points into tmp_path and every collaborator
(screen_metrics, auto_screener, validation, market_memory, fundamental_metrics) is a fake module.
"""

import datetime as dt
import json
import sys
import types

import pandas as pd
import pytest

import pipeline as p


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _install(monkeypatch, name, **attrs):
    mod = _module(name, **attrs)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture(autouse=True)
def registry(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "STATE", tmp_path / "pipeline_state.json")
    return tmp_path / "pipeline_state.json"


# ---- registry file ----------------------------------------------------------------------


def test_load_returns_empty_for_a_missing_or_corrupt_registry(registry):
    assert p._load() == {}
    registry.write_text("{ this is not json")
    assert p._load() == {}  # a damaged file must not crash the pipeline
    registry.write_text(json.dumps({"US": {"promoted": {"A": {}}}}))
    assert p._load() == {"US": {"promoted": {"A": {}}}}


def test_save_round_trips_and_stringifies_unserialisable_values(registry):
    p._save({"US": {"synced": dt.date(2025, 1, 2), "promoted": {}}})
    assert json.loads(registry.read_text()) == {"US": {"synced": "2025-01-02", "promoted": {}}}


def test_mkt_creates_the_entry_once():
    state = {}
    m = p._mkt(state, "US")
    assert m == {"promoted": {}, "synced": None} and state["US"] is m
    m["promoted"]["A"] = {}
    assert p._mkt(state, "US")["promoted"] == {"A": {}}  # an existing entry is returned untouched


def test_today_is_an_iso_date():
    assert p._today() == dt.date.today().isoformat()


def test_promoted_reads_the_registry_per_market():
    p.promote("US", {"A": ["f1"]}, verbose=False)
    assert list(p.promoted("US")) == ["A"] and p.promoted("JP") == {}


def test_repromoting_keeps_the_original_since_and_refreshed_dates(registry):
    registry.write_text(
        json.dumps(
            {
                "US": {
                    "promoted": {
                        "A": {"filters": ["f"], "since": "2020-01-01", "refreshed": "2021-02-03"}
                    },
                    "synced": None,
                }
            }
        )
    )
    assert p.promote("US", {"A": ["g", "h"]}, verbose=False) == 0
    assert p.promoted("US")["A"] == {
        "filters": ["g", "h"],
        "since": "2020-01-01",
        "refreshed": "2021-02-03",
    }


def test_promote_reports_new_names_and_the_watchlist_size(capsys):
    assert p.promote("US", {"A": ["f"], "B": ["f"]}) == 2
    assert "promote[US]: +2 new, 2 on watchlist" in capsys.readouterr().out
    assert (
        p.promote("US", {"A": ["f", "g"], "C": ["f"]}) == 1
    )  # A already tracked: filters refreshed, not new
    assert "promote[US]: +1 new, 3 on watchlist" in capsys.readouterr().out
    assert p.promoted("US")["A"]["filters"] == ["f", "g"]


# ---- filter_clearers --------------------------------------------------------------------

SCREENS = [
    "golden_crossover",
    "companies_creating_new_high",
    "price_volume_action",
    "breakout_stocks",
    "multibagger_momentum",
]


def _sources(monkeypatch, per_screen=None, fundamental=None, popular=None, calls=None):
    per_screen = per_screen or {}

    def run_screen(name, market, min_turnover_usd=None):
        if calls is not None:
            calls.append((name, market, min_turnover_usd))
        return pd.DataFrame({"Symbol": per_screen.get(name, [])})

    _install(monkeypatch, "screen_metrics", run_screen=run_screen)
    _install(monkeypatch, "auto_screener", fundamental_anchor=lambda m: set(fundamental or []))
    _install(monkeypatch, "validation", known_universe=lambda m: set(popular or []))


def test_filter_clearers_unions_every_source_with_sorted_filter_names(monkeypatch):
    calls = []
    _sources(
        monkeypatch,
        per_screen={
            "golden_crossover": ["A", "B"],
            "companies_creating_new_high": ["A"],
            "breakout_stocks": ["A"],
        },
        fundamental=["B", "C"],
        popular=["C"],
        calls=calls,
    )
    out = p.filter_clearers("US", min_turnover_usd=2_500_000)
    assert out == {
        "A": ["breakout_stocks", "companies_creating_new_high", "golden_crossover"],
        "B": ["fundamental", "golden_crossover"],
        "C": ["fundamental", "popular"],
    }
    assert [c[0] for c in calls] == SCREENS  # exactly the five cheap price screens, in order
    assert all(c[1:] == ("US", 2_500_000) for c in calls)  # the turnover floor reaches every screen


def test_filter_clearers_default_turnover_floor_is_one_million_usd(monkeypatch):
    calls = []
    _sources(monkeypatch, calls=calls)
    p.filter_clearers("US")
    assert calls and all(c[2] == 1_000_000 for c in calls)


def test_filter_clearers_ignores_screens_that_return_nothing(monkeypatch):
    _sources(monkeypatch)
    assert p.filter_clearers("US") == {}


def test_filter_clearers_keeps_the_other_sources_when_one_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("source down")

    _install(monkeypatch, "screen_metrics", run_screen=boom)
    _install(monkeypatch, "auto_screener", fundamental_anchor=lambda m: {"F"})
    _install(monkeypatch, "validation", known_universe=lambda m: {"P"})
    assert p.filter_clearers("US") == {"F": ["fundamental"], "P": ["popular"]}

    _install(monkeypatch, "auto_screener", fundamental_anchor=boom)
    assert p.filter_clearers("US") == {"P": ["popular"]}
    _install(monkeypatch, "validation", known_universe=boom)
    assert p.filter_clearers("US") == {}


# ---- demote(trim_ltm) -------------------------------------------------------------------


def test_demote_can_free_the_deep_history_of_demoted_names(monkeypatch, capsys):
    p.promote("US", {"A": ["f"], "B": ["f"], "C": ["f"]}, verbose=False)
    dropped = []
    _install(
        monkeypatch,
        "market_memory",
        drop=lambda market, syms, verbose=True: dropped.append((market, list(syms), verbose)),
    )
    assert p.demote("US", ["A", "B", "UNKNOWN"], trim_ltm=True) == 2
    assert dropped == [("US", ["A", "B"], False)]  # only names that were actually on the watchlist
    assert list(p.promoted("US")) == ["C"]
    assert "demote[US]: -2, 1 remain" in capsys.readouterr().out


def test_demote_does_not_touch_the_ltm_unless_asked_or_nothing_was_removed(monkeypatch):
    p.promote("US", {"A": ["f"]}, verbose=False)
    dropped = []
    _install(monkeypatch, "market_memory", drop=lambda *a, **k: dropped.append(a))
    p.demote("US", ["A"], verbose=False)  # trim_ltm defaults to False
    p.demote("US", ["NOT_THERE"], trim_ltm=True, verbose=False)  # nothing to remove
    assert dropped == []


def test_demote_survives_a_failing_ltm_trim(monkeypatch):
    p.promote("US", {"A": ["f"]}, verbose=False)

    def boom(*a, **k):
        raise RuntimeError("disk error")

    _install(monkeypatch, "market_memory", drop=boom)
    assert p.demote("US", ["A"], trim_ltm=True, verbose=False) == 1
    assert p.promoted("US") == {}  # the registry change still stands


# ---- sync -------------------------------------------------------------------------------


def test_sync_records_when_it_ran_and_reports_the_diff(monkeypatch, capsys):
    p.promote("US", {"OLD": ["a", "b"], "KEEP": ["a", "b"]}, verbose=False)
    monkeypatch.setattr(
        p,
        "filter_clearers",
        lambda market, min_turnover_usd: {"KEEP": ["a", "b"], "NEW": ["a", "c"], "WEAK": ["a"]},
    )
    out = p.sync("US")
    assert out == {
        "market": "US",
        "watchlist": 2,
        "promoted": 1,
        "demoted": 1,
    }  # WEAK clears only 1 filter
    assert sorted(p.promoted("US")) == ["KEEP", "NEW"]
    synced = p._load()["US"]["synced"]
    assert (
        dt.datetime.fromisoformat(synced).date() == dt.date.today() and "." not in synced
    )  # whole seconds
    assert "sync[US]: watchlist 2 (+1 promoted, -1 demoted)" in capsys.readouterr().out


def test_sync_passes_the_turnover_floor_and_honours_min_filters(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        p,
        "filter_clearers",
        lambda market, min_turnover_usd: seen.update(floor=min_turnover_usd)
        or {"A": ["x"], "B": ["x", "y"]},
    )
    assert p.sync("US", min_turnover_usd=5e6, min_filters=1, verbose=False)["watchlist"] == 2
    assert seen["floor"] == 5e6
    assert (
        p.sync("US", min_filters=3, verbose=False)["watchlist"] == 0
    )  # a stricter conviction gate demotes all
    assert p.promoted("US") == {}


def test_sync_leaves_the_ltm_alone(monkeypatch):
    p.promote("US", {"GONE": ["a", "b"]}, verbose=False)
    dropped = []
    _install(monkeypatch, "market_memory", drop=lambda *a, **k: dropped.append(a))
    monkeypatch.setattr(p, "filter_clearers", lambda market, min_turnover_usd: {})
    assert p.sync("US", verbose=False)["demoted"] == 1 and dropped == []


# ---- refresh ----------------------------------------------------------------------------


def test_refresh_with_an_empty_watchlist_only_reports(monkeypatch, capsys):
    called = []
    _install(monkeypatch, "fundamental_metrics", load_fundamentals=lambda *a, **k: called.append(1))
    out = p.refresh("US")
    assert out == {"market": "US", "promoted": 0, "fundamentals_fetched": 0} and called == []
    assert "refresh[US]: 0 promoted stocks" in capsys.readouterr().out


def test_refresh_fetches_fundamentals_for_promoted_names_and_stamps_them(monkeypatch, capsys):
    p.promote("US", {"A": ["f"], "B": ["f"]}, verbose=False)
    seen = {}
    counts = iter([10, 14])
    monkeypatch.setattr(p, "_fund_count", lambda market: next(counts))
    _install(
        monkeypatch,
        "fundamental_metrics",
        load_fundamentals=lambda market, **kw: seen.update(market=market, **kw),
    )
    out = p.refresh("US", fund_limit=25)
    assert out == {
        "market": "US",
        "promoted": 2,
        "fundamentals_fetched": 4,
    }  # cache grew from 10 to 14
    assert seen == {"market": "US", "limit": 25, "verbose": False, "symbols": ["A", "B"]}
    assert {v["refreshed"] for v in p.promoted("US").values()} == {dt.date.today().isoformat()}
    assert "fundamentals now cover +4 promoted names" in capsys.readouterr().out


def test_refresh_can_skip_the_fundamentals_fetch(monkeypatch):
    p.promote("US", {"A": ["f"]}, verbose=False)
    called = []
    _install(monkeypatch, "fundamental_metrics", load_fundamentals=lambda *a, **k: called.append(1))
    out = p.refresh("US", fetch_fundamentals=False, verbose=False)
    assert called == [] and out["fundamentals_fetched"] == 0
    assert p.promoted("US")["A"]["refreshed"] == dt.date.today().isoformat()  # still stamped


def test_refresh_survives_a_failing_fundamentals_fetch(monkeypatch, capsys):
    p.promote("US", {"A": ["f"]}, verbose=False)

    def boom(*a, **k):
        raise RuntimeError("SEC unreachable")

    _install(monkeypatch, "fundamental_metrics", load_fundamentals=boom, CACHE=None)
    out = p.refresh("US")
    assert out["fundamentals_fetched"] == 0 and p.promoted("US")["A"]["refreshed"] is not None
    assert "fundamentals fetch skipped (SEC unreachable)" in capsys.readouterr().out


def test_fund_count_reads_the_cached_row_count(monkeypatch, tmp_path):
    _install(monkeypatch, "fundamental_metrics", CACHE=tmp_path)
    assert p._fund_count("US") == 0  # nothing cached yet
    pd.DataFrame({"Symbol": ["A", "B", "C"]}).to_parquet(tmp_path / "US.parquet", index=False)
    assert p._fund_count("US") == 3
    monkeypatch.setitem(sys.modules, "fundamental_metrics", None)  # the module cannot be imported
    assert p._fund_count("US") == 0


# ---- status -----------------------------------------------------------------------------


def test_status_counts_the_watchlist_and_refreshed_names_per_market(monkeypatch):
    p.promote("US", {"A": ["f"], "B": ["f"]}, verbose=False)
    p.promote("JP", {"J": ["f"]}, verbose=False)
    state = p._load()
    state["US"]["promoted"]["A"]["refreshed"] = "2025-01-01"
    state["US"]["synced"] = "2025-01-02T03:04:05"
    p._save(state)
    df = p.status().set_index("market")
    assert df.loc["US", "watchlist"] == 2 and df.loc["US", "fund_refreshed"] == 1
    assert df.loc["US", "synced"] == "2025-01-02T03:04:05" and df.loc["JP", "fund_refreshed"] == 0
    assert list(p.status(["JP"])["market"]) == ["JP"]  # a market filter
    assert p.status(["DE"]).empty  # no watchlist -> no row


# ---- main -------------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch):
    calls = []
    monkeypatch.setattr(p, "MARKETS", ["US", "JP"])
    monkeypatch.setattr(
        p, "sync", lambda m, floor, gate, verbose=True: calls.append(("sync", m, floor, gate))
    )
    monkeypatch.setattr(p, "refresh", lambda m, **k: calls.append(("refresh", m)))
    return calls


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["pipeline.py", *args])


def test_main_defaults_to_a_sync_of_the_chosen_market(cli, monkeypatch):
    _argv(monkeypatch, "--market", "US", "--min-turnover", "2000000", "--min-filters", "3")
    assert p.main() == 0 and cli == [("sync", "US", 2_000_000.0, 3)]


def test_main_refresh_alone_does_not_sync_and_both_flags_do_both(cli, monkeypatch):
    _argv(monkeypatch, "--market", "US", "--refresh")
    p.main()
    assert cli == [("refresh", "US")]
    cli.clear()
    _argv(monkeypatch, "--market", "US", "--sync", "--refresh")
    p.main()
    assert [c[0] for c in cli] == ["sync", "refresh"]


def test_main_all_applies_to_every_market(cli, monkeypatch):
    _argv(monkeypatch, "--all", "--sync")
    p.main()
    assert [c[:2] for c in cli] == [("sync", "US"), ("sync", "JP")]


def test_main_status_prints_the_table_or_a_hint(monkeypatch, capsys):
    _argv(monkeypatch, "--status")
    assert p.main() == 0 and "no watchlist yet" in capsys.readouterr().out
    p.promote("US", {"A": ["f"]}, verbose=False)
    assert p.main() == 0
    out = capsys.readouterr().out
    assert "market" in out and "US" in out and "watchlist" in out
