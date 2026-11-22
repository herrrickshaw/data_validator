"""Tests for market_memory.py: the two-tier (LTM / STM) store -- paths, UPSERT, eviction, tier reads,
STM derivation, commit, acquisition, per-market update, drop, batch driver, status and CLI.

Offline and sandboxed: SEED_DIR / LTM_DIR point into tmp_path, BHAV_CACHE is redirected (its
default is a folder under the user's home), and every collaborator (ingest_gate, data_sources,
universe_sources, market_holidays) is a fake module. All dates are relative to today because the
retention windows are measured from today.
"""

import datetime as dt
import sys
import types

import pandas as pd
import pytest

import market_memory as mm

TODAY = pd.Timestamp(dt.date.today())


def days_ago(n):
    return TODAY - pd.Timedelta(int(n), unit="D")


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _long(rows):
    """rows: (symbol, date, close) -> long frame in the layout the store uses."""
    return pd.DataFrame(
        [
            {"Date": d, "Open": c, "High": c, "Low": c, "Close": c, "Volume": 1000, "Symbol": s}
            for s, d, c in rows
        ]
    )


def _bars(symbol, n, end=None, close0=100.0):
    end = end if end is not None else TODAY
    return _long(
        [(symbol, end - pd.Timedelta(int(n - 1 - i), unit="D"), close0 + i) for i in range(n)]
    )


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path):
    seed = tmp_path / "cache_seed"
    (seed / "ltm").mkdir(parents=True)
    monkeypatch.setattr(mm, "SEED_DIR", seed)
    monkeypatch.setattr(mm, "LTM_DIR", seed / "ltm")
    monkeypatch.setenv("BHAV_CACHE", str(tmp_path / "bhav"))
    gate = types.SimpleNamespace(calls=[], reports=[])
    gate.module = _module(
        "ingest_gate",
        gate_frame=lambda market, frame, path, verbose=True: gate.calls.append(
            (market, len(frame), path)
        )
        or frame,
        report_after=lambda market, path, before, verbose=True: gate.reports.append(
            (market, before)
        ),
    )
    monkeypatch.setitem(sys.modules, "ingest_gate", gate.module)
    return types.SimpleNamespace(dir=seed, gate=gate, tmp=tmp_path)


def _put_ltm(market, long):
    mm._write(long, mm._ltm_path(market))


def _put_stm(market, long):
    mm._write(long, mm._stm_path(market))


# ---- paths and reads --------------------------------------------------------------------


def test_paths_name_india_differently():
    assert mm._ltm_path("US") == mm.LTM_DIR / "US.parquet"
    assert mm._stm_path("IN").name == "cleaned_long.parquet"
    assert mm._stm_path("US").name == "cleaned_long_US.parquet"


def test_read_long_returns_an_empty_frame_with_the_store_columns_when_missing():
    df = mm._read_long(mm.LTM_DIR / "NOPE.parquet")
    assert df.empty and list(df.columns) == ["Date", *mm.OHLCV, "Symbol"]


def test_read_long_parses_dates():
    _put_ltm("US", _bars("A", 3))
    out = mm._read_long(mm._ltm_path("US"))
    assert pd.api.types.is_datetime64_any_dtype(out["Date"]) and len(out) == 3


def test_read_returns_each_tier_and_honours_min_bars():
    _put_ltm("US", pd.concat([_bars("A", 30), _bars("B", 3)]))
    _put_stm("US", _bars("S", 5))
    assert sorted(mm.read("US", tier="ltm")) == ["A", "B"]
    assert list(mm.read("US", tier="stm")) == ["S"]  # stm is the default tier
    assert list(mm.read("US", tier="ltm", min_bars=10)) == ["A"]
    assert mm.read("JP") == {}  # nothing stored for that market


# ---- _upsert ----------------------------------------------------------------------------


def test_upsert_lets_the_newest_write_win_and_keeps_everything_else():
    old = _long([("A", days_ago(2), 10.0), ("A", days_ago(1), 11.0), ("B", days_ago(1), 50.0)])
    new = _long([("A", days_ago(1), 11.5), ("A", days_ago(0), 12.0)])  # revises one bar, adds one
    out = mm._upsert(old, new)
    a = out[out["Symbol"] == "A"].set_index("Date")["Close"]
    assert a[days_ago(1)] == 11.5 and a[days_ago(2)] == 10.0 and a[days_ago(0)] == 12.0
    assert len(out) == 4 and out[out["Symbol"] == "B"]["Close"].tolist() == [50.0]


def test_upsert_is_idempotent_and_sorted():
    old = _long([("B", days_ago(1), 5.0), ("A", days_ago(2), 1.0)])
    new = _long([("A", days_ago(1), 2.0), ("B", days_ago(1), 6.0)])
    once = mm._upsert(old, new)
    twice = mm._upsert(once, new)
    pd.testing.assert_frame_equal(once, twice)  # re-running the same update changes nothing
    assert list(zip(once["Symbol"], once["Date"])) == sorted(zip(once["Symbol"], once["Date"]))


def test_upsert_orders_by_symbol_then_date_even_when_a_backfill_arrives_late():
    existing = _long([("B", days_ago(1), 5.0), ("A", days_ago(1), 1.0)])
    backfill = _long(
        [("A", days_ago(3), 0.5), ("B", days_ago(2), 4.0)]
    )  # older bars appended after
    out = mm._upsert(existing, backfill)
    assert list(zip(out["Symbol"], out["Date"])) == [
        ("A", days_ago(3)),
        ("A", days_ago(1)),
        ("B", days_ago(2)),
        ("B", days_ago(1)),
    ]


def test_upsert_passes_through_when_there_is_nothing_to_merge():
    existing = _bars("A", 3)
    assert mm._upsert(existing, None) is existing
    assert mm._upsert(existing, pd.DataFrame()) is existing
    seeded = mm._upsert(pd.DataFrame(), _bars("A", 3))
    assert len(seeded) == 3 and mm._upsert(None, _bars("A", 2)).shape[0] == 2


# ---- _evict -----------------------------------------------------------------------------


def test_evict_keeps_bars_up_to_the_retention_cutoff():
    # cutoff = today - (int(years * 365.25) + 5) days; one year -> 370 days
    long = _long([("A", days_ago(371), 1.0), ("A", days_ago(370), 2.0), ("A", days_ago(10), 3.0)])
    out = mm._evict(long, 1)
    assert out["Close"].tolist() == [
        2.0,
        3.0,
    ]  # exactly at the cutoff is kept, a day older is dropped


def test_evict_window_scales_with_years_and_empty_passes_through():
    long = _long(
        [("A", days_ago(3657), 1.0), ("A", days_ago(3658), 2.0)]
    )  # ten years: 3652 + 5 days
    assert mm._evict(long, 10)["Close"].tolist() == [1.0]
    empty = pd.DataFrame(columns=["Date", *mm.OHLCV, "Symbol"])
    assert mm._evict(empty, 1) is empty


# ---- _write / _derive_stm / ensure_stm_seeds --------------------------------------------


def test_write_returns_the_symbol_count_and_stores_compact_floats():
    n = mm._write(pd.concat([_bars("A", 3), _bars("B", 3)]), mm._ltm_path("US"))
    stored = pd.read_parquet(mm._ltm_path("US"))
    assert n == 2 and set(stored["Symbol"]) == {"A", "B"}
    assert str(stored["Close"].dtype) == "float32"  # same compact layout as every other seed


def test_write_of_an_empty_frame_writes_nothing():
    assert mm._write(pd.DataFrame(columns=["Date", *mm.OHLCV, "Symbol"]), mm._ltm_path("US")) == 0
    assert not mm._ltm_path("US").exists()


def test_derive_stm_is_the_trailing_year_of_the_ltm():
    ltm = pd.concat(
        [_bars("A", 5, end=days_ago(500)), _bars("A", 5, end=days_ago(3), close0=200.0)]
    )
    n = mm._derive_stm(ltm, "US")
    stm = mm._read_long(mm._stm_path("US"))
    assert n == 1 and len(stm) == 5 and stm["Close"].min() >= 200.0  # the 500-day-old bars are gone


def test_ensure_stm_seeds_derives_only_what_is_missing(capsys):
    _put_ltm("US", _bars("A", 5))
    _put_ltm("JP", _bars("J", 5))
    _put_stm("JP", _bars("OLD", 2))  # JP already has an STM seed
    assert mm.ensure_stm_seeds(["US", "JP", "DE"]) == 1  # DE has no LTM; JP's STM is left alone
    assert list(mm.read("US")) == ["A"] and list(mm.read("JP")) == ["OLD"]
    assert "derived 1 STM seed(s) from LTM" in capsys.readouterr().out


def test_ensure_stm_seeds_force_rebuilds_and_stays_quiet_when_nothing_to_do(capsys):
    _put_ltm("JP", _bars("J", 5))
    _put_stm("JP", _bars("OLD", 2))
    assert mm.ensure_stm_seeds(["JP"], force=True) == 1
    assert list(mm.read("JP")) == ["J"]
    capsys.readouterr()
    assert mm.ensure_stm_seeds(["DE"]) == 0 and capsys.readouterr().out == ""
    assert mm.ensure_stm_seeds(["JP"], force=True, verbose=False) == 1


def test_ensure_stm_seeds_defaults_to_every_market(monkeypatch):
    monkeypatch.setattr(mm, "MARKETS", ["US", "JP"])
    _put_ltm("US", _bars("A", 5))
    _put_ltm("JP", _bars("J", 5))
    assert mm.ensure_stm_seeds(verbose=False) == 2


# ---- commit -----------------------------------------------------------------------------


def test_commit_upserts_evicts_and_derives_both_tiers(sandbox, capsys):
    _put_ltm(
        "US",
        pd.concat([_bars("A", 3, end=days_ago(4000)), _bars("A", 3, end=days_ago(5), close0=50.0)]),
    )
    out = mm.commit("US", _bars("A", 2, end=TODAY, close0=70.0))
    ltm = mm._read_long(mm._ltm_path("US"))
    assert len(ltm) == 5 and ltm["Date"].min() >= days_ago(
        3657
    )  # the 4000-day-old bars were evicted
    assert len(mm._read_long(mm._stm_path("US"))) == 5  # STM derived from the committed LTM
    assert out == {"market": "US", "ltm_symbols": 1, "stm_symbols": 1, "latest": str(TODAY)[:10]}
    assert f"US: LTM 1 syms / STM 1 syms (latest {str(TODAY)[:10]})" in capsys.readouterr().out


def test_commit_keeps_ten_years_in_the_ltm_but_only_one_in_the_stm():
    mm.commit(
        "US",
        pd.concat([_bars("A", 2, end=days_ago(500)), _bars("A", 3, end=days_ago(5), close0=50.0)]),
        verbose=False,
    )
    assert (
        len(mm._read_long(mm._ltm_path("US"))) == 5
    )  # the 500-day-old bars are well inside ten years
    stm = mm._read_long(mm._stm_path("US"))
    assert len(stm) == 3 and stm["Date"].min() >= days_ago(
        370
    )  # ... but past the one-year STM window


def test_commit_runs_the_ingest_gate_and_writes_what_it_returns(sandbox):
    def gate(market, frame, path, verbose=True):
        sandbox.gate.calls.append((market, len(frame), path))
        return frame[frame["Symbol"] != "BAD"]  # the gate may drop rows

    sandbox.gate.module.gate_frame = gate
    mm.commit("US", pd.concat([_bars("GOOD", 3), _bars("BAD", 3)]), verbose=False)
    assert sandbox.gate.calls == [("US", 6, mm._ltm_path("US"))]
    assert set(mm._read_long(mm._ltm_path("US"))["Symbol"]) == {"GOOD"}  # LTM holds the gated frame
    assert set(mm._read_long(mm._stm_path("US"))["Symbol"]) == {"GOOD"}


def test_commit_reports_the_size_change_to_the_gate(sandbox):
    mm.commit("US", _bars("A", 3), verbose=False)
    assert sandbox.gate.reports[-1] == ("US", 0)  # no LTM file existed before the first commit
    size = mm._ltm_path("US").stat().st_size
    mm.commit("US", _bars("A", 4), verbose=False)
    assert sandbox.gate.reports[-1] == (
        "US",
        size,
    )  # the previous size, for the before/after report


def test_commit_still_works_when_the_ingest_gate_is_unavailable(sandbox, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("gate offline")

    sandbox.gate.module.gate_frame = boom
    out = mm.commit("US", _bars("A", 3))
    assert out["ltm_symbols"] == 1 and mm._ltm_path("US").exists()  # the core never hard-breaks
    assert "ingest-gate skipped (gate offline)" in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "ingest_gate", None)  # the module cannot even be imported
    assert mm.commit("JP", _bars("J", 3), verbose=False)["ltm_symbols"] == 1


def test_commit_of_nothing_reports_no_latest_date():
    out = mm.commit("US", pd.DataFrame(columns=["Date", *mm.OHLCV, "Symbol"]), verbose=False)
    assert out["latest"] == "—" and out["ltm_symbols"] == 0


# ---- _fetch_new -------------------------------------------------------------------------


def test_fetch_new_india_prefers_the_live_bhavcopy_cache(sandbox):
    _put_stm("IN", _bars("COMMITTED", 3))
    assert list(mm._fetch_new("IN", "5d", False)["Symbol"].unique()) == ["COMMITTED"]
    live = sandbox.tmp / "bhav"
    live.mkdir()
    _bars("LIVE", 4).to_parquet(live / "cleaned_long.parquet", index=False)
    assert list(mm._fetch_new("IN", "5d", False)["Symbol"].unique()) == ["LIVE"]


def _hist(symbols, n=3):
    out = {}
    for s in symbols:
        long = _bars(s, n)
        out[s] = long.set_index("Date")[mm.OHLCV]
    return out


def test_fetch_new_refreshes_the_symbols_already_tracked(monkeypatch):
    _put_ltm("US", pd.concat([_bars("A", 3), _bars("B", 3)]))
    seen = {}

    def fetch(symbols, period, min_bars, verbose):
        seen.update(symbols=sorted(symbols), period=period, min_bars=min_bars, verbose=verbose)
        return _hist(symbols)

    monkeypatch.setitem(sys.modules, "data_sources", _module("data_sources", fetch=fetch))
    out = mm._fetch_new("US", "5d", True)
    assert seen == {"symbols": ["A", "B"], "period": "5d", "min_bars": 1, "verbose": False}
    assert set(out["Symbol"]) == {"A", "B"} and str(out["Close"].dtype) == "float32"


def test_fetch_new_seeds_from_the_universe_when_nothing_is_tracked(monkeypatch, capsys):
    monkeypatch.setitem(
        sys.modules, "data_sources", _module("data_sources", fetch=lambda s, **k: _hist(s))
    )
    monkeypatch.setitem(
        sys.modules,
        "universe_sources",
        _module("universe_sources", get_universe=lambda m: ["U1", "U2"]),
    )
    out = mm._fetch_new("JP", "10y", True)
    assert set(out["Symbol"]) == {"U1", "U2"}
    assert "JP: fetching 2 symbols (period=10y)" in capsys.readouterr().out


def test_fetch_new_without_a_universe_returns_an_empty_frame(monkeypatch, capsys):
    def boom(market):
        raise RuntimeError("no universe source")

    monkeypatch.setitem(
        sys.modules, "data_sources", _module("data_sources", fetch=lambda s, **k: _hist(s))
    )
    monkeypatch.setitem(
        sys.modules, "universe_sources", _module("universe_sources", get_universe=boom)
    )
    out = mm._fetch_new("JP", "5d", True)
    assert out.empty and list(out.columns) == ["Date", *mm.OHLCV, "Symbol"]
    assert "JP: no universe" in capsys.readouterr().out


# ---- update / init ----------------------------------------------------------------------


def _fake_fetch(monkeypatch, hist):
    monkeypatch.setitem(
        sys.modules, "data_sources", _module("data_sources", fetch=lambda s, **k: hist)
    )


def test_update_bootstraps_a_thin_ltm_from_the_committed_seed_first(sandbox, monkeypatch):
    # A first run would otherwise derive a thin STM from 2 fetched bars and overwrite the good seed
    seed = _bars("A", 120, close0=100.0)
    seed.to_parquet(sandbox.dir / "cleaned_long_US.parquet", index=False)
    thin = _bars("A", 2, close0=999.0).set_index("Date")[mm.OHLCV]  # revises the last two bars
    _fake_fetch(monkeypatch, {"A": thin})
    mm.update("US", verbose=False)
    ltm = mm._read_long(mm._ltm_path("US")).set_index("Date")
    assert len(ltm) == 120  # seed history + the revised bars, not 2 rows
    assert ltm["Close"].iloc[-1] == pytest.approx(1000.0, abs=0.01) and ltm["Close"].iloc[
        0
    ] == pytest.approx(100.0)
    assert len(mm._read_long(mm._stm_path("US"))) == 120


def test_update_bootstraps_until_the_ltm_spans_60_days(sandbox, monkeypatch):
    _bars("SEEDED", 120).to_parquet(sandbox.dir / "cleaned_long_US.parquet", index=False)
    _fake_fetch(monkeypatch, {"A": _bars("A", 2, close0=500.0).set_index("Date")[mm.OHLCV]})
    _put_ltm("US", _bars("A", 46))  # spans 45 days: still thin
    mm.update("US", verbose=False)
    assert "SEEDED" in set(
        mm._read_long(mm._ltm_path("US"))["Symbol"]
    )  # bootstrapped from the seed

    _put_ltm("JP", _bars("A", 62))  # spans 61 days: seeded enough
    _bars("SEEDED", 120).to_parquet(sandbox.dir / "cleaned_long_JP.parquet", index=False)
    mm.update("JP", verbose=False)
    assert "SEEDED" not in set(mm._read_long(mm._ltm_path("JP"))["Symbol"])


def test_update_skips_the_bootstrap_for_a_deep_ltm_and_for_india(sandbox, monkeypatch):
    _put_ltm("US", _bars("A", 90))  # spans more than 60 days: already seeded
    _bars("SEEDONLY", 120).to_parquet(sandbox.dir / "cleaned_long_US.parquet", index=False)
    _fake_fetch(monkeypatch, {"A": _bars("A", 2, close0=500.0).set_index("Date")[mm.OHLCV]})
    mm.update("US", verbose=False)
    assert set(mm._read_long(mm._ltm_path("US"))["Symbol"]) == {"A"}  # the seed was NOT merged in

    _put_stm("IN", _bars("INDIA", 5))
    out = mm.update("IN", verbose=False)  # India reads its own bhavcopy as the fetch
    assert out["market"] == "IN" and out["ltm_symbols"] == 1


def test_update_without_a_seed_just_commits_what_was_fetched(sandbox, monkeypatch):
    _put_ltm("US", _bars("A", 5))  # thin, but there is no committed seed to bootstrap from
    _fake_fetch(monkeypatch, {"A": _bars("A", 2, close0=60.0).set_index("Date")[mm.OHLCV]})
    assert mm.update("US", verbose=False)["ltm_symbols"] == 1


def test_init_runs_an_update_over_the_whole_ltm_window(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        mm, "update", lambda market, period="5d", verbose=True: seen.update(m=market, p=period)
    )
    mm.init("US")
    assert seen == {"m": "US", "p": f"{mm.LTM_YEARS}y"}


# ---- drop / seed_from_committed ---------------------------------------------------------


def test_drop_removes_symbols_from_both_tiers(capsys):
    _put_ltm("US", pd.concat([_bars("KEEP", 5), _bars("GONE", 5)]))
    _put_stm("US", pd.concat([_bars("KEEP", 5), _bars("GONE", 5)]))
    out = mm.drop("US", ["GONE", "NOT_THERE"])
    assert out == {"market": "US", "ltm_symbols": 1, "stm_symbols": 1}
    assert list(mm.read("US", "ltm")) == ["KEEP"] and list(mm.read("US", "stm")) == ["KEEP"]
    assert "dropped 2 syms" in capsys.readouterr().out
    assert mm.drop("US", [], verbose=False)["ltm_symbols"] == 1  # nothing to drop


@pytest.mark.xfail(
    strict=True,
    reason="known bug: write_seed writes nothing for an empty frame, so dropping EVERY symbol leaves "
    "the old file (with the 'dropped' symbols) in place instead of emptying the tier",
)
def test_dropping_every_symbol_empties_the_tier():
    _put_ltm("US", _bars("A", 5))
    _put_stm("US", _bars("A", 5))
    mm.drop("US", ["A"], verbose=False)
    assert mm.read("US", "ltm") == {} and mm.read("US", "stm") == {}


def test_seed_from_committed_initialises_the_ltm_offline(sandbox):
    assert mm.seed_from_committed("US") == {"market": "US", "error": "no committed seed"}
    _bars("A", 10).to_parquet(sandbox.dir / "cleaned_long_US.parquet", index=False)
    out = mm.seed_from_committed("US", verbose=False)
    assert out["ltm_symbols"] == 1 and list(mm.read("US", "ltm")) == ["A"]
    _bars("B", 5).to_parquet(
        sandbox.dir / "cleaned_long.parquet", index=False
    )  # India's seed has its own name
    assert mm.seed_from_committed("IN", verbose=False)["ltm_symbols"] == 1


# ---- update_all -------------------------------------------------------------------------


@pytest.fixture
def batch(monkeypatch):
    calls = []
    monkeypatch.setattr(mm, "MARKETS", ["US", "JP", "DE"])
    monkeypatch.setattr(
        mm,
        "update",
        lambda m, period="5d", verbose=True: calls.append(("update", m, period)) or {"market": m},
    )
    monkeypatch.setattr(
        mm, "init", lambda m, verbose=True: calls.append(("init", m)) or {"market": m}
    )
    return calls


def test_update_all_updates_each_market_with_the_period(batch, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "market_holidays",
        _module("market_holidays", is_market_open_today=lambda m: True),
    )
    out = mm.update_all(period="1mo", verbose=False)
    assert batch == [("update", "US", "1mo"), ("update", "JP", "1mo"), ("update", "DE", "1mo")]
    assert [r["market"] for r in out] == ["US", "JP", "DE"]


def test_update_all_skips_closed_markets_unless_creating(batch, monkeypatch, capsys):
    monkeypatch.setitem(
        sys.modules,
        "market_holidays",
        _module("market_holidays", is_market_open_today=lambda m: m != "JP"),
    )
    mm.update_all()
    assert [c[1] for c in batch] == ["US", "DE"]
    assert "skipping closed markets: ['JP']" in capsys.readouterr().out
    batch.clear()
    mm.update_all(create=True, verbose=False)  # a first-time backfill ignores holidays
    assert batch == [("init", "US"), ("init", "JP"), ("init", "DE")]


def test_update_all_runs_everything_when_the_holiday_calendar_is_unavailable(batch, monkeypatch):
    monkeypatch.setitem(sys.modules, "market_holidays", None)
    mm.update_all(markets=["US", "JP"], verbose=False)
    assert [c[1] for c in batch] == ["US", "JP"]


def test_update_all_isolates_a_failing_market(monkeypatch, capsys):
    monkeypatch.setitem(
        sys.modules,
        "market_holidays",
        _module("market_holidays", is_market_open_today=lambda m: True),
    )

    def update(market, period="5d", verbose=True):
        if market == "JP":
            raise RuntimeError("yahoo down")
        return {"market": market, "ok": True}

    monkeypatch.setattr(mm, "update", update)
    out = mm.update_all(markets=["US", "JP", "DE"])
    assert [r["market"] for r in out] == ["US", "JP", "DE"]
    assert out[1] == {"market": "JP", "error": "yahoo down"} and out[2]["ok"] is True
    assert "JP: FAILED (yahoo down)" in capsys.readouterr().out


def test_update_all_announces_the_mode(batch, monkeypatch, capsys):
    monkeypatch.setitem(
        sys.modules,
        "market_holidays",
        _module("market_holidays", is_market_open_today=lambda m: True),
    )
    mm.update_all(markets=["US"], create=True)
    assert "CREATE 5y backfill for 1 markets" in capsys.readouterr().out


# ---- status -----------------------------------------------------------------------------


def test_status_reports_symbols_span_and_rows_per_tier():
    _put_ltm("US", pd.concat([_bars("A", 10, end=days_ago(0)), _bars("B", 4, end=days_ago(0))]))
    _put_stm("US", _bars("A", 3))
    df = mm.status(["US", "JP"]).set_index(["market", "tier"])
    ltm = df.loc[("US", "ltm")]
    assert ltm["symbols"] == 2 and ltm["rows"] == 14 and ltm["end"] == str(TODAY)[:10]
    assert ltm["start"] == str(days_ago(9))[:10]
    assert df.loc[("US", "stm")]["rows"] == 3
    empty = df.loc[("JP", "ltm")]
    assert (
        empty["symbols"] == 0
        and empty["rows"] == 0
        and empty["start"] == "—"
        and empty["end"] == "—"
    )
    assert len(df) == 4  # two tiers per market


def test_status_defaults_to_every_market(monkeypatch):
    monkeypatch.setattr(mm, "MARKETS", ["US", "JP"])
    assert len(mm.status()) == 4


# ---- main -------------------------------------------------------------------------------


def test_main_status_prints_the_coverage_table(monkeypatch, capsys):
    _put_ltm("US", _bars("A", 5))
    monkeypatch.setattr(sys, "argv", ["market_memory.py", "--status", "--market", "US"])
    assert mm.main() == 0
    out = capsys.readouterr().out
    assert "market" in out and "ltm" in out and "stm" in out


def test_main_from_seeds_initialises_the_requested_markets(monkeypatch):
    seen = []
    monkeypatch.setattr(
        mm, "seed_from_committed", lambda m, verbose=True: seen.append((m, verbose))
    )
    monkeypatch.setattr(
        sys, "argv", ["market_memory.py", "--from-seeds", "--market", "US", "JP", "--quiet"]
    )
    assert mm.main() == 0 and seen == [("US", False), ("JP", False)]


def test_main_runs_the_daily_update_with_the_parsed_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(mm, "update_all", lambda **kw: seen.update(kw))
    monkeypatch.setattr(
        sys,
        "argv",
        ["market_memory.py", "--daily", "--market", "US", "--period", "1mo", "--create"],
    )
    assert mm.main() == 0
    assert seen == {"markets": ["US"], "period": "1mo", "create": True, "verbose": True}


def test_main_returns_1_when_the_update_fails(monkeypatch, capsys):
    def boom(**kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mm, "update_all", boom)
    monkeypatch.setattr(sys, "argv", ["market_memory.py"])
    assert mm.main() == 1 and "market memory update FAILED: disk full" in capsys.readouterr().out
