"""Tests for datalink.py: path resolution, the memoized market loader, the cached India CCC map,
the data manifest and the incremental store build.

Everything runs against tmp_path: every path constant is redirected (LIVE_CACHE defaults to a
folder under the user's home), the process-level caches are reset, and the optional collaborators
(liquidity, screener_in, market_memory, bhavcopy_store) are fake modules.
"""

import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest

import datalink as dl


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path):
    seed = tmp_path / "cache_seed"
    for d in (seed, seed / "ltm", tmp_path / "reference_seed", tmp_path / "live"):
        d.mkdir()
    monkeypatch.setattr(dl, "HERE", tmp_path)
    monkeypatch.setattr(dl, "SEED_DIR", seed)
    monkeypatch.setattr(dl, "LTM_DIR", seed / "ltm")
    monkeypatch.setattr(dl, "REF_DIR", tmp_path / "reference_seed")
    monkeypatch.setattr(dl, "LIVE_CACHE", tmp_path / "live")
    monkeypatch.setattr(dl, "MANIFEST", seed / "data_manifest.json")
    monkeypatch.setattr(dl, "STORE_STATE", seed / ".store_state.json")
    monkeypatch.setattr(dl, "CCC_PARQUET", seed / "india_ccc_screen.parquet")
    monkeypatch.setattr(dl, "_LOAD_CACHE", {})
    monkeypatch.setattr(dl, "_CCC_CACHE", None)
    return tmp_path


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _long(symbols=("AAA", "BBB"), n=5, start="2024-01-01", close=10.0):
    rows = []
    for s in symbols:
        for i, d in enumerate(pd.date_range(start, periods=n)):
            c = close + i
            rows.append(
                {
                    "Date": d,
                    "Symbol": s,
                    "Open": c,
                    "High": c + 1,
                    "Low": c - 1,
                    "Close": c,
                    "Volume": 1000,
                }
            )
    return pd.DataFrame(rows)


def _seed(path, **kw):
    _long(**kw).to_parquet(path, index=False)
    return path


# ---- path resolution --------------------------------------------------------------------


def test_seed_names_differ_for_india():
    assert dl._seed_name("IN") == "cleaned_long.parquet"
    assert dl._seed_name("US") == "cleaned_long_US.parquet"
    assert dl._ltm_path("JP") == dl.LTM_DIR / "JP.parquet"


def test_stm_path_prefers_the_live_cache_then_the_committed_seed(sandbox):
    assert (
        dl._stm_path("US") == dl.SEED_DIR / "cleaned_long_US.parquet"
    )  # nothing exists: seed path
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet")
    assert dl._stm_path("US") == dl.SEED_DIR / "cleaned_long_US.parquet"
    _seed(dl.LIVE_CACHE / "cleaned_long_US.parquet")
    assert dl._stm_path("US") == dl.LIVE_CACHE / "cleaned_long_US.parquet"  # live wins


def test_stm_path_derives_the_stm_seed_from_the_ltm_when_missing(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "market_memory",
        _module(
            "market_memory",
            ensure_stm_seeds=lambda markets, verbose: calls.append((markets, verbose)),
        ),
    )
    _seed(dl._ltm_path("US"))
    assert dl._stm_path("US") == dl.SEED_DIR / "cleaned_long_US.parquet"
    assert calls == [(["US"], False)]  # asked to derive just this market, quietly
    calls.clear()
    dl._stm_path("DE")  # no LTM either: nothing to derive from
    assert calls == []


def test_stm_path_survives_a_failing_seed_derivation(monkeypatch):
    def boom(markets, verbose):
        raise RuntimeError("disk full")

    monkeypatch.setitem(
        sys.modules, "market_memory", _module("market_memory", ensure_stm_seeds=boom)
    )
    _seed(dl._ltm_path("US"))
    assert (
        dl._stm_path("US").name == "cleaned_long_US.parquet"
    )  # falls back to the (absent) seed path


def test_sig_is_size_and_whole_second_mtime(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"12345")
    os.utime(p, (1_700_000_000, 1_700_000_000))
    assert dl._sig(p) == "5:1700000000"


# ---- load_market -------------------------------------------------------------------------


def test_load_market_groups_by_symbol_and_returns_empty_when_absent():
    assert dl.load_market("US") == {}
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet", symbols=("AAA", "BBB"), n=5)
    out = dl.load_market("US")
    assert sorted(out) == ["AAA", "BBB"]
    assert list(out["AAA"].columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out["AAA"]) == 5 and out["AAA"].index.is_monotonic_increasing


def test_load_market_ltm_tier_reads_the_ltm_file():
    _seed(dl._ltm_path("US"), symbols=("LONG",), n=8)
    assert dl.load_market("US") == {}  # stm tier: no short-term seed
    assert list(dl.load_market("US", tier="ltm")) == ["LONG"]


def test_load_market_is_memoized_until_the_file_changes():
    p = _seed(dl.SEED_DIR / "cleaned_long_US.parquet", symbols=("AAA",))
    first = dl.load_market("US")
    assert dl.load_market("US") is first  # second call is free: same objects

    _seed(p, symbols=("AAA", "NEW"))  # the seed changes (new size/mtime signature)
    os.utime(p, (time.time() + 10, time.time() + 10))
    again = dl.load_market("US")
    assert again is not first and sorted(again) == ["AAA", "NEW"]

    dl.clear_cache()
    assert dl._LOAD_CACHE == {}


def test_load_market_cache_key_includes_the_turnover_floor(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "liquidity", _module("liquidity", liquid_symbols=lambda m, min_usd: ["AAA"])
    )
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet", symbols=("AAA", "BBB"))
    everything = dl.load_market("US")
    liquid = dl.load_market("US", min_turnover_usd=1e6)
    assert sorted(everything) == ["AAA", "BBB"] and sorted(liquid) == ["AAA"]
    assert len(dl._LOAD_CACHE) == 2  # the two calls are cached separately


def test_load_market_passes_market_and_floor_to_the_liquidity_filter(monkeypatch):
    seen = {}

    def liquid_symbols(market, min_usd):
        seen.update(market=market, min_usd=min_usd)
        return ["BBB"]

    monkeypatch.setitem(
        sys.modules, "liquidity", _module("liquidity", liquid_symbols=liquid_symbols)
    )
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet", symbols=("AAA", "BBB"))
    assert list(dl.load_market("US", min_turnover_usd=2_500_000)) == ["BBB"]
    assert seen == {"market": "US", "min_usd": 2_500_000.0}


def test_load_market_ignores_the_floor_when_the_liquidity_index_fails(monkeypatch):
    def boom(market, min_usd):
        raise RuntimeError("no index built")

    monkeypatch.setitem(sys.modules, "liquidity", _module("liquidity", liquid_symbols=boom))
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet", symbols=("AAA", "BBB"))
    assert sorted(dl.load_market("US", min_turnover_usd=1e6)) == [
        "AAA",
        "BBB",
    ]  # unfiltered, not an error


# ---- ccc_map_cached ----------------------------------------------------------------------


def _ccc(rows):
    return pd.DataFrame(rows, columns=["Symbol", "Cash_Cycle"])


def _fake_screener(monkeypatch, result=None, boom=None):
    calls = []

    def ccc_screen():
        calls.append(1)
        if boom:
            raise boom
        return result

    monkeypatch.setitem(sys.modules, "screener_in", _module("screener_in", ccc_screen=ccc_screen))
    return calls


def test_ccc_map_reads_a_fresh_parquet_without_touching_the_network(monkeypatch):
    calls = _fake_screener(monkeypatch, result=_ccc([("ZZZ", 1.0)]))
    _ccc([("AAA", 45.0), ("BBB", -10.5)]).to_parquet(dl.CCC_PARQUET, index=False)
    assert dl.ccc_map_cached() == {"AAA": 45.0, "BBB": -10.5}
    assert calls == []


def test_ccc_map_refetches_a_stale_parquet_and_persists_it(monkeypatch):
    calls = _fake_screener(monkeypatch, result=_ccc([("NEW", 12.0)]))
    _ccc([("OLD", 99.0)]).to_parquet(dl.CCC_PARQUET, index=False)
    old = time.time() - 3 * 86400
    os.utime(dl.CCC_PARQUET, (old, old))  # three days old; the default limit is one day
    assert dl.ccc_map_cached() == {"NEW": 12.0}
    assert calls == [1]
    assert pd.read_parquet(dl.CCC_PARQUET)["Symbol"].tolist() == ["NEW"]


def test_ccc_map_keeps_the_committed_parquet_when_the_refetch_fails_or_is_empty(monkeypatch):
    for kind in ("raises", "empty"):
        monkeypatch.setattr(dl, "_CCC_CACHE", None)
        _fake_screener(
            monkeypatch,
            result=_ccc([]) if kind == "empty" else None,
            boom=RuntimeError("blocked") if kind == "raises" else None,
        )
        _ccc([("OLD", 99.0)]).to_parquet(dl.CCC_PARQUET, index=False)
        old = time.time() - 9 * 86400
        os.utime(dl.CCC_PARQUET, (old, old))
        assert dl.ccc_map_cached() == {"OLD": 99.0}, kind


def test_ccc_map_does_not_use_the_network_when_disallowed(monkeypatch):
    calls = _fake_screener(monkeypatch, result=_ccc([("NEW", 1.0)]))
    _ccc([("OLD", 5.0)]).to_parquet(dl.CCC_PARQUET, index=False)
    old = time.time() - 9 * 86400
    os.utime(dl.CCC_PARQUET, (old, old))
    assert dl.ccc_map_cached(allow_network=False) == {"OLD": 5.0}
    assert calls == []


def test_ccc_map_max_age_is_in_days(monkeypatch):
    calls = _fake_screener(monkeypatch, result=_ccc([("NEW", 1.0)]))
    _ccc([("OLD", 5.0)]).to_parquet(dl.CCC_PARQUET, index=False)
    old = time.time() - 2 * 86400
    os.utime(dl.CCC_PARQUET, (old, old))
    assert dl.ccc_map_cached(max_age_days=5) == {"OLD": 5.0}  # 2 days old, limit 5: still fresh
    assert calls == []


def test_ccc_map_drops_non_numeric_and_nan_values_and_tolerates_missing_files(monkeypatch):
    assert dl.ccc_map_cached(allow_network=False) == {}  # no parquet at all
    monkeypatch.setattr(dl, "_CCC_CACHE", None)
    # a text column (as scraped): "30.0" parses, "n/a" and a missing value are dropped
    _ccc([("AAA", "30.0"), ("BAD", "n/a"), ("NAN", None)]).to_parquet(dl.CCC_PARQUET, index=False)
    assert dl.ccc_map_cached(allow_network=False) == {"AAA": 30.0}
    monkeypatch.setattr(dl, "_CCC_CACHE", None)
    _ccc([("AAA", 30.0), ("NAN", np.nan)]).to_parquet(
        dl.CCC_PARQUET, index=False
    )  # numeric NaN too
    assert dl.ccc_map_cached(allow_network=False) == {"AAA": 30.0}
    monkeypatch.setattr(dl, "_CCC_CACHE", None)
    pd.DataFrame({"Symbol": ["AAA"], "Other": [1]}).to_parquet(dl.CCC_PARQUET, index=False)
    assert dl.ccc_map_cached(allow_network=False) == {}  # no Cash_Cycle column


def test_ccc_map_is_memoized_per_process():
    _ccc([("AAA", 1.0)]).to_parquet(dl.CCC_PARQUET, index=False)
    first = dl.ccc_map_cached(allow_network=False)
    _ccc([("CHANGED", 2.0)]).to_parquet(dl.CCC_PARQUET, index=False)
    assert dl.ccc_map_cached(allow_network=False) is first


# ---- manifest ----------------------------------------------------------------------------


def test_relpath_is_relative_inside_the_repo_and_absolute_outside(sandbox, tmp_path_factory):
    assert dl._relpath(dl.SEED_DIR / "x.parquet") == os.path.join("cache_seed", "x.parquet")
    outside = tmp_path_factory.mktemp("elsewhere") / "x.parquet"
    assert dl._relpath(outside) == str(outside)  # the live cache lives outside the repo


def test_describe_summarises_symbols_rows_and_date_span():
    p = _seed(
        dl.SEED_DIR / "cleaned_long_US.parquet",
        symbols=("AAA", "BBB", "CCC"),
        n=4,
        start="2024-03-01",
    )
    d = dl._describe(p, "US", "stm")
    assert d["market"] == "US" and d["tier"] == "stm"
    assert d["symbols"] == 3 and d["rows"] == 12
    assert (d["start"], d["end"]) == ("2024-03-01", "2024-03-04")
    assert d["bytes"] == p.stat().st_size and d["sig"] == dl._sig(p)
    assert d["path"] == os.path.join("cache_seed", "cleaned_long_US.parquet")


def test_build_manifest_indexes_every_asset_and_writes_json(capsys):
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet", symbols=("AAA",))
    _seed(dl._ltm_path("US"), symbols=("AAA", "BBB"), n=10)
    _seed(dl._ltm_path("JP"), symbols=("TOY",))
    (dl.REF_DIR / "ref_b.parquet").write_bytes(b"b")
    (dl.REF_DIR / "ref_a.parquet").write_bytes(b"aa")
    (dl.REF_DIR / "ignore.txt").write_text("not parquet")

    man = dl.build_manifest()
    keyed = {(a["market"], a["tier"]): a for a in man["markets"]}
    assert set(keyed) == {("US", "stm"), ("US", "ltm"), ("JP", "ltm")}
    assert keyed[("US", "ltm")]["symbols"] == 2 and keyed[("US", "ltm")]["rows"] == 20
    assert [r["path"] for r in man["reference"]] == [  # sorted, parquet only
        os.path.join("reference_seed", "ref_a.parquet"),
        os.path.join("reference_seed", "ref_b.parquet"),
    ]
    assert json.loads(dl.MANIFEST.read_text()) == man
    assert "3 market assets, 2 reference" in capsys.readouterr().out


def test_build_manifest_skips_unreadable_assets():
    (dl.SEED_DIR / "cleaned_long_US.parquet").write_bytes(b"this is not parquet")
    _seed(dl._ltm_path("JP"))
    man = dl.build_manifest(verbose=False)
    assert [(a["market"], a["tier"]) for a in man["markets"]] == [("JP", "ltm")]


def test_read_manifest_returns_empty_when_absent_else_the_json():
    assert dl.read_manifest() == {}
    dl.MANIFEST.write_text(json.dumps({"markets": [{"market": "US"}]}))
    assert dl.read_manifest() == {"markets": [{"market": "US"}]}


def test_status_table_uses_the_manifest_or_builds_one():
    _seed(dl._ltm_path("US"), symbols=("AAA",))
    table = dl._status_table()  # no manifest yet: built on the fly
    assert table["market"].tolist() == ["US"] and dl.MANIFEST.exists()
    dl.MANIFEST.write_text(json.dumps({"markets": [{"market": "ZZ", "symbols": 1}]}))
    assert dl._status_table()["market"].tolist() == ["ZZ"]  # the existing manifest wins


def test_main_manifest_and_status(monkeypatch, capsys):
    _seed(dl._ltm_path("US"), symbols=("AAA",))
    monkeypatch.setattr(sys, "argv", ["datalink.py", "--manifest"])
    assert dl.main() == 0 and dl.MANIFEST.exists()
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["datalink.py", "--status"])
    assert dl.main() == 0
    out = capsys.readouterr().out
    assert "market" in out and "US" in out and "ltm" in out
    monkeypatch.setattr(sys, "argv", ["datalink.py"])  # no flags: rebuilds the manifest
    dl.MANIFEST.unlink()
    assert dl.main() == 0 and dl.MANIFEST.exists()


# ---- incremental store build -------------------------------------------------------------


@pytest.fixture
def store(monkeypatch, sandbox):
    import lmdb

    path = sandbox / "store"
    calls = {"build": 0}

    def build(verbose=True):
        calls["build"] += 1
        path.mkdir(exist_ok=True)
        return 7

    mod = _module(
        "bhavcopy_store",
        STORE=path,
        _MAP_SIZE=10 * 1024 * 1024,
        _ser=lambda df: df.to_csv().encode(),
        build=build,
    )
    monkeypatch.setitem(sys.modules, "bhavcopy_store", mod)
    mod.calls, mod.lmdb = calls, lmdb
    return mod


def test_changed_markets_compares_signatures_against_the_last_build():
    p = _seed(dl.SEED_DIR / "cleaned_long_US.parquet")
    _seed(dl.SEED_DIR / "cleaned_long_JP.parquet")
    assert sorted(dl.changed_markets()) == ["JP", "US"]  # no state yet: everything is new
    dl.STORE_STATE.write_text(json.dumps(dl._stm_signatures()))
    assert dl.changed_markets() == []
    os.utime(p, (time.time() + 50, time.time() + 50))
    assert dl.changed_markets() == ["US"]  # only the touched seed


def test_stm_signatures_only_cover_markets_with_a_seed():
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet")
    sigs = dl._stm_signatures()
    assert list(sigs) == ["US"] and sigs["US"] == dl._sig(dl.SEED_DIR / "cleaned_long_US.parquet")


def test_first_store_build_is_a_full_build_and_records_state(store):
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet")
    out = dl.build_store_incremental(verbose=False)
    assert out["mode"] == "full" and out["symbols"] == 7 and out["changed"] == dl.MARKETS
    assert store.calls["build"] == 1
    assert json.loads(dl.STORE_STATE.read_text()) == dl._stm_signatures()


def test_store_build_is_a_noop_when_nothing_changed(store, capsys):
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet")
    dl.build_store_incremental(verbose=False)  # full build, state written, store dir exists
    out = dl.build_store_incremental()
    assert out == {"mode": "noop", "changed": []} and store.calls["build"] == 1
    assert "no markets changed" in capsys.readouterr().out


def test_store_build_rewrites_only_the_changed_markets(store, capsys):
    us = _seed(dl.SEED_DIR / "cleaned_long_US.parquet", symbols=("USA",))
    _seed(dl.SEED_DIR / "cleaned_long_JP.parquet", symbols=("JPN",))
    dl.build_store_incremental(verbose=False)  # baseline

    _seed(us, symbols=("USA", "USB"))  # US changes, JP does not
    os.utime(us, (time.time() + 100, time.time() + 100))
    out = dl.build_store_incremental()
    assert out == {"mode": "incremental", "symbols": 2, "changed": ["US"]}
    assert "refreshed 1 market(s)" in capsys.readouterr().out

    env = store.lmdb.open(str(store.STORE), subdir=True, readonly=True)
    with env.begin() as txn:
        assert sorted(k.decode() for k, _ in txn.cursor()) == ["USA", "USB"]  # JP was not written
    env.close()
    assert json.loads(dl.STORE_STATE.read_text()) == dl._stm_signatures()  # state advanced
    assert dl.changed_markets() == []


def test_full_build_when_the_store_directory_is_missing(store):
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet")
    dl.STORE_STATE.write_text(json.dumps(dl._stm_signatures()))  # state says "built" ...
    assert not store.STORE.exists()  # ... but the store itself is gone
    assert dl.build_store_incremental(verbose=False)["mode"] == "full"


def test_main_store_flag_runs_the_incremental_build(store, monkeypatch):
    _seed(dl.SEED_DIR / "cleaned_long_US.parquet")
    monkeypatch.setattr(sys, "argv", ["datalink.py", "--store"])
    assert dl.main() == 0 and store.calls["build"] == 1
