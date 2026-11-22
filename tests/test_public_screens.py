"""Tests for public_screens.py: URL overrides, the cache-first fetcher, the offline reader and the CLI.

The module imports ``screener_in`` at import time (a network scraper that not every repo ships), so
the fixture installs a fake ``screener_in``, imports public_screens fresh against it, and removes
it again afterwards so nothing leaks into other tests. The cache directory and the overrides file
location are redirected into tmp_path and ``time.sleep`` is recorded instead of waited on.
"""

import importlib
import json
import sys
import types

import pandas as pd
import pytest


@pytest.fixture
def ps(monkeypatch, tmp_path):
    fake = types.ModuleType("screener_in")
    fake.responses = {}
    fake.fetched = []

    fake.verbose_flags = []

    def fetch_screen(url, verbose=True):
        fake.fetched.append(url)
        fake.verbose_flags.append(verbose)
        r = fake.responses.get(url, pd.DataFrame())
        if isinstance(r, Exception):
            raise r
        return r

    fake.fetch_screen = fetch_screen
    monkeypatch.setitem(sys.modules, "screener_in", fake)
    previous = sys.modules.pop("public_screens", None)
    mod = importlib.import_module("public_screens")
    cache = tmp_path / "cache"
    cache.mkdir()
    sleeps = []
    monkeypatch.setattr(mod, "CACHE", cache)
    monkeypatch.setattr(
        mod, "__file__", str(tmp_path / "public_screens.py")
    )  # where the overrides file is read
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))
    mod.fake, mod.sleeps, mod.tmp = fake, sleeps, tmp_path
    yield mod
    sys.modules.pop("public_screens", None)
    if previous is not None:
        sys.modules["public_screens"] = previous


def _screen(*symbols, extra=None):
    df = pd.DataFrame({"Symbol": list(symbols)})
    if extra:
        df[extra] = 1.0
    return df


# ---- the URL table and overrides ----------------------------------------------------------


def test_every_public_url_lives_under_the_screens_base(ps):
    assert ps.PUBLIC_URLS and all(
        u.startswith(ps.BASE + "/") and u.endswith("/") for u in ps.PUBLIC_URLS.values()
    )
    assert len(set(ps.PUBLIC_URLS.values())) == len(ps.PUBLIC_URLS)  # no two keys share a screen


def test_overrides_default_to_a_copy_of_the_builtin_table(ps):
    cfg = ps._load_overrides()
    assert cfg == ps.PUBLIC_URLS and cfg is not ps.PUBLIC_URLS
    cfg["x"] = "y"
    assert "x" not in ps.PUBLIC_URLS  # callers cannot corrupt the module's table


def test_overrides_file_adds_and_replaces_urls(ps):
    (ps.tmp / "public_screens.json").write_text(
        json.dumps(
            {"piotroski_9": "https://example.test/mine/", "brand_new": "https://example.test/new/"}
        )
    )
    cfg = ps._load_overrides()
    assert (
        cfg["piotroski_9"] == "https://example.test/mine/"
        and cfg["brand_new"] == "https://example.test/new/"
    )
    assert (
        cfg["magic_formula"] == ps.PUBLIC_URLS["magic_formula"]
    )  # untouched keys keep their default


def test_a_corrupt_overrides_file_is_ignored(ps):
    (ps.tmp / "public_screens.json").write_text("{ not json")
    assert ps._load_overrides() == ps.PUBLIC_URLS


# ---- fetch ------------------------------------------------------------------------------


def test_fetch_caches_each_screen_and_reports_symbol_counts(ps, capsys):
    ps.fake.responses = {
        ps.PUBLIC_URLS["piotroski_9"]: _screen("A", "B", "C", extra="ROCE"),
        ps.PUBLIC_URLS["magic_formula"]: _screen("X"),
    }
    got = ps.fetch(only=["piotroski_9", "magic_formula"], pause=0.25)
    assert got == {"piotroski_9": 3, "magic_formula": 1}
    assert list(pd.read_parquet(ps.CACHE / "piotroski_9.parquet")["Symbol"]) == ["A", "B", "C"]
    assert (
        "ROCE" in pd.read_parquet(ps.CACHE / "piotroski_9.parquet").columns
    )  # the whole table is kept
    assert ps.sleeps == [0.25, 0.25]  # polite pause after each fetched screen
    assert "piotroski_9: 3 symbols cached" in capsys.readouterr().out


def test_fetch_only_requests_the_named_screens_and_ignores_unknown_keys(ps):
    ps.fake.responses = {ps.PUBLIC_URLS["piotroski_9"]: _screen("A")}
    assert ps.fetch(only=["piotroski_9", "not_a_screen"], verbose=False) == {"piotroski_9": 1}
    assert ps.fake.fetched == [ps.PUBLIC_URLS["piotroski_9"]]


def test_fetch_without_a_filter_tries_every_mapped_screen(ps):
    ps.fetch(verbose=False)
    assert ps.fake.fetched == list(ps.PUBLIC_URLS.values())  # all of them, in table order


def test_fetch_skips_empty_or_symbol_less_tables(ps, capsys):
    ps.fake.responses = {
        ps.PUBLIC_URLS["piotroski_9"]: pd.DataFrame(),  # blocked / empty
        ps.PUBLIC_URLS["magic_formula"]: pd.DataFrame({"Name": ["a"]}),  # no Symbol column
        ps.PUBLIC_URLS["coffee_can"]: _screen("OK"),
    }
    got = ps.fetch(only=["piotroski_9", "magic_formula", "coffee_can"])
    assert got == {"coffee_can": 1}
    assert (
        not (ps.CACHE / "piotroski_9.parquet").exists()
        and not (ps.CACHE / "magic_formula.parquet").exists()
    )
    out = capsys.readouterr().out
    assert (
        "piotroski_9: empty/invalid — skipped" in out
        and "magic_formula: empty/invalid — skipped" in out
    )


def test_fetch_skips_a_table_that_has_a_symbol_column_but_no_rows(ps):
    ps.fake.responses = {ps.PUBLIC_URLS["piotroski_9"]: pd.DataFrame({"Symbol": []})}
    assert ps.fetch(only=["piotroski_9"], verbose=False) == {}
    assert not (ps.CACHE / "piotroski_9.parquet").exists()


def test_fetch_keeps_the_scrapers_own_output_quiet(ps):
    ps.fake.responses = {ps.PUBLIC_URLS["piotroski_9"]: _screen("A")}
    ps.fetch(only=["piotroski_9"], verbose=True)
    assert ps.fake.verbose_flags == [False]  # only this module reports progress


def test_fetch_isolates_a_failing_screen(ps, capsys):
    ps.fake.responses = {
        ps.PUBLIC_URLS["piotroski_9"]: RuntimeError("HTTP 429 too many requests"),
        ps.PUBLIC_URLS["magic_formula"]: _screen("X", "Y"),
    }
    assert ps.fetch(only=["piotroski_9", "magic_formula"]) == {
        "magic_formula": 2
    }  # the failure does not stop the rest
    assert "piotroski_9: fetch failed (HTTP 429 too many requests)" in capsys.readouterr().out
    assert ps.sleeps  # and the pacing between requests is still applied


def test_fetch_is_quiet_when_not_verbose(ps, capsys):
    ps.fake.responses = {ps.PUBLIC_URLS["piotroski_9"]: _screen("A")}
    ps.fetch(only=["piotroski_9"], verbose=False)
    assert capsys.readouterr().out == ""


def test_fetch_uses_an_overridden_url(ps):
    (ps.tmp / "public_screens.json").write_text(
        json.dumps({"piotroski_9": "https://example.test/mine/"})
    )
    ps.fake.responses = {"https://example.test/mine/": _screen("Z")}
    assert ps.fetch(only=["piotroski_9"], verbose=False) == {"piotroski_9": 1}
    assert ps.fake.fetched == ["https://example.test/mine/"]


# ---- cached_screens ---------------------------------------------------------------------


def test_cached_screens_reads_symbols_per_screen_without_the_network(ps):
    _screen("A", "B").to_parquet(ps.CACHE / "piotroski_9.parquet", index=False)
    pd.DataFrame({"Symbol": [101, 202, 101]}).to_parquet(
        ps.CACHE / "numeric.parquet", index=False
    )  # non-text symbols
    out = ps.cached_screens()
    assert out == {
        "piotroski_9": {"A", "B"},
        "numeric": {"101", "202"},
    }  # keyed by file stem; symbols as text, de-duplicated
    assert ps.fake.fetched == []


def test_cached_screens_skips_unreadable_or_symbol_less_files(ps):
    _screen("A").to_parquet(ps.CACHE / "good.parquet", index=False)
    (ps.CACHE / "broken.parquet").write_bytes(b"not a parquet file")
    pd.DataFrame({"Name": ["x"]}).to_parquet(ps.CACHE / "nosymbol.parquet", index=False)
    assert ps.cached_screens() == {"good": {"A"}}
    (ps.CACHE / "ignored.txt").write_text("not a cache file")
    assert set(ps.cached_screens()) == {"good"}


def test_cached_screens_is_empty_for_an_empty_cache(ps):
    assert ps.cached_screens() == {}


# ---- main -------------------------------------------------------------------------------


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["public_screens.py", *args])


def test_main_without_flags_lists_the_cache_and_does_not_fetch(ps, monkeypatch, capsys):
    _screen("A", "B").to_parquet(ps.CACHE / "small.parquet", index=False)
    _screen("A", "B", "C").to_parquet(ps.CACHE / "large.parquet", index=False)
    _argv(monkeypatch)
    assert ps.main() == 0
    out = capsys.readouterr().out
    assert "cached public screens: 2" in out and out.index("large") < out.index(
        "small"
    )  # biggest first
    assert ps.fake.fetched == []


def test_main_fetch_pulls_the_named_screens_and_reports_totals(ps, monkeypatch, capsys):
    ps.fake.responses = {ps.PUBLIC_URLS["piotroski_9"]: _screen("A", "B")}
    _argv(monkeypatch, "--fetch", "--only", "piotroski_9")
    assert ps.main() == 0
    out = capsys.readouterr().out
    assert "cached public screens" not in out  # --fetch alone does not also list
    assert f"fetched 1 screens; total mapped: {len(ps.PUBLIC_URLS)}" in out
    assert ps.fake.fetched == [ps.PUBLIC_URLS["piotroski_9"]]  # --only limits the fetch


def test_main_counts_override_urls_in_the_mapped_total(ps, monkeypatch, capsys):
    (ps.tmp / "public_screens.json").write_text(
        json.dumps({"brand_new": "https://example.test/n/"})
    )
    _argv(monkeypatch, "--fetch", "--only", "brand_new")
    assert ps.main() == 0
    assert f"total mapped: {len(ps.PUBLIC_URLS) + 1}" in capsys.readouterr().out


def test_main_list_and_fetch_together_do_both(ps, monkeypatch, capsys):
    _screen("A").to_parquet(ps.CACHE / "old.parquet", index=False)
    ps.fake.responses = {ps.PUBLIC_URLS["piotroski_9"]: _screen("A", "B")}
    _argv(monkeypatch, "--list", "--fetch", "--only", "piotroski_9")
    assert ps.main() == 0
    out = capsys.readouterr().out
    assert "cached public screens: 1" in out and "fetched 1 screens" in out
