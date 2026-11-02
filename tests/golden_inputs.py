"""Deterministic synthetic inputs for the data_validator golden tests (no network, no real data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _ohlc(n: int = 30) -> pd.DataFrame:
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.arange(n) + 1000,
        }
    )


def golden_frames() -> dict:
    """Name -> DataFrame, each triggering a distinct subset of rules."""
    clean = _ohlc()
    clean["date"] = pd.date_range("2020-01-01", periods=30, freq="D").astype(str)
    dup = pd.concat([clean, clean.iloc[:2]], ignore_index=True)
    nulls = clean.copy()
    nulls.loc[:5, "close"] = np.nan
    badohlc = clean.copy()
    badohlc.loc[3, ["high", "low"]] = [90.0, 95.0]
    badtype = clean.copy()
    badtype["Volume"] = ["x"] * 30
    gaps = _ohlc(14)
    gaps["Date"] = pd.to_datetime(["2020-01-01"]).repeat(14) + pd.to_timedelta(
        np.arange(14) * 9, "D"
    )
    nodate = _ohlc(5)
    return {
        "clean_stale": clean,
        "dup": dup,
        "nulls": nulls,
        "badohlc": badohlc,
        "badtype": badtype,
        "gaps_capital_date": gaps,
        "nodate": nodate,
    }


def write_base(base: Path) -> Path:
    """Write the golden frames as CSVs under ``base/cache_seed`` plus two leakage-flagged scripts."""
    seed = base / "cache_seed"
    seed.mkdir(parents=True, exist_ok=True)
    for name, df in golden_frames().items():
        df.to_csv(seed / f"{name}.csv", index=False)
    (base / "model.py").write_text(
        "x = df.shift(-1)\ntrain = all_data\nr = df.sample().frac\nok = 1\n"
    )
    (base / "clean.py").write_text("print('hello clean')\n")
    return base
