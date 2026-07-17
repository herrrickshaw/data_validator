#!/usr/bin/env python3
# sector_split.py
# ===============
# Repack the flat per-market LTM into a READABLE, sector + market-cap layout:
#
#   cache_seed/ltm_by_sector/<MARKET>/<Sector>__<CapTier>.parquet   (zstd-9)
#
# so you can browse/query a market by sector AND cap size instead of one 235 MB
# blob. Sector + market cap come from a single yfinance .info call per symbol,
# cached to cache_seed/sector_map.json (so the slow lookup runs once). Cap tiers
# are percentile-based WITHIN each market (currency-agnostic): Large = top 20% by
# market cap, Mid = next 30%, Small = rest; unlabelled → Unknown. Each file is
# deduped + zstd-9, so total size is unchanged — only readability improves.
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import market_memory as mm

HERE = Path(__file__).parent
OUT = HERE / "cache_seed" / "ltm_by_sector"
CACHE = HERE / "cache_seed" / "sector_map.json"

# yfinance suffix per market (LTM symbols are bare for US/IN, suffixed for JP/KR/CN)
_SUFFIX = {"IN": ".NS", "US": "", "JP": ".T", "KR": ".KS", "CN": ".SS", "HK": ".HK", "UK": ".L"}


def _load_cache() -> Dict[str, dict]:
    if CACHE.exists():
        try:
            raw = json.loads(CACHE.read_text())
            # migrate legacy {key: "Sector"} → {key: {"s": sector, "c": None}}
            return {k: (v if isinstance(v, dict) else {"s": v, "c": None}) for k, v in raw.items()}
        except Exception:
            return {}
    return {}


def _save_cache(m: Dict[str, dict]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(m))


def collect_info(
    symbols: List[str], market: str, limit: int = 1500, verbose: bool = True
) -> Dict[str, dict]:
    """Return {symbol: {'s': sector, 'c': market_cap}}, fetching missing via yfinance."""
    cache = _load_cache()
    suf = _SUFFIX.get(market, "")
    want = [s for s in symbols if cache.get(f"{market}:{s}", {}).get("c") is None][:limit]
    if want:
        import yfinance as yf

        t0 = time.time()
        for i, s in enumerate(want):
            tk = s if (suf and s.endswith(suf)) else s + suf
            try:
                info = yf.Ticker(tk).info
                sec = (
                    re.sub(r"[^A-Za-z0-9]+", "_", info.get("sector") or "Unknown").strip("_")
                    or "Unknown"
                )
                cap = info.get("marketCap")
            except Exception:
                sec, cap = "Unknown", None
            cache[f"{market}:{s}"] = {"s": sec, "c": cap}
            if verbose and (i + 1) % 200 == 0:
                print(f"  labelled {i+1}/{len(want)} ({time.time()-t0:.0f}s)", flush=True)
        _save_cache(cache)
    return {s: cache.get(f"{market}:{s}", {"s": "Unknown", "c": None}) for s in symbols}


def cap_tiers(info: Dict[str, dict]) -> Dict[str, str]:
    """Percentile cap tiers within the market: Large top 20%, Mid next 30%, Small rest."""
    caps = {s: d.get("c") for s, d in info.items()}
    known = {s: float(c) for s, c in caps.items() if c not in (None, 0) and np.isfinite(c)}
    if len(known) < 5:
        return {s: "Unknown" for s in caps}
    vals = np.array(list(known.values()))
    p80, p50 = np.percentile(vals, 80), np.percentile(vals, 50)
    out = {}
    for s, c in caps.items():
        if s not in known:
            out[s] = "Unknown"
        elif known[s] >= p80:
            out[s] = "Large"
        elif known[s] >= p50:
            out[s] = "Mid"
        else:
            out[s] = "Small"
    return out


def split_market(market: str, sector_limit: int = 1500, verbose: bool = True) -> dict:
    """Repack a market's LTM into per-(sector × cap-tier) zstd-9 parquet files."""
    from frames import long_to_dict, write_seed

    long = mm._read_long(mm._ltm_path(market))
    if long.empty:
        return {"market": market, "error": "empty LTM"}
    syms = sorted(long["Symbol"].unique())
    info = collect_info(syms, market, limit=sector_limit, verbose=verbose)
    sec_of = {s: d.get("s", "Unknown") for s, d in info.items()}
    tier_of = cap_tiers(info)

    long = long.copy()
    long["Sector"] = long["Symbol"].map(sec_of).fillna("Unknown")
    long["Cap"] = long["Symbol"].map(tier_of).fillna("Unknown")

    outdir = OUT / market
    outdir.mkdir(parents=True, exist_ok=True)
    counts: Dict[Tuple[str, str], dict] = {}
    for (sector, cap), grp in long.groupby(["Sector", "Cap"]):
        path = outdir / f"{sector}__{cap}.parquet"
        write_seed(long_to_dict(grp.drop(columns=["Sector", "Cap"])), path)
        counts[f"{sector}/{cap}"] = {
            "symbols": int(grp["Symbol"].nunique()),
            "rows": int(len(grp)),
            "mb": round(path.stat().st_size / 1e6, 1),
        }
    if verbose:
        print(f"\n  {market} → {len(counts)} sector×cap files under {outdir.relative_to(HERE)}")
        # cap-tier summary
        cap_summary = long.groupby("Cap")["Symbol"].nunique().to_dict()
        print(
            "  cap tiers:", {k: cap_summary.get(k, 0) for k in ("Large", "Mid", "Small", "Unknown")}
        )
        for name, c in sorted(counts.items(), key=lambda x: -x[1]["rows"])[:15]:
            print(f"    {name:34} {c['symbols']:>4} syms  {c['rows']:>9,} rows  {c['mb']:>5.1f} MB")
    return {"market": market, "files": counts}


def main() -> int:
    ap = argparse.ArgumentParser(description="Repack LTM by sector × market-cap tier")
    ap.add_argument("--market", default="US")
    ap.add_argument("--sector-limit", type=int, default=1500, help="max new yfinance lookups")
    a = ap.parse_args()
    split_market(a.market, sector_limit=a.sector_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
