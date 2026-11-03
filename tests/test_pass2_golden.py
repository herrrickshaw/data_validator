"""Pass-2 characterisation tests: outputs recorded from the code BEFORE the long functions were split.

Record with ``GOLDEN_RECORD=1 pytest tests/test_pass2_golden.py`` (only ever on the pre-split commit).
"""

from __future__ import annotations

import io
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOLDEN_DIR = Path(__file__).parent / "golden_pass2"
RECORD = bool(os.environ.get("GOLDEN_RECORD"))


def check(name: str, value) -> None:
    """Compare ``value`` (JSON-serialisable) to the stored golden, or record it."""
    path = GOLDEN_DIR / f"{name}.json"
    text = json.dumps(value, indent=1, sort_keys=False, default=repr)
    if RECORD:
        path.write_text(text + "\n")
        return
    assert text + "\n" == path.read_text()


# --- _define_countries -------------------------------------------------------------


def test_countries_config_golden():
    from global_20country_universe_screener import Global20CountryScreener

    check("countries", Global20CountryScreener().countries_config)


# --- screener_in_auth.company_financials --------------------------------------------

SHEET_ROWS = {
    "full": [
        ["Net profit", 100, 80, 60, 50],
        ["Sales", 1000, 900, 800, 700, 600, 500],
        ["Interest", 10, 12],
        ["Profit before tax", 150, 120],
        ["EPS in Rs", 5, 4],
        ["Borrowings", 200, 250],
        ["Reserves", 400, 350],
        ["Equity Capital", 50, 50],
        ["Other Liabilities", 100, 90],
        ["Net Block", 300, 280],
        ["Inventory", 40, 35],
        ["Trade receivables", 60, 55],
        ["Cash & Bank", 30, 25],
        ["Cash from Operating Activity", 120, 100],
        ["No. of Equity Shares", 10, 10],
        ["Dividend Amount", 5, 0, 4],
    ],
    "sparse": [["Sales", 500, 400], ["Net profit", 20]],
    "norows": [["Unrelated", 1, 2]],
}
PAGE = {
    "roe_page": 18.5,
    "roce_page": 22.1,
    "market_cap_cr": 5000,
    "current_price": 250.0,
    "pe": 20.0,
    "book_value": 80.0,
    "dividend_yield": 1.2,
}


def _xlsx(rows):
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, sheet_name="Data Sheet", header=False, index=False)
    return buf.getvalue()


@pytest.mark.parametrize("case", ["full", "sparse", "norows", "empty", "garbage"])
@pytest.mark.parametrize("page", [True, False])
def test_company_financials_golden(monkeypatch, case, page):
    import screener_in_auth as sa

    raw = {"empty": b"", "garbage": b"not an excel file"}.get(case)
    if raw is None:
        raw = _xlsx(SHEET_ROWS[case])
    monkeypatch.setattr(sa, "company_export_bytes", lambda s: raw)
    monkeypatch.setattr(sa, "page_ratios", lambda s: dict(PAGE) if page else {})
    check(f"fin_{case}_{page}", sa.company_financials("TEST"))


# --- backtest_weight_optimization.generate_report -----------------------------------


def _metrics(scale):
    return dict(
        n_signals=120,
        n_outperform=300,
        tp=70,
        fp=50,
        fn=230,
        tn=650,
        precision=0.58 * scale,
        recall=0.23 * scale,
        f1=0.33 * scale,
        accuracy=0.72 * scale,
        avg_buy_return=4.2 * scale,
        avg_sell_return=-1.1,
        return_spread=5.3 * scale,
    )


@pytest.mark.parametrize(
    "case", ["both_big", "both_modest", "both_worse", "baseline", "optimized", "none"]
)
def test_generate_report_golden(capsys, case):
    from backtest_weight_optimization import WeightBacktester

    bt = WeightBacktester(pd.DataFrame({"a": range(1000)}))
    base, opt = _metrics(1.0), _metrics(1.0)
    if case == "both_big":
        opt["f1"] = base["f1"] + 0.1
    elif case == "both_modest":
        opt["f1"] = base["f1"] + 0.01
    elif case == "both_worse":
        opt["f1"] = base["f1"] - 0.02
        opt["precision"] = base["precision"]
    results = (
        {
            "baseline": {
                "both_big": base,
                "both_modest": base,
                "both_worse": base,
                "baseline": base,
            }.get(case)
        }
        if case != "optimized"
        else {}
    )
    if case in ("both_big", "both_modest", "both_worse", "optimized"):
        results["optimized"] = opt
    results = {k: v for k, v in results.items() if v is not None}
    bt.generate_report(results)
    check(f"report_{case}", capsys.readouterr().out.split("\n"))


# --- quarterly_data_analyzer.fetch_company_data -------------------------------------


class _FakeTicker:
    def __init__(self, inc, cf, bs):
        self.quarterly_income_stmt, self.quarterly_cashflow, self.quarterly_balance_sheet = (
            inc,
            cf,
            bs,
        )


def _frames(n, drop=()):
    cols = pd.to_datetime(
        [
            f"2024-{12 - 3 * i:02d}-30" if i < 4 else f"{2024 - i // 4}-{12 - 3 * (i % 4):02d}-30"
            for i in range(n)
        ]
    )
    rng = np.random.default_rng(7)

    def mk(names):
        names = [x for x in names if x not in drop]
        return pd.DataFrame(
            rng.integers(100, 1000, (len(names), n)).astype(float), index=names, columns=cols
        )

    inc = mk(["Total Revenue", "Operating Income", "Net Income", "Interest Expense"])
    cf = mk(["Operating Cash Flow", "Capital Expenditure"])
    bs = mk(
        [
            "Total Assets",
            "Total Debt",
            "Total Stockholder Equity",
            "Current Assets",
            "Current Liabilities",
        ]
    )
    return inc, cf, bs


@pytest.mark.parametrize("case", ["ok", "few", "norev", "nocf", "none", "raises"])
def test_fetch_company_data_golden(monkeypatch, case):
    fake = types.ModuleType("yfinance")
    if case == "few":
        frames = _frames(3)
    elif case == "norev":
        frames = _frames(8, drop=("Total Revenue",))
    elif case == "nocf":
        frames = _frames(8, drop=("Operating Cash Flow", "Capital Expenditure"))
    elif case == "none":
        frames = (None, None, None)
    else:
        frames = _frames(8)

    def ticker(sym):
        if case == "raises":
            raise RuntimeError("boom")
        return _FakeTicker(*frames)

    fake.Ticker = ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    sys.modules.pop("quarterly_data_analyzer", None)
    import quarterly_data_analyzer as qa

    out = qa.QuarterlyAnalyzer().fetch_company_data("TEST")
    check(f"qtr_{case}", out)
    sys.modules.pop("quarterly_data_analyzer", None)
