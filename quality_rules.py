"""Per-source data-quality rules for ``data_validator.DataValidator``.

Each rule is a pure function ``(df) -> list[str]`` returning human-readable error
strings (empty list = pass).  Rules are held in the ordered ``RULES`` registry so a
new rule is added with ``@register_rule`` without editing the validator.  Rule order
matters: ``check_date_gaps`` converts the date column to datetime in place, exactly
as the original inline code did, and later rules/consumers see that conversion.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, List, Optional

import pandas as pd

Rule = Callable[[pd.DataFrame], List[str]]

HIGH_NULL_FRACTION = 0.1        # a column is flagged when > 10% of its values are null
MAX_DATE_GAP_DAYS = 5           # a gap is a consecutive-row step of more than this
MAX_DATE_GAPS = 10              # more gaps than this is flagged
STALE_AFTER_DAYS = 30           # last observation older than this is flagged
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
OHLC_COLUMNS = ("open", "high", "low", "close")
QUALITY_PENALTY_PER_ERROR = 0.1

RULES: List[Rule] = []


def register_rule(fn: Rule) -> Rule:
    """Append ``fn`` to the ordered rule registry and return it unchanged."""
    RULES.append(fn)
    return fn


def _date_column(df: pd.DataFrame) -> Optional[str]:
    """Return 'date' or 'Date' if present (lower-case preferred), else None."""
    if "date" in df.columns:
        return "date"
    if "Date" in df.columns:
        return "Date"
    return None


@register_rule
def check_missing(df: pd.DataFrame) -> List[str]:
    """Flag columns with more than HIGH_NULL_FRACTION nulls."""
    missing_pct = df.isnull().sum() / len(df)
    if (missing_pct > HIGH_NULL_FRACTION).any():
        return [f"High nulls in {missing_pct[missing_pct > HIGH_NULL_FRACTION].index.tolist()}"]
    return []


@register_rule
def check_duplicates(df: pd.DataFrame) -> List[str]:
    """Flag fully duplicated rows."""
    if df.duplicated().sum() > 0:
        return [f"{df.duplicated().sum()} duplicate rows found"]
    return []


@register_rule
def check_numeric_types(df: pd.DataFrame) -> List[str]:
    """Flag OHLCV columns (case-insensitive name match) that are not numeric."""
    errors = []
    for col in df.columns:
        if col.lower() in PRICE_COLUMNS and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' should be numeric but is {df[col].dtype}")
    return errors


@register_rule
def check_ohlc_order(df: pd.DataFrame) -> List[str]:
    """Flag rows where high < low (lower-case open/high/low/close columns only)."""
    if all(c in df.columns for c in OHLC_COLUMNS):
        invalid = (df["high"] < df["low"]).sum()
        if invalid > 0:
            return [f"{invalid} rows with high < low (invalid OHLC)"]
    return []


@register_rule
def check_date_gaps(df: pd.DataFrame) -> List[str]:
    """Flag time series with many gaps; converts the date column to datetime in place."""
    date_col = _date_column(df)
    if date_col is None:
        return []
    df[date_col] = pd.to_datetime(df[date_col])
    gaps = (df[date_col].diff() > timedelta(days=MAX_DATE_GAP_DAYS)).sum()
    if gaps > MAX_DATE_GAPS:
        return [f"Many date gaps detected ({gaps} gaps > {MAX_DATE_GAP_DAYS} days)"]
    return []


@register_rule
def check_freshness(df: pd.DataFrame) -> List[str]:
    """Flag series whose latest date is more than STALE_AFTER_DAYS old."""
    date_col = _date_column(df)
    if date_col is None:
        return []
    max_date = pd.to_datetime(df[date_col]).max()
    days_stale = (datetime.now().date() - max_date.date()).days
    if days_stale > STALE_AFTER_DAYS:
        return [f"Data is {days_stale} days stale (last date: {max_date.date()})"]
    return []


def run_rules(df: pd.DataFrame, rules: Optional[List[Rule]] = None) -> List[str]:
    """Run every rule in order and concatenate their error strings."""
    errors: List[str] = []
    for rule in RULES if rules is None else rules:
        errors.extend(rule(df))
    return errors


def quality_score(n_errors: int) -> float:
    """Score in [0, 1]: 1.0 minus 0.1 per error, floored at 0."""
    return max(0, 1.0 - n_errors * QUALITY_PENALTY_PER_ERROR)
