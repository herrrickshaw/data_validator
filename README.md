# data_validator

A market-data validation framework plus a broader screening and backtesting workbench for equities (India-centred, with a
20-market data layer). The core validator inventories local parquet/CSV data sources, runs per-source quality rules,
cross-checks sources against each other, flags possible look-ahead leakage in analysis scripts, and writes a JSON report.
The surrounding code holds screeners, backtest weight optimisation, data collectors, ML/RL experiments and a
domain-driven-design package under `src/`.

> Status: research-only snapshot, rescued 2026-07-17 from a stray `working-files` repo (subfolder `python_files/`),
> where it was the only surviving copy. Educational use only; not investment advice. No licence file yet — all rights reserved.

## What it does
- `DataValidator.run_full_validation()` inventories `cache_seed/*.parquet`, CSVs and a Kaggle watch-list, then validates each source.
- Six per-source quality rules (nulls, duplicates, OHLCV dtypes, high < low, date gaps, staleness) feed a 0–1 quality score.
- Cross-source close-price comparison on a sample of shared symbols (flags > 2% differences).
- Regex heuristics for leakage in `*.py` scripts (negative shifts, train/test overlap, unseeded splits).
- Screeners and ML layers: `auto_screener.py`, `validation.py`, `ml_supervised.py`, `rl_trader.py`, `serving_layer.py`.
- Backtest weight comparison (`backtest_weight_optimization.py`, `backtest_weight_validation.py`), tiered OHLCV memory
  (`market_memory.py`, `bulk_seed.py`) and an ingest gate (`ingest_gate.py`).

## Quick start
```bash
git clone https://github.com/herrrickshaw/data_validator.git
cd data_validator
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy pyarrow pytest      # the validator needs only pandas, numpy (+ pyarrow for parquet)
python data_validator.py                     # validates DataValidator's default base path, writes /tmp/data_validation_report.json
```
Programmatic use, pointing at your own data directory (it must contain a `cache_seed/` folder):
```python
from data_validator import DataValidator
report = DataValidator(base_path="/path/to/data").run_full_validation()
report.save_report("data_validation_report.json")
```

## How it works
| Stage | Module | Output |
|---|---|---|
| 1. Inventory | `data_validator.py` (`_scan_parquet_files`, `_scan_csv_files`, `_scan_kaggle_datasets`) | `DataSourceMetadata` per source |
| 2. Per-source rules | `quality_rules.py` (ordered `RULES` registry) | error strings + quality score |
| 3. Cross-source check | `data_validator.py` (`_check_cross_source_consistency`) | `CrossSourceIssue` list |
| 4. Leakage scan | `leakage_scan.py` (`LEAKAGE_PATTERNS`, pure `scan_lines`) | `DataLeakageIssue` list |
| 5. Repo→data map and report | `data_validator.py` (`_map_repos_to_data`, `_compile_report`) | `ValidationReport` (JSON) |

Design notes: each quality rule is a pure `DataFrame -> list[str]` function added to the registry with `@register_rule`, so
new rules need no edit to the validator. Rule order is part of the contract (the date-gap rule converts the date column to
datetime in place, as the original inline code did). Quality score is `max(0, 1 - 0.1 * n_errors)`; thresholds are named
constants at the top of `quality_rules.py`.

## Repository layout
| Path | Purpose |
|---|---|
| `data_validator.py`, `quality_rules.py`, `leakage_scan.py` | Validation framework (entry point and extracted rule/pattern modules) |
| `data_config.py`, `repo_data_analyzer.py`, `inventory_all_data.py` | Data configuration and repository/data inventory |
| top-level `*.py` (about 45) | Screeners, backtests, collectors, ML/RL experiments |
| `src/{backtesting,market_data,portfolio_strategy,reporting,risk_signals,shared,stock_analysis}/` | Domain-driven-design package |
| `.vcrud_bin/`, `.bt_bin/` | vCRUD repository-management and backtest tooling (see caution below) |
| `cache_seed/` | Small seed data and `DATA_CATALOG.md` / `CHECKSUMS.sha256` describing the missing parquet files |
| `schedule/` | launchd-style scheduling configs |
| `tests/` | pytest suite; `tests/test_data_validator_golden.py` pins the validator's verdicts on synthetic inputs |
| `*.md` at root | About 30 architecture notes, backtest findings, roadmaps and status write-ups (not all current) |

## Data sources
Quality rules run on whatever parquet/CSV files sit under the base path. The Kaggle watch-list in
`_scan_kaggle_datasets` (e.g. `sudalairajkumar/stock-market-data`, `rohanrao/nifty50-stock-market-data`) only reports
"Not yet downloaded"; the validator performs no network calls. Other modules use yfinance, NSE/BSE and screener.in
(`screener_in_auth.py` logs in with your own credentials); check each source's terms before use.

## Configuration
- `DataValidator(base_path=...)`: defaults to a hard-coded author path (`/Users/umashankar/Downloads/code/python_files`),
  so pass your own. `python data_validator.py` always writes its report to `/tmp/data_validation_report.json`.
- Credentials (screener.in, brokers) are read by individual modules and are never committed.

## Testing
```bash
pytest -q
```
Four test modules (`test_cleaning.py`, `test_liquidity_and_custom.py`, `test_strategies.py`, `test_store.py`) import modules
that did not survive the extraction (`stock_utils`, `custom_screener`, `strategies`, `bhavcopy_store`) and fail at
collection; the rest pass. CI (`.github/workflows/tests.yml`) runs ruff, black and pytest.

## Limitations and known issues
- Data not recovered: `cache_seed/*.parquet` (47 files) and `india_stocks_2021_2026.db` were Git-LFS-tracked in the source
  repo, and the extraction only fetched ~130-byte pointer files, so they were dropped rather than committed as misleading
  stubs. `cache_seed/DATA_CATALOG.md` and `CHECKSUMS.sha256` document what they should contain.
- Scope vs. name: `MOVE_TO_STANDALONE_REPO.txt` names `herrrickshaw/repository-systems` ("Repository management with vCRUD,
  deduplication, indexing") as the intended home. The name `data_validator` was kept because the repo already existed.
- `src/` bounded-context names match the archived `stock-screener-ddd` repo ("abandoned scaffold, mostly empty stubs");
  the two trees have not been compared.
- vCRUD caution: `.vcrud_bin/vcrud_manager.py` and `db_handler.py` are a variant of vCRUD tooling flagged elsewhere on this
  account as a git-hijacking hazard (auto-commits, branch switches). Nothing here activates them; review before running.
- The leakage scan and repo mapping sample only the first 50 / 30 `*.py` files; leakage hits are regex heuristics for review.
- `_map_repos_to_data` matches by substring, and the cross-source check samples 20 symbols and 5 dates each.

## Related repositories
`stock-screener-ddd` (archived; same bounded-context layout) and the intended `repository-systems` home, both noted above.

## Licence
No licence file yet — all rights reserved.
