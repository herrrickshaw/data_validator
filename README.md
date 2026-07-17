# data_validator

Rescued 2026-07-17 from a stray `working-files` dumping-ground repo (subfolder
`python_files/`), where this was the only surviving copy. Populated into this
repo because it already existed under this name, empty.

**Scope note**: despite the repo name, this is broader than a data-validation
library — it's a market-data screening/backtesting project with a
domain-driven-design layout (`src/{backtesting,market_data,portfolio_strategy,
reporting,risk_signals,shared,stock_analysis}/`), plus `data_validator.py`/
`data_config.py`/`repo_data_analyzer.py` (the actual validation framework),
plus vCRUD repository-management tooling (`.vcrud_bin/`, `.bt_bin/`).

The project's own `MOVE_TO_STANDALONE_REPO.txt` (kept below for reference)
names its originally-intended destination as `herrrickshaw/repository-systems`
with the description "Repository management with vCRUD, deduplication,
indexing" — a different name/framing than "data_validator". Both names are
defensible depending on which part of this repo you're using it for; kept as
`data_validator` since that repo already existed.

**`src/` overlap note**: the bounded-context names here
(`backtesting`/`market_data`/`portfolio_strategy`/`reporting`/`risk_signals`/
`stock_analysis`) match the archived `stock-screener-ddd` repo's structure
exactly. `stock-screener-ddd` was archived as "abandoned scaffold, mostly
empty stubs" — worth comparing the two `src/` trees before assuming this one
supersedes or duplicates that one; not verified as part of this extraction.

**vCRUD caution**: `.vcrud_bin/vcrud_manager.py` and `.vcrud_bin/db_handler.py`
are a variant of the same vCRUD tooling flagged elsewhere on this account as a
git-hijacking hazard (auto-commits, branch switches). Not activated by
anything in this repo as committed — review before running.

## Data not recovered

`cache_seed/*.parquet` (47 files) and `india_stocks_2021_2026.db` were
Git-LFS-tracked in the source `working-files` repo. The extraction used a
`--filter=blob:none` shallow clone that never fetched the actual LFS blob
content — only the ~130-byte pointer text was present locally, and the `.db`
file was 0 bytes. Rather than commit misleading stub files that look like
real data but aren't, they were removed from this extraction. `cache_seed/
DATA_CATALOG.md` and `CHECKSUMS.sha256` (kept) document what the real files
should contain if they're ever recovered from wherever their actual LFS
storage lives.

## What's real code vs. documentation

223 files total: ~45 top-level Python scripts (screeners, backtest weight
optimization/validation, data collectors, ML/RL experiments), a `src/`
DDD package, `.vcrud_bin`/`.bt_bin` tooling, `cache_seed/` (268KB of seed
parquet/data), `schedule/` (launchd-style scheduling configs), `tests/`, and
~30 root-level markdown docs (architecture notes, backtest findings,
data-deduplication strategy, implementation status/roadmap writeups).
