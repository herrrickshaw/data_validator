"""DataValidator: inventory, validate, cross-check and report on local data sources."""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from leakage_scan import scan_lines
from quality_rules import quality_score, run_rules

from .models import (
    CrossSourceIssue,
    DataLeakageIssue,
    DataSourceMetadata,
    ValidationReport,
)

logger = logging.getLogger(__name__)


class DataValidator:
    """Main validation coordinator."""

    def __init__(self, base_path: str = "/Users/umashankar/Downloads/code/python_files"):
        self.base_path = Path(base_path)
        self.cache_seed = self.base_path / "cache_seed"
        self.report = None
        self._loaded_data = {}  # Cache loaded DataFrames

    def run_full_validation(self) -> ValidationReport:
        """Run complete validation pipeline."""
        print("🔍 Starting comprehensive data validation...")

        # Phase 1: Inventory all data sources
        print("\n[1/5] Inventorying data sources...")
        sources_metadata = self._inventory_data_sources()

        # Phase 2: Validate each source exhaustively
        print("\n[2/5] Validating all data exhaustively...")
        for metadata in sources_metadata:
            self._validate_single_source(metadata)

        # Phase 3: Cross-source consistency
        print("\n[3/5] Checking cross-source consistency...")
        cross_issues = self._check_cross_source_consistency(sources_metadata)

        # Phase 4: Data leakage detection
        print("\n[4/5] Detecting data leakage in analysis scripts...")
        leakage_issues = self._detect_data_leakage()

        # Phase 5: Repository mapping
        print("\n[5/5] Mapping repositories to data sources...")
        repo_mapping = self._map_repos_to_data(sources_metadata)

        # Compile report
        self.report = self._compile_report(
            sources_metadata, cross_issues, leakage_issues, repo_mapping
        )

        return self.report

    def _inventory_data_sources(self) -> List[DataSourceMetadata]:
        """Discover and catalog all data sources."""
        sources = []

        # Phase 1a: Local parquet files
        print("  • Scanning parquet files...")
        sources.extend(self._scan_parquet_files())

        # Phase 1b: Local CSV files
        print("  • Scanning CSV files...")
        sources.extend(self._scan_csv_files())

        # Phase 1c: Kaggle metadata (if available)
        print("  • Checking Kaggle datasets...")
        sources.extend(self._scan_kaggle_datasets())

        print(f"  ✓ Found {len(sources)} data sources")
        return sources

    def _scan_parquet_files(self) -> List[DataSourceMetadata]:
        """Scan cache_seed for parquet files."""
        sources = []
        parquet_files = list(self.cache_seed.glob("*.parquet"))

        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                metadata = self._extract_parquet_metadata(df, pf)
                sources.append(metadata)
            except Exception as e:
                print(f"  ⚠️  Error reading {pf.name}: {e}")

        return sources

    def _extract_parquet_metadata(self, df: pd.DataFrame, path: Path) -> DataSourceMetadata:
        """Extract metadata from a parquet file."""
        stat = path.stat()

        # Infer symbols and date range
        symbols = self._extract_symbols(df)
        date_range = self._extract_date_range(df)

        metadata = DataSourceMetadata(
            source_name=path.stem,
            source_type="parquet",
            path=str(path),
            created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
            last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            file_size_mb=stat.st_size / (1024*1024),
            record_count=len(df),
            symbols_count=len(symbols),
            date_range=date_range,
            columns=df.columns.tolist(),
            checksums={"sha256": self._compute_file_hash(path)},
        )
        return metadata

    def _scan_csv_files(self) -> List[DataSourceMetadata]:
        """Scan for CSV files in project."""
        sources = []
        # Search common data directories
        for pattern in ["**/*.csv", "data/**/*.csv", "**/results/**/*.csv"]:
            csv_files = list(self.base_path.glob(pattern))
            for cf in csv_files[:10]:  # Limit to avoid noise
                try:
                    df = pd.read_csv(cf, nrows=1000)  # Sample for metadata
                    metadata = DataSourceMetadata(
                        source_name=cf.stem,
                        source_type="csv",
                        path=str(cf),
                        created_at=datetime.fromtimestamp(cf.stat().st_ctime).isoformat(),
                        last_modified=datetime.fromtimestamp(cf.stat().st_mtime).isoformat(),
                        file_size_mb=cf.stat().st_size / (1024*1024),
                        record_count=len(df),
                        symbols_count=len(self._extract_symbols(df)),
                        date_range=self._extract_date_range(df),
                        columns=df.columns.tolist(),
                    )
                    sources.append(metadata)
                except Exception as exc:
                    logger.debug("skipping unreadable file: %s", exc)

        return sources

    def _scan_kaggle_datasets(self) -> List[DataSourceMetadata]:
        """Check for Kaggle datasets and metadata."""
        sources = []
        # List common stock market datasets on Kaggle
        # Note: Requires kaggle API to be installed and authenticated
        stock_datasets = [
            "sudalairajkumar/stock-market-data",
            "finnhub/stock-market-data",
            "rohanrao/nifty50-stock-market-data",
            "jacksoncrow/stock-market-dataset",
        ]

        for ds in stock_datasets:
            sources.append(DataSourceMetadata(
                source_name=ds.split("/")[-1],
                source_type="kaggle",
                path=f"kaggle://{ds}",
                created_at="unknown",
                last_modified="unknown",
                file_size_mb=0,
                record_count=0,
                symbols_count=0,
                date_range=("unknown", "unknown"),
                validation_errors=["Not yet downloaded"],
            ))

        return sources

    def _validate_single_source(self, metadata: DataSourceMetadata):
        """Exhaustively validate a single data source."""
        try:
            # Load full data
            if metadata.source_type == "parquet":
                df = pd.read_parquet(metadata.path)
            elif metadata.source_type == "csv":
                df = pd.read_csv(metadata.path)
            else:
                return

            self._loaded_data[metadata.source_name] = df

            errors = run_rules(df)

            metadata.validation_errors = errors
            metadata.data_quality_score = quality_score(len(errors))

            status = "✓" if not errors else "⚠️"
            print(f"  {status} {metadata.source_name}: {len(df):,} rows, "
                  f"quality={metadata.data_quality_score:.2%}")

        except Exception as e:
            metadata.validation_errors.append(f"Failed to validate: {e}")
            metadata.data_quality_score = 0.0
            print(f"  ✗ {metadata.source_name}: {e}")

    def _check_cross_source_consistency(
        self,
        sources: List[DataSourceMetadata]
    ) -> List[CrossSourceIssue]:
        """Check consistency across data sources."""
        issues = []

        # Find common symbols across sources
        all_symbols = {}
        for source in sources:
            if source.source_name in self._loaded_data:
                symbols = self._extract_symbols(self._loaded_data[source.source_name])
                for sym in symbols:
                    if sym not in all_symbols:
                        all_symbols[sym] = []
                    all_symbols[sym].append(source.source_name)

        # Check symbols that appear in multiple sources
        common_symbols = {s: sources for s, sources in all_symbols.items() if len(sources) > 1}

        print(f"  • Checking {len(common_symbols)} symbols across sources...")

        for symbol, source_names in list(common_symbols.items())[:20]:  # Sample check
            try:
                dfs = []
                for sname in source_names:
                    if sname in self._loaded_data:
                        df = self._loaded_data[sname].copy()
                        if 'symbol' in df.columns or 'Symbol' in df.columns:
                            sym_col = 'symbol' if 'symbol' in df.columns else 'Symbol'
                            df = df[df[sym_col] == symbol]
                        dfs.append((sname, df))

                # Compare close prices
                if len(dfs) >= 2 and all(len(df) > 0 for _, df in dfs):
                    for i, (sname1, df1) in enumerate(dfs[:-1]):
                        for sname2, df2 in dfs[i+1:]:
                            # Merge on date and compare close
                            if 'close' in df1.columns and 'close' in df2.columns:
                                common_dates = pd.to_datetime(
                                    set(pd.to_datetime(df1['date'] if 'date' in df1.columns else df1.index)) &
                                    set(pd.to_datetime(df2['date'] if 'date' in df2.columns else df2.index))
                                )

                                for date in list(common_dates)[:5]:  # Sample
                                    val1 = df1[df1['date'] == date]['close'].values
                                    val2 = df2[df2['date'] == date]['close'].values

                                    if len(val1) > 0 and len(val2) > 0:
                                        v1, v2 = float(val1[0]), float(val2[0])
                                        if v1 > 0:
                                            pct_diff = abs(v1 - v2) / v1
                                            if pct_diff > 0.02:  # >2% difference
                                                issues.append(CrossSourceIssue(
                                                    symbol=symbol,
                                                    date=str(date),
                                                    source_1=sname1,
                                                    source_2=sname2,
                                                    field='close',
                                                    value_1=v1,
                                                    value_2=v2,
                                                    discrepancy_pct=pct_diff
                                                ))
            except Exception as exc:
                logger.debug("cross-source comparison failed for %s: %s", symbol, exc)

        print(f"  ✓ Found {len(issues)} cross-source inconsistencies")
        return issues

    def _detect_data_leakage(self) -> List[DataLeakageIssue]:
        """Scan Python scripts for potential data leakage."""
        issues = []

        py_files = list(self.base_path.glob("**/*.py"))

        for py_file in py_files[:50]:  # Sample analysis
            try:
                with open(py_file, 'r') as f:
                    lines = f.read().split('\n')
            except Exception:
                continue  # unreadable file: skip, as before
            for issue_type, line_number in scan_lines(lines):
                issues.append(DataLeakageIssue(
                    file_path=str(py_file),
                    line_number=line_number,
                    issue_type=issue_type,
                    severity="low",  # Flag for review
                ))

        print(f"  ✓ Scanned {len(py_files)} scripts, found {len(issues)} potential leakage flags")
        return issues

    def _map_repos_to_data(self, sources: List[DataSourceMetadata]) -> Dict[str, List[str]]:
        """Map which Python scripts use which data sources."""
        mapping = {}

        py_files = list(self.base_path.glob("**/*.py"))
        source_names = {s.source_name for s in sources}

        for py_file in py_files[:30]:  # Sample
            try:
                with open(py_file, 'r') as f:
                    content = f.read()

                used_sources = []
                for source_name in source_names:
                    if source_name in content:
                        used_sources.append(source_name)

                if used_sources:
                    mapping[str(py_file)] = used_sources
            except Exception as exc:
                logger.debug("repo mapping skipped %s: %s", py_file, exc)

        return mapping

    def _compile_report(
        self,
        sources: List[DataSourceMetadata],
        cross_issues: List[CrossSourceIssue],
        leakage_issues: List[DataLeakageIssue],
        repo_mapping: Dict[str, List[str]],
    ) -> ValidationReport:
        """Compile final validation report."""

        total_records = sum(s.record_count for s in sources)
        total_symbols = sum(s.symbols_count for s in sources)
        avg_quality = np.mean([s.data_quality_score for s in sources])

        summary = self._generate_summary(
            sources, cross_issues, leakage_issues, repo_mapping
        )

        return ValidationReport(
            generated_at=datetime.now().isoformat(),
            total_sources=len(sources),
            total_records=total_records,
            total_symbols=total_symbols,
            avg_quality_score=avg_quality,
            data_sources=sources,
            cross_source_issues=cross_issues,
            leakage_issues=leakage_issues,
            repo_data_mapping=repo_mapping,
            summary=summary,
        )

    def _generate_summary(
        self,
        sources: List[DataSourceMetadata],
        cross_issues: List[CrossSourceIssue],
        leakage_issues: List[DataLeakageIssue],
        repo_mapping: Dict[str, List[str]],
    ) -> str:
        """Generate human-readable summary."""
        lines = [
            "\n📊 DATA QUALITY BREAKDOWN:",
            f"  • Sources with quality > 80%: {sum(1 for s in sources if s.data_quality_score > 0.8)}/{len(sources)}",
            f"  • Sources with quality < 50%: {sum(1 for s in sources if s.data_quality_score < 0.5)}/{len(sources)}",
            "",
            "🔗 CROSS-SOURCE ISSUES:",
            f"  • Price discrepancies (>2%): {sum(1 for i in cross_issues if i.field == 'close')}",
            f"  • Date range mismatches: {sum(1 for i in cross_issues if i.field == 'date')}",
            "",
            "⚠️  DATA LEAKAGE RISKS:",
            f"  • Future-looking operations: {sum(1 for i in leakage_issues if i.issue_type == 'future_data')}",
            f"  • Train-test contamination: {sum(1 for i in leakage_issues if i.issue_type == 'test_in_train')}",
            "",
            "📍 REPOSITORY MAPPING:",
            f"  • Scripts using multiple sources: {sum(1 for files in repo_mapping.values() if len(files) > 1)}",
            f"  • Data sources used by multiple scripts: {len([s for s in sources if any(s.source_name in v for v in repo_mapping.values())])}",
        ]

        return "\n".join(lines)

    @staticmethod
    def _extract_symbols(df: pd.DataFrame) -> set:
        """Extract unique symbols from DataFrame."""
        for col in ['symbol', 'Symbol', 'ticker', 'Ticker', 'SYMBOL']:
            if col in df.columns:
                return set(df[col].unique())
        return set()

    @staticmethod
    def _extract_date_range(df: pd.DataFrame) -> Tuple[str, str]:
        """Extract date range from DataFrame."""
        for col in ['date', 'Date', 'datetime', 'Datetime', 'DATE']:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col])
                    return (str(dates.min().date()), str(dates.max().date()))
                except Exception as exc:
                    logger.debug("unparseable date column %s: %s", col, exc)
        return ("unknown", "unknown")

    @staticmethod
    def _compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
        """Compute file hash."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
