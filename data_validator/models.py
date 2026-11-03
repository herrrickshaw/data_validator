"""Dataclasses shared by the validator: source metadata, issues and the final report."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DataSourceMetadata:
    """Metadata for a single data source."""

    source_name: str
    source_type: str  # "parquet", "csv", "api", "kaggle"
    path: str
    created_at: str
    last_modified: str
    file_size_mb: float
    record_count: int
    symbols_count: int
    date_range: Tuple[str, str]
    checksums: Dict[str, str] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class CrossSourceIssue:
    """Issue found when comparing across sources."""

    symbol: str
    date: str
    source_1: str
    source_2: str
    field: str
    value_1: Any
    value_2: Any
    discrepancy_pct: float


@dataclass
class DataLeakageIssue:
    """Data leakage issue in backtest/analysis."""

    file_path: str
    line_number: int
    issue_type: str  # "future_data", "test_in_train", "unfair_split"
    symbol: Optional[str] = None
    date_range: Optional[Tuple[str, str]] = None
    severity: str = "medium"


@dataclass
class ValidationReport:
    """Complete validation report."""

    generated_at: str
    total_sources: int
    total_records: int
    total_symbols: int
    avg_quality_score: float

    data_sources: List[DataSourceMetadata] = field(default_factory=list)
    cross_source_issues: List[CrossSourceIssue] = field(default_factory=list)
    leakage_issues: List[DataLeakageIssue] = field(default_factory=list)
    data_freshness_check: Dict[str, Any] = field(default_factory=dict)
    repo_data_mapping: Dict[str, List[str]] = field(default_factory=dict)

    missing_data_analysis: Dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def save_report(self, path: str):
        """Save report to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Report saved to {path}")

    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT SUMMARY")
        print("=" * 60)
        print(f"Generated: {self.generated_at}")
        print(f"\nTotal Data Sources: {self.total_sources}")
        print(f"Total Records: {self.total_records:,}")
        print(f"Unique Symbols: {self.total_symbols:,}")
        print(f"Average Quality Score: {self.avg_quality_score:.2%}")
        print(f"\n⚠️  Cross-Source Inconsistencies: {len(self.cross_source_issues)}")
        print(f"⚠️  Data Leakage Issues: {len(self.leakage_issues)}")
        print(f"\n{self.summary}")
