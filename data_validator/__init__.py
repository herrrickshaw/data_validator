"""Data Validation & Verification Framework (package; old ``import data_validator`` still works).

Usage:
    validator = DataValidator(base_path="/path/to/data")
    report = validator.run_full_validation()
    report.save_report("data_validation_report.json")

CLI: ``python3 -m data_validator`` (was ``python3 data_validator.py``).
"""

from .kaggle import KaggleDataIntegrator
from .lineage import DataLineageTracker
from .models import (
    CrossSourceIssue,
    DataLeakageIssue,
    DataSourceMetadata,
    ValidationReport,
)
from .validator import DataValidator

__all__ = [
    "CrossSourceIssue",
    "DataLeakageIssue",
    "DataLineageTracker",
    "DataSourceMetadata",
    "DataValidator",
    "KaggleDataIntegrator",
    "ValidationReport",
]
