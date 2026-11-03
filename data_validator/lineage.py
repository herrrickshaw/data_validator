"""Data lineage tracking through loads and transforms."""

from datetime import datetime
from typing import Dict, List

import pandas as pd


class DataLineageTracker:
    """Track data lineage through transformations."""

    def __init__(self):
        self.lineage = {}

    def track_load(self, source: str, target_name: str, df: pd.DataFrame):
        """Track data loading."""
        self.lineage[target_name] = {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "records": len(df),
            "columns": df.columns.tolist(),
        }

    def track_transform(self, source_name: str, target_name: str, operation: str):
        """Track data transformation."""
        if source_name not in self.lineage:
            self.lineage[source_name] = {}

        self.lineage[target_name] = {
            "parent": source_name,
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
        }

    def get_lineage(self, data_name: str) -> List[Dict]:
        """Get full lineage chain for a dataset."""
        chain = []
        current = data_name

        while current in self.lineage:
            chain.append(self.lineage[current])
            current = self.lineage[current].get("parent")

        return chain
