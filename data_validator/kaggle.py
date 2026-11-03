"""Kaggle dataset discovery (static catalogue; no network)."""

from typing import Dict, List


class KaggleDataIntegrator:
    """Discover and integrate Kaggle datasets."""

    @staticmethod
    def discover_relevant_datasets(
        markets: List[str] = None,
        categories: List[str] = None,
    ) -> List[Dict]:
        """Discover relevant stock datasets on Kaggle."""

        if markets is None:
            markets = ["India", "US", "Japan", "Korea"]
        if categories is None:
            categories = ["stock-market", "financial-data", "trading"]

        # These would be fetched from Kaggle API in production
        datasets = [
            {
                "title": "NSE/BSE Historical Stock Data",
                "url": "kaggle.com/datasets/rohanrao/nifty50-stock-market-data",
                "records": 1_500_000,
                "last_updated": "2024-06-30",
                "coverage": "India (NSE/BSE)",
            },
            {
                "title": "S&P 500 Historical Data",
                "url": "kaggle.com/datasets/camnugent/sandp500",
                "records": 8_000_000,
                "last_updated": "2024-06-30",
                "coverage": "US (S&P 500)",
            },
            {
                "title": "stock market data",
                "url": "kaggle.com/datasets/jacksoncrow/stock-market-dataset",
                "records": 50_000_000,
                "last_updated": "2024-06-30",
                "coverage": "US (Large Cap)",
            },
        ]

        return datasets
