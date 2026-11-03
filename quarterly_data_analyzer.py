#!/usr/bin/env python3
"""
Quarterly Data Analysis - Extract & Summarize 5-Year Trends
Analyzes collected quarterly data for 8-D and 11-D expansion scoring
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class QuarterlyAnalyzer:
    """Extract expansion metrics from quarterly financial data"""

    def __init__(self):
        self.companies = []

    @staticmethod
    def _stmt_value(stmt, name: str, qtr_date):
        """stmt.loc[name, qtr_date], or None when the statement has no such line."""
        return stmt.loc[name, qtr_date] if name in stmt.index else None

    def _income_metrics(self, qtr_is, qtr_date, qtr_metrics: dict, v: dict) -> None:
        """Revenue, operating/net income, interest expense and margins; keeps locals in `v`."""
        v['total_revenue'] = self._stmt_value(qtr_is, 'Total Revenue', qtr_date)
        v['operating_income'] = self._stmt_value(qtr_is, 'Operating Income', qtr_date)
        v['net_income'] = self._stmt_value(qtr_is, 'Net Income', qtr_date)
        v['interest_expense'] = self._stmt_value(qtr_is, 'Interest Expense', qtr_date)

        qtr_metrics['revenue'] = v['total_revenue']
        qtr_metrics['operating_income'] = v['operating_income']
        qtr_metrics['net_income'] = v['net_income']
        qtr_metrics['interest_expense'] = v['interest_expense']

        if v['total_revenue'] and v['operating_income']:
            qtr_metrics['oi_margin'] = (v['operating_income'] / v['total_revenue'] * 100)
        if v['total_revenue'] and v['net_income']:
            qtr_metrics['ni_margin'] = (v['net_income'] / v['total_revenue'] * 100)

    def _cashflow_metrics(self, qtr_cf, qtr_date, qtr_metrics: dict, v: dict) -> None:
        """Operating cash flow, capex, FCF and FCF margin."""
        ocf = self._stmt_value(qtr_cf, 'Operating Cash Flow', qtr_date)
        capex = self._stmt_value(qtr_cf, 'Capital Expenditure', qtr_date)

        qtr_metrics['ocf'] = ocf
        qtr_metrics['capex'] = abs(capex) if capex else None

        if ocf and capex:
            qtr_metrics['fcf'] = ocf - abs(capex)
            total_revenue = v['total_revenue']  # unset if the income block failed, as before
            qtr_metrics['fcf_margin'] = (qtr_metrics['fcf'] / total_revenue * 100) if total_revenue else None

    def _balance_metrics(self, qtr_bs, qtr_date, qtr_metrics: dict, v: dict) -> None:
        """Assets, debt, equity, current ratio, D/E and asset turnover; keeps locals in `v`."""
        v['total_assets'] = self._stmt_value(qtr_bs, 'Total Assets', qtr_date)
        v['total_debt'] = self._stmt_value(qtr_bs, 'Total Debt', qtr_date)
        v['total_equity'] = self._stmt_value(qtr_bs, 'Total Stockholder Equity', qtr_date)
        has_both = 'Current Assets' in qtr_bs.index and 'Current Liabilities' in qtr_bs.index
        current_ratio = (
            (qtr_bs.loc['Current Assets', qtr_date] / qtr_bs.loc['Current Liabilities', qtr_date])
            if has_both else None
        )

        qtr_metrics['total_assets'] = v['total_assets']
        qtr_metrics['total_debt'] = v['total_debt']
        qtr_metrics['total_equity'] = v['total_equity']
        qtr_metrics['current_ratio'] = current_ratio

        if v['total_equity'] and v['total_equity'] > 0:
            qtr_metrics['de_ratio'] = v['total_debt'] / v['total_equity'] if v['total_debt'] else 0
        if v['total_assets'] and v['total_assets'] > 0:
            qtr_metrics['asset_turnover'] = (
                v['total_revenue'] / v['total_assets'] if v['total_revenue'] else None
            )

    @staticmethod
    def _roic_metric(qtr_metrics: dict, v: dict) -> None:
        """ROIC from NOPAT over invested capital (tax rate estimated from NI / operating income)."""
        operating_income, total_debt, total_equity = (
            v['operating_income'], v['total_debt'], v['total_equity']
        )
        if operating_income and total_debt and total_equity:
            tax_rate = 1 - (v['net_income'] / operating_income) if operating_income > 0 else 0.25
            nopat = operating_income * (1 - tax_rate)
            invested_capital = total_debt + total_equity
            if invested_capital > 0:
                qtr_metrics['roic'] = (nopat / invested_capital * 100)

    def _quarter_metrics(self, qtr_date, statements: tuple, v: dict) -> dict:
        """All metrics for one quarter. Each block is independent: a failure skips only that block."""
        qtr_is, qtr_cf, qtr_bs = statements
        qtr_metrics = {
            'date': qtr_date,
            'quarter_label': qtr_date.strftime('%Y-Q%q'),
        }
        blocks = (
            (self._income_metrics, qtr_is),
            (self._cashflow_metrics, qtr_cf),
            (self._balance_metrics, qtr_bs),
        )
        for block, stmt in blocks:
            try:
                block(stmt, qtr_date, qtr_metrics, v)
            except Exception as exc:
                logger.debug("%s skipped for %s: %s", block.__name__, qtr_date, exc)
        try:
            self._roic_metric(qtr_metrics, v)
        except Exception as exc:
            logger.debug("roic skipped for %s: %s", qtr_date, exc)
        return qtr_metrics

    def fetch_company_data(self, ticker: str) -> dict:
        """Fetch 5 years of quarterly data for a company"""
        try:
            stock = yf.Ticker(ticker)

            # Get quarterly statements (yfinance provides up to 4 years easily)
            qtr_is = stock.quarterly_income_stmt
            qtr_cf = stock.quarterly_cashflow
            qtr_bs = stock.quarterly_balance_sheet

            if qtr_is is None or qtr_cf is None or qtr_bs is None:
                return None

            # Get at least 20 quarters
            if len(qtr_is.columns) < 4:
                return None

            quarters = min(20, len(qtr_is.columns))
            quarters_dates = qtr_is.columns[:quarters].tolist()

            data = {
                'ticker': ticker,
                'quarters': [],
            }

            # `v` plays the role of the old function-scope locals: values from an earlier
            # quarter/block stay visible to later ones, and a never-assigned name raises KeyError
            # (formerly NameError, swallowed per block).
            v: dict = {}
            for qtr_date in quarters_dates:
                data['quarters'].append(
                    self._quarter_metrics(qtr_date, (qtr_is, qtr_cf, qtr_bs), v)
                )

            # Calculate 5-year trends
            data['trends'] = self._calculate_trends(data['quarters'])
            return data

        except Exception as exc:
            logger.debug("fetch_company_data(%s) failed: %s", ticker, exc)
            return None

    def _calculate_trends(self, quarters: list) -> dict:
        """Calculate expansion metrics from quarterly data"""
        trends = {}

        if len(quarters) < 4:
            return trends

        # Sort by date (most recent first)
        quarters_sorted = sorted(quarters, key=lambda x: x['date'], reverse=True)
        most_recent = quarters_sorted[0]
        oldest = quarters_sorted[-1]

        # Extract values
        def safe_cagr(newest_val, oldest_val, periods=5):
            if newest_val is None or oldest_val is None or oldest_val <= 0:
                return None
            return ((newest_val / oldest_val) ** (1/periods) - 1) * 100

        # CAGR calculations
        revenue_values = [q.get('revenue') for q in quarters_sorted if q.get('revenue')]
        capex_values = [q.get('capex') for q in quarters_sorted if q.get('capex')]
        debt_values = [q.get('total_debt') for q in quarters_sorted if q.get('total_debt')]
        fcf_values = [q.get('fcf') for q in quarters_sorted if q.get('fcf')]
        ocf_values = [q.get('ocf') for q in quarters_sorted if q.get('ocf')]

        if len(revenue_values) >= 2:
            trends['revenue_cagr'] = safe_cagr(revenue_values[0], revenue_values[-1])
        if len(capex_values) >= 2:
            trends['capex_cagr'] = safe_cagr(capex_values[0], capex_values[-1])
        if len(debt_values) >= 2:
            trends['debt_cagr'] = safe_cagr(debt_values[0], debt_values[-1])

        # Averages
        trends['avg_fcf'] = np.mean([v for v in fcf_values if v])
        trends['avg_ocf'] = np.mean([v for v in ocf_values if v])
        trends['avg_capex'] = np.mean([v for v in capex_values if v])

        # Margin trends
        oi_margins = [q.get('oi_margin') for q in quarters_sorted if q.get('oi_margin')]
        ni_margins = [q.get('ni_margin') for q in quarters_sorted if q.get('ni_margin')]
        roic_values = [q.get('roic') for q in quarters_sorted if q.get('roic')]

        if len(oi_margins) >= 2:
            trends['oi_margin_change'] = oi_margins[0] - oi_margins[-1]
        if len(ni_margins) >= 2:
            trends['ni_margin_change'] = ni_margins[0] - ni_margins[-1]
        if len(roic_values) >= 2:
            trends['roic_change'] = roic_values[0] - roic_values[-1]

        # Latest metrics
        trends['latest_de'] = most_recent.get('de_ratio')
        trends['latest_fcf_margin'] = most_recent.get('fcf_margin')
        trends['latest_asset_turnover'] = most_recent.get('asset_turnover')
        trends['latest_roic'] = most_recent.get('roic')

        # Debt service coverage (approximation)
        if most_recent.get('ocf') and most_recent.get('interest_expense'):
            trends['interest_coverage'] = most_recent['ocf'] / (most_recent['interest_expense'] + 0.01)

        return trends

    def score_company(self, data: dict) -> float:
        """Score company on 8-D expansion model"""
        if not data or not data.get('trends'):
            return 0

        score = 0
        trends = data['trends']

        # 1. Debt Expansion (should be low but positive) - 10%
        if trends.get('debt_cagr'):
            if 0 < trends['debt_cagr'] < 15:
                score += 10

        # 2. Capex Acceleration (should be high) - 20%
        if trends.get('capex_cagr'):
            if trends['capex_cagr'] > 15:
                score += 20

        # 3. Profit Reinvestment (revenue should grow) - 15%
        if trends.get('revenue_cagr'):
            if trends['revenue_cagr'] > 10:
                score += 15

        # 4. Profitability Quality (margins should improve or be high) - 15%
        if trends.get('ni_margin_change'):
            if trends['ni_margin_change'] > -2:  # Not declining too much
                score += 15

        # 5. Sustainability (FCF positive) - 15%
        if trends.get('avg_fcf') and trends['avg_fcf'] > 0:
            score += 15

        # 6. Leverage Health (D/E ratio healthy) - 10%
        if trends.get('latest_de'):
            if 0 < trends['latest_de'] < 1:
                score += 10

        # 7. FCF Generation (positive and growing) - 10%
        if trends.get('latest_fcf_margin') and trends['latest_fcf_margin'] > 2:
            score += 5

        # NEW: ROIC Trend (quality of expansion) - BONUS
        if trends.get('roic_change') and trends['roic_change'] > 0:
            score += 5  # Bonus points for improving ROIC

        return min(score, 100)

    def analyze_portfolio(self, tickers: list) -> pd.DataFrame:
        """Analyze all companies"""
        print("\n" + "="*80)
        print("QUARTERLY DATA ANALYSIS - 5-YEAR EXPANSION METRICS")
        print("="*80)

        print(f"\n📊 Analyzing {len(tickers)} companies...")

        results = []
        successful = 0

        for i, ticker in enumerate(tickers, 1):
            print(f"   [{i:2d}/{len(tickers)}] {ticker}...", end='\r', flush=True)

            data = self.fetch_company_data(ticker)
            if data:
                score = self.score_company(data)
                trends = data['trends']

                result = {
                    'ticker': ticker,
                    'score': score,
                    'revenue_cagr': trends.get('revenue_cagr'),
                    'capex_cagr': trends.get('capex_cagr'),
                    'debt_cagr': trends.get('debt_cagr'),
                    'avg_fcf': trends.get('avg_fcf'),
                    'oi_margin_change': trends.get('oi_margin_change'),
                    'ni_margin_change': trends.get('ni_margin_change'),
                    'roic_change': trends.get('roic_change'),  # NEW
                    'latest_de': trends.get('latest_de'),
                    'latest_roic': trends.get('latest_roic'),  # NEW
                    'interest_coverage': trends.get('interest_coverage'),
                }
                results.append(result)
                successful += 1

        print(f"\n✅ Analysis complete: {successful}/{len(tickers)} successful")

        df_results = pd.DataFrame(results).sort_values('score', ascending=False)

        # Display top performers
        print(f"\n📈 TOP 15 EXPANSION CANDIDATES (8-D SCORING):")
        print(f"{'Rank':<5} {'Ticker':<8} {'Score':<8} {'Rev.CAGR':<10} {'Capex.CAGR':<12} {'Debt.CAGR':<10} {'FCF':<10}")
        print("-" * 80)

        for idx, (i, row) in enumerate(df_results.head(15).iterrows(), 1):
            rev_cagr = f"{row['revenue_cagr']:.1f}%" if row['revenue_cagr'] else "N/A"
            capex_cagr = f"{row['capex_cagr']:.1f}%" if row['capex_cagr'] else "N/A"
            debt_cagr = f"{row['debt_cagr']:.1f}%" if row['debt_cagr'] else "N/A"
            avg_fcf = f"${row['avg_fcf']/1e9:.2f}B" if row['avg_fcf'] else "N/A"

            print(f"{idx:<5} {row['ticker']:<8} {row['score']:<8.0f} {rev_cagr:<10} {capex_cagr:<12} {debt_cagr:<10} {avg_fcf:<10}")

        # Summary statistics
        print(f"\n📊 PORTFOLIO SUMMARY:")
        print(f"   High expansion (75+): {len(df_results[df_results['score'] >= 75])}")
        print(f"   Medium expansion (50-75): {len(df_results[(df_results['score'] >= 50) & (df_results['score'] < 75)])}")
        print(f"   Low expansion (<50): {len(df_results[df_results['score'] < 50])}")

        print(f"\n💡 METRICS OVERVIEW:")
        print(f"   Avg revenue CAGR: {df_results['revenue_cagr'].mean():.1f}%")
        print(f"   Avg capex CAGR: {df_results['capex_cagr'].mean():.1f}%")
        print(f"   Avg debt CAGR: {df_results['debt_cagr'].mean():.1f}%")
        print(f"   Companies with improving ROIC: {len(df_results[df_results['roic_change'] > 0])}")  # NEW

        return df_results


if __name__ == "__main__":
    tickers = [
        'MSFT', 'NVDA', 'CRM', 'ADBE', 'AVGO', 'MSTR', 'FTNT', 'NET', 'DDOG', 'CRWD',
        'SNOW', 'PANW', 'ORCL', 'AAPL', 'IBM',
        'CAT', 'GE', 'LMT', 'BA', 'RTX', 'ISRG', 'ABB', 'EMR', 'DE', 'CARR',
        'TT', 'IRM', 'WAFER', 'FLEX', 'LOGI',
        'CVX', 'COP', 'SLB', 'MPC', 'OKE', 'NUE', 'FCX', 'CF', 'ALB', 'LYB', 'DOW', 'HUN',
        'DAL', 'UAL', 'ALK', 'NCLH', 'CCL', 'RCL',
        'PLD', 'DLR', 'EQIX', 'VICI', 'PSA', 'UMC', 'CCI', 'AMT',
        'JNJ', 'PFE', 'MRK', 'BIIB', 'LLY', 'BMY', 'AMGN', 'GILD'
    ]

    analyzer = QuarterlyAnalyzer()
    df_results = analyzer.analyze_portfolio(tickers[:60])

    print(f"\n✅ DATA COLLECTION & ANALYSIS COMPLETE")
    print(f"   Companies: {len(df_results)}")
    print(f"   Ready for: Phase 2 - Daily price correlation analysis")
    print(f"   Ready for: Phase 3 - Backtest weight validation")

