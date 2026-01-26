# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yfinance",
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scipy",
# ]
# ///
"""
TOPIX-17 ETF Ranking Turnover Analysis

This script analyzes the ranking turnover of TOPIX-17 sector ETFs based on various return metrics.
It calculates returns using close-to-close and open-to-close methods across daily, weekly, and monthly frequencies,
then computes multiple turnover metrics to quantify ranking stability.

Turnover Metrics:
- Spearman rank correlation: Correlation between consecutive period rankings
- Kendall's tau: Concordance measure for ranking pairs
- Top-N turnover: Fraction of top-N sectors that change each period
- Mean absolute rank change: Average position change across all sectors
- Rank change RMS: Root mean square of rank changes

Usage:
    uv run 20260126_topix17_etf_turnover/20260126_topix17_etf_turnover.py
"""

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================================
# TOPIX-17 ETF Definitions
# ============================================================================

TOPIX17_ETFS = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービス他",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}


# ============================================================================
# Data Fetching
# ============================================================================


def fetch_etf_data(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """
    Fetch historical OHLCV data for TOPIX-17 ETFs using yfinance.

    Args:
        tickers: List of ticker symbols. If None, fetches all TOPIX-17 ETFs.

    Returns:
        Dictionary mapping ticker symbols to their OHLCV DataFrames.
    """
    if tickers is None:
        tickers = list(TOPIX17_ETFS.keys())

    data = {}
    print("Fetching ETF data...")
    for ticker in tickers:
        try:
            etf = yf.Ticker(ticker)
            df = etf.history(period="max")
            if not df.empty:
                data[ticker] = df
                print(f"  {ticker} ({TOPIX17_ETFS.get(ticker, 'Unknown')}): {len(df)} records")
            else:
                print(f"  {ticker}: No data available")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")

    return data


def create_price_dataframe(
    data: dict[str, pd.DataFrame], price_type: Literal["Open", "Close"] = "Close"
) -> pd.DataFrame:
    """
    Create a DataFrame with prices of all ETFs aligned by date.

    Args:
        data: Dictionary of ETF OHLCV data.
        price_type: Type of price to extract ('Open' or 'Close').

    Returns:
        DataFrame with dates as index and tickers as columns.
    """
    prices = {}
    for ticker, df in data.items():
        if price_type in df.columns:
            prices[ticker] = df[price_type]

    price_df = pd.DataFrame(prices)
    price_df.index = price_df.index.tz_localize(None)  # Remove timezone info
    return price_df


# ============================================================================
# Return Calculation
# ============================================================================


def calculate_returns(
    data: dict[str, pd.DataFrame],
    return_type: Literal["close_close", "open_close"] = "close_close",
    frequency: Literal["daily", "weekly", "monthly"] = "daily",
) -> pd.DataFrame:
    """
    Calculate returns based on specified type and frequency.

    Args:
        data: Dictionary of ETF OHLCV data.
        return_type: 'close_close' for Close-to-Close, 'open_close' for Open-to-Close.
        frequency: 'daily', 'weekly', or 'monthly'.

    Returns:
        DataFrame with returns for each ETF.
    """
    if return_type == "close_close":
        close_df = create_price_dataframe(data, "Close")

        if frequency == "daily":
            returns = close_df.pct_change()
        elif frequency == "weekly":
            weekly_close = close_df.resample("W-FRI").last()
            returns = weekly_close.pct_change()
        else:  # monthly
            monthly_close = close_df.resample("ME").last()
            returns = monthly_close.pct_change()

    else:  # open_close
        open_df = create_price_dataframe(data, "Open")
        close_df = create_price_dataframe(data, "Close")

        if frequency == "daily":
            returns = (close_df - open_df) / open_df
        elif frequency == "weekly":
            weekly_open = open_df.resample("W-FRI").first()
            weekly_close = close_df.resample("W-FRI").last()
            returns = (weekly_close - weekly_open) / weekly_open
        else:  # monthly
            monthly_open = open_df.resample("ME").first()
            monthly_close = close_df.resample("ME").last()
            returns = (monthly_close - monthly_open) / monthly_open

    return returns.dropna(how="all")


# ============================================================================
# Ranking Calculation
# ============================================================================


def calculate_rankings(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rankings based on returns (higher return = rank 1).

    Args:
        returns: DataFrame of returns.

    Returns:
        DataFrame with rankings for each period.
    """
    # Rank in descending order (highest return = rank 1)
    rankings = returns.rank(axis=1, ascending=False, method="average")
    return rankings


# ============================================================================
# Turnover Metrics
# ============================================================================


def spearman_rank_correlation(rankings: pd.DataFrame) -> pd.Series:
    """
    Calculate Spearman rank correlation between consecutive periods.

    Args:
        rankings: DataFrame of rankings.

    Returns:
        Series of Spearman correlations.
    """
    correlations = []
    dates = []

    for i in range(1, len(rankings)):
        prev_rank = rankings.iloc[i - 1].dropna()
        curr_rank = rankings.iloc[i].dropna()

        common_tickers = prev_rank.index.intersection(curr_rank.index)
        if len(common_tickers) >= 2:
            corr, _ = stats.spearmanr(prev_rank[common_tickers], curr_rank[common_tickers])
            correlations.append(corr)
            dates.append(rankings.index[i])

    return pd.Series(correlations, index=dates, name="Spearman Correlation")


def kendall_tau(rankings: pd.DataFrame) -> pd.Series:
    """
    Calculate Kendall's tau between consecutive periods.

    Args:
        rankings: DataFrame of rankings.

    Returns:
        Series of Kendall's tau values.
    """
    taus = []
    dates = []

    for i in range(1, len(rankings)):
        prev_rank = rankings.iloc[i - 1].dropna()
        curr_rank = rankings.iloc[i].dropna()

        common_tickers = prev_rank.index.intersection(curr_rank.index)
        if len(common_tickers) >= 2:
            tau, _ = stats.kendalltau(prev_rank[common_tickers], curr_rank[common_tickers])
            taus.append(tau)
            dates.append(rankings.index[i])

    return pd.Series(taus, index=dates, name="Kendall Tau")


def top_n_turnover(rankings: pd.DataFrame, n: int = 5) -> pd.Series:
    """
    Calculate the turnover rate in top-N rankings.

    Args:
        rankings: DataFrame of rankings.
        n: Number of top positions to consider.

    Returns:
        Series of turnover rates (0 = no change, 1 = complete change).
    """
    turnovers = []
    dates = []

    for i in range(1, len(rankings)):
        prev_rank = rankings.iloc[i - 1].dropna()
        curr_rank = rankings.iloc[i].dropna()

        prev_top_n = set(prev_rank[prev_rank <= n].index)
        curr_top_n = set(curr_rank[curr_rank <= n].index)

        if len(prev_top_n) > 0 and len(curr_top_n) > 0:
            # Count how many changed
            unchanged = len(prev_top_n.intersection(curr_top_n))
            turnover = 1 - (unchanged / n)
            turnovers.append(turnover)
            dates.append(rankings.index[i])

    return pd.Series(turnovers, index=dates, name=f"Top-{n} Turnover")


def mean_absolute_rank_change(rankings: pd.DataFrame) -> pd.Series:
    """
    Calculate the mean absolute rank change across all ETFs.

    Args:
        rankings: DataFrame of rankings.

    Returns:
        Series of mean absolute rank changes.
    """
    changes = []
    dates = []

    for i in range(1, len(rankings)):
        prev_rank = rankings.iloc[i - 1].dropna()
        curr_rank = rankings.iloc[i].dropna()

        common_tickers = prev_rank.index.intersection(curr_rank.index)
        if len(common_tickers) > 0:
            abs_change = np.abs(curr_rank[common_tickers] - prev_rank[common_tickers]).mean()
            changes.append(abs_change)
            dates.append(rankings.index[i])

    return pd.Series(changes, index=dates, name="Mean Absolute Rank Change")


def rank_change_rms(rankings: pd.DataFrame) -> pd.Series:
    """
    Calculate the root mean square of rank changes.

    Args:
        rankings: DataFrame of rankings.

    Returns:
        Series of RMS values.
    """
    rms_values = []
    dates = []

    for i in range(1, len(rankings)):
        prev_rank = rankings.iloc[i - 1].dropna()
        curr_rank = rankings.iloc[i].dropna()

        common_tickers = prev_rank.index.intersection(curr_rank.index)
        if len(common_tickers) > 0:
            squared_changes = (curr_rank[common_tickers] - prev_rank[common_tickers]) ** 2
            rms = np.sqrt(squared_changes.mean())
            rms_values.append(rms)
            dates.append(rankings.index[i])

    return pd.Series(rms_values, index=dates, name="Rank Change RMS")


def calculate_all_turnover_metrics(rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all turnover metrics for a given ranking DataFrame.

    Args:
        rankings: DataFrame of rankings.

    Returns:
        DataFrame with all turnover metrics.
    """
    metrics = pd.DataFrame(
        {
            "spearman_corr": spearman_rank_correlation(rankings),
            "kendall_tau": kendall_tau(rankings),
            "top5_turnover": top_n_turnover(rankings, n=5),
            "top3_turnover": top_n_turnover(rankings, n=3),
            "mean_abs_rank_change": mean_absolute_rank_change(rankings),
            "rank_change_rms": rank_change_rms(rankings),
        }
    )
    return metrics


# ============================================================================
# Visualization
# ============================================================================


def plot_turnover_time_series(
    metrics_dict: dict[str, pd.DataFrame],
    metric_name: str,
    title: str,
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot time series of a specific turnover metric across different return calculations.

    Args:
        metrics_dict: Dictionary mapping scenario names to metric DataFrames.
        metric_name: Name of the metric column to plot.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))

    for idx, (scenario, metrics) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        if metric_name in metrics.columns:
            data = metrics[metric_name].dropna()
            # Rolling average for smoother visualization
            rolling_mean = data.rolling(window=min(20, len(data) // 10 + 1), min_periods=1).mean()

            ax.plot(data.index, data.values, alpha=0.3, color=colors[idx], linewidth=0.5)
            ax.plot(rolling_mean.index, rolling_mean.values, color=colors[idx], linewidth=2, label="Rolling Mean")
            ax.set_title(scenario, fontsize=10)
            ax.set_xlabel("Date")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_all_metrics_comparison(metrics_dict: dict[str, pd.DataFrame], figsize: tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create a comprehensive comparison plot of all metrics.

    Args:
        metrics_dict: Dictionary mapping scenario names to metric DataFrames.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    metric_names = [
        "spearman_corr",
        "kendall_tau",
        "top5_turnover",
        "top3_turnover",
        "mean_abs_rank_change",
        "rank_change_rms",
    ]
    metric_labels = [
        "Spearman Correlation",
        "Kendall's Tau",
        "Top-5 Turnover",
        "Top-3 Turnover",
        "Mean |Rank Change|",
        "Rank Change RMS",
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    scenarios = list(metrics_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))

    for ax, metric_name, metric_label in zip(axes, metric_names, metric_labels):
        means = []
        stds = []

        for scenario in scenarios:
            if metric_name in metrics_dict[scenario].columns:
                data = metrics_dict[scenario][metric_name].dropna()
                means.append(data.mean())
                stds.append(data.std())
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(len(scenarios))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", alpha=0.8)

        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" ", "\n") for s in scenarios], fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Comparison of Turnover Metrics Across Return Types", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_ranking_heatmap(rankings: pd.DataFrame, title: str, figsize: tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Plot a heatmap of rankings over time.

    Args:
        rankings: DataFrame of rankings.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Rename columns to Japanese names
    rankings_renamed = rankings.rename(columns=TOPIX17_ETFS)

    # Transpose so that sectors are on y-axis
    rankings_t = rankings_renamed.T

    # Sample if too many dates
    if len(rankings_t.columns) > 100:
        sample_indices = np.linspace(0, len(rankings_t.columns) - 1, 100, dtype=int)
        rankings_t = rankings_t.iloc[:, sample_indices]

    im = ax.imshow(rankings_t.values, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=17)

    # Set labels
    ax.set_yticks(np.arange(len(rankings_t.index)))
    ax.set_yticklabels(rankings_t.index, fontsize=9)

    # X-axis labels (sampled dates)
    n_labels = min(10, len(rankings_t.columns))
    label_indices = np.linspace(0, len(rankings_t.columns) - 1, n_labels, dtype=int)
    ax.set_xticks(label_indices)
    ax.set_xticklabels([rankings_t.columns[i].strftime("%Y-%m") for i in label_indices], rotation=45, ha="right")

    cbar = plt.colorbar(im, ax=ax, label="Rank (1=Best)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sector")

    plt.tight_layout()
    return fig


def plot_rank_stability(rankings: pd.DataFrame, title: str, figsize: tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot the stability of each sector's ranking (variance of rank over time).

    Args:
        rankings: DataFrame of rankings.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Calculate statistics
    mean_ranks = rankings.mean().sort_values()
    std_ranks = rankings.std().sort_values()

    # Rename to Japanese
    mean_ranks.index = [TOPIX17_ETFS.get(t, t) for t in mean_ranks.index]
    std_ranks.index = [TOPIX17_ETFS.get(t, t) for t in std_ranks.index]

    # Mean rank plot
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(mean_ranks)))
    axes[0].barh(mean_ranks.index, mean_ranks.values, color=colors)
    axes[0].set_xlabel("Mean Rank")
    axes[0].set_title("Average Ranking by Sector")
    axes[0].axvline(x=9, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis="x")

    # Rank std plot
    std_sorted = rankings.std().sort_values()
    std_sorted.index = [TOPIX17_ETFS.get(t, t) for t in std_sorted.index]
    colors_std = plt.cm.Blues(np.linspace(0.3, 0.9, len(std_sorted)))
    axes[1].barh(std_sorted.index, std_sorted.values, color=colors_std)
    axes[1].set_xlabel("Rank Standard Deviation")
    axes[1].set_title("Ranking Volatility by Sector")
    axes[1].grid(True, alpha=0.3, axis="x")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_top5_composition_over_time(rankings: pd.DataFrame, title: str, figsize: tuple[int, int] = (16, 8)) -> plt.Figure:
    """
    Visualize how the top-5 composition changes over time.

    Args:
        rankings: DataFrame of rankings.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create binary matrix: 1 if in top 5, 0 otherwise
    top5_matrix = (rankings <= 5).astype(int)
    top5_matrix = top5_matrix.rename(columns=TOPIX17_ETFS)

    # Sample if too many dates
    if len(top5_matrix) > 200:
        sample_indices = np.linspace(0, len(top5_matrix) - 1, 200, dtype=int)
        top5_matrix = top5_matrix.iloc[sample_indices]

    # Stack plot
    ax.stackplot(
        top5_matrix.index,
        top5_matrix.T.values,
        labels=top5_matrix.columns,
        alpha=0.8,
    )

    ax.set_ylabel("Top-5 Composition")
    ax.set_xlabel("Date")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_ylim(0, 5.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rolling_turnover(metrics_dict: dict[str, pd.DataFrame], window: int = 20, figsize: tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Plot rolling average of turnover metrics.

    Args:
        metrics_dict: Dictionary mapping scenario names to metric DataFrames.
        window: Rolling window size.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    metric_pairs = [
        ("spearman_corr", "Spearman Correlation (Rolling Mean)"),
        ("top5_turnover", "Top-5 Turnover Rate (Rolling Mean)"),
        ("mean_abs_rank_change", "Mean |Rank Change| (Rolling Mean)"),
        ("rank_change_rms", "Rank Change RMS (Rolling Mean)"),
    ]

    for ax, (metric_name, metric_label) in zip(axes, metric_pairs):
        for scenario, metrics in metrics_dict.items():
            if metric_name in metrics.columns:
                data = metrics[metric_name].dropna()
                if len(data) > window:
                    rolling = data.rolling(window=window, min_periods=1).mean()
                    ax.plot(rolling.index, rolling.values, label=scenario, alpha=0.8)

        ax.set_title(metric_label)
        ax.set_xlabel("Date")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"Rolling {window}-Period Average of Turnover Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ============================================================================
# Main Analysis
# ============================================================================


def run_full_analysis(save_figures: bool = True, output_dir: str = ".") -> dict:
    """
    Run the complete analysis pipeline.

    Args:
        save_figures: Whether to save figures to files.
        output_dir: Directory to save figures.

    Returns:
        Dictionary containing all analysis results.
    """
    print("=" * 60)
    print("TOPIX-17 ETF Ranking Turnover Analysis")
    print("=" * 60)

    # 1. Fetch data
    print("\n[1/4] Fetching ETF data...")
    data = fetch_etf_data()

    if len(data) == 0:
        print("Error: No data fetched. Please check your internet connection.")
        return {}

    # 2. Calculate returns for all 6 scenarios
    print("\n[2/4] Calculating returns...")
    return_types = ["close_close", "open_close"]
    frequencies = ["daily", "weekly", "monthly"]

    returns_dict = {}
    rankings_dict = {}
    metrics_dict = {}

    for ret_type in return_types:
        for freq in frequencies:
            scenario_name = f"{ret_type.replace('_', '-')} {freq}"
            print(f"  Processing: {scenario_name}")

            returns = calculate_returns(data, return_type=ret_type, frequency=freq)
            rankings = calculate_rankings(returns)

            returns_dict[scenario_name] = returns
            rankings_dict[scenario_name] = rankings

    # 3. Calculate turnover metrics
    print("\n[3/4] Calculating turnover metrics...")
    for scenario_name, rankings in rankings_dict.items():
        print(f"  Processing: {scenario_name}")
        metrics = calculate_all_turnover_metrics(rankings)
        metrics_dict[scenario_name] = metrics

    # 4. Generate visualizations
    print("\n[4/4] Generating visualizations...")

    figures = {}

    # 4.1 Metrics comparison bar chart
    print("  - Metrics comparison chart")
    fig = plot_all_metrics_comparison(metrics_dict)
    figures["metrics_comparison"] = fig
    if save_figures:
        fig.savefig(f"{output_dir}/01_metrics_comparison.png", dpi=150, bbox_inches="tight")

    # 4.2 Time series for each metric
    for metric_name, metric_label in [
        ("spearman_corr", "Spearman Rank Correlation"),
        ("top5_turnover", "Top-5 Turnover Rate"),
        ("mean_abs_rank_change", "Mean Absolute Rank Change"),
    ]:
        print(f"  - Time series: {metric_label}")
        fig = plot_turnover_time_series(
            metrics_dict,
            metric_name,
            f"{metric_label} Over Time",
        )
        figures[f"timeseries_{metric_name}"] = fig
        if save_figures:
            fig.savefig(f"{output_dir}/02_timeseries_{metric_name}.png", dpi=150, bbox_inches="tight")

    # 4.3 Rolling turnover
    print("  - Rolling turnover metrics")
    fig = plot_rolling_turnover(metrics_dict, window=20)
    figures["rolling_turnover"] = fig
    if save_figures:
        fig.savefig(f"{output_dir}/03_rolling_turnover.png", dpi=150, bbox_inches="tight")

    # 4.4 Heatmaps for selected scenarios
    for scenario in ["close-close monthly", "close-close weekly"]:
        if scenario in rankings_dict:
            print(f"  - Ranking heatmap: {scenario}")
            fig = plot_ranking_heatmap(rankings_dict[scenario], f"Ranking Heatmap ({scenario})")
            figures[f"heatmap_{scenario.replace(' ', '_')}"] = fig
            if save_figures:
                fig.savefig(f"{output_dir}/04_heatmap_{scenario.replace(' ', '_').replace('-', '_')}.png", dpi=150, bbox_inches="tight")

    # 4.5 Rank stability
    print("  - Rank stability analysis")
    fig = plot_rank_stability(rankings_dict["close-close monthly"], "Rank Stability (Close-Close Monthly)")
    figures["rank_stability"] = fig
    if save_figures:
        fig.savefig(f"{output_dir}/05_rank_stability.png", dpi=150, bbox_inches="tight")

    # 4.6 Top-5 composition over time
    print("  - Top-5 composition over time")
    fig = plot_top5_composition_over_time(
        rankings_dict["close-close monthly"], "Top-5 Sector Composition Over Time (Close-Close Monthly)"
    )
    figures["top5_composition"] = fig
    if save_figures:
        fig.savefig(f"{output_dir}/06_top5_composition.png", dpi=150, bbox_inches="tight")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    summary_data = []
    for scenario, metrics in metrics_dict.items():
        summary_data.append(
            {
                "Scenario": scenario,
                "Spearman (mean)": f"{metrics['spearman_corr'].mean():.3f}",
                "Top-5 Turnover (mean)": f"{metrics['top5_turnover'].mean():.3f}",
                "Mean |Rank Δ|": f"{metrics['mean_abs_rank_change'].mean():.2f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    if save_figures:
        print(f"Figures saved to: {output_dir}/")
    print("=" * 60)

    return {
        "data": data,
        "returns": returns_dict,
        "rankings": rankings_dict,
        "metrics": metrics_dict,
        "figures": figures,
    }


if __name__ == "__main__":
    import os

    # Set output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Set matplotlib to use a font that supports Japanese
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans", "Yu Gothic", "sans-serif"]

    # Run analysis
    results = run_full_analysis(save_figures=True, output_dir=output_dir)

    # Show plots
    plt.show()
