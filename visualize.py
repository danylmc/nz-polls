#!/usr/bin/env python3
"""
NZ Election Polling Data Visualization
Generates cross-election trend graphs from scraped polling data
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np


# Election years
ELECTION_YEARS = [1993, 1996, 1999, 2002, 2005, 2008, 2011, 2014, 2017, 2020, 2023, 2026]

# Party colors (official or commonly used)
PARTY_COLORS = {
    "National": "#00529F",      # Blue
    "Labour": "#D82A20",        # Red
    "Green": "#098137",         # Green
    "ACT": "#FDE401",           # Yellow
    "NZ First": "#000000",      # Black
    "Te Pāti Māori": "#B2001A", # Maroon
    "TOP": "#32DAC3",           # Teal
    "United Future": "#501557", # Purple
    "Alliance": "#00AA4E",      # Green (darker)
    "Progressive": "#D82A20",   # Red variant
    "Mana": "#770808",          # Dark red
    "Conservative": "#00AEEF",  # Light blue
    "New Conservative": "#00AEEF",
}

# Major parties for main graphs
MAJOR_PARTIES = ["National", "Labour"]
THIRD_PARTIES = ["Green", "ACT", "NZ First", "Te Pāti Māori"]


def load_all_data(data_dir: Path) -> pd.DataFrame:
    """Load all polling data into a single DataFrame"""
    all_polls = []

    for year in ELECTION_YEARS:
        file_path = data_dir / f"{year}_polling.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for poll in data.get("polls", []):
            row = {
                "election_year": year,
                "date": poll.get("date"),
                "pollster": poll.get("pollster"),
                "sample_size": poll.get("sample_size"),
            }
            # Add party percentages
            for party, pct in poll.get("parties", {}).items():
                row[party] = pct

            all_polls.append(row)

    df = pd.DataFrame(all_polls)

    # Convert date to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def plot_national_vs_labour(df: pd.DataFrame, output_dir: Path):
    """Create National vs Labour historical comparison"""
    print("Creating National vs Labour history graph...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Filter to rows with both National and Labour data
    mask = df["National"].notna() & df["Labour"].notna() & df["date"].notna()
    plot_df = df[mask].sort_values("date")

    if plot_df.empty:
        print("  No data available for National vs Labour plot")
        return

    # Plot rolling averages for smoother visualization
    window = max(5, len(plot_df) // 50)  # Adaptive window size

    for party in MAJOR_PARTIES:
        if party in plot_df.columns:
            # Calculate rolling average
            rolling = plot_df.set_index("date")[party].rolling(window=window, min_periods=1).mean()
            ax.plot(rolling.index, rolling.values,
                   label=party, color=PARTY_COLORS.get(party, "#888888"),
                   linewidth=2, alpha=0.8)

            # Also plot raw data points with transparency
            ax.scatter(plot_df["date"], plot_df[party],
                      color=PARTY_COLORS.get(party, "#888888"),
                      alpha=0.15, s=10)

    # Add election date markers
    election_dates = {
        1993: "1993-11-06", 1996: "1996-10-12", 1999: "1999-11-27",
        2002: "2002-07-27", 2005: "2005-09-17", 2008: "2008-11-08",
        2011: "2011-11-26", 2014: "2014-09-20", 2017: "2017-09-23",
        2020: "2020-10-17", 2023: "2023-10-14", 2026: "2026-11-21"
    }

    for year, date_str in election_dates.items():
        try:
            date = pd.to_datetime(date_str)
            if plot_df["date"].min() <= date <= plot_df["date"].max():
                ax.axvline(x=date, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.text(date, ax.get_ylim()[1], f"'{str(year)[2:]}",
                       fontsize=8, ha='center', va='bottom', color='gray')
        except:
            pass

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Support (%)", fontsize=12)
    ax.set_title("National vs Labour Support: 1993-2026\nNew Zealand Election Polling", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 60)

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    output_path = output_dir / "national_vs_labour_history.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def plot_third_party_evolution(df: pd.DataFrame, output_dir: Path):
    """Create third party evolution graph"""
    print("Creating third party evolution graph...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Get available third parties
    available_parties = [p for p in THIRD_PARTIES if p in df.columns and df[p].notna().any()]

    if not available_parties:
        print("  No third party data available")
        return

    # Filter to rows with date
    mask = df["date"].notna()
    plot_df = df[mask].sort_values("date")

    window = max(5, len(plot_df) // 50)

    for party in available_parties:
        party_mask = plot_df[party].notna()
        if party_mask.sum() < 3:
            continue

        party_df = plot_df[party_mask]

        # Calculate rolling average
        rolling = party_df.set_index("date")[party].rolling(window=window, min_periods=1).mean()
        ax.plot(rolling.index, rolling.values,
               label=party, color=PARTY_COLORS.get(party, "#888888"),
               linewidth=2, alpha=0.8)

        # Raw data points
        ax.scatter(party_df["date"], party_df[party],
                  color=PARTY_COLORS.get(party, "#888888"),
                  alpha=0.15, s=10)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Support (%)", fontsize=12)
    ax.set_title("Third Party Evolution: 1993-2026\nRise of Minor Parties Under MMP", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 25)

    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    output_path = output_dir / "third_party_evolution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def plot_two_party_vote_share(df: pd.DataFrame, output_dir: Path):
    """Create combined National+Labour share graph showing vote fragmentation"""
    print("Creating two-party vote share graph...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Filter to rows with both parties
    mask = df["National"].notna() & df["Labour"].notna() & df["date"].notna()
    plot_df = df[mask].copy()

    if plot_df.empty:
        print("  No data available for two-party vote share plot")
        return

    plot_df = plot_df.sort_values("date")
    plot_df["combined"] = plot_df["National"] + plot_df["Labour"]

    window = max(5, len(plot_df) // 50)

    # Rolling average of combined share
    rolling = plot_df.set_index("date")["combined"].rolling(window=window, min_periods=1).mean()

    ax.fill_between(rolling.index, 0, rolling.values, alpha=0.3, color='purple', label='Combined share')
    ax.plot(rolling.index, rolling.values, color='purple', linewidth=2)

    # Add 50% line for reference
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')

    # Mark MMP introduction (1996)
    mmp_date = pd.to_datetime("1996-10-12")
    if plot_df["date"].min() <= mmp_date:
        ax.axvline(x=mmp_date, color='green', linestyle='-', alpha=0.7, linewidth=2)
        ax.text(mmp_date, ax.get_ylim()[1] * 0.95, 'MMP\nIntroduced',
               fontsize=9, ha='center', va='top', color='green')

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Combined Support (%)", fontsize=12)
    ax.set_title("Two-Party Vote Share: National + Labour Combined\nVote Fragmentation Since MMP (1996)", fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)

    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    output_path = output_dir / "two_party_vote_share.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def plot_party_support_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap of average party support by election year"""
    print("Creating party support heatmap...")

    # Get all party columns
    party_cols = [col for col in df.columns if col in PARTY_COLORS.keys()]

    if not party_cols:
        # Try to find any columns that look like parties
        exclude_cols = {'election_year', 'date', 'pollster', 'sample_size'}
        party_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]

    if not party_cols:
        print("  No party data available for heatmap")
        return

    # Calculate average support by election year
    avg_support = df.groupby("election_year")[party_cols].mean()

    # Filter to parties with sufficient data
    avg_support = avg_support.loc[:, avg_support.notna().sum() >= 3]

    if avg_support.empty:
        print("  Insufficient data for heatmap")
        return

    # Sort columns by total support
    col_order = avg_support.sum().sort_values(ascending=False).index
    avg_support = avg_support[col_order]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    sns.heatmap(avg_support.T, annot=True, fmt='.1f', cmap='YlOrRd',
               ax=ax, cbar_kws={'label': 'Average Support (%)'})

    ax.set_xlabel("Election Year", fontsize=12)
    ax.set_ylabel("Party", fontsize=12)
    ax.set_title("Average Party Support by Election Year\nNew Zealand 1993-2026", fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / "party_support_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def main():
    """Main function to generate all visualizations"""
    print("NZ Election Polling Visualization")
    print("=" * 40)

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "graphs"
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading polling data...")
    df = load_all_data(data_dir)

    if df.empty:
        print("No data found! Run scraper.py first.")
        return

    print(f"Loaded {len(df)} polling records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Available parties: {[col for col in df.columns if col in PARTY_COLORS]}")

    # Generate graphs
    print("\nGenerating visualizations...")
    plot_national_vs_labour(df, output_dir)
    plot_third_party_evolution(df, output_dir)
    plot_two_party_vote_share(df, output_dir)
    plot_party_support_heatmap(df, output_dir)

    print("\n" + "=" * 40)
    print("Visualization complete!")
    print(f"Graphs saved to: {output_dir}")


if __name__ == "__main__":
    main()
