"""
Walkability vs. housing prices: cleaning, EDA, statistics, and PDF report.

Run from project root:
  python scripts/generate_walkability_housing_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as mticker

# Project root (parent of scripts/)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIG_W, FIG_H = 11, 8.5  # letter-ish landscape for readability


def zfill_zip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    walk = pd.read_csv(DATA / "zip_walk_index.csv")
    zillow = pd.read_csv(DATA / "zillow_clean.csv")
    return walk, zillow


def clean_and_merge(walk: pd.DataFrame, zillow: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize ZIPs, merge, flag outliers, return cleaning log."""
    log: dict = {}

    walk = walk.copy()
    zillow = zillow.copy()

    walk["ZIP5"] = zfill_zip(walk["ZIP"])
    zillow["ZIP5"] = zfill_zip(zillow["ZIP"])

    log["walk_rows"] = len(walk)
    log["zillow_rows"] = len(zillow)

    merged = walk.merge(
        zillow,
        on="ZIP5",
        how="inner",
        suffixes=("_walk", "_zillow"),
    )
    log["merged_rows"] = len(merged)

    # Missing handling: drop rows with missing core fields
    before = len(merged)
    merged = merged.dropna(subset=["NatWalkInd", "home_value", "State"])
    log["dropped_missing_core"] = before - len(merged)

    # Extreme home values (likely data artifacts or ultra-luxury micro-markets)
    q1, q99 = merged["home_value"].quantile([0.01, 0.99])
    before_w = len(merged)
    merged_w = merged[(merged["home_value"] >= q1) & (merged["home_value"] <= q99)].copy()
    log["winsor_home_value_pct"] = (1, 99)
    log["home_value_q01"] = float(q1)
    log["home_value_q99"] = float(q99)
    log["dropped_extreme_home_value"] = before_w - len(merged_w)

    # Walkability bounds (already continuous index; clip tiny tails for robustness)
    w1, w99 = merged_w["NatWalkInd"].quantile([0.01, 0.99])
    merged_w = merged_w[
        (merged_w["NatWalkInd"] >= w1) & (merged_w["NatWalkInd"] <= w99)
    ].copy()
    log["walk_q01"] = float(w1)
    log["walk_q99"] = float(w99)

    return merged_w, log


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    desc = df[["NatWalkInd", "home_value"]].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    return desc


def state_level(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("State", as_index=False)
        .agg(
            n_zips=("ZIP5", "count"),
            mean_walk=("NatWalkInd", "mean"),
            median_home=("home_value", "median"),
            mean_home=("home_value", "mean"),
        )
        .sort_values("mean_walk", ascending=False)
    )
    return g


def correlations(df: pd.DataFrame) -> dict:
    x = df["NatWalkInd"].to_numpy()
    y = df["home_value"].to_numpy()
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    y_log = np.log1p(y)
    pl_r, pl_p = stats.pearsonr(x, y_log)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_log_price_r": float(pl_r),
        "pearson_log_price_p": float(pl_p),
    }


def ols_log_price(df: pd.DataFrame) -> dict:
    x = df[["NatWalkInd"]].to_numpy()
    y = np.log1p(df["home_value"].to_numpy())
    model = LinearRegression().fit(x, y)
    y_hat = model.predict(x)
    resid = y - y_hat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return {
        "intercept": float(model.intercept_),
        "slope_per_walk_unit": float(model.coef_[0]),
        "r2_log_price": float(r2),
    }


def add_title_page(pdf: PdfPages, log: dict, n_final: int) -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("#fafafa")
    title = "Walkability and Typical Home Values Across U.S. ZIP Codes"
    subtitle = (
        "Research question: In places where daily errands are easier to do on foot, "
        "are typical home values higher?\n"
        "Data: EPA National Walkability Index (ZIP) merged with Zillow-style typical "
        "home values (`zillow_clean.csv`)."
    )
    body = (
        f"Cleaning highlights:\n"
        f"  • ZIP codes standardized to 5 digits for a reliable merge.\n"
        f"  • Rows with missing walk score, price, or state removed "
        f"({log.get('dropped_missing_core', 0):,} rows).\n"
        f"  • Extreme home values trimmed at the 1st–99th percentile to limit "
        f"outlier leverage while keeping most of the market "
        f"(${log.get('home_value_q01', 0):,.0f} – ${log.get('home_value_q99', 0):,.0f}).\n"
        f"  • Walk scores trimmed at the 1st–99th percentile for robustness.\n\n"
        f"Final analytic sample: {n_final:,} ZIP codes after cleaning.\n\n"
        "Why it matters: Walkability is a proxy for land-use intensity, amenities, and "
        "transport options. If prices rise with walkability, that signals how households "
        "trade off housing costs against neighborhood accessibility—useful for planners, "
        "buyers, and researchers studying urban form and affordability."
    )
    fig.text(0.08, 0.88, title, fontsize=18, weight="bold", color="#1a1a1a")
    fig.text(0.08, 0.72, subtitle, fontsize=11, color="#333333", wrap=True)
    fig.text(0.08, 0.52, body, fontsize=10.5, color="#333333", va="top", linespacing=1.45)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_summary_table_page(pdf: PdfPages, desc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.axis("off")
    ax.set_title("Summary statistics (ZIP-level, cleaned sample)", fontsize=14, pad=12)
    table_df = desc.round(2)
    table = ax.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.scale(1.2, 2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, pdf: PdfPages) -> None:
    import matplotlib.ticker as mticker
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    sns.scatterplot(
        data=df.sample(min(8000, len(df)), random_state=42),
        x="NatWalkInd",
        y="home_value",
        alpha=0.25,
        s=12,
        ax=ax,
        color="#2c7fb8",
    )
    ax.set_title("Walk Score vs. Typical Home Value (random subsample)")
    ax.set_xlabel("National Walkability Index (higher = more walkable)")
    ax.set_ylabel("Typical home value ($)")
    ax.set_xlim(0, 17)

    dollar_fmt = mticker.FuncFormatter(
        lambda x, _: f"${x/1_000_000:.1f}M" if x >= 1_000_000 else f"${x/1000:.0f}k"
    )
    ax.yaxis.set_major_formatter(dollar_fmt)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_hex(df: pd.DataFrame, pdf: PdfPages) -> None:
    import matplotlib.ticker as mticker
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    hb = ax.hexbin(
        df["NatWalkInd"],
        df["home_value"],
        gridsize=45,
        cmap="viridis",
        mincnt=1,
        bins="log",
    )
    ax.set_title("Density of ZIP Codes (darker = more ZIPs)")
    ax.set_xlabel("National Walkability Index")
    ax.set_ylabel("Typical home value ($)")
    ax.set_xlim(2, 17)
    plt.colorbar(hb, ax=ax, label="log(count)")

    dollar_fmt = mticker.FuncFormatter(
        lambda x, _: f"${x/1_000_000:.1f}M" if x >= 1_000_000 else f"${x/1000:.0f}k"
    )
    ax.yaxis.set_major_formatter(dollar_fmt)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def plot_state_bars(state_df: pd.DataFrame, pdf: PdfPages) -> None:
    top = state_df.nlargest(15, "mean_walk")
    bottom = state_df.nsmallest(15, "mean_walk")
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=False)

    sns.barplot(
        data=top,
        y="State",
        x="mean_walk",
        hue="State",
        ax=axes[0],
        palette="crest",
        legend=False,
    )
    axes[0].set_xlim(0, 16)  # Force same scale
    axes[1].set_xlim(0, 16)  # Force same scale
    axes[0].set_title("States with highest mean ZIP walk scores")
    axes[0].set_xlabel("Mean walkability index")

    sns.barplot(
        data=bottom,
        y="State",
        x="mean_walk",
        hue="State",
        ax=axes[1],
        palette="flare",
        legend=False,
    )
    axes[1].set_title("States with lowest mean ZIP walk scores")
    axes[1].set_xlabel("Mean walkability index")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_state_walk_vs_price(state_df: pd.DataFrame, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.scatterplot(
        data=state_df,
        x="mean_walk",
        y="median_home",
        size="n_zips",
        sizes=(40, 400),
        alpha=0.75,
        ax=ax,
    )
    for _, row in state_df.iterrows():
        if row["n_zips"] >= 400:  # label larger states to reduce clutter
            ax.annotate(
                row["State"],
                (row["mean_walk"], row["median_home"]),
                textcoords="offset points",
                xytext=(4, 2),
                fontsize=8,
            )
    ax.set_title("State aggregates: mean walk score vs. median typical home value")
    ax.set_xlabel("Mean ZIP walkability index")
    ax.set_ylabel("Median typical home value ($)")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(df: pd.DataFrame, pdf: PdfPages) -> None:
    x = df["NatWalkInd"].to_numpy().reshape(-1, 1)
    y = np.log1p(df["home_value"].to_numpy())
    model = LinearRegression().fit(x, y)
    y_hat = model.predict(x)
    resid = y - y_hat
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.scatter(y_hat, resid, s=8, alpha=0.2, color="#8856a7")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Predicted log(1 + home value)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals from simple OLS: log home value ~ walk score")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_stats_page(pdf: PdfPages, corr: dict, ols: dict) -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("#fafafa")
    lines = [
        "Statistical analysis (ZIP-level, cleaned sample)",
        "",
        f"Pearson correlation (walk, price): r = {corr['pearson_r']:.3f}, "
        f"p = {corr['pearson_p']:.2e}",
        f"Spearman rank correlation: rho = {corr['spearman_r']:.3f}, "
        f"p = {corr['spearman_p']:.2e}",
        "",
        "Home values are right-skewed; log transform stabilizes variance.",
        f"Pearson correlation (walk, log1p(price)): r = {corr['pearson_log_price_r']:.3f}, "
        f"p = {corr['pearson_log_price_p']:.2e}",
        "",
        "Simple OLS: log1p(home_value) = beta0 + beta1 * NatWalkInd",
        f"  beta0 (intercept): {ols['intercept']:.4f}",
        f"  beta1 (per walk unit): {ols['slope_per_walk_unit']:.5f}",
        f"  R-squared on log scale: {ols['r2_log_price']:.3f}",
        "",
        "Interpretation: A positive association means ZIPs with higher walk scores "
        "tend to have higher typical prices, but correlation is not causation—dense, "
        "high-amenity places can be both walkable and expensive for many reasons.",
    ]
    text = "\n".join(lines)
    fig.text(0.08, 0.9, text, fontsize=11, va="top", family="monospace", color="#222")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORTS / "walkability_housing_report.pdf"

    walk, zillow = load_raw()
    df, log = clean_and_merge(walk, zillow)
    desc = summary_stats(df)
    state_df = state_level(df)
    corr = correlations(df)
    ols = ols_log_price(df)

    with PdfPages(pdf_path) as pdf:
        add_title_page(pdf, log, len(df))
        add_summary_table_page(pdf, desc)
        plot_scatter(df, pdf)
        plot_hex(df, pdf)
        plot_state_bars(state_df, pdf)
        plot_state_walk_vs_price(state_df, pdf)
        plot_residuals(df, pdf)
        add_stats_page(pdf, corr, ols)

    print(f"Wrote PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
