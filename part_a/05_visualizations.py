"""
part_a/05_visualizations.py
============================
Generates 10 publication-quality visualizations from the cleaned
Colombo restaurant dataset using pandas + matplotlib + seaborn.

All charts are saved as PNG files to:
    data/processed/visualizations/

Run
---
    python part_a/05_visualizations.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common.paths import CLEAN_CSV, VIZ_DIR

CLEAN_CSV = Path(CLEAN_CSV)
VIZ_DIR   = Path(VIZ_DIR)

# === STYLE ===
PALETTE   = "Set2"
BAR_SIZE  = (12, 6)
HEAT_SIZE = (10, 8)
WIDE_SIZE = (14, 6)
DPI       = 120


def setup_style() -> None:
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "axes.titlesize":  14,
        "axes.labelsize":  12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def save_fig(name: str) -> None:
    path = VIZ_DIR / name
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


def load_data() -> pd.DataFrame:
    """
    Load the cleaned CSV — handles both a plain file and a PySpark-written
    directory (reads the first part-*.csv inside).
    """
    path = Path(CLEAN_CSV)
    print(f"\n[Load] Reading {path}")

    if path.is_dir():
        # PySpark coalesce(1).write.csv() → directory with one part-*.csv inside
        parts = sorted(path.glob("part-*.csv"))
        if not parts:
            print(f"[ERROR] No part-*.csv found in {path}", file=sys.stderr)
            sys.exit(1)
        csv_file = parts[0]
        print(f"  Detected PySpark directory — using {csv_file.name}")
    elif path.is_file():
        csv_file = path
    else:
        print(f"[ERROR] Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(
            csv_file,
            quotechar='"',
            doublequote=True,
            engine="python",       # handles embedded newlines in quoted fields
            on_bad_lines="skip",   # skip any rows that still can't be parsed
        )
    except Exception as exc:
        print(f"[ERROR] Could not read CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Shape: {df.shape}")

    # Cast numeric columns — CSV has no type metadata so pandas infers everything as str
    for col in ("rating", "total_ratings", "price_level",
                "avg_review_rating", "popularity_score", "lat", "lng"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# === CHART 1: Restaurant Count by Cuisine Category ===

def chart_01_cuisine_count(df: pd.DataFrame) -> None:
    print("\n[Chart 01] Restaurant count by cuisine_category")
    counts = df["cuisine_category"].value_counts().reset_index()
    counts.columns = ["cuisine_category", "count"]

    fig, ax = plt.subplots(figsize=BAR_SIZE)
    sns.barplot(data=counts, x="cuisine_category", y="count", palette=PALETTE, ax=ax)
    ax.set_title("Restaurant Count by Cuisine Category")
    ax.set_xlabel("Cuisine Category")
    ax.set_ylabel("Number of Restaurants")
    ax.tick_params(axis="x", rotation=45)
    for bar in ax.patches:
        ax.annotate(
            f"{int(bar.get_height())}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )
    save_fig("01_cuisine_count.png")


# === CHART 2: Restaurant Count by District ===

def chart_02_district_count(df: pd.DataFrame) -> None:
    print("[Chart 02] Restaurant count by district")
    counts = df["district"].value_counts().reset_index()
    counts.columns = ["district", "count"]

    fig, ax = plt.subplots(figsize=BAR_SIZE)
    sns.barplot(data=counts, x="district", y="count", palette=PALETTE, ax=ax)
    ax.set_title("Restaurant Count by Colombo District")
    ax.set_xlabel("District")
    ax.set_ylabel("Number of Restaurants")
    ax.tick_params(axis="x", rotation=45)
    save_fig("02_district_count.png")


# === CHART 3: Top 10 Most Reviewed (Horizontal Bar) ===

def chart_03_top10_reviewed(df: pd.DataFrame) -> None:
    print("[Chart 03] Top 10 most reviewed restaurants")
    top10 = df.nlargest(10, "total_ratings")[["name", "total_ratings"]].copy()
    top10 = top10.sort_values("total_ratings")

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette(PALETTE, len(top10))
    ax.barh(top10["name"], top10["total_ratings"], color=colors)
    ax.set_title("Top 10 Most Reviewed Restaurants in Colombo")
    ax.set_xlabel("Total Ratings (Reviews)")
    ax.set_ylabel("Restaurant")
    for i, v in enumerate(top10["total_ratings"]):
        ax.text(v + 5, i, str(v), va="center", fontsize=9)
    save_fig("03_top10_most_reviewed.png")


# === CHART 4: Rating Distribution by Cuisine (Box Plot) ===

def chart_04_rating_boxplot(df: pd.DataFrame) -> None:
    print("[Chart 04] Rating distribution by cuisine_category (box plot)")
    fig, ax = plt.subplots(figsize=BAR_SIZE)
    order = df.groupby("cuisine_category")["rating"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="cuisine_category", y="rating", order=order, palette=PALETTE, ax=ax)
    ax.set_title("Rating Distribution by Cuisine Category")
    ax.set_xlabel("Cuisine Category")
    ax.set_ylabel("Rating (1–5)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 5.5)
    save_fig("04_rating_boxplot_by_cuisine.png")


# === CHART 5: Scatter — total_ratings vs rating (coloured by price_label) ===

def chart_05_scatter_ratings(df: pd.DataFrame) -> None:
    print("[Chart 05] Scatter: total_ratings vs rating by price_label")
    fig, ax = plt.subplots(figsize=BAR_SIZE)
    palette_map = {
        "Free": "#2ecc71", "Budget": "#3498db", "Moderate": "#f39c12",
        "Expensive": "#e74c3c", "Luxury": "#9b59b6", "Unknown": "#95a5a6",
    }
    for label, group in df.groupby("price_label"):
        ax.scatter(
            group["total_ratings"], group["rating"],
            label=label, alpha=0.6, s=40,
            color=palette_map.get(label, "#333333"),
        )
    ax.set_title("Total Ratings vs Rating (by Price Level)")
    ax.set_xlabel("Total Ratings (log scale)")
    ax.set_ylabel("Rating")
    ax.set_xscale("log")
    ax.legend(title="Price Level", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig("05_scatter_ratings_vs_total.png")


# === CHART 6: Price Label Pie Chart ===

def chart_06_price_pie(df: pd.DataFrame) -> None:
    print("[Chart 06] Price label distribution (pie chart)")
    counts = df["price_label"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette(PALETTE, len(counts))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.82,
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Price Level Distribution Across Colombo Restaurants")
    save_fig("06_price_label_pie.png")


# === CHART 7: Heatmap — Avg Rating by District × Cuisine ===

def chart_07_heatmap(df: pd.DataFrame) -> None:
    print("[Chart 07] Heatmap: avg rating by district × cuisine")
    pivot = df.pivot_table(
        values="rating",
        index="district",
        columns="cuisine_category",
        aggfunc="mean",
    ).round(2)

    # Drop cuisines with fewer than 2 data points across all districts
    pivot = pivot.dropna(axis=1, thresh=2)

    fig, ax = plt.subplots(figsize=HEAT_SIZE)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Avg Rating"},
    )
    ax.set_title("Average Rating by District and Cuisine Category")
    ax.set_xlabel("Cuisine Category")
    ax.set_ylabel("District")
    ax.tick_params(axis="x", rotation=45)
    save_fig("07_heatmap_rating_district_cuisine.png")


# === CHART 8: Rating Distribution Histogram ===

def chart_08_rating_histogram(df: pd.DataFrame) -> None:
    print("[Chart 08] Rating distribution histogram")
    fig, ax = plt.subplots(figsize=BAR_SIZE)
    sns.histplot(df["rating"].dropna(), bins=20, kde=True, color="#3498db", ax=ax)
    ax.axvline(df["rating"].mean(), color="red", linestyle="--", label=f"Mean: {df['rating'].mean():.2f}")
    ax.axvline(df["rating"].median(), color="green", linestyle="--", label=f"Median: {df['rating'].median():.2f}")
    ax.set_title("Rating Distribution Across All Restaurants")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.legend()
    save_fig("08_rating_histogram.png")


# === CHART 9: Avg Popularity Score by District ===

def chart_09_popularity_by_district(df: pd.DataFrame) -> None:
    print("[Chart 09] Avg popularity score by district")
    pop = df.groupby("district")["popularity_score"].mean().sort_values(ascending=False).reset_index()
    pop.columns = ["district", "avg_popularity_score"]

    fig, ax = plt.subplots(figsize=BAR_SIZE)
    sns.barplot(data=pop, x="district", y="avg_popularity_score", palette=PALETTE, ax=ax)
    ax.set_title("Average Popularity Score by Colombo District")
    ax.set_xlabel("District")
    ax.set_ylabel("Avg Popularity Score")
    ax.tick_params(axis="x", rotation=45)
    save_fig("09_popularity_by_district.png")


# === CHART 10: Rating Bucket Count Plot ===

def chart_10_rating_buckets(df: pd.DataFrame) -> None:
    print("[Chart 10] Rating bucket distribution")

    def bucket(r):
        if pd.isna(r):    return "Unknown"
        if r < 2:         return "Poor (1–2)"
        if r < 3:         return "Average (2–3)"
        if r < 4:         return "Good (3–4)"
        return "Excellent (4–5)"

    df = df.copy()
    df["rating_bucket"] = df["rating"].apply(bucket)

    order = ["Poor (1–2)", "Average (2–3)", "Good (3–4)", "Excellent (4–5)", "Unknown"]
    order = [o for o in order if o in df["rating_bucket"].unique()]

    fig, ax = plt.subplots(figsize=BAR_SIZE)
    sns.countplot(data=df, x="rating_bucket", order=order, palette=PALETTE, ax=ax)
    ax.set_title("Restaurant Count by Rating Bucket")
    ax.set_xlabel("Rating Bucket")
    ax.set_ylabel("Count")
    for bar in ax.patches:
        ax.annotate(
            f"{int(bar.get_height())}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=10,
        )
    save_fig("10_rating_bucket_countplot.png")


# === MAIN ===

def main() -> None:
    print("=" * 60)
    print("PART A — Step 05: Visualizations")
    print("=" * 60)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    df = load_data()

    chart_01_cuisine_count(df)
    chart_02_district_count(df)
    chart_03_top10_reviewed(df)
    chart_04_rating_boxplot(df)
    chart_05_scatter_ratings(df)
    chart_06_price_pie(df)
    chart_07_heatmap(df)
    chart_08_rating_histogram(df)
    chart_09_popularity_by_district(df)
    chart_10_rating_buckets(df)

    print(f"\n✅  All 10 visualizations saved to {VIZ_DIR}")


if __name__ == "__main__":
    main()
