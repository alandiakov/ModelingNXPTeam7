# VBSA_plots.py
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# 1) Top drivers per time bin
# -------------------------------
def plot_top_drivers_per_bin(
    vbsa_df: pd.DataFrame,
    time_bins: Optional[List[float]] = None,
    top_k: int = 10,
    figsize: Tuple[int, int] = (8, 5),
    save_prefix: Optional[str] = None,
):
    """
    For each time_bin, barplot the top_k features by ST with 95% CI bands.
    Saves one PNG per time bin if save_prefix is given.
    """
    if time_bins is None:
        time_bins = sorted(vbsa_df["time_bin"].unique())

    for tb in time_bins:
        df = (
            vbsa_df[vbsa_df["time_bin"] == tb]
            .sort_values("ST", ascending=False)
            .head(top_k)
            .copy()
        )
        if df.empty:
            continue

        # Compose CI whiskers for ST
        df["err_low"] = df["ST"] - df["ST_ci_lo"]
        df["err_high"] = df["ST_ci_hi"] - df["ST"]

        plt.figure(figsize=figsize)
        sns.barplot(data=df, x="ST", y="feature", color="steelblue")
        # Draw error bars manually
        for i, (_, r) in enumerate(df.iterrows()):
            plt.errorbar(
                x=r["ST"], y=i, xerr=np.array([[r["err_low"]], [r["err_high"]]]),
                fmt="none", ecolor="black", elinewidth=1, capsize=3
            )
        plt.title(f"Top {top_k} total-effect (ST) drivers — time_bin={tb}")
        plt.xlabel("Total-effect index (ST)")
        plt.ylabel("Feature")
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_topST_timebin_{tb}.png", dpi=170)
        plt.close()


# -------------------------------
# 2) Heatmaps over time
# -------------------------------
def plot_heatmaps_over_time(
    vbsa_df: pd.DataFrame,
    value: str = "ST",
    max_features: int = 30,
    figsize: Tuple[int, int] = (10, 10),
    savepath: Optional[str] = None,
):
    """
    Heatmap of sensitivity over time:
      - rows = features (top by overall 'value' sum across time),
      - cols = time_bin,
      - cell = chosen 'value' (ST or S1).
    """
    assert value in {"ST", "S1"}, "value must be 'ST' or 'S1'"

    # Select top features overall by sum of chosen metric
    totals = vbsa_df.groupby("feature")[value].sum().sort_values(ascending=False)
    keep = totals.head(max_features).index
    df = vbsa_df[vbsa_df["feature"].isin(keep)].copy()

    pivot = df.pivot_table(index="feature", columns="time_bin", values=value, aggfunc="mean")
    # Reorder rows by overall importance
    pivot = pivot.loc[totals.index.intersection(pivot.index)]
    pivot = pivot.iloc[:max_features]

    plt.figure(figsize=figsize)
    sns.heatmap(pivot, cmap="viridis", annot=False)
    plt.title(f"{value} heatmap across time")
    plt.xlabel("time_bin")
    plt.ylabel("feature (top by overall {})".format(value))
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=170)
    plt.close()


# -------------------------------
# 3) S1 vs ST scatter (per bin)
# -------------------------------
def plot_S1_vs_ST_scatter(
    vbsa_df: pd.DataFrame,
    time_bin: float,
    figsize: Tuple[int, int] = (6, 6),
    savepath: Optional[str] = None,
):
    """
    Scatter of S1 vs ST for one time bin, with y=x reference.
    Points far above diagonal → strong interactions (ST >> S1).
    """
    df = vbsa_df[vbsa_df["time_bin"] == time_bin].copy()
    if df.empty:
        return

    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x="S1", y="ST")
    lim = (0, max(1.0, df[["S1", "ST"]].max().max()))
    plt.plot(lim, lim, "--", color="gray")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("First-order index (S1)")
    plt.ylabel("Total-effect index (ST)")
    plt.title(f"S1 vs ST — time_bin={time_bin}")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=160)
    plt.close()


# -------------------------------
# 4) Sensitivity evolution over time for selected features
# -------------------------------
def plot_feature_time_evolution(
    vbsa_df: pd.DataFrame,
    features: List[str],
    value: str = "ST",
    figsize: Tuple[int, int] = (8, 5),
    savepath: Optional[str] = None,
):
    """
    Line plot of S1 or ST across time for a list of features.
    Useful to see how importance shifts over lifetime.
    """
    assert value in {"ST", "S1"}, "value must be 'ST' or 'S1'"
    plot_df = vbsa_df[vbsa_df["feature"].isin(features)].copy()
    if plot_df.empty:
        return

    plt.figure(figsize=figsize)
    sns.lineplot(data=plot_df, x="time_bin", y=value, hue="feature", marker="o")
    plt.title(f"{value} over time")
    plt.xlabel("time_bin")
    plt.ylabel(value)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=170)
    plt.close()


# -------------------------------
# 5) Output variance over time (from meta_df)
# -------------------------------
def plot_output_variance_over_time(
    meta_df: pd.DataFrame,
    figsize: Tuple[int, int] = (7, 4),
    savepath: Optional[str] = None,
):
    """
    Plots VarY (predicted output variance used in VBSA normalization) vs time_bin.
    Useful to see when the system is most/least variable.
    """
    df = meta_df.sort_values("time_bin").copy()
    if df.empty or "VarY" not in df.columns:
        return

    plt.figure(figsize=figsize)
    sns.lineplot(data=df, x="time_bin", y="VarY", marker="o")
    plt.title("Output variance (VarY) over time")
    plt.xlabel("time_bin")
    plt.ylabel("VarY")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=170)
    plt.close()
