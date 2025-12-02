# plots_viz.py
# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


# --- Model factory (used only if you want to recompute importances quickly) ---
def _default_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=1.0, random_state=random_state))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }


# --- Importances reader (needed if you compute importances here) -------------
def _extract_importances(fitted_model, feature_names: List[str]) -> Dict[str, float]:
    """Return {feature: importance} for tree models and |coef_| for linear models (e.g., Ridge in a Pipeline)."""
    if hasattr(fitted_model, "feature_importances_"):  # tree-based
        imps = np.asarray(fitted_model.feature_importances_, dtype=float)
        return dict(zip(feature_names, imps))

    if isinstance(fitted_model, Pipeline):  # typical: StandardScaler + Ridge
        try:
            inner = fitted_model.named_steps["model"]
        except KeyError:
            inner = None
        if inner is not None and hasattr(inner, "coef_"):
            coefs = np.abs(np.ravel(inner.coef_)).astype(float)
            if len(coefs) == len(feature_names):
                return dict(zip(feature_names, coefs))

    if hasattr(fitted_model, "coef_"):  # bare linear model
        coefs = np.abs(np.ravel(fitted_model.coef_)).astype(float)
        if len(coefs) == len(feature_names):
            return dict(zip(feature_names, coefs))

    return dict(zip(feature_names, np.zeros(len(feature_names), dtype=float)))


# --- Plotting: performance summary ------------------------------------------
def plot_performance_summary(
    metrics_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 4),
    savepath: Optional[str] = None,
):
    """Line plots of R² and RMSE vs time_bin, colored by model."""
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    sns.lineplot(data=metrics_df, x="time_bin", y="R2_mean", hue="model", marker="o", ax=ax[0])
    ax[0].set_title("R² vs time_bin")
    ax[0].set_xlabel("time_bin")
    ax[0].set_ylabel("R² (CV mean)")

    sns.lineplot(data=metrics_df, x="time_bin", y="RMSE_mean", hue="model", marker="o", ax=ax[1])
    ax[1].set_title("RMSE vs time_bin")
    ax[1].set_xlabel("time_bin")
    ax[1].set_ylabel("RMSE (CV mean)")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=170)
    plt.close()


# --- Plotting: feature importances (requires a precomputed importances_df) ---
def plot_feature_importances(
    importances_df: pd.DataFrame,
    time_bin: float,
    model: str,
    top_k: int = 20,
    figsize: Tuple[int, int] = (8, 6),
    savepath: Optional[str] = None,
):
    """Barplot of top-k features for a given (time_bin, model) from an importances_df."""
    df = (
        importances_df
        .query("time_bin == @time_bin and model == @model")
        .sort_values("importance", ascending=False)
        .head(top_k)
        .copy()
    )
    if df.empty:
        print(f"[plot_feature_importances] No data for time_bin={time_bin}, model={model}")
        return

    plt.figure(figsize=figsize)
    sns.barplot(data=df, x="importance", y="feature", orient="h")
    plt.title(f"Feature importances — {model} @ time_bin={time_bin}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=160)
    plt.close()


# --- Optional: compute importances here (no clone needed; fresh fit) ---------
def compute_importances_from_datasets(
    datasets: Dict[float, object],   # values are TimeBinDataset
    model_name: str = "RandomForest",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Convenience function: refit one model per time_bin on full data and return importances_df.
    Use this only if you didn't compute importances earlier.
    """
    models = _default_models(random_state)
    if model_name not in models:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from {list(models.keys())}.")

    rows = []
    for tb, data in datasets.items():
        X, y = data.X, data.y.astype(float)
        mdl = models[model_name]
        fitted = mdl.fit(X, y)
        imps = _extract_importances(fitted, X.columns.tolist())
        for feat, imp in imps.items():
            rows.append({"time_bin": tb, "model": model_name, "feature": feat, "importance": float(imp)})

    return pd.DataFrame(rows)


# --- Plotting: predicted vs true (requires oof_df) ---------------------------
def plot_pred_vs_true(
    oof_df: pd.DataFrame,
    time_bin: float,
    model: str,
    figsize: Tuple[int, int] = (6, 6),
    savepath: Optional[str] = None,
):
    """OOF predicted vs true with 45° line + R² / RMSE annotations."""
    df = oof_df.query("time_bin == @time_bin and model == @model").copy()
    if df.empty:
        print(f"[plot_pred_vs_true] No data for time_bin={time_bin}, model={model}")
        return

    r2 = r2_score(df["y_true"], df["y_pred"])
    rmse = np.sqrt(mean_squared_error(df["y_true"], df["y_pred"]))

    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x="y_true", y="y_pred", alpha=0.6, s=30)
    lims = [min(df["y_true"].min(), df["y_pred"].min()),
            max(df["y_true"].max(), df["y_pred"].max())]
    plt.plot(lims, lims, linestyle="--")
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("True GHz"); plt.ylabel("Predicted GHz")
    plt.title(f"{model} @ time_bin={time_bin}  |  R²={r2:.3f}, RMSE={rmse:.3f}")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=160)
    plt.close()


# --- Plotting: residual histogram (requires oof_df) --------------------------
def plot_residual_hist(
    oof_df: pd.DataFrame,
    time_bin: float,
    model: str,
    bins: int = 30,
    figsize: Tuple[int, int] = (7, 4),
    savepath: Optional[str] = None,
):
    """Residual histogram for a given (time_bin, model)."""
    df = oof_df.query("time_bin == @time_bin and model == @model").copy()
    if df.empty:
        print(f"[plot_residual_hist] No data for time_bin={time_bin}, model={model}")
        return
    df["resid"] = df["y_true"] - df["y_pred"]

    plt.figure(figsize=figsize)
    sns.histplot(df["resid"], bins=bins, kde=True)
    plt.title(f"Residuals — {model} @ time_bin={time_bin}")
    plt.xlabel("Residual (GHz)")
    plt.ylabel("Count")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=160)
    plt.close()
