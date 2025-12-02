"""
Per-time (binned) dataset builder + simple surrogate fitting/validation.

What it does (in short):
1) Loads Excel ("Input", "Output").
2) Reduces dimensionality by aggregating *per transistor type* (row col 0),
   then *keeps only MP* devices (drops MN), and (optionally) log1p-rescales.
3) Produces per-time(-bin) datasets: X (features = MP transistor types), y = GHz.
4) Fits a few light surrogate models per time-bin with CV and reports R^2 / RMSE.

Assumptions:
- Input sheet: col 0 = transistor id (e.g., "I0.MN0", "I4.MP1"),
               col 1 = parameter name (ignored for aggregation),
               cols 2.. = datapoints aligned to Output sheet columns 2..
- Output sheet: row 0 = datapoint labels (e.g., 1, 1.1, ...),
                row 1 = Time, row 2 = Temperature, row 3 = Frequency (GHz).
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ---- Helpers ----------------------------------------------------------------

def _parse_output_sheet(out_raw: pd.DataFrame) -> pd.DataFrame:
    """Parse the Output sheet (no header) into a tidy dataframe with (datapoint_id, time, temp, frequency, instance)."""
    col_idx = list(range(2, out_raw.shape[1]))
    datapoint_id = out_raw.iloc[0, col_idx].astype(str).tolist()
    time_vals     = pd.to_numeric(out_raw.iloc[1, col_idx], errors="coerce").values
    temp_vals     = pd.to_numeric(out_raw.iloc[2, col_idx], errors="coerce").values
    freq_vals     = pd.to_numeric(out_raw.iloc[3, col_idx], errors="coerce").values

    def parse_instance(s: str) -> int:
        try:
            return int(str(s).split(".")[0])
        except Exception:
            return -1

    instances = [parse_instance(s) for s in datapoint_id]

    out_df = pd.DataFrame({
        "datapoint_id": datapoint_id,
        "time": time_vals,
        "temperature": temp_vals,
        "frequency": freq_vals,
        "instance": instances,
    })
    return out_df


def _aggregate_input_by_transistor(inp_raw: pd.DataFrame,
                                   datapoint_id: List[str],
                                   keep_family: str = "MP",
                                   drop_family: str = "MN",
                                   log1p: bool = True) -> pd.DataFrame:
    """
    From Input sheet (no header):
    - group by transistor id (col 0) across all parameters (col 1),
    - aggregate values (mean) across parameters for each datapoint,
    - keep only transistor ids containing keep_family (e.g., 'MP'),
      optionally drop those containing drop_family (e.g., 'MN'),
    - return a wide DataFrame: rows = datapoints, cols = transistor ids (kept),
      values = (optionally) log1p-rescaled aggregated degradation.
    """
    # cols: 0=transistor id, 1=param, 2..=datapoints
    trans_id = inp_raw.iloc[:, 0].astype(str)
    values = inp_raw.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    values.columns = datapoint_id  # align with Output

    # group by transistor id → aggregate across parameters (mean)
    agg = values.groupby(trans_id).mean()

    # filter transistor families
    keep_mask = agg.index.str.contains(keep_family)
    if drop_family:
        keep_mask &= ~agg.index.str.contains(drop_family)
    agg = agg.loc[keep_mask].copy()

    # (optional) log-rescaling for stability/comparability
    if log1p:
        agg = np.log1p(agg.clip(lower=0))

    # transpose to datapoints x features (so we can join with y easily)
    aggT = agg.T.copy()
    aggT.index.name = "datapoint_id"
    return aggT  # shape: (n_datapoints) x (n_features_kept)


def _bin_time(arr: np.ndarray, width: Optional[float]) -> np.ndarray:
    """Return binned times; if width is None, return original times."""
    if width is None or width <= 0:
        return arr
    return np.round(arr / width) * width


# ---- Public API -------------------------------------------------------------

@dataclass
class TimeBinDataset:
    X: pd.DataFrame      # features per datapoint (columns = transistor types kept)
    y: pd.Series         # GHz
    meta: pd.DataFrame   # datapoint_id, time, time_bin, temperature, instance


def build_per_time_datasets(
    path: str = "Complete_data.xlsx",
    input_sheet: str = "Input",
    output_sheet: str = "Output",
    keep_family: str = "MP",
    drop_family: str = "MN",
    log1p_inputs: bool = True,
    time_bin_width: Optional[float] = None,
    min_samples_per_bin: int = 5,
    max_time_hours: float = 25000.0,
) -> Dict[float, TimeBinDataset]:
    """
    Load + clean + reduce:
      - aggregate to transistor types,
      - keep only MP (drop MN),
      - log1p-rescale inputs (optional),
      - join with Output to get GHz,
      - **filter to time <= max_time_hours**, drop NaN times,
      - bin by time (optional),
      - return per-time-bin datasets with (X, y, meta).
    """
    inp_raw = pd.read_excel(path, sheet_name=input_sheet, header=None)
    out_raw = pd.read_excel(path, sheet_name=output_sheet, header=None)

    out_df = _parse_output_sheet(out_raw)
    datapoint_id = out_df["datapoint_id"].tolist()

    X_all = _aggregate_input_by_transistor(
        inp_raw,
        datapoint_id=datapoint_id,
        keep_family=keep_family,
        drop_family=drop_family,
        log1p=log1p_inputs,
    )

    # Join features with Output meta & target
    merged = X_all.merge(out_df, left_index=True, right_on="datapoint_id", how="left")

    # ---- NEW: filter out redundant late timepoints ----
    merged = merged.dropna(subset=["time"])
    merged = merged[merged["time"] <= max_time_hours]

    # Build time bins (after filtering)
    merged["time_bin"] = _bin_time(merged["time"].values, time_bin_width)

    # Split into per-time-bin datasets
    datasets: Dict[float, TimeBinDataset] = {}
    feature_cols = X_all.columns.tolist()

    for tb, dfb in merged.groupby("time_bin"):
        if len(dfb) < min_samples_per_bin:
            continue
        X = dfb[feature_cols].copy()
        y = dfb["frequency"].astype(float).copy()
        meta = dfb[["datapoint_id", "time", "time_bin", "temperature", "instance"]].copy()
        datasets[tb] = TimeBinDataset(X=X, y=y, meta=meta)

    return datasets



def _default_models(random_state: int = 42) -> Dict[str, object]:
    """Small, readable defaults."""
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


def fit_validate_surrogates(
    datasets: Dict[float, TimeBinDataset],
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cross-validate surrogate models per time-bin, returning:
      - metrics_df: aggregated CV results (R², RMSE)
      - oof_df: out-of-fold predictions per sample
      - importances_df: feature importances / coefficients per model
    """

    results = []
    oof_rows = []
    imp_rows = []

    # Helper for importances
    def _extract_importances(fitted_model, feature_names: List[str]) -> Dict[str, float]:
        if hasattr(fitted_model, "feature_importances_"):  # tree models
            imps = np.asarray(fitted_model.feature_importances_, dtype=float)
            return dict(zip(feature_names, imps))
        if isinstance(fitted_model, Pipeline):
            inner = fitted_model.named_steps.get("model", None)
            if inner is not None and hasattr(inner, "coef_"):
                coefs = np.abs(np.ravel(inner.coef_))
                if len(coefs) == len(feature_names):
                    return dict(zip(feature_names, coefs))
        if hasattr(fitted_model, "coef_"):
            coefs = np.abs(np.ravel(fitted_model.coef_))
            if len(coefs) == len(feature_names):
                return dict(zip(feature_names, coefs))
        return dict(zip(feature_names, np.zeros(len(feature_names))))

    for tb, data in datasets.items():
        X, y = data.X, data.y
        ids = data.meta["datapoint_id"].tolist()

        n_splits = max(2, min(cv_splits, len(y)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        models = {
            "Ridge": Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]),
            "RandomForest": RandomForestRegressor(
                n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        }

        for name, model in models.items():
            r2_list, rmse_list = [], []
            y_pred_oof = np.zeros(len(y))
            fold_idx = np.zeros(len(y), dtype=int)

            for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
                Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
                ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

                mdl = Pipeline(model.steps) if isinstance(model, Pipeline) else type(model)(**model.get_params())
                mdl.fit(Xtr, ytr)
                yhat = mdl.predict(Xva)

                y_pred_oof[va_idx] = yhat
                fold_idx[va_idx] = fold

                r2_list.append(r2_score(yva, yhat))
                rmse_list.append(mean_squared_error(yva, yhat) ** 0.5)

            # Save OOF predictions
            for i in range(len(y)):
                oof_rows.append({
                    "time_bin": tb,
                    "model": name,
                    "datapoint_id": ids[i],
                    "fold": int(fold_idx[i]),
                    "y_true": float(y.iloc[i]),
                    "y_pred": float(y_pred_oof[i]),
                })

            # Compute metrics
            results.append({
                "time_bin": tb,
                "n_samples": len(y),
                "n_features": X.shape[1],
                "model": name,
                "R2_mean": float(np.mean(r2_list)),
                "R2_std": float(np.std(r2_list)),
                "RMSE_mean": float(np.mean(rmse_list)),
                "RMSE_std": float(np.std(rmse_list)),
            })

            # Refit on full data for importances
            final_model = Pipeline(model.steps) if isinstance(model, Pipeline) else type(model)(**model.get_params())
            final_model.fit(X, y)
            imps = _extract_importances(final_model, X.columns.tolist())

            for feat, imp in imps.items():
                imp_rows.append({
                    "time_bin": tb,
                    "model": name,
                    "feature": feat,
                    "importance": float(imp),
                })

    metrics_df = pd.DataFrame(results).sort_values(["time_bin", "model"]).reset_index(drop=True)
    oof_df = pd.DataFrame(oof_rows)
    importances_df = pd.DataFrame(imp_rows)

    return metrics_df, oof_df, importances_df



def _clone_model(model):
    """Safe clone for simple estimators/pipelines without importing sklearn.base.clone."""
    import copy
    return copy.deepcopy(model)


def _extract_importances(fitted_model, feature_names: List[str]) -> Dict[str, float]:
    """
    Try feature_importances_ (tree models).
    If Pipeline[Ridge], fall back to |coef_|.
    Returns dict: feature -> importance (non-negative).
    """
    # Tree-style importances
    if hasattr(fitted_model, "feature_importances_"):
        imps = fitted_model.feature_importances_
        return dict(zip(feature_names, imps))

    # Pipeline case (e.g., StandardScaler + Ridge)
    if isinstance(fitted_model, Pipeline):
        # Try step named "model"
        try:
            inner = fitted_model.named_steps["model"]
        except KeyError:
            inner = None

        if inner is not None and hasattr(inner, "coef_"):
            coefs = np.abs(np.ravel(inner.coef_))
            # Ensure length consistency
            if len(coefs) == len(feature_names):
                return dict(zip(feature_names, coefs))

    # Direct linear (rare if not pipeline)
    if hasattr(fitted_model, "coef_"):
        coefs = np.abs(np.ravel(fitted_model.coef_))
        if len(coefs) == len(feature_names):
            return dict(zip(feature_names, coefs))

    # Fallback: zeros
    return dict(zip(feature_names, np.zeros(len(feature_names), dtype=float)))

# ---- Example usage ----------------------------------------------------------
if __name__ == "__main__":
    # 1) Build per-time datasets.
    #    Tip: choose a bin width that reflects your time scale (e.g., 1000, 5000, 10000).
    datasets = build_per_time_datasets(
        path="Complete_data.xlsx",
        input_sheet="Input",
        output_sheet="Output",
        keep_family="MP",     # keep only MP
        drop_family="MN",     # drop MN
        log1p_inputs=True,    # log-rescale inputs (safe for zeros)
        time_bin_width=5000,  # <-- adjust / set to None to use exact times
        min_samples_per_bin=5
    )

    # 2) Fit/validate surrogates per time-bin (Ridge / RF / GB).
    results_df = fit_validate_surrogates(datasets, cv_splits=5, random_state=42)
    print(results_df)


