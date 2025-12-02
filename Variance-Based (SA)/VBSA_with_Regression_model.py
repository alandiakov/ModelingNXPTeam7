# --- Add this near your other imports ---
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from VBSA import _make_batches, _sample_independent_marginals, run_vbsa_random_forest, VBSAConfig, _batch_ci

# assumes _sample_independent_marginals, _make_batches, _batch_ci, VBSAConfig, TimeBinDataset already exist


# ---------- Load coefficients from the attached Excel ----------
# ---- robust loader for your Excel layout ----
def _load_linear_coefs_from_excel(
    coef_path: str = "/mnt/data/regression_coeffients.xlsx",
    feature_col_candidates: tuple = ("feature", "term", "name", "variable"),
    coef_col_candidates: tuple = ("coef", "coefficient", "beta", "value"),
    intercept_names: tuple = ("Intercept", "intercept", "B0", "b0", "const"),
    strip_suffix: str = " Average value",
) -> Tuple[float, pd.Series]:
    """
    Supports:
      A) tidy two-column with headers (feature, coef)
      B) wide row with an intercept column
      C) your case: two columns *without* headers; last row 'Intercept'
    """
    df0 = pd.read_excel(coef_path, sheet_name=0)

    # Case A (tidy with headers)
    feat_col = next((c for c in df0.columns if c in feature_col_candidates), None)
    coef_col = next((c for c in df0.columns if c in coef_col_candidates), None)
    if feat_col and coef_col:
        mask_inter = df0[feat_col].astype(str).str.strip().isin(intercept_names)
        if not mask_inter.any():
            raise ValueError(f"No intercept row found in column '{feat_col}'.")
        intercept_val = float(df0.loc[mask_inter, coef_col].iloc[0])
        feats_df = df0.loc[~mask_inter, [feat_col, coef_col]].copy()
        feats_df[feat_col] = feats_df[feat_col].astype(str).str.strip()
        if strip_suffix:
            feats_df[feat_col] = feats_df[feat_col].str.replace(strip_suffix, "", regex=False)
        beta = pd.Series(feats_df[coef_col].astype(float).to_numpy(),
                         index=feats_df[feat_col].to_numpy())
        return intercept_val, beta

    # Case C (no headers): read again header=None
    df = pd.read_excel(coef_path, sheet_name=0, header=None)
    if df.shape[1] == 2:
        terms = df.iloc[:, 0].astype(str).str.strip()
        vals  = df.iloc[:, 1].astype(float)
        # intercept is a dedicated row
        mask_inter = terms.isin(intercept_names)
        if mask_inter.any():
            intercept_val = float(vals.loc[mask_inter].iloc[0])
            feat_terms = terms.loc[~mask_inter]
            if strip_suffix:
                feat_terms = feat_terms.str.replace(strip_suffix, "", regex=False)
            beta = pd.Series(vals.loc[~mask_inter].to_numpy(), index=feat_terms.to_numpy())
            return intercept_val, beta

    # Case B (wide)
    inter_cols = [c for c in df0.columns if str(c) in intercept_names]
    if inter_cols:
        row = df0.iloc[0]
        intercept_val = float(row[inter_cols[0]])
        feature_cols = [c for c in df0.columns if c != inter_cols[0]]
        beta = row[feature_cols].astype(float)
        beta.index = beta.index.astype(str)
        if strip_suffix:
            beta.index = beta.index.str.replace(strip_suffix, "", regex=False)
        return intercept_val, beta

    raise ValueError(
        "Unrecognized coefficient layout. Supported: two-column (with or without headers) "
        "with an Intercept row, or wide with an intercept column."
    )



# ---------- Simple linear surrogate ----------
class _FixedLinearModel:
    def __init__(self, intercept: float, beta: pd.Series):
        self.intercept = float(intercept)
        self.beta = beta.astype(float)
        self.beta.index = self.beta.index.astype(str)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # align by name; missing features contribute 0
        common = [c for c in X.columns if c in self.beta.index]
        return self.intercept + (X[common] @ self.beta.loc[common]).to_numpy()


# ---------- VBSA with regression surrogate ----------
def run_vbsa_regression(
    datasets: Dict[float, "TimeBinDataset"],
    coef_path: str = "regression_coeffients.xlsx",   # <- imports your Excel
    save_path: str = "vbsa_results.xlsx",
    cfg: "VBSAConfig" = VBSAConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Same outputs/Excel as run_vbsa_random_forest(), but the surrogate is a fixed linear model
    built from coefficients in `coef_path` (sheet 0).
    """
    all_rows: List[dict] = []
    meta_rows: List[dict] = []

    rng = np.random.default_rng(cfg.random_state)

    # Load intercept and per-feature betas once
    intercept, beta_all = _load_linear_coefs_from_excel(coef_path)

    for tb, data in datasets.items():
        X = data.X.copy()
        y = data.y.astype(float).copy()
        feat_names = X.columns.tolist()
        d = X.shape[1]

        # Align betas to this dataset's features (extras ignored; missing treated as 0)
        lin = _FixedLinearModel(intercept=intercept, beta=beta_all.copy())

        # A, B via independent-marginal bootstrap
        A_np = _sample_independent_marginals(X, cfg.n_samples, rng)   # likely returns ndarray
        B_np = _sample_independent_marginals(X, cfg.n_samples, rng)
        A = pd.DataFrame(A_np, columns=X.columns)
        B = pd.DataFrame(B_np, columns=X.columns)

        fA = lin.predict(A)
        fB = lin.predict(B)

        VarY = np.var(np.concatenate([fA, fB]), ddof=1)
        f0 = 0.5 * (np.mean(fA) + np.mean(fB))

        S1 = np.zeros(d)
        ST = np.zeros(d)

        batches = _make_batches(cfg.n_samples, cfg.n_batches)
        S1_batches = np.zeros((cfg.n_batches, d))
        ST_batches = np.zeros((cfg.n_batches, d))

        for i in range(d):
            ABi = A.copy()
            col_i = A.columns[i]
            ABi[col_i] = B[col_i].to_numpy()     
            fABi = lin.predict(ABi)

            S1[i] = np.mean(fB * (fABi - fA)) / VarY
            ST[i] = 0.5 * np.mean((fA - fABi) ** 2) / VarY

            for b, sl in enumerate(batches):
                fA_b = fA[sl]
                fB_b = fB[sl]
                fABi_b = fABi[sl]
                VarY_b = np.var(np.concatenate([fA_b, fB_b]), ddof=1)
                if VarY_b <= 0:
                    S1_batches[b, i] = np.nan
                    ST_batches[b, i] = np.nan
                else:
                    S1_batches[b, i] = np.mean(fB_b * (fABi_b - fA_b)) / VarY_b
                    ST_batches[b, i] = 0.5 * np.mean((fA_b - fABi_b) ** 2) / VarY_b

        S1_ci_lo, S1_ci_hi = _batch_ci(S1_batches, alpha=0.05)
        ST_ci_lo, ST_ci_hi = _batch_ci(ST_batches, alpha=0.05)

        for i, feat in enumerate(feat_names):
            all_rows.append({
                "time_bin": tb,
                "feature": feat,
                "S1": S1[i],
                "S1_ci_lo": S1_ci_lo[i],
                "S1_ci_hi": S1_ci_hi[i],
                "ST": ST[i],
                "ST_ci_lo": ST_ci_lo[i],
                "ST_ci_hi": ST_ci_hi[i],
                "d": d,
                "N_base": cfg.n_samples,
            })

        meta_rows.append({
            "time_bin": tb,
            "n_samples_train": len(y),
            "n_features": d,
            "rf_n_estimators": None,   # keep same columns as RF version
            "rf_max_depth": None,
            "N_base": cfg.n_samples,
            "n_batches": cfg.n_batches,
            "f0_mean": f0,
            "VarY": VarY,
        })

    vbsa_df = pd.DataFrame(all_rows).sort_values(["time_bin", "ST"], ascending=[True, False]).reset_index(drop=True)
    meta_df = pd.DataFrame(meta_rows).sort_values("time_bin").reset_index(drop=True)

    with pd.ExcelWriter(save_path, engine="xlsxwriter") as writer:
        vbsa_df.to_excel(writer, sheet_name="vbsa_indices", index=False)
        meta_df.to_excel(writer, sheet_name="meta", index=False)

    return vbsa_df, meta_df
