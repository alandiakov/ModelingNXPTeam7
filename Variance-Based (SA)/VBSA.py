# VBSA.py
# -*- coding: utf-8 -*-
"""
Variance-Based Sensitivity Analysis (Sobol' first-order & total indices)
using a RandomForest surrogate, per time-bin dataset.

Assumptions:
- Inputs are treated as INDEPENDENT for sampling. We sample each feature
  from its empirical marginal (with replacement). If inputs are correlated,
  indices may be biased — use copulas/Shapley if needed.
- `datasets` is a dict[time_bin -> TimeBinDataset] from your existing code.
- Features X are already cleaned/reduced (e.g., only MP transistor groups),
  optionally log-rescaled.

Outputs:
- A single Excel file with two sheets:
  1) "vbsa_indices": per (time_bin, feature) with S1, ST, CIs, and metadata.
  2) "meta": per time_bin with model fit info and sampling stats.
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------

@dataclass
class VBSAConfig:
    n_samples: int = 2000          # base Monte-Carlo sample size (A/B)
    n_estimators: int = 500        # RF trees
    max_depth: Optional[int] = None
    random_state: int = 42
    n_batches: int = 10            # for simple CI via batching (>=5 recommended)
    n_jobs: int = -1               # RF parallelism


def run_vbsa_random_forest(
    datasets: Dict[float, "TimeBinDataset"],
    save_path: str = "vbsa_results.xlsx",
    cfg: VBSAConfig = VBSAConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each time_bin in `datasets`:
      - Fit RandomForest surrogate on full data.
      - Build Saltelli/Jansen sampling matrices A, B (independent marginals).
      - Compute first-order (S1) and total-effect (ST) Sobol' indices.
      - Estimate simple (batch) 95% CIs.
    Saves results to Excel and returns (vbsa_indices_df, meta_df).
    """
    all_rows: List[dict] = []
    meta_rows: List[dict] = []

    rng = np.random.default_rng(cfg.random_state)

    for tb, data in datasets.items():
        X = data.X.copy()
        y = data.y.astype(float).copy()
        feat_names = X.columns.tolist()
        d = X.shape[1]

        # ----- Fit surrogate (with standard scaling for stability)
        rf = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("rf", RandomForestRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                random_state=cfg.random_state,
                n_jobs=cfg.n_jobs
            ))
        ])
        rf.fit(X, y)

        # ----- Build A and B sampling matrices (independent marginals via bootstrap)
        A = _sample_independent_marginals(X, cfg.n_samples, rng)
        B = _sample_independent_marginals(X, cfg.n_samples, rng)

        # Predict model outputs
        fA = rf.predict(A)
        fB = rf.predict(B)

        # Output variance for normalization
        # Use pooled variance from fA & fB (ddof=1)
        VarY = np.var(np.concatenate([fA, fB]), ddof=1)
        f0 = 0.5 * (np.mean(fA) + np.mean(fB))

        # ----- Compute S1, ST with Saltelli/Jansen estimators
        # S1_i   (Saltelli 2002):   S1 = (1/N) sum f(B) * (f(AB_i) - f(A)) / VarY
        # ST_i   (Jansen 1999):     ST = (1/(2N)) sum (f(A) - f(AB_i))^2 / VarY
        S1 = np.zeros(d)
        ST = np.zeros(d)

        # For CI via batching, precompute batch indices
        batches = _make_batches(cfg.n_samples, cfg.n_batches)
        S1_batches = np.zeros((cfg.n_batches, d))
        ST_batches = np.zeros((cfg.n_batches, d))

        for i in range(d):
            ABi = A.copy()
            ABi[:, i] = B[:, i]
            fABi = rf.predict(ABi)

            # Full-sample indices
            S1[i] = np.mean(fB * (fABi - fA)) / VarY
            ST[i] = 0.5 * np.mean((fA - fABi) ** 2) / VarY

            # Batch-wise indices (for simple CIs)
            for b, sl in enumerate(batches):
                fA_b = fA[sl]
                fB_b = fB[sl]
                fABi_b = fABi[sl]
                VarY_b = np.var(np.concatenate([fA_b, fB_b]), ddof=1)
                # guard against degenerate variance
                if VarY_b <= 0:
                    S1_batches[b, i] = np.nan
                    ST_batches[b, i] = np.nan
                else:
                    S1_batches[b, i] = np.mean(fB_b * (fABi_b - fA_b)) / VarY_b
                    ST_batches[b, i] = 0.5 * np.mean((fA_b - fABi_b) ** 2) / VarY_b

        # Compute 95% CIs from batch estimates (normal approx; drop NaNs)
        S1_ci_lo, S1_ci_hi = _batch_ci(S1_batches, alpha=0.05)
        ST_ci_lo, ST_ci_hi = _batch_ci(ST_batches, alpha=0.05)

        # Collect rows
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
            "rf_n_estimators": cfg.n_estimators,
            "rf_max_depth": cfg.max_depth,
            "N_base": cfg.n_samples,
            "n_batches": cfg.n_batches,
            "f0_mean": f0,
            "VarY": VarY,
        })

    vbsa_df = pd.DataFrame(all_rows).sort_values(["time_bin", "ST"], ascending=[True, False]).reset_index(drop=True)
    meta_df = pd.DataFrame(meta_rows).sort_values("time_bin").reset_index(drop=True)

    # Save to Excel (two sheets)
    with pd.ExcelWriter(save_path, engine="xlsxwriter") as writer:
        vbsa_df.to_excel(writer, sheet_name="vbsa_indices", index=False)
        meta_df.to_excel(writer, sheet_name="meta", index=False)

    return vbsa_df, meta_df


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _sample_independent_marginals(X: pd.DataFrame, N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw N samples assuming independence between columns by bootstrapping
    each feature from its empirical marginal (with replacement).
    Returns shape (N, d).
    """
    cols = X.columns
    d = len(cols)
    A = np.empty((N, d), dtype=float)
    for j, c in enumerate(cols):
        col_vals = X[c].values
        idx = rng.integers(0, len(col_vals), size=N)
        A[:, j] = col_vals[idx]
    return A


def _make_batches(N: int, n_batches: int) -> List[slice]:
    """
    Split 0..N-1 into ~equal contiguous slices for batch-based CI estimation.
    """
    n_batches = max(1, min(n_batches, N))
    edges = np.linspace(0, N, n_batches + 1, dtype=int)
    return [slice(edges[i], edges[i + 1]) for i in range(n_batches)]


def _batch_ci(batch_vals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an array of shape (n_batches, d), compute normal-approx 95% CI per column.
    Ignores NaNs within each column. Returns (lo, hi) arrays of length d.
    """
    lo = np.zeros(batch_vals.shape[1])
    hi = np.zeros(batch_vals.shape[1])
    for j in range(batch_vals.shape[1]):
        col = batch_vals[:, j]
        col = col[~np.isnan(col)]
        if len(col) < 2:
            lo[j] = np.nan
            hi[j] = np.nan
            continue
        m = np.mean(col)
        s = np.std(col, ddof=1) / np.sqrt(len(col))
        z = 1.96  # ~95%
        lo[j] = m - z * s
        hi[j] = m + z * s
    return lo, hi

