import pandas as pd
from surrogate_model import fit_validate_surrogates, build_per_time_datasets
from plots_viz import (
    plot_performance_summary,
    plot_feature_importances,
    compute_importances_from_datasets,
    plot_pred_vs_true,
    plot_residual_hist,
)
from VBSA import VBSAConfig, run_vbsa_random_forest
from VBSA_plots import (
    plot_top_drivers_per_bin,
    plot_heatmaps_over_time,
    plot_S1_vs_ST_scatter,
    plot_feature_time_evolution,
    plot_output_variance_over_time,
)
from VBSA_with_Regression_model import run_vbsa_regression


# Build datasets
datasets = build_per_time_datasets(
    path="Complete_data.xlsx",
    input_sheet="Input",
    output_sheet="Output",
    keep_family="MP",
    drop_family="MN",
    log1p_inputs=True,
    time_bin_width=8760,   # try different bin widths or None
    min_samples_per_bin=5,
    max_time_hours=25000.0
)

# 1) Fit & validate surrogates
metrics_df, oof_df, importances_df = fit_validate_surrogates(
    datasets,
    cv_splits=5,
    random_state=42
)

metrics_df.to_excel("cv_results.xlsx")  # or print(cv_results.to_string(index=False))


# # 2) Plots to diagnose
plot_performance_summary(metrics_df, savepath="Figures/performance_summary.png")
plot_feature_importances(importances_df, time_bin=5000, model="RandomForest", top_k=15, savepath="Figures/feature_importance.png")

plot_pred_vs_true(oof_df, time_bin=5000, model="RandomForest", savepath="Figures/pred_vs_true.png")
plot_residual_hist(oof_df, time_bin=5000, model="RandomForest", savepath="Figures/residual_hist.png")

# plot_performance_summary(metrics_df)
# plot_feature_importances(importances_df, time_bin=5000, model="RandomForest", top_k=15)
# plot_pred_vs_true(oof_df, time_bin=5000, model="RandomForest")
# plot_residual_hist(oof_df, time_bin=5000, model="RandomForest")

# Then call VBSA:
cfg = VBSAConfig(n_samples=100000, n_estimators=600, random_state=123, n_batches=12)
vbsa_df, meta_df = run_vbsa_random_forest(
    datasets=datasets,
    save_path="vbsa_results.xlsx",
    cfg=cfg,
)

# Optionally run the VBSA with the regression model
# vbsa_df, meta_df = run_vbsa_regression(
#     datasets=datasets,
#     save_path="vbsa_results_regression.xlsx",
#     cfg=cfg,
# )



print("Saved VBSA results to vbsa_results.xlsx")
print(vbsa_df.head())
print(meta_df.head())


# -------------------------
# PLOT RESULTS
# -------------------------

# Load VBSA results produced earlier
vbsa_df = pd.read_excel("vbsa_results.xlsx", sheet_name="vbsa_indices")
meta_df = pd.read_excel("vbsa_results.xlsx", sheet_name="meta")

# 1) Top drivers per bin (saves one PNG per time bin)
plot_top_drivers_per_bin(
    vbsa_df,
    time_bins=None,     # or e.g. [0, 5000, 10000]
    top_k=10,
    save_prefix="vbsa_top"
)

# 2) Heatmaps over time
plot_heatmaps_over_time(vbsa_df, value="ST", max_features=25, savepath="vbsa_heatmap_ST.png")
plot_heatmaps_over_time(vbsa_df, value="S1", max_features=25, savepath="vbsa_heatmap_S1.png")

# 3) S1 vs ST scatter for a specific bin
plot_S1_vs_ST_scatter(vbsa_df, time_bin=0, savepath="vbsa_scatter_S1_vs_ST_t0.png")

# 4) Track a few key features over time
key_feats = ["I0.MP2", "I0.MP1"]  # adjust to your top drivers
plot_feature_time_evolution(vbsa_df, features=key_feats, value="ST", savepath="vbsa_evolution_ST.png")
plot_feature_time_evolution(vbsa_df, features=key_feats, value="S1", savepath="vbsa_evolution_S1.png")

# 5) Output variance trend
plot_output_variance_over_time(meta_df, savepath="vbsa_VarY_over_time.png")