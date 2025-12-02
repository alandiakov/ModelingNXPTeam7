Transistor Degradation – Surrogate Models & VBSA Pipeline
=========================================================

This folder provides a modular pipeline for:
1. Exploring the degradation dataset.
2. Building per-time-bin datasets for GHz prediction.
3. Training surrogate models (Ridge, RandomForest, GradientBoosting).
4. Running Variance-Based Sensitivity Analysis (VBSA).
5. Generating diagnostic and VBSA plots.

All functionality is driven by the Excel file “Complete_data.xlsx”, using the
sheets “Input” and “Output”. The structure and data expectations are defined in
the top docstrings of surrogate_model.py.


---------------------------------------------------------------------------
Folder Overview
---------------------------------------------------------------------------

surrogate_model.py
    - Reads and parses the data.
    - Builds per-time-bin datasets (TimeBinDataset objects).
    - Fits and cross-validates surrogate models per time-bin.
    - Produces model metrics, out-of-fold predictions, and feature importances.

plots_viz.py
    - Visual diagnostics for surrogate models:
      * Performance over time (R² / RMSE).
      * Feature importances.
      * Prediction vs truth.
      * Residual distributions.

VBSA.py
    - VBSA using RandomForest surrogates.
    - Generates first-order (S1) and total-effect (ST) Sobol indices.
    - Bootstraps uncertainty and writes results to vbsa_results.xlsx.

VBSA_plots.py
    - Reads vbsa_results.xlsx.
    - Plots:
      * Top drivers per time-bin.
      * Heatmaps (S1 or ST) over time.
      * S1 vs ST scatter.
      * Time-evolution of key features.
      * Output variance vs time.

VBSA_with_Regression_model.py
    - Optional VBSA backend using a fixed linear regression surrogate.
    - This was a fast exploratory try-out.
    - Future research can replace this with stronger regression-based
      surrogates (e.g., Lasso, Elastic Net, GAMs, spline models) following
      the same interface and plug-in structure.

explore.py / explore_plots.py
    - Basic data exploration.
    - Histograms, time distributions, degradation vs GHz, etc.
    - Results saved under Figures_exploration/.

main.py
    - The main orchestrator: builds data, fits surrogates, saves metrics,
      generates plots, runs VBSA, and creates all VBSA visualisations.
    - Contains an optional call to VBSA_with_Regression_model.py
      (commented out by default).


---------------------------------------------------------------------------
Workflow
---------------------------------------------------------------------------

1. Optional: Data exploration
   --------------------------------------
   Run:
       python explore.py
   This generates exploratory plots in Figures_exploration/.


2. Build datasets + fit surrogate models
   --------------------------------------
   Run:
       python main.py

   main.py performs:
   - Reading and parsing of Complete_data.xlsx.
   - Creating per-time-bin datasets (configurable: log1p, time_bin_width,
     min_samples_per_bin, etc.).
   - Training and validating surrogate models via fit_validate_surrogates().
   - Saving cv_results.xlsx containing performance metrics.


3. Surrogate model diagnostics
   --------------------------------------
   main.py automatically calls:
   - plot_performance_summary
   - plot_feature_importances
   - plot_pred_vs_true
   - plot_residual_hist

   Output images are saved in the Figures/ directory.


4. VBSA using RandomForest surrogates
   --------------------------------------
   main.py constructs a VBSAConfig and calls run_vbsa_random_forest().
   This produces:
       - vbsa_results.xlsx (S1, ST, confidence intervals)
       - metadata (sampling info, output variance per time-bin)


5. VBSA visualisations
   --------------------------------------
   main.py then loads vbsa_results.xlsx and calls:
   - plot_top_drivers_per_bin
   - plot_heatmaps_over_time (S1 / ST)
   - plot_S1_vs_ST_scatter
   - plot_feature_time_evolution
   - plot_output_variance_over_time

   All figures are saved automatically in the working directory.


---------------------------------------------------------------------------
Optional: VBSA Using Regression Surrogate
---------------------------------------------------------------------------

VBSA_with_Regression_model.py provides an alternative VBSA implementation based
on a fixed linear regression surrogate loaded from an external Excel file.

This was a fast exploratory try-out and not a fully tuned model. For future
research, stronger regression-type surrogates (e.g., Lasso, Elastic Net,
nonlinear splines, GAMs, or domain-specific physics-aware regressors) can be
added in exactly the same plug-in manner: implement a predict() method and feed
it into the VBSA routine identical to the RandomForest workflow.


---------------------------------------------------------------------------
Dependencies
---------------------------------------------------------------------------

Required packages based on imports:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- xlsxwriter (or another ExcelWriter engine)

---------------------------------------------------------------------------
Usage Summary
---------------------------------------------------------------------------

The minimal usage flow is simply:

    python main.py

This executes:
    - data preparation
    - surrogate training
    - surrogate diagnostics
    - VBSA (RandomForest)
    - VBSA visualisation

Plots, CV metrics, and VBSA results are written into the working directory.


---------------------------------------------------------------------------
End of README
---------------------------------------------------------------------------
