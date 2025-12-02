import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_against_time():
    # === 1) Load ===
    path = "Complete_data.xlsx"  # adjust path if needed
    inp_raw = pd.read_excel(path, sheet_name="Input", header=None)
    out_raw = pd.read_excel(path, sheet_name="Output", header=None)

    # === 2) Parse Output sheet → datapoint table ===
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

    instance_vals = [parse_instance(s) for s in datapoint_id]

    out_df = pd.DataFrame({
        "datapoint_id": datapoint_id,
        "instance": instance_vals,
        "time": time_vals,
        "temperature": temp_vals,
        "frequency": freq_vals,
    })

    # === 3) Global axis ranges (used later to fix all plots) ===
    GLOBAL_TMIN = np.nanmin(out_df["time"].values)
    GLOBAL_TMAX = np.nanmax(out_df["time"].values)

    # === 4) Parse Input sheet → long table ===
    param_id = inp_raw.iloc[:, 0].astype(str) + "|" + inp_raw.iloc[:, 1].astype(str)
    inp_vals = inp_raw.iloc[:, col_idx].copy()
    inp_vals.columns = datapoint_id

    inp_long = (
        inp_vals
        .assign(param=param_id.values)
        .melt(id_vars="param", var_name="datapoint_id", value_name="value_raw")
    )

    # --- Log-rescale degradation values ---
    # log1p handles zeros safely: log(1 + x)
    inp_long["value"] = np.log1p(inp_long["value_raw"].astype(float))

    # Merge with output meta
    data_long = inp_long.merge(out_df, on="datapoint_id", how="left")

    # === 5) Compute global y-axis range (to keep all plots comparable) ===
    ymin_global = np.nanmin(data_long["value"])
    ymax_global = np.nanmax(data_long["value"])

    # === 6) Plot 1: all parameters per instance, same color, same axes ===
    instances = sorted(out_df["instance"].unique())

    for inst in instances:
        sub = data_long[data_long["instance"] == inst]
        plt.figure(figsize=(10, 6))
        for _, dfp in sub.groupby("param"):
            dfp = dfp.sort_values("time")
            plt.plot(dfp["time"].values, dfp["value"].values, linewidth=0.5, color="gray")
        plt.xlabel("Time")
        plt.ylabel("log(1 + degradation)")
        plt.title(f"Instance {inst}: parameter degradations (log-scaled)")
        plt.xlim(GLOBAL_TMIN, GLOBAL_TMAX)
        plt.ylim(ymin_global, ymax_global)
        plt.tight_layout()
        plt.savefig(f"Figures_exploration/vs Time/plot_instance_{inst}_all_params_log.png", dpi=150)
        plt.close()

    # === 7) Plot 2: aggregated per-instance + overall mean (red) ===
    inst_series: Dict[int, pd.DataFrame] = {}
    for inst in instances:
        sub = data_long[data_long["instance"] == inst]
        agg = sub.groupby("time", as_index=False)["value"].mean().sort_values("time")
        inst_series[inst] = agg

    grid = np.linspace(GLOBAL_TMIN, GLOBAL_TMAX, 400)
    grid_vals = []
    for inst, series in inst_series.items():
        t = series["time"].values
        v = series["value"].values
        if len(t) < 2:
            continue
        v_interp = np.interp(grid, t, v, left=np.nan, right=np.nan)
        mask = (grid >= np.nanmin(t)) & (grid <= np.nanmax(t))
        arr = np.full_like(grid, np.nan, dtype=float)
        arr[mask] = v_interp[mask]
        grid_vals.append(arr)

    mean_overall = np.nanmean(np.vstack(grid_vals), axis=0) if grid_vals else np.full_like(grid, np.nan)

    plt.figure(figsize=(11, 7))
    for inst, series in inst_series.items():
        plt.plot(series["time"].values, series["value"].values, linewidth=0.8, color="gray")
    plt.plot(grid, mean_overall, linewidth=2.0, color="red", label="Overall mean")
    plt.xlabel("Time")
    plt.ylabel("Mean log(1 + degradation)")
    plt.title("Aggregated degradation per instance (grey) + overall mean (red)")
    #plt.xlim(GLOBAL_TMIN, GLOBAL_TMAX)
    #plt.ylim(ymin_global, ymax_global)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figures_exploration/vs Time/plot_aggregated_instances_with_overall_mean_log.png", dpi=160)
    plt.close()



def plot_against_GHz():

    # === 1) Load ===
    path = "Complete_data.xlsx"  # adjust if needed
    inp_raw = pd.read_excel(path, sheet_name="Input", header=None)
    out_raw = pd.read_excel(path, sheet_name="Output", header=None)

    # === 2) Parse Output sheet → datapoint table ===
    col_idx = list(range(2, out_raw.shape[1]))

    datapoint_id = out_raw.iloc[0, col_idx].astype(str).tolist()
    time_vals     = pd.to_numeric(out_raw.iloc[1, col_idx], errors="coerce").values
    temp_vals     = pd.to_numeric(out_raw.iloc[2, col_idx], errors="coerce").values
    freq_vals     = pd.to_numeric(out_raw.iloc[3, col_idx], errors="coerce").values  # GHz

    def parse_instance(s: str) -> int:
        try:
            return int(str(s).split(".")[0])
        except Exception:
            return -1

    instance_vals = [parse_instance(s) for s in datapoint_id]

    out_df = pd.DataFrame({
        "datapoint_id": datapoint_id,
        "instance": instance_vals,
        "time": time_vals,
        "temperature": temp_vals,
        "frequency": freq_vals,
    })

    # === 3) Global axis ranges (for consistency) ===
    GLOBAL_FMIN = np.nanmin(out_df["frequency"].values)
    GLOBAL_FMAX = np.nanmax(out_df["frequency"].values)

    # === 4) Parse Input sheet → long table ===
    param_id = inp_raw.iloc[:, 0].astype(str) + "|" + inp_raw.iloc[:, 1].astype(str)
    inp_vals = inp_raw.iloc[:, col_idx].copy()
    inp_vals.columns = datapoint_id

    inp_long = (
        inp_vals
        .assign(param=param_id.values)
        .melt(id_vars="param", var_name="datapoint_id", value_name="value_raw")
    )

    # Log-scale the degradation values for better comparability
    inp_long["value"] = np.log1p(inp_long["value_raw"].astype(float))

    # Merge with frequency data instead of time
    data_long = inp_long.merge(out_df, on="datapoint_id", how="left")

    # === 5) Global y-axis range (same over all plots) ===
    ymin_global = np.nanmin(data_long["value"])
    ymax_global = np.nanmax(data_long["value"])

    # === 6) Plot 1: for each instance, all parameters (same color), over Frequency ===
    instances = sorted(out_df["instance"].unique())

    for inst in instances:
        sub = data_long[data_long["instance"] == inst]
        plt.figure(figsize=(10, 6))
        for _, dfp in sub.groupby("param"):
            dfp = dfp.sort_values("frequency")
            plt.plot(dfp["frequency"].values, dfp["value"].values, linewidth=0.5, color="gray")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("log(1 + degradation)")
        plt.title(f"Instance {inst}: parameter degradations vs Frequency (log-scaled)")
        plt.xlim(GLOBAL_FMIN, GLOBAL_FMAX)
        plt.ylim(ymin_global, ymax_global)
        plt.tight_layout()
        plt.savefig(f"Figures_exploration/vs GHz/plot_instance_{inst}_degradation_vs_GHz.png", dpi=150)
        plt.close()

    # === 7) Plot 2: aggregated per-instance mean (vs GHz) + overall mean (red) ===
    inst_series: Dict[int, pd.DataFrame] = {}
    for inst in instances:
        sub = data_long[data_long["instance"] == inst]
        agg = sub.groupby("frequency", as_index=False)["value"].mean().sort_values("frequency")
        inst_series[inst] = agg

    # Create common grid across all frequency values
    grid = np.linspace(GLOBAL_FMIN, GLOBAL_FMAX, 400)
    grid_vals = []
    for inst, series in inst_series.items():
        x = series["frequency"].values
        y = series["value"].values
        if len(x) < 2:
            continue
        y_interp = np.interp(grid, x, y, left=np.nan, right=np.nan)
        mask = (grid >= np.nanmin(x)) & (grid <= np.nanmax(x))
        arr = np.full_like(grid, np.nan, dtype=float)
        arr[mask] = y_interp[mask]
        grid_vals.append(arr)

    mean_overall = np.nanmean(np.vstack(grid_vals), axis=0) if grid_vals else np.full_like(grid, np.nan)

    plt.figure(figsize=(11, 7))
    # Per-instance thin grey lines
    for inst, series in inst_series.items():
        plt.plot(series["frequency"].values, series["value"].values, linewidth=0.8, color="gray")
    # Overall mean red line
    plt.plot(grid, mean_overall, linewidth=2.0, color="red", label="Overall mean")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Mean log(1 + degradation)")
    plt.title("Aggregated degradation vs Frequency (grey = instances, red = overall mean)")
    #plt.xlim(GLOBAL_FMIN, GLOBAL_FMAX)
    #plt.ylim(ymin_global, ymax_global)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figures_exploration/vs GHz/plot_aggregated_degradation_vs_GHz.png", dpi=160)
    plt.close()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from typing import List, Tuple

def plot_degradation_histograms_by_transistor(
    path: str = "Complete_data.xlsx",
    input_sheet: str = "Input",
    rows: int = 3,
    cols: int = 6,
    bins: int = 30,
    save_prefix: str = "hist_transistor_page",
    use_log1p: bool = True,
) -> List[str]:
    """
    Build histograms of rescaled (log) degradation values per transistor type (col 0 in Input).
    Creates a grid of size rows x cols per page (6x3 by default), covering all unique transistor types.
    Uses seaborn for the histograms. Returns list of saved figure filenames.
    """
    # --- Load input sheet (no header) ---
    inp = pd.read_excel(path, sheet_name=input_sheet, header=None)
    # Columns: 0=transistor id (e.g., "I0.MN0"), 1=parameter name, 2..end = datapoints
    if inp.shape[1] < 3:
        raise ValueError("Expected at least 3 columns in the Input sheet.")

    # --- Extract transistor IDs and stack all datapoint values per transistor ---
    transistor_col = inp.iloc[:, 0].astype(str)
    values_mat = inp.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")  # to numeric
    # Keep only non-negative values for log1p stability (drop negatives if any noise present)
    values_mat = values_mat.where(values_mat >= 0)

    # Build long format: (transistor, value)
    long = values_mat.copy()
    long["transistor"] = transistor_col.values
    long = long.melt(id_vars="transistor", value_name="value").drop(columns="variable")
    long = long.dropna(subset=["value"])

    # --- Rescale: log ---
    if use_log1p:
        long["value_log"] = np.log1p(long["value"].astype(float))
        yvals = "value_log"
        ylabel = "log(1 + degradation)"
    else:
        # If strict log is desired, uncomment the next two lines and set a small epsilon
        # eps = 1e-12
        # long["value_log"] = np.log(long["value"].astype(float) + eps)
        long["value_log"] = np.log1p(long["value"].astype(float))  # safe default
        yvals = "value_log"
        ylabel = "log(1 + degradation)"

    # --- Unique transistor types and pagination across pages of rows*cols ---
    unique_trans = sorted(long["transistor"].unique())
    per_page = rows * cols
    n_pages = ceil(len(unique_trans) / per_page)

    saved_files: List[str] = []
    for page in range(n_pages):
        chunk = unique_trans[page * per_page : (page + 1) * per_page]

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = np.array(axes).reshape(rows, cols)

        for i, trans in enumerate(chunk):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            dat = long.loc[long["transistor"] == trans, yvals]
            sns.histplot(dat, bins=bins, ax=ax)
            ax.set_title(f"{trans}  (n={len(dat)})", fontsize=10)
            ax.set_xlabel(ylabel)
            ax.set_ylabel("Count")

        # Hide any unused subplots on the last page
        for j in range(len(chunk), per_page):
            r, c = divmod(j, cols)
            axes[r, c].axis("off")

        plt.tight_layout()
        fname = f"{save_prefix}_{page+1}.png"
        fig.savefig(fname, dpi=160)
        plt.close(fig)
        saved_files.append(fname)

    return saved_files

