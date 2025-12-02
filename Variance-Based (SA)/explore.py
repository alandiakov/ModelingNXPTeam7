from explore_plots import plot_against_time, plot_against_GHz, plot_degradation_histograms_by_transistor
from surrogate_model import _parse_output_sheet
import pandas as pd

#plot_against_time()
#plot_against_GHz()

files = plot_degradation_histograms_by_transistor(
    path="Complete_data.xlsx",
    input_sheet="Input",
    rows=3, cols=6, bins=30,
    save_prefix="Figures_exploration/hist_transistor_page",
    use_log1p=True
)


def plot_time_distribution(out_df: pd.DataFrame):
    """Quick histogram of all measurement time points."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(7, 4))
    sns.histplot(out_df["time"].dropna(), bins=30, kde=True)
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.title("Distribution of measurement time points")
    plt.tight_layout()
    plt.savefig('Figures_exploration/Timepoints.png', dpi=160)
    plt.close()

out_df = _parse_output_sheet(pd.read_excel("Complete_data.xlsx", sheet_name="Output", header=None))
plot_time_distribution(out_df)
