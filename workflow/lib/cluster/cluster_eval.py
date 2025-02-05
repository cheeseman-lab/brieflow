from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from lib.shared.file_utils import parse_filename


def plot_cell_histogram(df, cutoff, bins=50, figsize=(12, 6)):
    """
    Plot a histogram of cell numbers with a vertical cutoff line and return genes below the cutoff.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'cell_number' and 'gene_symbol_0' columns
    cutoff : float
        Vertical line position and threshold for identifying genes
    bins : int, optional
        Number of bins for histogram (default: 50)
    figsize : tuple, optional
        Figure size as (width, height) (default: (12, 6))

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object of the generated plot
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram using seaborn for better styling
    sns.histplot(data=df, x="cell_number", bins=bins, color="skyblue", alpha=0.6, ax=ax)

    # Add vertical line at cutoff
    ax.axvline(x=cutoff, color="red", linestyle="--", label=f"Cutoff: {cutoff}")

    # Customize the plot
    ax.set_title("Distribution of Cell Numbers", fontsize=12, pad=15)
    ax.set_xlabel("Cell Number", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend()

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Get genes below cutoff
    genes_below_cutoff = df[df["cell_number"] <= cutoff]["gene_symbol_0"].tolist()

    # Print genes below cutoff
    print(f"Number of genes below cutoff: {len(genes_below_cutoff)}")
    print(genes_below_cutoff)

    # Return the figure object
    return fig


def aggregate_global_metrics(global_metrics_files: list) -> pd.DataFrame:
    """Aggregate global metrics from multiple files into a unified DataFrame.

    Args:
        global_metrics_files (list): List of file paths to global metrics TSV files.

    Returns:
        pd.DataFrame: A DataFrame with one row per file, containing columns for dataset,
                      channel combo, and each metric.
    """
    combined_data = []

    for global_metrics_fp in global_metrics_files:
        # Extract dataset and channel combo
        metadata = parse_filename(global_metrics_fp)[0]
        dataset = metadata["dataset"]
        channel_combo = Path(global_metrics_fp).parent.parent.name

        # Read metrics and convert to a dictionary
        metrics = (
            pd.read_csv(global_metrics_fp, sep="\t")
            .set_index("metric")["value"]
            .to_dict()
        )

        # Append row data
        combined_data.append(
            {"dataset": dataset, "channel_combo": channel_combo, **metrics}
        )

    # Create the final DataFrame
    return pd.DataFrame(combined_data)
