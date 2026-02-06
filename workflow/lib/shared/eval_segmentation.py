"""Functions for evaluating segmentation results."""

from pathlib import Path

import pandas as pd
from lib.shared.io import read_image
from microfilm.microplot import Microimage
import matplotlib.pyplot as plt

from lib.shared.file_utils import parse_filename, parse_nested_path
from lib.shared.eval import plot_plate_heatmap
from lib.shared.configuration_utils import create_micropanel
from lib.shared.configuration_utils import image_segmentation_annotations
from lib.shared.segment_cellpose import prepare_cellpose


def segmentation_overview(segmentation_stats_paths):
    """Compile segmentation statistics across multiple files and aggregate by well.

    Processes each segmentation stats file to extract counts for initial nuclei, initial cells,
    after-edge-removal nuclei, after-edge-removal cells, final cells, and final nuclei.
    Aggregates the counts across tiles for each well.

    Args:
        segmentation_stats_paths (list of str): List of file paths to segmentation statistics files,
            each containing well and tile segmentation data.

    Returns:
        pandas.DataFrame: A DataFrame with aggregated segmentation counts for each well.
            Columns include 'well', 'initial_nuclei', 'initial_cells', 'after_edge_removal_nuclei',
            'after_edge_removal_cells', 'final_cells', and 'final_nuclei', with values summed across tiles.
    """
    # Initialize an empty list to store individual DataFrames
    data_frames = []

    # Process each segmentation stats file
    for segmentation_stats_path in segmentation_stats_paths:
        # Read the segmentation stats file
        segmentation_stats = pd.read_csv(segmentation_stats_path, sep="\t")

        # Parse filename to get well and tile information
        segmentation_filename = Path(segmentation_stats_path).name
        data_location, _, _ = parse_filename(segmentation_filename)
        if not data_location:
            data_location, _, _ = parse_nested_path(
                segmentation_stats_path, ["plate", "well", "tile"]
            )
        well = data_location["well"]

        # Add the well information as a column
        segmentation_stats["well"] = well

        # Append this DataFrame to the list
        data_frames.append(segmentation_stats)

    # Concatenate all data frames
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Group by well and sum across tiles
    segmentation_overview = combined_df.groupby("well").sum().reset_index()

    return segmentation_overview


def plot_cell_density_heatmap(df_cells, shape="square", plate="6W", **kwargs):
    """Plot a heatmap of cell density by well and tile in a plate layout.

    This function calculates and visualizes the number of cells per well-tile combination in a plate layout.
    It uses `plot_plate_heatmap` to plot cell density across the specified plate type (e.g., '6W', '24W', '96W').

    Args:
        df_cells (pandas.DataFrame):
            DataFrame containing cell data with columns 'well' and 'tile', where each row represents a cell
            located in a specific well and tile.
        shape (str, optional):
            Shape of the subplot for each well within the plate. Options include 'square', '6W_ph', '6W_sbs',
            or a list defining custom row layouts. Defaults to 'square'.
        plate (str):
            Type of plate for plotting layout. Options are {'6W', '24W', '96W'}, which determine the overall
            layout of wells in the plot.
        **kwargs:
            Additional keyword arguments to customize the appearance, passed directly to `plot_plate_heatmap()`.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: A summary DataFrame with cell counts (`cell count`) per well-tile combination.
            - matplotlib.figure.Figure: The figure object containing the plotted heatmap.

    """
    # Calculate cell counts by well and tile
    df_summary = (
        df_cells.groupby(["well", "tile"]).size().reset_index(name="cell count")
    )

    # Ensure all wells and tiles are represented
    max_tile = df_summary["tile"].max()
    all_tiles = pd.DataFrame(
        [(w, t) for w in df_summary["well"].unique() for t in range(max_tile + 1)],
        columns=["well", "tile"],
    )

    # Merge and sort
    df_summary = (
        all_tiles.merge(df_summary, on=["well", "tile"], how="left")
        .fillna({"cell count": 0})
        .sort_values(["well", "tile"])
    )

    # Plot heatmap
    fig, _ = plot_plate_heatmap(
        df_summary, metric="cell count", shape=shape, plate=plate, **kwargs
    )

    return df_summary, fig


def evaluate_segmentation_paramsearch(
    df,
    segmentation_process="sbs",
    default_cell_diameter=None,
    default_nuclei_diameter=None,
    default_cellprob_threshold=None,
    default_flow_threshold=None,
    channel_cmaps=None,
    prepare_cellpose_kwargs=None,
):
    """Calculate parameter optimization metrics and visualize segmentation results.

    Processes segmentation results to identify optimal parameters and compare against defaults.
    Generates statistical summaries and visual comparisons of segmentation quality.

    Args:
        df (pandas.DataFrame): DataFrame containing segmentation results with columns:
            nuclei_diameter, cell_diameter, flow_threshold, cellprob_threshold,
            initial_nuclei, initial_cells, final_cells, final_nuclei,
            after_edge_removal_cells, after_edge_removal_nuclei, path
        segmentation_process (str, optional): Process type to evaluate.
            Must be either "sbs" or "phenotype". Defaults to "sbs".
        default_cell_diameter (float, optional): Reference cell diameter for comparison.
            If None, only optimal parameters are shown. Defaults to None.
        default_nuclei_diameter (float, optional): Reference nuclei diameter.
            Required if default_cell_diameter is provided. Defaults to None.
        default_cellprob_threshold (float, optional): Reference cell probability threshold.
            Required if default_cell_diameter is provided. Defaults to None.
        default_flow_threshold (float, optional): Reference flow threshold.
            Required if default_cell_diameter is provided. Defaults to None.
        channel_cmaps (list of str, optional): List of colormap names for each channel.
            Only used when segmentation_process is "phenotype".
        prepare_cellpose_kwargs (dict, optional): Keywords for prepare_cellpose function.
            Only used when segmentation_process is "sbs".

    Returns:
        pandas.DataFrame: Statistics grouped by parameter combinations, containing columns:
            initial_nuclei_mean, initial_cells_mean, final_cells_mean, final_nuclei_mean,
            cell_retention_mean, nuclei_retention_mean, measurement_count, combined_score
        str: Formatted summary text containing performance metrics for optimal and default parameters
        Micropanel: Visualization comparing optimal and default segmentation results (if defaults provided)
    """

    def paths_segmentation_paramsearch(base_path):
        """Generate paths for nuclei, cell, and aligned image files from a stats path.

        Takes a segmentation stats file path and converts it to corresponding paths
        for the nuclei, cell mask, and illumination-corrected image files.

        Args:
            base_path (str): Path to segmentation stats TSV file, typically containing
                '/tsvs/' and ending with '_segmentation_stats.tsv'

        Returns:
            str: Path to nuclei mask TIFF file (_nuclei.tiff)
            str: Path to cell mask TIFF file (_cells.tiff)
            str: Path to illumination-corrected TIFF file (__illumination_corrected.tiff)
        """
        # Convert Path to string if needed
        base_path = str(base_path)

        # Replace /tsvs/ with /images/
        base_path = base_path.replace("/tsvs/", "/images/")

        # Remove extension "_segmentation_stats.tsv"
        prefix = base_path.replace("_segmentation_stats.tsv", "")

        # Create paths for nuclei and cells
        nuclei_path = f"{prefix}_nuclei.tiff"
        cells_path = f"{prefix}_cells.tiff"

        # Get aligned image path by replacing /paramsearch/tsvs/ with /
        aligned_path = base_path.replace("/paramsearch/", "/")
        aligned_path = (
            aligned_path.split("__paramsearch")[0] + "__illumination_corrected.tiff"
        )

        return nuclei_path, cells_path, aligned_path

    param_cols = [
        "nuclei_diameter",
        "cell_diameter",
        "flow_threshold",
        "cellprob_threshold",
    ]

    df["cell_retention"] = df["final_cells"] / df["after_edge_removal_cells"]
    df["nuclei_retention"] = df["final_nuclei"] / df["after_edge_removal_nuclei"]

    metrics = [
        "initial_nuclei",
        "initial_cells",
        "final_cells",
        "final_nuclei",
        "cell_retention",
        "nuclei_retention",
    ]

    grouped_stats = (
        df.groupby(param_cols)[metrics]
        .agg(
            {
                "initial_nuclei": ["mean"],
                "initial_cells": ["mean"],
                "final_cells": ["mean"],
                "final_nuclei": ["mean"],
                "cell_retention": ["mean"],
                "nuclei_retention": ["mean"],
            }
        )
        .round(2)
    )

    grouped_stats.columns = [f"{col[0]}_{col[1]}" for col in grouped_stats.columns]
    grouped_stats["measurement_count"] = df.groupby(param_cols).size()

    grouped_stats["combined_score"] = (
        grouped_stats["cell_retention_mean"]
        * grouped_stats["nuclei_retention_mean"]
        * grouped_stats["final_cells_mean"]
    )

    grouped_stats = grouped_stats.sort_values("combined_score", ascending=False)
    best_params = grouped_stats.index[0]
    best_stats = grouped_stats.iloc[0]

    # Generate summary text
    base_summary = f"""=== Segmentation Parameter Optimization Summary ===

    Optimal Parameters:
    - Nuclei Diameter: {best_params[0]:.2f}
    - Cell Diameter: {best_params[1]:.2f}
    - Flow Threshold: {best_params[2]:.2f}
    - Cell Probability Threshold: {best_params[3]:.2f}

    Performance Metrics:
    - Cell Retention: {best_stats["cell_retention_mean"] * 100:.1f}%
    - Nuclei Retention: {best_stats["nuclei_retention_mean"] * 100:.1f}%
    - Final Cells (avg): {best_stats["final_cells_mean"]:.0f}
    - Final Nuclei (avg): {best_stats["final_nuclei_mean"]:.0f}
    - Number of measurements: {best_stats["measurement_count"]}
    - Combined Score: {best_stats["combined_score"]:.1f}"""

    if default_cell_diameter is not None:
        default_params = (
            default_nuclei_diameter,
            default_cell_diameter,
            default_flow_threshold,
            default_cellprob_threshold,
        )
        default_stats = grouped_stats.loc[default_params]

        default_summary = f"""
    Default Parameters:
    - Nuclei Diameter: {default_nuclei_diameter:.2f}
    - Cell Diameter: {default_cell_diameter:.2f}
    - Flow Threshold: {default_flow_threshold:.2f}
    - Cell Probability Threshold: {default_cellprob_threshold:.2f}

    Performance Metrics:
    - Cell Retention: {default_stats["cell_retention_mean"] * 100:.1f}%
    - Nuclei Retention: {default_stats["nuclei_retention_mean"] * 100:.1f}%
    - Final Cells (avg): {default_stats["final_cells_mean"]:.0f}
    - Final Nuclei (avg): {default_stats["final_nuclei_mean"]:.0f}
    - Number of measurements: {default_stats["measurement_count"]}
    - Combined Score: {default_stats["combined_score"]:.1f}"""

        summary_text = base_summary + default_summary
    else:
        summary_text = base_summary

    print(summary_text)
    # Get a row with optimal parameters
    optimal_example = df[
        (df["nuclei_diameter"] == best_params[0])
        & (df["cell_diameter"] == best_params[1])
        & (df["flow_threshold"] == best_params[2])
        & (df["cellprob_threshold"] == best_params[3])
    ].iloc[0]

    optimal_nuclei_path, optimal_cells_path, corrected_full_path = (
        paths_segmentation_paramsearch(optimal_example["path"])
    )

    if default_cell_diameter is not None:
        # Get a row with default parameters
        default_example = df[
            (df["nuclei_diameter"] == default_nuclei_diameter)
            & (df["cell_diameter"] == default_cell_diameter)
            & (df["flow_threshold"] == default_flow_threshold)
            & (df["cellprob_threshold"] == default_cellprob_threshold)
        ].iloc[0]

        default_nuclei_path, default_cells_path, _ = paths_segmentation_paramsearch(
            default_example["path"]
        )

        # Create visualization
        optimal_nuclei = read_image(optimal_nuclei_path)
        optimal_cells = read_image(optimal_cells_path)
        default_nuclei = read_image(default_nuclei_path)
        default_cells = read_image(default_cells_path)
        corrected_image_data = read_image(corrected_full_path)

        if segmentation_process == "phenotype":
            annotated_optimal = image_segmentation_annotations(
                corrected_image_data, optimal_nuclei, optimal_cells
            )
            annotated_default = image_segmentation_annotations(
                corrected_image_data, default_nuclei, default_cells
            )

            seg_microimages = [
                Microimage(
                    annotated_optimal,
                    channel_names="Optimal",
                    cmaps=channel_cmaps + ["pure_cyan"],
                ),
                Microimage(
                    annotated_default,
                    channel_names="Default",
                    cmaps=channel_cmaps + ["pure_cyan"],
                ),
            ]
            seg_panel = create_micropanel(seg_microimages, add_channel_label=True)
            plt.show()

        elif segmentation_process == "sbs":
            cellpose_rgb = prepare_cellpose(
                corrected_image_data, **prepare_cellpose_kwargs
            )

            annotated_optimal = image_segmentation_annotations(
                cellpose_rgb[1:], optimal_nuclei, optimal_cells
            )
            annotated_default = image_segmentation_annotations(
                cellpose_rgb[1:], default_nuclei, default_cells
            )

            seg_microimages = [
                Microimage(
                    annotated_optimal,
                    channel_names="Optimal",
                    cmaps=["pure_blue", "pure_red", "pure_cyan"],
                ),
                Microimage(
                    annotated_default,
                    channel_names="Default",
                    cmaps=["pure_blue", "pure_red", "pure_cyan"],
                ),
            ]
            seg_panel = create_micropanel(seg_microimages, add_channel_label=True)
            plt.show()

    return grouped_stats, summary_text, seg_panel
