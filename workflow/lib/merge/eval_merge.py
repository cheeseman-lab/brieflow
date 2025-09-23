"""Helper functions for evaluating results of merge process."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
import yaml

from lib.shared.eval import plot_plate_heatmap


def plot_sbs_ph_matching_heatmap(
    df_merge,
    df_info,
    target="sbs",
    shape="square",
    plate="6W",
    return_plot=True,
    return_summary=False,
    **kwargs,
):
    """Plots the rate of matching segmented cells between phenotype and SBS datasets by well and tile in a convenient plate layout.

    Args:
        df_merge: DataFrame of all matched cells, e.g., concatenated outputs for all tiles and wells
            of merge_triangle_hash. Expects 'tile' and 'cell_0' columns to correspond to phenotype data and
            'site', 'cell_1' columns to correspond to SBS data.
        df_info: DataFrame of all cells segmented from either phenotype or SBS images, e.g., concatenated outputs for all tiles
            and wells of extract_phenotype_minimal(data_phenotype=nulcei, nuclei=nuclei), often used as `sbs_cell_info`
            rule in Snakemake.
        target: Which dataset to use as the target, e.g., if target='sbs', plots the fraction of cells in each SBS tile
            that match to a phenotype cell. Should match the information stored in df_info; if df_info is a table of all
            segmented cells from SBS tiles, then target should be set as 'sbs'.
        shape: Shape of subplot for each well used in `plot_plate_heatmap`. Defaults to 'square' and infers shape based on
            the value of `target`.
        plate: Plate type for `plot_plate_heatmap`, options are {'6W', '24W', '96W'}.
        return_plot: If True, returns `df_summary`.
        return_summary: If True, returns `df_summary`.
        **kwargs: Additional keyword arguments passed to `plot_plate_heatmap()`.

    Returns:
        df_summary: DataFrame used for plotting, returned if `return_summary=True`.
        axes: Numpy array of matplotlib Axes objects.
    """
    # Determine the merge columns and source based on the target
    if target == "sbs":
        merge_cols = ["site", "cell_1"]
        source = "phenotype"
        # Determine the default shape if not provided
        if not shape:
            shape = "6W_sbs"
    elif target == "phenotype":
        merge_cols = ["tile", "cell_0"]
        source = "sbs"
        # Determine the default shape if not provided
        if not shape:
            shape = "6W_ph"
    else:
        raise ValueError("target = {} not implemented".format(target))

    # Calculate the summary dataframe
    df_summary = (
        df_info.rename(columns={"tile": merge_cols[0], "cell": merge_cols[1]})[
            ["well"] + merge_cols
        ]
        .merge(
            df_merge[["well"] + merge_cols + ["distance"]],
            how="left",
            on=["well"] + merge_cols,
        )
        .assign(matched=lambda x: x["distance"].notna())
        .groupby(["well"] + merge_cols[:1])["matched"]
        .value_counts(normalize=True)
        .rename("fraction of {} cells matched to {} cells".format(target, source))
        .to_frame()
        .reset_index()
        .query("matched==True")
        .drop(columns="matched")
        .rename(columns={merge_cols[0]: "tile"})
    )

    if return_summary and return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return df_summary, axes[0]
    elif return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return axes[0]
    elif return_summary:
        return df_summary
    else:
        return None


def plot_channel_histogram(df_before, df_after, channel_min_cutoff=0):
    """Generates a histogram of channel values with raw counts and consistent bin edges.

    Args:
        df_before: DataFrame containing channel values before cleaning.
        df_after: DataFrame containing channel values after cleaning.
        channel_min_cutoff: Threshold value to mark with a red vertical line. Defaults to 0.

    Returns:
        The generated matplotlib figure object.
    """
    fig = plt.figure(figsize=(10, 6))

    # Calculate bin edges based on the full range of data
    min_val = min(df_before["channels_min"].min(), df_after["channels_min"].min())
    max_val = max(df_before["channels_min"].max(), df_after["channels_min"].max())
    bins = np.linspace(min_val, max_val, 201)  # 201 edges make 200 bins

    # Plot histograms with raw counts instead of density
    plt.hist(
        df_before["channels_min"].dropna(),
        bins=bins,
        color="blue",
        alpha=0.5,
        label="Before clean",
    )
    plt.hist(
        df_after["channels_min"].dropna(),
        bins=bins,
        color="orange",
        alpha=0.5,
        label="After clean",
    )

    # Add vertical line for channel_min_cutoff
    plt.axvline(channel_min_cutoff, color="red", linestyle="--", label="Cutoff")

    plt.title("Histogram of channels_min Values")
    plt.xlabel("channels_min")
    plt.ylabel("Count")
    plt.legend()
    return fig


def plot_cell_positions(df_merge, title, color=None, hue="channels_min"):
    """Generates a scatter plot of cell positions in the i_0, j_0 coordinate space.

    Args:
        df_merge: DataFrame containing cell position data with i_0, j_0 columns.
        title: Plot title.
        color: Fixed color for all points. If specified, overrides hue.
        hue: Column name for color variation. Defaults to 'channels_min'.

    Returns:
        The generated matplotlib figure object.
    """
    fig = plt.figure(figsize=(20, 20))

    # Plot scatter with either fixed color or hue-based coloring
    if color is not None:
        sns.scatterplot(data=df_merge, x="i_0", y="j_0", color=color, alpha=0.5)
    else:
        sns.scatterplot(data=df_merge, x="i_0", y="j_0", hue=hue, alpha=0.5)

    plt.title(title)
    plt.xlabel("i_0")
    plt.ylabel("j_0")
    return fig


def display_matched_and_unmatched_cells_for_site(
    root_fp,
    plate,
    well,
    selected_site=None,
    distance_threshold=15.0,
    verbose=False,
):
    """Display matched and unmatched cells using stitched_cell_id for matching.

    Args:
        root_fp (str/Path): Root analysis directory
        plate (str): Plate identifier
        well (str): Well identifier
        selected_site (str, optional): Site to filter by (if None, shows first available site)
        distance_threshold (float): Maximum distance to show matches
        verbose (bool): Whether to print detailed logs
    """
    # Construct paths to required files
    root_path = Path(root_fp)
    merge_fp = root_path / "merge"

    merged_cells_path = (
        merge_fp / "parquets" / f"P-{plate}_W-{well}__merge_final.parquet"
    )
    phenotype_transformed_path = (
        merge_fp
        / "well_alignment"
        / f"P-{plate}_W-{well}__phenotype_transformed.parquet"
    )
    sbs_positions_path = (
        merge_fp / "parquets" / f"P-{plate}_W-{well}__sbs_cell_positions.parquet"
    )

    # Check if all required files exist
    missing_files = []
    if not merged_cells_path.exists():
        missing_files.append(f"Raw matches: {merged_cells_path}")
    if not phenotype_transformed_path.exists():
        missing_files.append(f"Transformed phenotype: {phenotype_transformed_path}")
    if not sbs_positions_path.exists():
        missing_files.append(f"SBS positions: {sbs_positions_path}")

    if missing_files:
        print("Missing required files:")
        for f in missing_files:
            print(f" - {f}")
        return None

    try:
        # Load all datasets
        merged_df = pd.read_parquet(merged_cells_path)
        phenotype_transformed = pd.read_parquet(phenotype_transformed_path)
        sbs_positions = pd.read_parquet(sbs_positions_path)

        # Get available sites from merged data
        available_sites = (
            sorted(merged_df["site"].unique()) if "site" in merged_df.columns else []
        )

        # Select site to display
        if selected_site is None:
            if available_sites:
                selected_site = available_sites[0]
            else:
                return None
        elif selected_site not in available_sites:
            return None

        # Filter merged data by site and distance threshold
        site_merged = merged_df[merged_df["site"] == selected_site].copy()
        filtered_merged = site_merged[
            site_merged["distance"] <= distance_threshold
        ].copy()

        # Filter position datasets based on spatial coordinates

        # For SBS positions, filter by site
        if "site" in sbs_positions.columns:
            site_sbs = sbs_positions[sbs_positions["site"] == selected_site].copy()
        elif "tile" in sbs_positions.columns:
            # Sometimes SBS data uses 'tile' instead of 'site'
            site_sbs = sbs_positions[sbs_positions["tile"] == selected_site].copy()
        else:
            site_sbs = sbs_positions.copy()

        # Get the coordinate range of SBS cells in the selected site
        if len(site_sbs) > 0:
            sbs_i_min, sbs_i_max = site_sbs["i"].min(), site_sbs["i"].max()
            sbs_j_min, sbs_j_max = site_sbs["j"].min(), site_sbs["j"].max()

            # For phenotype_transformed, filter by coordinates within SBS range
            site_phenotype = phenotype_transformed[
                (phenotype_transformed["i"] >= sbs_i_min)
                & (phenotype_transformed["i"] <= sbs_i_max)
                & (phenotype_transformed["j"] >= sbs_j_min)
                & (phenotype_transformed["j"] <= sbs_j_max)
            ].copy()

        else:
            return None

        # For phenotype: match using stitched_cell_id_0 from raw_matches vs stitched_cell_id from phenotype_transformed
        if "stitched_cell_id_0" not in filtered_merged.columns:
            return None

        if "stitched_cell_id" not in site_phenotype.columns:
            return None

        matched_phenotype_stitched_ids = set(
            filtered_merged["stitched_cell_id_0"].dropna().unique()
        )
        unmatched_phenotype = site_phenotype[
            ~site_phenotype["stitched_cell_id"].isin(matched_phenotype_stitched_ids)
        ].copy()

        # For SBS: match using stitched_cell_id_1 from raw_matches vs stitched_cell_id from sbs_positions
        if "stitched_cell_id_1" not in filtered_merged.columns:
            return None

        if "stitched_cell_id" not in site_sbs.columns:
            return None

        matched_sbs_stitched_ids = set(
            filtered_merged["stitched_cell_id_1"].dropna().unique()
        )
        unmatched_sbs = site_sbs[
            ~site_sbs["stitched_cell_id"].isin(matched_sbs_stitched_ids)
        ].copy()

        # Create enhanced visualization
        create_enhanced_match_visualization(
            matched_data=filtered_merged,
            site_phenotype=site_phenotype,
            site_sbs=site_sbs,
            unmatched_phenotype=unmatched_phenotype,
            unmatched_sbs=unmatched_sbs,
            site=selected_site,
            distance_threshold=distance_threshold,
        )

        # Return summary statistics for programmatic use
        total_phenotype = len(site_phenotype)
        total_sbs = len(site_sbs)

        summary_stats = {
            "site": selected_site,
            "total_phenotype_cells": total_phenotype,
            "total_sbs_cells": total_sbs,
            "matched_cells": len(filtered_merged),
            "unmatched_phenotype_cells": len(unmatched_phenotype),
            "unmatched_sbs_cells": len(unmatched_sbs),
            "phenotype_match_rate": len(filtered_merged) / total_phenotype
            if total_phenotype > 0
            else 0,
            "sbs_match_rate": len(filtered_merged) / total_sbs if total_sbs > 0 else 0,
            "mean_match_distance": filtered_merged["distance"].mean()
            if len(filtered_merged) > 0
            else None,
            "median_match_distance": filtered_merged["distance"].median()
            if len(filtered_merged) > 0
            else None,
        }

        return summary_stats

    except Exception as e:
        return None


def create_enhanced_match_visualization(
    matched_data,
    site_phenotype,
    site_sbs,
    unmatched_phenotype,
    unmatched_sbs,
    site,
    distance_threshold,
):
    """Create enhanced visualization showing matched and unmatched cells.

    Args:
        matched_data (pd.DataFrame): Matched cell data with columns: global_i_0, global_j_0, global_i_1, global_j_1, distance
        site_phenotype (pd.DataFrame): Raw phenotype cells with columns: i, j
        site_sbs (pd.DataFrame): Raw SBS cells with columns: i, j
        unmatched_phenotype (pd.DataFrame): Unmatched phenotype cells
        unmatched_sbs (pd.DataFrame): Unmatched SBS cells
        site (str): Site name for title
        distance_threshold (float): Distance threshold used
    """
    # Check which coordinate column names exist in matched_data
    i_0_col = "global_i_0" if "global_i_0" in matched_data.columns else "i_0"
    j_0_col = "global_j_0" if "global_j_0" in matched_data.columns else "j_0"
    i_1_col = "global_i_1" if "global_i_1" in matched_data.columns else "i_1"
    j_1_col = "global_j_1" if "global_j_1" in matched_data.columns else "j_1"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Cell Matching Analysis - Site: {site}", fontsize=16)

    # 1. Distance distribution histogram
    ax1 = axes[0, 0]
    if len(matched_data) > 0:
        distances = matched_data["distance"]
        ax1.hist(distances, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.axvline(
            distance_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {distance_threshold}px",
        )
        ax1.axvline(
            distances.mean(),
            color="orange",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {distances.mean():.1f}px",
        )
        ax1.set_xlabel("Distance (pixels)")
        ax1.set_ylabel("Count")
        ax1.set_title("Match Distance Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(
            0.5,
            0.5,
            "No matches found",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
        )
        ax1.set_title("Match Distance Distribution")

    # 2. Spatial overview - Raw positions + Matched cells colored by distance
    ax2 = axes[0, 1]

    # Plot raw phenotype positions
    if len(site_phenotype) > 0:
        sample_size = min(2000, len(site_phenotype))
        sample_ph = (
            site_phenotype.sample(n=sample_size)
            if len(site_phenotype) > sample_size
            else site_phenotype
        )
        ax2.scatter(
            sample_ph["j"],
            sample_ph["i"],
            c="lightcoral",
            s=8,
            alpha=0.4,
            label=f"Raw Phenotype ({len(site_phenotype)})",
        )

    # Plot raw SBS positions
    if len(site_sbs) > 0:
        sample_size = min(2000, len(site_sbs))
        sample_sbs = (
            site_sbs.sample(n=sample_size) if len(site_sbs) > sample_size else site_sbs
        )
        ax2.scatter(
            sample_sbs["j"],
            sample_sbs["i"],
            c="lightblue",
            s=8,
            alpha=0.4,
            label=f"Raw SBS ({len(site_sbs)})",
        )

    # Plot matched cell pairs with connecting lines, colored by distance
    if len(matched_data) > 0:
        sample_size = min(1000, len(matched_data))  # Reduced for performance with lines
        sample_matched = (
            matched_data.sample(n=sample_size)
            if len(matched_data) > sample_size
            else matched_data
        )

        # Draw lines connecting matched pairs
        for _, row in sample_matched.iterrows():
            ax2.plot(
                [row[j_0_col], row[j_1_col]],
                [row[i_0_col], row[i_1_col]],
                "k-",
                alpha=0.8,
                linewidth=1,
            )

        # Plot matched SBS positions
        ax2.scatter(
            sample_matched[j_1_col],
            sample_matched[i_1_col],
            c="lightblue",
            s=20,
            alpha=0.8,
            marker="o",
        )

        # Plot matched phenotype positions
        scatter_ph = ax2.scatter(
            sample_matched[j_0_col],
            sample_matched[i_0_col],
            c=sample_matched["distance"],
            s=20,
            alpha=0.8,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.8,
            marker="o",
            label=f"Matched Phenotype ({len(matched_data)})",
        )

        plt.colorbar(scatter_ph, ax=ax2, label="Match Distance (px)", shrink=0.8)

    ax2.set_xlabel("j (pixels)")
    ax2.set_ylabel("i (pixels)")
    ax2.set_title(
        "Raw Phenotype Positions, Raw SBS Positions, Matched Cells Colored by Distance"
    )
    ax2.legend(bbox_to_anchor=(1.20, 1), loc="upper left", fontsize=10)
    ax2.invert_yaxis()

    # 3. CORRECTED: Pie chart showing % of phenotype cells with a match
    ax3 = axes[1, 0]

    total_phenotype = len(site_phenotype)
    matched_phenotype_count = len(matched_data)
    unmatched_phenotype_count = total_phenotype - matched_phenotype_count

    labels = ["Matched", "Unmatched"]
    sizes = [matched_phenotype_count, unmatched_phenotype_count]
    colors = ["#2ecc71", "#e74c3c"]

    if total_phenotype > 0:
        wedges, texts, autotexts = ax3.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
    else:
        ax3.text(
            0.5,
            0.5,
            "No phenotype cells",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
        )

    ax3.set_title(
        f"% of Phenotype Cells with a Match\n({total_phenotype} total phenotype cells)"
    )

    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis("off")

    total_sbs = len(site_sbs)

    # Create summary table
    summary_data = [
        ["Metric", "Phenotype", "SBS"],
        ["Total Cells", f"{total_phenotype}", f"{total_sbs}"],
        ["Matched Cells", f"{matched_phenotype_count}", f"{len(matched_data)}"],
        [
            "Unmatched Cells",
            f"{unmatched_phenotype_count}",
            f"{total_sbs - len(matched_data)}",
        ],
        [
            "Match Rate",
            f"{matched_phenotype_count / total_phenotype:.1%}"
            if total_phenotype > 0
            else "N/A",
            f"{len(matched_data) / total_sbs:.1%}" if total_sbs > 0 else "N/A",
        ],
    ]

    if len(matched_data) > 0:
        distances = matched_data["distance"]
        summary_data.extend(
            [
                ["Mean Distance", f"{distances.mean():.1f}px", ""],
                ["Median Distance", f"{distances.median():.1f}px", ""],
                ["Excellent (≤2px)", f"{(distances <= 2).sum()}", ""],
                ["Very Good (≤5px)", f"{(distances <= 5).sum()}", ""],
                ["Good (≤10px)", f"{(distances <= 10).sum()}", ""],
                ["Fair (>10px)", f"{(distances > 10).sum()}", ""],
            ]
        )

    # Create table
    table = ax4.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header row
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor("#3498db")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax4.set_title("Summary Statistics")

    plt.tight_layout()
    plt.show()


def run_well_alignment_qc(
    root_fp,
    plate,
    well,
    det_range,
    score,
    threshold,
    selected_site=None,
    distance_threshold=15.0,
    verbose=False,
):
    """Run complete QC visualization for a well alignment with merged cells display.

    Args:
        root_fp (str/Path): Root analysis directory
        plate (str): Plate identifier
        well (str): Well identifier
        det_range (tuple): Determinant range from config (DEPRECATED - no longer used)
        score (float): Score threshold from config
        threshold (float): Distance threshold from config (DEPRECATED - no longer used)
        selected_site (str, optional): Specific site to display merged cells for
        distance_threshold (float): Maximum distance to show matches (default 15.0)
        verbose (bool): Whether to print detailed logs

    Returns:
        dict: Alignment data dictionary
    """
    # Load alignment data
    alignment_data = load_well_alignment_outputs(root_fp, plate, well)

    return alignment_data


def load_well_alignment_outputs(root_fp, plate, well, verbose=False):
    """Load all alignment outputs for a specific well.

    Args:
        root_fp (str/Path): Root analysis directory path
        plate (str): Plate identifier
        well (str): Well identifier
        verbose (bool): Whether to print detailed logs

    Returns:
        dict: Dictionary containing all loaded alignment data
    """
    root_fp = Path(root_fp)
    merge_fp = root_fp / "merge"

    outputs = {}

    # Load alignment parameters
    alignment_path = (
        merge_fp / "well_alignment" / f"P-{plate}_W-{well}__alignment.parquet"
    )
    if alignment_path.exists():
        outputs["alignment_params"] = pd.read_parquet(alignment_path)
    else:
        raise FileNotFoundError(f"Alignment parameters not found: {alignment_path}")

    # Load alignment summary (TSV format, not YAML)
    summary_path = merge_fp / "tsvs" / f"P-{plate}_W-{well}__alignment_summary.tsv"
    if summary_path.exists():
        outputs["alignment_summary"] = pd.read_csv(summary_path, sep="\t").to_dict(
            orient="records"
        )[0]
    else:
        outputs["alignment_summary"] = {}

    # Load original cell positions
    pheno_pos_path = (
        merge_fp / "parquets" / f"P-{plate}_W-{well}__phenotype_cell_positions.parquet"
    )
    if pheno_pos_path.exists():
        outputs["phenotype_positions"] = pd.read_parquet(pheno_pos_path)
    else:
        raise FileNotFoundError(f"Phenotype positions not found: {pheno_pos_path}")

    sbs_pos_path = (
        merge_fp / "parquets" / f"P-{plate}_W-{well}__sbs_cell_positions.parquet"
    )
    if sbs_pos_path.exists():
        outputs["sbs_positions"] = pd.read_parquet(sbs_pos_path)
    else:
        raise FileNotFoundError(f"SBS positions not found: {sbs_pos_path}")

    # Load scaled phenotype positions
    scaled_path = (
        merge_fp / "well_alignment" / f"P-{plate}_W-{well}__phenotype_scaled.parquet"
    )
    if scaled_path.exists():
        outputs["phenotype_scaled"] = pd.read_parquet(scaled_path)
    else:
        raise FileNotFoundError(f"Scaled phenotype positions not found: {scaled_path}")

    # Load transformed phenotype positions
    transformed_path = (
        merge_fp
        / "well_alignment"
        / f"P-{plate}_W-{well}__phenotype_transformed.parquet"
    )
    if transformed_path.exists():
        outputs["phenotype_transformed"] = pd.read_parquet(transformed_path)
    else:
        raise FileNotFoundError(
            f"Transformed phenotype positions not found: {transformed_path}"
        )

    return outputs


def display_well_alignment_summary(alignment_data):
    """Display a summary of the well alignment results.

    Args:
        alignment_data (dict): Output from load_well_alignment_outputs

    Note:
        This function is kept for backward compatibility but does nothing.
        All summary printing has been removed.
    """
    pass
