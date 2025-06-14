"""Generate metrics for a brieflow run."""

import pandas as pd
from pathlib import Path
import numpy as np
import json
from scipy import stats
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_preprocess_stats(config):
    """Get preprocessing statistics including tile counts.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Statistics including sbs_tiles, phenotype_tiles, nd2_files
    """
    # Get paths from config
    sbs_samples_fp = Path(config["preprocess"]["sbs_samples_fp"])
    phenotype_samples_fp = Path(config["preprocess"]["phenotype_samples_fp"])

    # Load the sample TSV files with pandas
    sbs_samples = pd.read_csv(sbs_samples_fp, sep="\t")
    phenotype_samples = pd.read_csv(phenotype_samples_fp, sep="\t")

    # Count rows in each TSV file
    sbs_input_count = len(sbs_samples)
    phenotype_input_count = len(phenotype_samples)

    # Count output TIFF files in the respective directories
    root_fp = Path(config["all"]["root_fp"])
    sbs_tiff_dir = root_fp / "preprocess" / "images" / "sbs"
    phenotype_tiff_dir = root_fp / "preprocess" / "images" / "phenotype"

    # Count TIFF files in the SBS directory
    sbs_tiff_count = 0
    if sbs_tiff_dir.exists():
        sbs_tiff_count = len(list(sbs_tiff_dir.glob("**/*.tiff")))

    # Count TIFF files in the phenotype directory
    phenotype_tiff_count = 0
    if phenotype_tiff_dir.exists():
        phenotype_tiff_count = len(list(phenotype_tiff_dir.glob("**/*.tiff")))

    return {
        "sbs_tiles": sbs_tiff_count,
        "phenotype_tiles": phenotype_tiff_count,
        "nd2_files": sbs_input_count + phenotype_input_count,
    }


def get_sbs_stats(config):
    """Get SBS (Sequencing by Synthesis) statistics.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Statistics including total cells, barcode percentages, and mapping info
    """
    # Extract paths from config
    root_fp = Path(config["all"]["root_fp"])
    sbs_eval_dir = root_fp / "sbs" / "eval"

    # Load segmentation overview data
    seg_overview_files = list(
        sbs_eval_dir.glob("**/segmentation/*__segmentation_overview.tsv")
    )

    total_cells = 0
    if seg_overview_files:
        # Combine all segmentation overview files
        seg_dfs = []
        for file in seg_overview_files:
            df = pd.read_csv(file, sep="\t")
            seg_dfs.append(df)

        if seg_dfs:
            seg_combined = pd.concat(seg_dfs)
            # Sum the final_cells column to get total cells
            total_cells = seg_combined["final_cells"].sum()

    # Load mapping overview data
    mapping_overview_files = list(
        sbs_eval_dir.glob("**/mapping/*__mapping_overview.tsv")
    )

    # Initialize metrics
    one_gene_cells_percent = 0
    one_or_more_barcodes_percent = 0
    total_with_barcode = 0
    total_with_unique_gene = 0

    if mapping_overview_files:
        # Combine all mapping overview files
        map_dfs = []
        for file in mapping_overview_files:
            df = pd.read_csv(file, sep="\t")
            map_dfs.append(df)

        if map_dfs:
            map_combined = pd.concat(map_dfs)

            # Calculate averages across all wells
            one_or_more_barcodes_percent = map_combined[
                "1_or_more_barcodes__percent"
            ].mean()
            one_gene_cells_percent = map_combined["1_gene_cells__percent"].mean()

            # Sum the counts
            total_with_barcode = map_combined["1_or_more_barcodes__count"].sum()
            total_with_unique_gene = map_combined["1_gene_cells__count"].sum()

    return {
        "total_cells": total_cells,
        "percent_cells_with_barcode": one_or_more_barcodes_percent,
        "percent_cells_with_unique_mapping": one_gene_cells_percent,
        "total_with_barcode": total_with_barcode,
        "total_with_unique_gene": total_with_unique_gene,
    }


def get_phenotype_stats(config):
    """Get phenotype statistics including cell counts and feature counts.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Statistics including total cells and feature count
    """
    from lib.aggregate.cell_data_utils import DEFAULT_METADATA_COLS

    # Extract paths from config
    root_fp = Path(config["all"]["root_fp"])
    phenotype_eval_dir = root_fp / "phenotype" / "eval" / "segmentation"
    phenotype_parquet_dir = root_fp / "phenotype" / "parquets"

    # Initialize the total cells counter
    total_cells = 0

    # Find all segmentation overview files
    seg_overview_files = list(phenotype_eval_dir.glob("*__segmentation_overview.tsv"))

    if seg_overview_files:
        # Combine all segmentation overview files
        seg_dfs = []
        for file in seg_overview_files:
            df = pd.read_csv(file, sep="\t")
            seg_dfs.append(df)

        if seg_dfs:
            seg_combined = pd.concat(seg_dfs)
            # Sum the final_cells column to get total cells
            total_cells = seg_combined["final_cells"].sum()

    # Get feature count by reading column names from a sample parquet file
    # without loading the entire file into memory
    feature_count = 0
    sample_parquet_files = list(
        phenotype_parquet_dir.glob("**/*__phenotype_cp.parquet")
    )

    if sample_parquet_files:
        # Get just the schema (column names) from the first parquet file
        parquet_file = sample_parquet_files[0]
        parquet_schema = pq.read_schema(parquet_file)
        all_columns = parquet_schema.names

        # Calculate feature count by subtracting metadata columns
        metadata_cols_found = [
            col for col in DEFAULT_METADATA_COLS if col in all_columns
        ]
        feature_count = len(all_columns) - len(metadata_cols_found)

    return {"total_cells": total_cells, "feature_count": feature_count}


def get_merge_stats(config):
    """Get merge statistics for mapped vs unmapped cells.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Statistics including total cells, mapped cells, and mapping percentages
    """
    # Extract paths from config
    root_fp = Path(config["all"]["root_fp"])
    merge_eval_dir = root_fp / "merge" / "eval"

    # Find all cell mapping stats files
    mapping_stats_files = list(merge_eval_dir.glob("**/*__cell_mapping_stats.tsv"))

    if not mapping_stats_files:
        return {"error": "No cell mapping statistics files found"}

    # Initialize counters
    total_mapped_cells = 0
    total_cells = 0
    mapping_percentages = []

    # Process each file
    for file in mapping_stats_files:
        df = pd.read_csv(file, sep="\t")

        # Get counts for mapped and unmapped cells
        mapped_row = df[df["category"] == "mapped_cells"]
        unmapped_row = df[df["category"] == "unmapped_cells"]

        if not mapped_row.empty and not unmapped_row.empty:
            # Add to totals
            mapped_count = mapped_row["count"].iloc[0]
            unmapped_count = unmapped_row["count"].iloc[0]
            file_total = mapped_count + unmapped_count

            total_mapped_cells += mapped_count
            total_cells += file_total

            # Calculate and store mapping percentage for this file
            mapping_percentage = mapped_count / file_total * 100
            mapping_percentages.append(mapping_percentage)

    # Calculate overall percentage and average percentage
    avg_mapping_percentage = np.mean(mapping_percentages) if mapping_percentages else 0

    return {
        "total_cells": total_cells,
        "total_mapped_cells": total_mapped_cells,
        "average_mapping_percentage_across_plates": avg_mapping_percentage,
    }


def get_aggregate_stats(config, n_rows=1000):
    """Get aggregation statistics including perturbation counts and batch effect metrics.

    Args:
        config: Configuration dictionary
        n_rows: Optional number of rows to sample for memory efficiency

    Returns:
        dict: Dictionary with statistics for each cell_class/channel_combo combination
    """
    # Extract paths from config
    root_fp = Path(config["all"]["root_fp"])
    aggregate_dir = root_fp / "aggregate"

    # Load the aggregate combo TSV file from config
    aggregate_combo_fp = Path(config["aggregate"]["aggregate_combo_fp"])
    aggregate_combos = pd.read_csv(aggregate_combo_fp, sep="\t")

    # Get unique cell_class and channel_combo combinations
    unique_combos = aggregate_combos[["cell_class", "channel_combo"]].drop_duplicates()

    # Store results for each combination
    all_results = {}

    for _, combo in unique_combos.iterrows():
        cell_class = combo["cell_class"]
        channel_combo = combo["channel_combo"]

        try:
            # Process this combination
            result = _get_single_aggregate_stats(
                config, cell_class, channel_combo, n_rows, root_fp, aggregate_dir
            )
            all_results[f"{cell_class}_{channel_combo}"] = result
        except Exception as e:
            print(f"Error processing {cell_class}/{channel_combo}: {str(e)}")
            all_results[f"{cell_class}_{channel_combo}"] = {"error": str(e)}

    return all_results


def _get_single_aggregate_stats(
    config, cell_class, channel_combo, n_rows, root_fp, aggregate_dir
):
    """Helper function to get stats for a single cell_class/channel_combo combination."""
    from lib.aggregate.cell_data_utils import DEFAULT_METADATA_COLS
    from lib.shared.file_utils import load_parquet_subset
    from sklearn.feature_selection import f_classif

    # Load the aggregated TSV file
    aggregated_path = (
        aggregate_dir
        / "tsvs"
        / f"CeCl-{cell_class}_ChCo-{channel_combo}__aggregated.tsv"
    )
    aggregated = pd.read_csv(aggregated_path, sep="\t")

    # Calculate distinct perturbation count and median perturbation cells
    distinct_perturbation_count = len(aggregated["gene_symbol_0"].unique())
    median_perturbation_cells = int(aggregated["cell_count"].median())

    # Count total cells in the final aggregate TSV
    total_aggregated_cells = aggregated["cell_count"].sum()

    # Count features in the final aggregate TSV (columns starting with PC_)
    feature_cols = [col for col in aggregated.columns if col.startswith("PC_")]
    feature_count = len(feature_cols)

    # Count total cells from merge_data parquets (pre-filter)
    merge_data_paths = list(
        root_fp.glob(f"**/*_CeCl-{cell_class}_ChCo-{channel_combo}__merge_data.parquet")
    )

    # Count total cells by getting metadata only (not loading full dataframes)
    total_pre_filtered_cells = 0
    for path in merge_data_paths:
        # Get row count from parquet metadata without loading the data
        parquet_file = pq.ParquetFile(path)
        total_pre_filtered_cells += parquet_file.metadata.num_rows

    # Process filtered data
    filtered_dir = aggregate_dir / "parquets"
    filtered_paths = list(
        filtered_dir.glob(
            f"**/*_CeCl-{cell_class}_ChCo-{channel_combo}__filtered.parquet"
        )
    )

    # Handle filtered data based on n_rows parameter
    if n_rows is None:
        # Original approach - load all data
        filtered = ds.dataset(filtered_paths, format="parquet")
        filtered = filtered.to_table(use_threads=True, memory_pool=None).to_pandas()
    else:
        # Use load_parquet_subset when n_rows is specified
        if len(filtered_paths) == 1:
            # If there's only one file, load with subset function
            filtered = load_parquet_subset(filtered_paths[0], n_rows)
        else:
            # If multiple files, load and concatenate with row limit per file
            filtered_dfs = []
            for path in filtered_paths:
                df = load_parquet_subset(path, n_rows)
                filtered_dfs.append(df)
            filtered = pd.concat(filtered_dfs, ignore_index=True)

    # Count total cells in the filtered dataset
    total_filtered_cells = len(filtered)

    # Calculate batch effect metrics for pre-alignment data
    filtered = filtered.dropna(axis=1)
    old_control_data = filtered[
        filtered["gene_symbol_0"].str.startswith("nontargeting")
    ]
    metadata_cols = DEFAULT_METADATA_COLS + ["class", "confidence"]
    old_control_data["batch"] = (
        old_control_data["plate"].astype(str) + "_" + old_control_data["well"]
    )
    old_control_data = old_control_data.drop(columns=metadata_cols)

    # Drop features with zero variance
    old_feature_data = old_control_data.drop(columns=["batch"])
    non_constant_features = old_feature_data.columns[old_feature_data.var() > 0]
    old_feature_data_filtered = old_feature_data[non_constant_features]

    # Run ANOVA on the filtered features
    X = old_feature_data_filtered.values
    y = old_control_data["batch"].values

    # Run ANOVA F-test
    f_stats, p_vals = f_classif(X, y)

    # Filter out the NaN values
    valid_idx = ~np.isnan(p_vals)
    valid_p_vals = p_vals[valid_idx]
    old_p_vals_median = np.median(valid_p_vals)

    # Read aligned data and calculate post-alignment batch effects
    aligned_path = (
        aggregate_dir
        / "parquets"
        / f"CeCl-{cell_class}_ChCo-{channel_combo}__aligned.parquet"
    )

    if n_rows is None:
        aligned = pd.read_parquet(aligned_path)
    else:
        aligned = load_parquet_subset(aligned_path, n_rows)

    new_control_data = aligned[aligned["gene_symbol_0"].str.startswith("nontargeting")]
    feature_cols = [col for col in new_control_data.columns if col.startswith("PC_")]
    new_control_data["batch"] = (
        new_control_data["plate"].astype(str) + "_" + new_control_data["well"]
    )
    new_control_data = new_control_data[feature_cols + ["batch"]]

    # Separate features and batch labels
    X = new_control_data.drop(columns=["batch"]).values
    y = new_control_data["batch"].values

    # Run ANOVA F-test
    f_stats, p_vals = f_classif(X, y)

    # Filter out the NaN values
    valid_idx = ~np.isnan(p_vals)
    valid_p_vals = p_vals[valid_idx]
    new_p_vals_median = np.median(valid_p_vals)

    return {
        "distinct_perturbation_count": distinct_perturbation_count,
        "median_perturbation_cells": median_perturbation_cells,
        "old_control_p_vals_median": old_p_vals_median,
        "new_control_p_vals_median": new_p_vals_median,
        "total_pre_filtered_cells": total_pre_filtered_cells,
        "total_filtered_cells": total_filtered_cells,
        "total_aggregated_cells": total_aggregated_cells,
        "feature_count": feature_count,
    }


def get_cluster_stats(config):
    """Get clustering metrics including enrichment analysis results.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Contains detailed_results, best_resolutions, and summary_by_cell_class DataFrames
    """
    from statsmodels.stats.multitest import fdrcorrection

    # Extract paths from config
    root_fp = Path(config["all"]["root_fp"])
    cluster_dir = root_fp / "cluster"

    # Read the cluster_combos.tsv file from config
    cluster_combo_fp = Path(config["cluster"]["cluster_combo_fp"])
    cluster_combos = pd.read_csv(cluster_combo_fp, sep="\t")

    # Initialize a dictionary to store results
    results = []

    # Process each cluster combination
    for _, row in cluster_combos.iterrows():
        cell_class = row["cell_class"]
        channel_combo = row["channel_combo"]
        leiden_resolution = row["leiden_resolution"]

        # Build path to the specific cluster directory
        cluster_specific_dir = (
            cluster_dir / channel_combo / cell_class / str(leiden_resolution)
        )

        # Path to the combined table and global metrics (real and shuffled)
        combined_table_path = cluster_specific_dir / "CB-Real__combined_table.tsv"
        real_metrics_path = cluster_specific_dir / "CB-Real__global_metrics.json"
        shuffled_metrics_path = (
            cluster_specific_dir / "CB-Shuffled__global_metrics.json"
        )

        if not combined_table_path.exists() or not real_metrics_path.exists():
            # Skip if essential files don't exist
            continue

        # Count unique clusters from combined table
        try:
            combined_table = pd.read_csv(combined_table_path, sep="\t")
            unique_clusters = len(combined_table["cluster"].unique())
        except Exception as e:
            print(
                f"Error reading combined table for {cell_class}/{channel_combo}/{leiden_resolution}: {str(e)}"
            )
            unique_clusters = 0

        # Read real enrichment metrics from global metrics JSON
        try:
            with open(real_metrics_path, "r") as f:
                real_metrics = json.load(f)

            # Extract CORUM and KEGG enriched clusters (real)
            real_corum_enriched = real_metrics.get("CORUM", {}).get(
                "num_enriched_clusters", 0
            )
            real_kegg_enriched = real_metrics.get("KEGG", {}).get(
                "num_enriched_clusters", 0
            )
            real_corum_proportion = real_metrics.get("CORUM", {}).get(
                "proportion_enriched", 0
            )
            real_kegg_proportion = real_metrics.get("KEGG", {}).get(
                "proportion_enriched", 0
            )

        except Exception as e:
            print(
                f"Error reading real metrics for {cell_class}/{channel_combo}/{leiden_resolution}: {str(e)}"
            )
            real_corum_enriched = 0
            real_kegg_enriched = 0
            real_corum_proportion = 0
            real_kegg_proportion = 0

        # Read shuffled enrichment metrics if available
        shuffled_corum_enriched = 0
        shuffled_kegg_enriched = 0
        shuffled_corum_proportion = 0
        shuffled_kegg_proportion = 0

        if shuffled_metrics_path.exists():
            try:
                with open(shuffled_metrics_path, "r") as f:
                    shuffled_metrics = json.load(f)

                # Extract CORUM and KEGG enriched clusters (shuffled)
                shuffled_corum_enriched = shuffled_metrics.get("CORUM", {}).get(
                    "num_enriched_clusters", 0
                )
                shuffled_kegg_enriched = shuffled_metrics.get("KEGG", {}).get(
                    "num_enriched_clusters", 0
                )
                shuffled_corum_proportion = shuffled_metrics.get("CORUM", {}).get(
                    "proportion_enriched", 0
                )
                shuffled_kegg_proportion = shuffled_metrics.get("KEGG", {}).get(
                    "proportion_enriched", 0
                )

            except Exception as e:
                print(
                    f"Error reading shuffled metrics for {cell_class}/{channel_combo}/{leiden_resolution}: {str(e)}"
                )

        # Calculate total enriched clusters and proportions for each type
        total_real_enriched = real_corum_enriched + real_kegg_enriched
        total_shuffled_enriched = shuffled_corum_enriched + shuffled_kegg_enriched

        # Calculate average proportion for total (CORUM + KEGG)
        # Use weighted average based on the number of clusters in each category
        if real_corum_enriched + real_kegg_enriched > 0:
            real_avg_proportion = (
                real_corum_proportion * real_corum_enriched
                + real_kegg_proportion * real_kegg_enriched
            ) / (real_corum_enriched + real_kegg_enriched)
        else:
            real_avg_proportion = 0

        if shuffled_corum_enriched + shuffled_kegg_enriched > 0:
            shuffled_avg_proportion = (
                shuffled_corum_proportion * shuffled_corum_enriched
                + shuffled_kegg_proportion * shuffled_kegg_enriched
            ) / (shuffled_corum_enriched + shuffled_kegg_enriched)
        else:
            shuffled_avg_proportion = 0

        # Calculate enrichment fold change (real vs. shuffled)
        # Avoid division by zero
        fold_change_corum = real_corum_proportion / max(
            shuffled_corum_proportion, 1e-10
        )
        fold_change_kegg = real_kegg_proportion / max(shuffled_kegg_proportion, 1e-10)
        fold_change_total = real_avg_proportion / max(shuffled_avg_proportion, 1e-10)

        # Statistical test - Fisher's exact test for proportion comparison
        # For CORUM
        if unique_clusters > 0:
            real_corum_not_enriched = unique_clusters - real_corum_enriched
            shuffled_corum_not_enriched = unique_clusters - shuffled_corum_enriched

            corum_contingency = np.array(
                [
                    [real_corum_enriched, real_corum_not_enriched],
                    [shuffled_corum_enriched, shuffled_corum_not_enriched],
                ]
            )
            _, corum_pvalue = stats.fisher_exact(corum_contingency)

            # For KEGG
            real_kegg_not_enriched = unique_clusters - real_kegg_enriched
            shuffled_kegg_not_enriched = unique_clusters - shuffled_kegg_enriched

            kegg_contingency = np.array(
                [
                    [real_kegg_enriched, real_kegg_not_enriched],
                    [shuffled_kegg_enriched, shuffled_kegg_not_enriched],
                ]
            )
            _, kegg_pvalue = stats.fisher_exact(kegg_contingency)

            # For combined
            real_total_not_enriched = unique_clusters - total_real_enriched
            shuffled_total_not_enriched = unique_clusters - total_shuffled_enriched

            # Handle case where total enriched might exceed unique_clusters
            real_total_not_enriched = max(0, real_total_not_enriched)
            shuffled_total_not_enriched = max(0, shuffled_total_not_enriched)

            total_contingency = np.array(
                [
                    [total_real_enriched, real_total_not_enriched],
                    [total_shuffled_enriched, shuffled_total_not_enriched],
                ]
            )
            _, total_pvalue = stats.fisher_exact(total_contingency)
        else:
            corum_pvalue = 1.0
            kegg_pvalue = 1.0
            total_pvalue = 1.0

        # Store results for this combination
        results.append(
            {
                "cell_class": cell_class,
                "channel_combo": channel_combo,
                "leiden_resolution": leiden_resolution,
                "unique_clusters": unique_clusters,
                "real_corum_enriched": real_corum_enriched,
                "real_kegg_enriched": real_kegg_enriched,
                "total_real_enriched": total_real_enriched,
                "real_corum_proportion": real_corum_proportion,
                "real_kegg_proportion": real_kegg_proportion,
                "real_avg_proportion": real_avg_proportion,
                "shuffled_corum_enriched": shuffled_corum_enriched,
                "shuffled_kegg_enriched": shuffled_kegg_enriched,
                "total_shuffled_enriched": total_shuffled_enriched,
                "shuffled_corum_proportion": shuffled_corum_proportion,
                "shuffled_kegg_proportion": shuffled_kegg_proportion,
                "shuffled_avg_proportion": shuffled_avg_proportion,
                "fold_change_corum": fold_change_corum,
                "fold_change_kegg": fold_change_kegg,
                "fold_change_total": fold_change_total,
                "corum_pvalue": corum_pvalue,
                "kegg_pvalue": kegg_pvalue,
                "total_pvalue": total_pvalue,
            }
        )

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # If results dataframe is empty, return early
    if results_df.empty:
        return {
            "detailed_results": results_df,
            "best_resolutions": pd.DataFrame(),
            "summary_by_cell_class": pd.DataFrame(),
        }

    # Calculate adjusted p-values using Benjamini-Hochberg correction
    if not results_df.empty and len(results_df) > 1:
        # Apply FDR correction to each set of p-values
        results_df["corum_pvalue_adj"] = fdrcorrection(
            results_df["corum_pvalue"].values
        )[1]
        results_df["kegg_pvalue_adj"] = fdrcorrection(results_df["kegg_pvalue"].values)[
            1
        ]
        results_df["total_pvalue_adj"] = fdrcorrection(
            results_df["total_pvalue"].values
        )[1]
    else:
        # If there's only one result, adjusted p-value equals the original p-value
        if not results_df.empty:
            results_df["corum_pvalue_adj"] = results_df["corum_pvalue"]
            results_df["kegg_pvalue_adj"] = results_df["kegg_pvalue"]
            results_df["total_pvalue_adj"] = results_df["total_pvalue"]

    # Find the best resolution for each cell_class and channel_combo based on fold change and significance
    best_resolutions = []

    for (cell, channel), group in results_df.groupby(["cell_class", "channel_combo"]):
        # Sort by real average proportion (descending) to find the highest proportion of enriched pathways
        sorted_group = group.sort_values(by="real_avg_proportion", ascending=False)

        # Take the top resolution
        if not sorted_group.empty:
            best_resolution = sorted_group.iloc[0]
            best_resolutions.append(best_resolution)

    best_resolutions_df = (
        pd.DataFrame(best_resolutions) if best_resolutions else pd.DataFrame()
    )

    # Summary by cell class
    if not results_df.empty:
        summary_by_cell_class = (
            results_df.groupby("cell_class")
            .agg(
                {
                    "unique_clusters": "mean",
                    "real_corum_enriched": "sum",
                    "real_kegg_enriched": "sum",
                    "total_real_enriched": "sum",
                    "real_corum_proportion": "mean",
                    "real_kegg_proportion": "mean",
                    "real_avg_proportion": "mean",
                    "shuffled_corum_enriched": "sum",
                    "shuffled_kegg_enriched": "sum",
                    "total_shuffled_enriched": "sum",
                    "shuffled_corum_proportion": "mean",
                    "shuffled_kegg_proportion": "mean",
                    "shuffled_avg_proportion": "mean",
                    "fold_change_corum": "mean",
                    "fold_change_kegg": "mean",
                    "fold_change_total": "mean",
                    "corum_pvalue": "min",  # Take the most significant p-value
                    "kegg_pvalue": "min",
                    "total_pvalue": "min",
                }
            )
            .reset_index()
        )
    else:
        summary_by_cell_class = pd.DataFrame()

    return {
        "detailed_results": results_df,
        "best_resolutions": best_resolutions_df,
        "summary_by_cell_class": summary_by_cell_class,
    }


def get_all_stats(config):
    """Convenience function to get all pipeline statistics at once with formatted output.

    Args:
        config: Loaded configuration file

    Returns:
        dict: Dictionary containing all statistics from different pipeline stages
    """
    print("=" * 80)
    print("PIPELINE STATISTICS REPORT")
    print("=" * 80)

    stats = {}

    # Preprocessing Statistics
    print("\n[1/6] Gathering preprocessing statistics...")
    stats["preprocess"] = get_preprocess_stats(config)
    print("\n PREPROCESSING STATISTICS:")
    print(f"   • ND2 input files: {stats['preprocess']['nd2_files']:,}")
    print(f"   • SBS tiles generated: {stats['preprocess']['sbs_tiles']:,}")
    print(f"   • Phenotype tiles generated: {stats['preprocess']['phenotype_tiles']:,}")

    # SBS Statistics
    print("\n[2/6] Gathering SBS statistics...")
    stats["sbs"] = get_sbs_stats(config)
    print("\n SBS STATISTICS:")
    print(f"   • Total cells segmented: {stats['sbs']['total_cells']:,}")
    print(
        f"   • Cells with barcode: {stats['sbs']['percent_cells_with_barcode']:.1f}% ({stats['sbs']['total_with_barcode']:,} cells)"
    )
    print(
        f"   • Cells with unique gene mapping: {stats['sbs']['percent_cells_with_unique_mapping']:.1f}% ({stats['sbs']['total_with_unique_gene']:,} cells)"
    )

    # Phenotype Statistics
    print("\n[3/6] Gathering phenotype statistics...")
    stats["phenotype"] = get_phenotype_stats(config)
    print("\n PHENOTYPE STATISTICS:")
    print(f"   • Total cells segmented: {stats['phenotype']['total_cells']:,}")
    print(
        f"   • Number of morphological features: {stats['phenotype']['feature_count']:,}"
    )

    # Merge Statistics
    print("\n[4/6] Gathering merge statistics...")
    stats["merge"] = get_merge_stats(config)
    print("\n MERGE STATISTICS:")
    if "error" not in stats["merge"]:
        print(f"   • Total cells processed: {stats['merge']['total_cells']:,}")
        print(
            f"   • Successfully mapped cells: {stats['merge']['total_mapped_cells']:,}"
        )
        print(
            f"   • Average mapping rate: {stats['merge']['average_mapping_percentage_across_plates']:.1f}%"
        )
    else:
        print(f"   ⚠️  {stats['merge']['error']}")

    # Aggregate Statistics
    print("\n[5/6] Gathering aggregation statistics...")
    print(
        "   (This may take a moment as it processes multiple cell class/channel combinations...)"
    )
    stats["aggregate"] = get_aggregate_stats(config)
    print("\n AGGREGATION STATISTICS:")

    for combo_key, combo_stats in stats["aggregate"].items():
        cell_class, channel_combo = combo_key.split("_", 1)
        print(f"\n   {cell_class} cells - {channel_combo}:")

        if "error" not in combo_stats:
            print(
                f"      • Distinct perturbations: {combo_stats['distinct_perturbation_count']:,}"
            )
            print(
                f"      • Median cells per perturbation: {combo_stats['median_perturbation_cells']:,}"
            )
            print(
                f"      • Total cells (pre-filter): {combo_stats['total_pre_filtered_cells']:,}"
            )
            print(
                f"      • Total cells (post-filter): {combo_stats['total_filtered_cells']:,}"
            )
            print(
                f"      • Total cells (aggregated): {combo_stats['total_aggregated_cells']:,}"
            )
            print(f"      • PC features: {combo_stats['feature_count']}")
            print(
                f"      • Batch effect reduction: {combo_stats['old_control_p_vals_median']:.4f} → {combo_stats['new_control_p_vals_median']:.4f}"
            )
        else:
            print(f"      ⚠️  Error: {combo_stats['error']}")

    # Cluster Statistics
    print("\n[6/6] Gathering clustering statistics...")
    print("   (Analyzing enrichment results across multiple resolutions...)")
    stats["cluster"] = get_cluster_stats(config)
    print("\n🌐 CLUSTERING STATISTICS:")

    if not stats["cluster"]["best_resolutions"].empty:
        print("\n   Best resolution for each cell class/channel combination:")
        for _, row in stats["cluster"]["best_resolutions"].iterrows():
            print(f"\n   {row['cell_class']} cells - {row['channel_combo']}:")
            print(f"      • Best resolution: {row['leiden_resolution']}")
            print(f"      • Number of clusters: {row['unique_clusters']}")
            print(f"      • Enriched clusters (real): {row['real_avg_proportion']:.1%}")
            print(
                f"      • Enriched clusters (shuffled): {row['shuffled_avg_proportion']:.1%}"
            )
            print(f"      • Fold enrichment: {row['fold_change_total']:.2f}x")

            # Add significance indicator
            if "total_pvalue_adj" in row and row["total_pvalue_adj"] < 0.05:
                print(
                    f"      • Statistical significance: *** (p_adj = {row['total_pvalue_adj']:.3e})"
                )
            elif "total_pvalue" in row and row["total_pvalue"] < 0.05:
                print(
                    f"      • Statistical significance: * (p = {row['total_pvalue']:.3e})"
                )
    else:
        print("   ⚠️  No clustering results found")

    print("\n" + "=" * 80)
    print("REPORT COMPLETE")
    print("=" * 80)

    return stats
