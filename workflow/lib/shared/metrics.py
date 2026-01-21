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
    """Get merge statistics from merge summary files.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Statistics including total cells, matched cells, and mapping rates
    """
    root_fp = Path(config["all"]["root_fp"])
    merge_eval_dir = root_fp / "merge" / "eval"

    # Find merge summary files (new format: P-{plate}__merge_summary.tsv)
    summary_files = list(merge_eval_dir.glob("*__merge_summary.tsv"))

    if not summary_files:
        # Fall back to old cell_mapping_stats format
        return _get_merge_stats_legacy(config)

    # Aggregate across all plates
    total_ph_cells = 0
    total_sbs_cells = 0
    total_matched = 0
    total_with_barcode = 0
    total_single_gene = 0
    plate_summaries = []

    for file in summary_files:
        df = pd.read_csv(file, sep="\t")

        # Get TOTAL row if present, otherwise sum all wells
        if "TOTAL" in df["well"].values:
            totals = df[df["well"] == "TOTAL"].iloc[0]
        else:
            totals = df.sum(numeric_only=True)
            totals["ph_match_rate"] = df["ph_match_rate"].mean()
            totals["sbs_match_rate"] = df["sbs_match_rate"].mean()

        total_ph_cells += int(totals.get("ph_cells", 0))
        total_sbs_cells += int(totals.get("sbs_cells", 0))
        total_matched += int(totals.get("matched_final", totals.get("matched_raw", 0)))
        total_with_barcode += int(totals.get("cells_with_barcode", 0))
        total_single_gene += int(totals.get("single_gene_count", 0))

        plate_summaries.append(
            {
                "plate": file.stem.split("__")[0],
                "ph_match_rate": float(totals.get("ph_match_rate", 0)),
                "sbs_match_rate": float(totals.get("sbs_match_rate", 0)),
            }
        )

    # Calculate overall rates
    avg_ph_match_rate = (
        np.mean([p["ph_match_rate"] for p in plate_summaries]) if plate_summaries else 0
    )
    avg_sbs_match_rate = (
        np.mean([p["sbs_match_rate"] for p in plate_summaries])
        if plate_summaries
        else 0
    )
    single_gene_rate = (
        total_single_gene / total_with_barcode if total_with_barcode > 0 else 0
    )

    return {
        "phenotype_cells": total_ph_cells,
        "sbs_cells": total_sbs_cells,
        "matched_cells": total_matched,
        "cells_with_barcode": total_with_barcode,
        "cells_with_single_gene": total_single_gene,
        "phenotype_match_rate": avg_ph_match_rate,
        "sbs_match_rate": avg_sbs_match_rate,
        "single_gene_rate": single_gene_rate,
    }


def _get_merge_stats_legacy(config):
    """Legacy merge stats using cell_mapping_stats files."""
    root_fp = Path(config["all"]["root_fp"])
    merge_eval_dir = root_fp / "merge" / "eval"

    mapping_stats_files = list(merge_eval_dir.glob("**/*__cell_mapping_stats.tsv"))

    if not mapping_stats_files:
        return {"error": "No merge statistics files found"}

    total_mapped_cells = 0
    total_cells = 0
    mapping_percentages = []

    for file in mapping_stats_files:
        df = pd.read_csv(file, sep="\t")
        mapped_row = df[df["category"] == "mapped_cells"]
        unmapped_row = df[df["category"] == "unmapped_cells"]

        if not mapped_row.empty and not unmapped_row.empty:
            mapped_count = mapped_row["count"].iloc[0]
            unmapped_count = unmapped_row["count"].iloc[0]
            file_total = mapped_count + unmapped_count

            total_mapped_cells += mapped_count
            total_cells += file_total
            mapping_percentages.append(mapped_count / file_total * 100)

    return {
        "phenotype_cells": total_cells,
        "matched_cells": total_mapped_cells,
        "phenotype_match_rate": np.mean(mapping_percentages) / 100
        if mapping_percentages
        else 0,
    }


def get_aggregate_stats(config, n_rows=1000, include_batch_effects=False):
    """Get aggregation statistics including perturbation counts.

    Args:
        config: Configuration dictionary
        n_rows: Optional number of rows to sample for memory efficiency
        include_batch_effects: Whether to calculate batch effect metrics (slow)

    Returns:
        dict: Dictionary with statistics for each cell_class/channel_combo combination
    """
    root_fp = Path(config["all"]["root_fp"])
    aggregate_dir = root_fp / "aggregate"

    # Load the aggregate combo TSV file from config
    aggregate_combo_fp = Path(config["aggregate"]["aggregate_combo_fp"])
    aggregate_combos = pd.read_csv(aggregate_combo_fp, sep="\t")

    # Get unique cell_class and channel_combo combinations
    unique_combos = aggregate_combos[["cell_class", "channel_combo"]].drop_duplicates()

    # Get perturbation column name from config
    perturbation_col = config["aggregate"].get("perturbation_name_col", "gene_symbol_0")
    control_key = config["aggregate"].get("control_key", "nontargeting")

    all_results = {}

    for _, combo in unique_combos.iterrows():
        cell_class = combo["cell_class"]
        channel_combo = combo["channel_combo"]

        try:
            result = _get_single_aggregate_stats(
                config,
                cell_class,
                channel_combo,
                n_rows,
                root_fp,
                aggregate_dir,
                perturbation_col,
                control_key,
                include_batch_effects,
            )
            all_results[f"{cell_class}_{channel_combo}"] = result
        except Exception as e:
            print(f"Error processing {cell_class}/{channel_combo}: {str(e)}")
            all_results[f"{cell_class}_{channel_combo}"] = {"error": str(e)}

    return all_results


def _get_single_aggregate_stats(
    config,
    cell_class,
    channel_combo,
    n_rows,
    root_fp,
    aggregate_dir,
    perturbation_col,
    control_key,
    include_batch_effects,
):
    """Helper function to get stats for a single cell_class/channel_combo combination."""
    from lib.shared.file_utils import load_parquet_subset

    # Load the aggregated TSV file
    aggregated_path = (
        aggregate_dir
        / "tsvs"
        / f"CeCl-{cell_class}_ChCo-{channel_combo}__aggregated.tsv"
    )
    aggregated = pd.read_csv(aggregated_path, sep="\t")

    # Calculate distinct perturbation count and median perturbation cells
    distinct_perturbation_count = len(aggregated[perturbation_col].unique())
    median_perturbation_cells = int(aggregated["cell_count"].median())

    # Count total cells in the final aggregate TSV
    total_aggregated_cells = int(aggregated["cell_count"].sum())

    # Count features in the final aggregate TSV (columns starting with PC_)
    feature_cols = [col for col in aggregated.columns if col.startswith("PC_")]
    feature_count = len(feature_cols)

    # Count total cells from merge_data parquets (pre-filter)
    merge_data_paths = list(
        root_fp.glob(f"**/*_CeCl-{cell_class}_ChCo-{channel_combo}__merge_data.parquet")
    )

    total_pre_filtered_cells = 0
    for path in merge_data_paths:
        parquet_file = pq.ParquetFile(path)
        total_pre_filtered_cells += parquet_file.metadata.num_rows

    # Count filtered cells from parquet metadata (fast)
    filtered_dir = aggregate_dir / "parquets"
    filtered_paths = list(
        filtered_dir.glob(
            f"**/*_CeCl-{cell_class}_ChCo-{channel_combo}__filtered.parquet"
        )
    )

    total_filtered_cells = 0
    for path in filtered_paths:
        parquet_file = pq.ParquetFile(path)
        total_filtered_cells += parquet_file.metadata.num_rows

    result = {
        "cell_class": cell_class,
        "distinct_perturbation_count": distinct_perturbation_count,
        "median_perturbation_cells": median_perturbation_cells,
        "total_pre_filtered_cells": total_pre_filtered_cells,
        "total_filtered_cells": total_filtered_cells,
        "total_aggregated_cells": total_aggregated_cells,
        "feature_count": feature_count,
    }

    # Optional batch effect calculation (slow)
    if include_batch_effects:
        batch_stats = _calculate_batch_effects(
            config,
            cell_class,
            channel_combo,
            n_rows,
            aggregate_dir,
            perturbation_col,
            control_key,
        )
        result.update(batch_stats)

    return result


def _calculate_batch_effects(
    config,
    cell_class,
    channel_combo,
    n_rows,
    aggregate_dir,
    perturbation_col,
    control_key,
):
    """Calculate batch effect metrics (pre/post alignment)."""
    from lib.aggregate.cell_data_utils import DEFAULT_METADATA_COLS
    from lib.shared.file_utils import load_parquet_subset
    from sklearn.feature_selection import f_classif

    filtered_dir = aggregate_dir / "parquets"
    filtered_paths = list(
        filtered_dir.glob(
            f"**/*_CeCl-{cell_class}_ChCo-{channel_combo}__filtered.parquet"
        )
    )

    # Load filtered data
    if n_rows is None:
        filtered = ds.dataset(filtered_paths, format="parquet")
        filtered = filtered.to_table(use_threads=True, memory_pool=None).to_pandas()
    else:
        if len(filtered_paths) == 1:
            filtered = load_parquet_subset(filtered_paths[0], n_rows)
        else:
            filtered_dfs = [
                load_parquet_subset(path, n_rows) for path in filtered_paths
            ]
            filtered = pd.concat(filtered_dfs, ignore_index=True)

    # Pre-alignment batch effects
    filtered = filtered.dropna(axis=1)
    control_data = filtered[filtered[perturbation_col].str.startswith(control_key)]
    metadata_cols = DEFAULT_METADATA_COLS + ["class", "confidence"]
    control_data = control_data.copy()
    control_data["batch"] = (
        control_data["plate"].astype(str) + "_" + control_data["well"]
    )
    cols_to_drop = [c for c in metadata_cols if c in control_data.columns]
    control_data = control_data.drop(columns=cols_to_drop, errors="ignore")

    feature_data = control_data.drop(columns=["batch"])
    non_constant = feature_data.columns[feature_data.var() > 0]
    feature_data = feature_data[non_constant]

    X = feature_data.values
    y = control_data["batch"].values
    _, p_vals = f_classif(X, y)
    valid_p_vals = p_vals[~np.isnan(p_vals)]
    pre_align_p_median = np.median(valid_p_vals) if len(valid_p_vals) > 0 else np.nan

    # Post-alignment batch effects
    aligned_path = (
        aggregate_dir
        / "parquets"
        / f"CeCl-{cell_class}_ChCo-{channel_combo}__aligned.parquet"
    )

    if n_rows is None:
        aligned = pd.read_parquet(aligned_path)
    else:
        aligned = load_parquet_subset(aligned_path, n_rows)

    control_aligned = aligned[
        aligned[perturbation_col].str.startswith(control_key)
    ].copy()
    pc_cols = [col for col in control_aligned.columns if col.startswith("PC_")]
    control_aligned["batch"] = (
        control_aligned["plate"].astype(str) + "_" + control_aligned["well"]
    )
    control_aligned = control_aligned[pc_cols + ["batch"]]

    X = control_aligned.drop(columns=["batch"]).values
    y = control_aligned["batch"].values
    _, p_vals = f_classif(X, y)
    valid_p_vals = p_vals[~np.isnan(p_vals)]
    post_align_p_median = np.median(valid_p_vals) if len(valid_p_vals) > 0 else np.nan

    return {
        "pre_alignment_batch_p_median": pre_align_p_median,
        "post_alignment_batch_p_median": post_align_p_median,
    }


def get_cluster_stats(config):
    """Get clustering metrics including enrichment analysis results.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Contains detailed_results and summary DataFrames
    """
    root_fp = Path(config["all"]["root_fp"])
    cluster_dir = root_fp / "cluster"

    # Read the cluster_combos.tsv file from config
    cluster_combo_fp = Path(config["cluster"]["cluster_combo_fp"])
    cluster_combos = pd.read_csv(cluster_combo_fp, sep="\t")

    results = []

    for _, row in cluster_combos.iterrows():
        cell_class = row["cell_class"]
        channel_combo = row["channel_combo"]
        leiden_resolution = row["leiden_resolution"]

        cluster_specific_dir = (
            cluster_dir / channel_combo / cell_class / str(leiden_resolution)
        )

        # Path to metrics files
        combined_table_path = cluster_specific_dir / "CB-Real__combined_table.tsv"
        real_metrics_path = cluster_specific_dir / "CB-Real__global_metrics.json"
        shuffled_metrics_path = (
            cluster_specific_dir / "CB-Shuffled__global_metrics.json"
        )

        if not real_metrics_path.exists():
            continue

        # Count unique clusters
        unique_clusters = 0
        if combined_table_path.exists():
            try:
                combined_table = pd.read_csv(combined_table_path, sep="\t")
                unique_clusters = len(combined_table["cluster"].unique())
            except Exception:
                pass

        # Read real metrics
        try:
            with open(real_metrics_path, "r") as f:
                real_metrics = json.load(f)

            result = {
                "cell_class": cell_class,
                "channel_combo": channel_combo,
                "leiden_resolution": leiden_resolution,
                "unique_clusters": unique_clusters,
                # CORUM metrics
                "corum_enriched": real_metrics.get("CORUM", {}).get(
                    "num_enriched_clusters", 0
                ),
                "corum_proportion": real_metrics.get("CORUM", {}).get(
                    "proportion_enriched", 0
                ),
                # KEGG metrics
                "kegg_enriched": real_metrics.get("KEGG", {}).get(
                    "num_enriched_clusters", 0
                ),
                "kegg_proportion": real_metrics.get("KEGG", {}).get(
                    "proportion_enriched", 0
                ),
                # STRING metrics
                "string_precision": real_metrics.get("STRING", {}).get("precision", 0),
                "string_recall": real_metrics.get("STRING", {}).get("recall", 0),
                "string_f1": real_metrics.get("STRING", {}).get("f1_score", 0),
                "string_true_positives": real_metrics.get("STRING", {}).get(
                    "true_positives", 0
                ),
            }

            # Read shuffled metrics if available
            if shuffled_metrics_path.exists():
                with open(shuffled_metrics_path, "r") as f:
                    shuffled_metrics = json.load(f)
                result["shuffled_corum_proportion"] = shuffled_metrics.get(
                    "CORUM", {}
                ).get("proportion_enriched", 0)
                result["shuffled_kegg_proportion"] = shuffled_metrics.get(
                    "KEGG", {}
                ).get("proportion_enriched", 0)
                result["shuffled_string_f1"] = shuffled_metrics.get("STRING", {}).get(
                    "f1_score", 0
                )

            results.append(result)

        except Exception as e:
            print(
                f"Error reading metrics for {cell_class}/{channel_combo}/{leiden_resolution}: {e}"
            )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return {
            "detailed_results": results_df,
            "summary": pd.DataFrame(),
        }

    # Create summary grouped by cell_class
    summary = (
        results_df.groupby("cell_class")
        .agg(
            {
                "unique_clusters": "sum",
                "corum_enriched": "sum",
                "corum_proportion": "mean",
                "kegg_enriched": "sum",
                "kegg_proportion": "mean",
                "string_f1": "mean",
                "string_precision": "mean",
                "string_recall": "mean",
            }
        )
        .reset_index()
    )

    return {
        "detailed_results": results_df,
        "summary": summary,
    }


def get_all_stats(config, include_batch_effects=False):
    """Convenience function to get all pipeline statistics at once with formatted output.

    Args:
        config: Loaded configuration file
        include_batch_effects: Whether to calculate batch effect metrics (slow)

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
    print(f"   - ND2 input files: {stats['preprocess']['nd2_files']:,}")
    print(f"   - SBS tiles generated: {stats['preprocess']['sbs_tiles']:,}")
    print(f"   - Phenotype tiles generated: {stats['preprocess']['phenotype_tiles']:,}")

    # SBS Statistics
    print("\n[2/6] Gathering SBS statistics...")
    stats["sbs"] = get_sbs_stats(config)
    print("\n SBS STATISTICS:")
    print(f"   - Total cells segmented: {stats['sbs']['total_cells']:,}")
    print(
        f"   - Cells with barcode: {stats['sbs']['percent_cells_with_barcode']:.1f}% ({stats['sbs']['total_with_barcode']:,} cells)"
    )
    print(
        f"   - Cells with unique gene mapping: {stats['sbs']['percent_cells_with_unique_mapping']:.1f}% ({stats['sbs']['total_with_unique_gene']:,} cells)"
    )

    # Phenotype Statistics
    print("\n[3/6] Gathering phenotype statistics...")
    stats["phenotype"] = get_phenotype_stats(config)
    print("\n PHENOTYPE STATISTICS:")
    print(f"   - Total cells segmented: {stats['phenotype']['total_cells']:,}")
    print(
        f"   - Number of morphological features: {stats['phenotype']['feature_count']:,}"
    )

    # Merge Statistics
    print("\n[4/6] Gathering merge statistics...")
    stats["merge"] = get_merge_stats(config)
    print("\n MERGE STATISTICS:")
    if "error" not in stats["merge"]:
        merge = stats["merge"]
        print(f"   - Phenotype cells: {merge.get('phenotype_cells', 0):,}")
        print(f"   - SBS cells: {merge.get('sbs_cells', 0):,}")
        print(f"   - Matched cells: {merge.get('matched_cells', 0):,}")
        print(f"   - Phenotype match rate: {merge.get('phenotype_match_rate', 0):.1%}")
        print(f"   - SBS match rate: {merge.get('sbs_match_rate', 0):.1%}")
        if "cells_with_single_gene" in merge:
            print(
                f"   - Cells with single gene: {merge['cells_with_single_gene']:,} ({merge.get('single_gene_rate', 0):.1%})"
            )
    else:
        print(f"   [!] {stats['merge']['error']}")

    # Aggregate Statistics
    print("\n[5/6] Gathering aggregation statistics...")
    stats["aggregate"] = get_aggregate_stats(
        config, include_batch_effects=include_batch_effects
    )
    print("\n AGGREGATION STATISTICS:")

    for combo_key, combo_stats in stats["aggregate"].items():
        cell_class = combo_stats.get("cell_class", combo_key.split("_")[0])
        print(f"\n   {combo_key}:")

        if "error" not in combo_stats:
            print(
                f"      - Distinct perturbations: {combo_stats['distinct_perturbation_count']:,}"
            )
            print(
                f"      - Median cells per perturbation: {combo_stats['median_perturbation_cells']:,}"
            )
            print(
                f"      - Total cells (pre-filter): {combo_stats['total_pre_filtered_cells']:,}"
            )
            print(
                f"      - Total cells (post-filter): {combo_stats['total_filtered_cells']:,}"
            )
            print(
                f"      - Total cells (aggregated): {combo_stats['total_aggregated_cells']:,}"
            )
            print(f"      - PC features: {combo_stats['feature_count']}")
            if include_batch_effects and "pre_alignment_batch_p_median" in combo_stats:
                print(
                    f"      - Batch effect (pre/post alignment): {combo_stats['pre_alignment_batch_p_median']:.4f} -> {combo_stats['post_alignment_batch_p_median']:.4f}"
                )
        else:
            print(f"      [!] Error: {combo_stats['error']}")

    # Cluster Statistics
    print("\n[6/6] Gathering clustering statistics...")
    stats["cluster"] = get_cluster_stats(config)
    print("\n CLUSTERING STATISTICS:")

    if not stats["cluster"]["detailed_results"].empty:
        for _, row in stats["cluster"]["detailed_results"].iterrows():
            print(
                f"\n   {row['cell_class']} - {row['channel_combo']} (resolution={row['leiden_resolution']}):"
            )
            print(f"      - Clusters: {row['unique_clusters']}")
            print(
                f"      - CORUM enriched: {row['corum_enriched']} ({row['corum_proportion']:.1%})"
            )
            print(
                f"      - KEGG enriched: {row['kegg_enriched']} ({row['kegg_proportion']:.1%})"
            )
            print(
                f"      - STRING F1: {row['string_f1']:.3f} (P={row['string_precision']:.3f}, R={row['string_recall']:.3f})"
            )
    else:
        print("   [!] No clustering results found")

    print("\n" + "=" * 80)
    print("REPORT COMPLETE")
    print("=" * 80)

    return stats
