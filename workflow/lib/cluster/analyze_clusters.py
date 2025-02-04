from itertools import combinations

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, fisher_exact
from statsmodels.stats.multitest import multipletests


def create_cluster_gene_table(
    df, cluster_col="cluster", columns_to_combine=["gene_symbol_0"]
):
    """
    Creates a table with cluster number and combined gene information.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with cluster assignments and gene information
    cluster_col : str, default='cluster'
    columns_to_combine : list, default=['gene_symbol_0']
        Columns to combine for each cluster

    Returns:
    --------
    pandas.DataFrame
        DataFrame with cluster number, combined gene information, and gene count

    """
    # Combine gene information for each cluster
    cluster_summary = (
        df.groupby(cluster_col)
        .agg(
            {
                col: lambda x: ", ".join(
                    sorted([str(val) for val in set(x) if pd.notna(val)])
                )
                for col in columns_to_combine
            }
        )
        .reset_index()
    )

    # Count number of unique genes in each cluster
    cluster_summary["gene_number"] = df.groupby(cluster_col)[columns_to_combine[0]].agg(
        lambda x: len([val for val in set(x) if pd.notna(val)])
    )

    # Sort by cluster number
    cluster_summary = cluster_summary.rename(columns={cluster_col: "cluster_number"})
    cluster_summary = cluster_summary.sort_values("cluster_number").reset_index(
        drop=True
    )

    return cluster_summary


def analyze_differential_features(
    cluster_gene_table,
    feature_df,
    n_top=5,
    exclude_cols=["gene_symbol_0", "cell_number"],
):
    """
    Analyze differential features between clusters

    Parameters:
    -----------
    cluster_gene_table : pandas.DataFrame
        DataFrame with cluster assignments and gene information
    feature_df : pandas.DataFrame
        DataFrame with feature values for each gene
    n_top : int, default=5
        Number of top features to select
    exclude_cols : list, default=['gene_symbol_0', 'cell_number']
        Columns to exclude from feature analysis

    Returns:
    --------
    tuple
        (DataFrame with top features for each cluster, dictionary of feature analysis results)

    """
    # Get feature columns
    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
    results = {}

    # Copy the cluster gene table
    cluster_gene_table = cluster_gene_table.copy()
    cluster_gene_table[f"top_{n_top}_up"] = ""
    cluster_gene_table[f"top_{n_top}_down"] = ""

    # Analyze each cluster
    total_clusters = len(cluster_gene_table)
    print(f"Analyzing {total_clusters} clusters...")

    # Iterate over each cluster
    for idx, row in enumerate(cluster_gene_table.iterrows(), 1):
        cluster_num = row[1]["cluster_number"]
        cluster_genes = set(row[1]["gene_symbol_0"].split(", "))

        print(f"Processing cluster {idx}/{total_clusters} (#{cluster_num})", end="\r")

        # Split data into cluster and non-cluster
        cluster_data = feature_df[feature_df["gene_symbol_0"].isin(cluster_genes)][
            feature_cols
        ]
        non_cluster_data = feature_df[~feature_df["gene_symbol_0"].isin(cluster_genes)][
            feature_cols
        ]

        # Perform t-test for each feature
        t_stats, p_values, effect_sizes = [], [], []

        # Calculate t-statistic, p-value, and effect size for each feature
        for feature in feature_cols:
            t_stat, p_val = ttest_ind(cluster_data[feature], non_cluster_data[feature])
            cohens_d = (
                cluster_data[feature].mean() - non_cluster_data[feature].mean()
            ) / np.sqrt(
                (
                    (
                        cluster_data[feature].std() ** 2
                        + non_cluster_data[feature].std() ** 2
                    )
                    / 2
                )
            )

            # Store results
            t_stats.append(abs(t_stat))
            p_values.append(p_val)
            effect_sizes.append(cohens_d)

        # Store results in DataFrame
        feature_results = pd.DataFrame(
            {
                "feature": feature_cols,
                "t_statistic": t_stats,
                "p_value": p_values,
                "effect_size": effect_sizes,
            }
        )

        # Adjust p-values using Benjamini-Hochberg method
        feature_results["p_value_adj"] = multipletests(
            feature_results["p_value"], method="fdr_bh"
        )[1]
        feature_results["abs_effect_size"] = feature_results["effect_size"].abs()

        # Select top features based on effect size
        top_up = feature_results.nlargest(n_top, "effect_size")
        top_down = feature_results.nsmallest(n_top, "effect_size")

        # Update cluster gene table with top features
        cluster_idx = cluster_gene_table.index[
            cluster_gene_table["cluster_number"] == cluster_num
        ][0]
        cluster_gene_table.at[cluster_idx, f"top_{n_top}_up"] = ", ".join(
            top_up["feature"]
        )
        cluster_gene_table.at[cluster_idx, f"top_{n_top}_down"] = ", ".join(
            top_down["feature"]
        )

        # Store feature analysis results
        results[cluster_num] = feature_results

    return cluster_gene_table, results


def process_interactions(df_clusters, string_data_fp, corum_data_fp):
    """
    Process cluster data against STRING and CORUM databases

    Parameters:
    -----------
    df_clusters : pandas.DataFrame
        DataFrame with cluster information
    string_data_fp : str
        File path to STRING data
    corum_data_fp : str
        File path to CORUM data

    Returns:
    --------
    tuple:
        - DataFrame with cluster information and validation results
        - Dictionary with global metrics for both STRING and CORUM
    """
    # load string and corum data
    string_df = pd.read_csv(string_data_fp, sep="\t")
    corum_df = pd.read_csv(corum_data_fp, sep="\t")

    # Process STRING data
    string_pairs = set(map(tuple, string_df[["protein1", "protein2"]].values))

    # Process CORUM data - keep both pair and complex information
    corum_complexes = []
    corum_pairs = set()
    for _, complex_row in corum_df.iterrows():
        if pd.isna(complex_row["subunits_gene_name"]):
            continue
        # Get all genes in complex
        genes = [gene.strip() for gene in complex_row["subunits_gene_name"].split(";")]
        if len(genes) >= 2:
            # Store complete complex
            corum_complexes.append(
                {
                    "name": complex_row["complex_name"],
                    "genes": set(genes),
                    "size": len(genes),
                }
            )
            # Store pairs for pair-based analysis
            pairs = set(combinations(sorted(genes), 2))
            corum_pairs.update(pairs)

    # Get all screened genes (from both STRING and CORUM columns)
    screened_genes = set()
    for _, row in df_clusters.iterrows():
        screened_genes.update(
            gene.strip() for gene in row["gene_symbol_0"].replace(", ", ",").split(",")
        )
        screened_genes.update(
            gene.strip() for gene in row["STRING"].replace(", ", ",").split(",")
        )

    # Process cluster data
    all_string_predicted_pairs = set()
    all_cluster_pairs = set()
    all_corum_cluster_pairs = set()
    results = []

    for _, row in df_clusters.iterrows():
        cluster_num = row["cluster_number"]
        cluster_genes = set(
            gene.strip() for gene in row["gene_symbol_0"].replace(", ", ",").split(",")
        )
        genes_string = set(
            gene.strip() for gene in row["STRING"].replace(", ", ",").split(",")
        )

        # STRING analysis
        string_cluster_pairs = set()
        if len(genes_string) >= 2:
            string_cluster_pairs = set(combinations(sorted(genes_string), 2))
            all_string_predicted_pairs.update(string_cluster_pairs)
        matching_string_pairs = (
            string_cluster_pairs & string_pairs if string_cluster_pairs else set()
        )

        # CORUM complex-level analysis
        enriched_complexes = []
        for complex_info in corum_complexes:
            complex_genes = complex_info["genes"]
            screened_complex_genes = complex_genes & screened_genes

            # Apply complex filtering criteria
            if len(screened_complex_genes) >= 3 and len(screened_complex_genes) >= (
                2 / 3 * len(complex_genes)
            ):
                # Calculate overlap with cluster
                overlap_genes = cluster_genes & screened_complex_genes

                # Fisher's exact test
                table = [
                    [len(overlap_genes), len(screened_complex_genes - cluster_genes)],
                    [
                        len(cluster_genes - screened_complex_genes),
                        len(screened_genes - cluster_genes - screened_complex_genes),
                    ],
                ]
                odds_ratio, pvalue = fisher_exact(table)

                if pvalue < 0.05:  # Store for FDR correction
                    enriched_complexes.append(
                        {
                            "complex_name": complex_info["name"],
                            "pvalue": pvalue,
                            "overlap_size": len(overlap_genes),
                            "complex_size": len(screened_complex_genes),
                        }
                    )

        # Apply FDR correction to enriched complexes
        if enriched_complexes:
            pvals = [x["pvalue"] for x in enriched_complexes]
            _, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
            significant_complexes = [
                enr
                for enr, p_adj in zip(enriched_complexes, pvals_corrected)
                if p_adj < 0.05
            ]
        else:
            significant_complexes = []

        # CORUM pair-based analysis for this cluster
        if len(cluster_genes) >= 2:
            cluster_pairs = set(combinations(sorted(cluster_genes), 2))
            all_cluster_pairs.update(cluster_pairs)
            # Find which pairs are also in CORUM
            matching_corum = cluster_pairs & corum_pairs
            all_corum_cluster_pairs.update(matching_corum)

        # Store results
        results.append(
            {
                "cluster_number": cluster_num,
                "total_string_pairs": len(string_cluster_pairs),
                "string_validated_pairs": len(matching_string_pairs),
                "string_validation_ratio": len(matching_string_pairs)
                / len(string_cluster_pairs)
                if string_cluster_pairs
                else 0,
                "enriched_corum_complexes": ", ".join(
                    x["complex_name"] for x in significant_complexes
                ),
                "num_enriched_complexes": len(significant_complexes),
            }
        )

    # Calculate global STRING metrics
    string_true_positives = all_string_predicted_pairs & string_pairs
    string_precision = (
        len(string_true_positives) / len(all_string_predicted_pairs)
        if all_string_predicted_pairs
        else 0
    )
    string_recall = (
        len(string_true_positives) / len(string_pairs) if string_pairs else 0
    )
    string_f1 = (
        2 * (string_precision * string_recall) / (string_precision + string_recall)
        if (string_precision + string_recall)
        else 0
    )

    # Calculate CORUM pair-based metrics (as shown in the plot)
    corum_precision = (
        len(all_corum_cluster_pairs) / len(all_cluster_pairs)
        if all_cluster_pairs
        else 0
    )
    corum_recall = len(all_corum_cluster_pairs) / len(corum_pairs) if corum_pairs else 0
    corum_f1 = (
        2 * (corum_precision * corum_recall) / (corum_precision + corum_recall)
        if (corum_precision + corum_recall)
        else 0
    )

    results_df = pd.DataFrame(results)
    cluster_results = df_clusters.merge(results_df, on="cluster_number")

    # Global metrics including both STRING and CORUM results
    global_metrics = {
        "num_clusters": df_clusters["cluster_number"].nunique(),
        "string_global_precision": string_precision,
        "string_global_recall": string_recall,
        "string_global_f1": string_f1,
        "string_total_predicted_pairs": len(all_string_predicted_pairs),
        "string_total_reference_pairs": len(string_pairs),
        "string_total_correct_pairs": len(string_true_positives),
        "corum_precision": corum_precision,  # Fraction of cluster pairs in CORUM
        "corum_recall": corum_recall,  # Fraction of CORUM pairs in clusters
        "corum_f1": corum_f1,
        "corum_total_cluster_pairs": len(all_cluster_pairs),
        "corum_total_complex_pairs": len(corum_pairs),
        "corum_matching_pairs": len(all_corum_cluster_pairs),
        "num_enriched_complexes": sum(
            len(row["enriched_corum_complexes"]) for row in results
        ),
    }
    # Convert global metrics to a DataFrame with 'metric' and 'value' columns
    global_metrics_df = pd.DataFrame(
        list(global_metrics.items()), columns=["metric", "value"]
    )

    return cluster_results, global_metrics_df
