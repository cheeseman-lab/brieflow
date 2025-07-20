"""Standardize barcode design tables for SBS analysis.

This module provides functions to standardize barcode design tables from various sources
into the format expected by the SBS pipeline. Includes gene validation and annotation
functions for quality control.
"""

import pandas as pd
import warnings
from typing import Optional, List, Callable


def standardize_barcode_design(
    df_design: pd.DataFrame,
    prefix_map: str = "iBAR2",
    prefix_recomb: Optional[str] = None,
    gene_symbol_col: Optional[str] = "gene_symbol",
    gene_id_col: Optional[str] = None,
    prefix_map_func: Optional[Callable] = None,
    prefix_recomb_func: Optional[Callable] = None,
    map_prefix_length: Optional[int] = None,
    recomb_prefix_length: Optional[int] = None,
    skip_cycles_map: Optional[List[int]] = None,
    skip_cycles_recomb: Optional[List[int]] = None,
    filter_func: Optional[Callable] = None,
    drop_duplicates: bool = True,
    keep_extra_cols: bool = False,
    handle_ampersand_genes: bool = True,
    standardize_nontargeting: bool = True,
    nontargeting_patterns: List[str] = ["nontargeting", "sg_nt", "non-targeting"],
    nontargeting_format: str = "nontargeting_{prefix}",
    nontargeting_pattern_map: Optional[dict] = None,
    uniprot_data_path: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Standardize a barcode design table for use with SBS analysis.

    This function handles the most common manipulations needed to prepare barcode
    design tables for the call_cells function. UniProt validation is required to
    ensure gene symbols are valid.

    Args:
        df_design (pd.DataFrame): Input barcode design table
        prefix_map (str): Name of column containing barcode sequences for mapping (default: "iBAR2")
        prefix_recomb (str, optional): Name of column containing barcodes used to calculate recombination (optional)
        gene_symbol_col (str, optional): Name of column containing gene symbols.
            If None, will create empty gene_symbol column
        gene_id_col (str, optional): Name of column containing gene IDs.
            Optional - only included if user wants additional gene identifiers
        prefix_map_func (callable, optional): Custom function to generate prefixes for mapping that match
            experimental read structure. Function should take a row and return the prefix string.
        prefix_recomb_func (callable, optional): Custom function to generate prefixes for recombination that match
            experimental read structure. Function should take a row and return the prefix string.
        map_prefix_length (int, optional): Length of barcode prefix for prefix_map (mapping).
            Should match the length of experimental read barcodes for mapping. Ignored if prefix_map_func is provided.
        recomb_prefix_length (int, optional): Length of barcode prefix for prefix_recomb (recombination).
            Should match the length of experimental read barcodes for recombination. Ignored if prefix_recomb_func is provided.
        skip_cycles_map (List[int], optional): List of 1-based cycle numbers to skip in prefix_map when generating prefix.
        skip_cycles_recomb (List[int], optional): List of 1-based cycle numbers to skip in recomb_prefix when generating prefix_recomb.
        filter_func (callable, optional): Custom function to filter rows.
            Function should take dataframe and return filtered dataframe.
            If None, no filtering is applied.
        drop_duplicates (bool): Whether to drop duplicate barcodes (default: True)
        keep_extra_cols (bool): Whether to keep additional columns (default: True)
        handle_ampersand_genes (bool): Whether to split ampersand-separated genes (default: True)
        standardize_nontargeting (bool): Whether to standardize non-targeting controls (default: True)
        nontargeting_patterns (List[str]): Patterns to identify non-targeting controls
            Case-insensitive matching against gene symbols (default: ["nontargeting", "sg_nt", "non-targeting"])
        nontargeting_format (str): Format string for standardized non-targeting names
            Use {prefix} placeholder for barcode prefix, {original} for original name
        nontargeting_pattern_map (dict, optional): Mapping of specific patterns to custom formats
            Example: {
                "intergenic": "nontargeting_intergenic_{prefix}",
                "nontargeting": "nontargeting_noncutting_{prefix}"
            }
            If None, uses default nontargeting_format for all patterns
        uniprot_data_path (str): Path to UniProt annotation file (REQUIRED for gene validation)
        verbose (bool): Whether to print processing information (default: True)

    Returns:
        pd.DataFrame: Standardized barcode design table with columns:
            - prefix_map: barcode sequences
            - gene_symbol: gene symbols
            - gene_id: gene identifiers (optional, only if gene_id_col provided)
            - prefix: barcode prefixes for matching
            - uniprot_entry: UniProt entry ID (REQUIRED)
            - prefix_recomb: recombined barcode sequences (optional, only if prefix_recomb provided)
            - (additional columns if keep_extra_cols=True)

    Raises:
        ValueError: If required columns are missing, UniProt data not provided, or validation fails
    """
    if verbose:
        print(f"Standardizing barcode design table with {len(df_design)} entries...")

    # UniProt validation is required
    if uniprot_data_path is None:
        raise ValueError(
            "uniprot_data_path is required for gene validation. "
            "Please provide path to UniProt annotation file."
        )

    # Create a copy to avoid modifying original
    df = df_design.copy()

    # Validate required barcode column exists
    if prefix_map not in df.columns:
        raise ValueError(
            f"Barcode column '{prefix_map}' not found in design table. "
            f"Available columns: {list(df.columns)}"
        )

    # Apply custom filtering if provided
    if filter_func is not None:
        initial_count = len(df)
        df = filter_func(df)
        if verbose:
            print(f"Applied custom filter: {initial_count} → {len(df)} entries")

    # Rename barcode column to standard name "prefix_map"
    if prefix_map != "prefix_map":
        df = df.rename(columns={prefix_map: "prefix_map"})
        if verbose:
            print(f"Renamed '{prefix_map}' column to 'prefix_map'")

    # Handle prefix_recomb column if provided
    if prefix_recomb is not None:
        if prefix_recomb not in df.columns:
            raise ValueError(
                f"Recombined barcode column '{prefix_recomb}' not found in design table. "
                f"Available columns: {list(df.columns)}"
            )
        if prefix_recomb != "prefix_recomb":
            df = df.rename(columns={prefix_recomb: "prefix_recomb"})
            if verbose:
                print(f"Renamed '{prefix_recomb}' column to 'prefix_recomb'")

    # Handle gene symbol column
    if gene_symbol_col is not None and gene_symbol_col in df.columns:
        if gene_symbol_col != "gene_symbol":
            df = df.rename(columns={gene_symbol_col: "gene_symbol"})
            if verbose:
                print(f"Renamed '{gene_symbol_col}' column to 'gene_symbol'")
    else:
        df["gene_symbol"] = gene_symbol_col if gene_symbol_col is not None else ""
        if verbose:
            print(
                f"Created gene_symbol column with value: '{df['gene_symbol'].iloc[0] if len(df) > 0 else ''}'"
            )

    # Handle gene ID column (optional)
    if gene_id_col is not None and gene_id_col in df.columns:
        if gene_id_col != "gene_id":
            df = df.rename(columns={gene_id_col: "gene_id"})
            if verbose:
                print(f"Renamed '{gene_id_col}' column to 'gene_id'")
    elif gene_id_col is not None:
        # User specified a column that doesn't exist
        warnings.warn(
            f"Gene ID column '{gene_id_col}' not found. Skipping gene_id column."
        )

    # Remove any rows with missing barcodes
    initial_count = len(df)
    df = df.dropna(subset=["prefix_map"])
    if len(df) < initial_count and verbose:
        print(f"Removed {initial_count - len(df)} entries with missing barcodes")

    # Drop duplicates if requested (before prefix generation to avoid issues)
    if drop_duplicates:
        initial_count = len(df)
        df = df.drop_duplicates(subset=["prefix_map"])
        if len(df) < initial_count and verbose:
            print(f"Removed {initial_count - len(df)} duplicate barcodes")

    # [PREFIX GENERATION CODE REMAINS THE SAME...]
    # Generate prefix_map column for mapping (truncate or modify in place)
    if prefix_map_func is not None:
        # Use custom prefix function
        df["prefix_map"] = df.apply(prefix_map_func, axis=1)
        if verbose:
            print("Modified 'prefix_map' using custom function")
    elif skip_cycles_map is not None:
        # Use skip_cycles_prefix_function for prefix_map
        if map_prefix_length is not None and skip_cycles_map is not None:
            prefix_length = map_prefix_length - len(skip_cycles_map)
        else:
            prefix_length = map_prefix_length
        prefix_func_map = create_skip_cycles_prefix_function(
            skip_cycles=skip_cycles_map,
            prefix_length=prefix_length,
            column_name="prefix_map",
        )
        df["prefix_map"] = df.apply(prefix_func_map, axis=1)
        if verbose:
            print(
                f"Modified 'prefix_map' using skip_cycles_prefix_function (skip_cycles={skip_cycles_map}, length={map_prefix_length})"
            )
    else:
        # Use map_prefix_length for prefix_map
        if map_prefix_length is None:
            # Use full barcode length as default
            barcode_lengths = df["prefix_map"].astype(str).str.len()
            map_prefix_length = int(barcode_lengths.mode()[0])  # Most common length
            if verbose:
                print(
                    f"Using full barcode length ({map_prefix_length}) for 'prefix_map'"
                )
        df["prefix_map"] = df["prefix_map"].astype(str).str[:map_prefix_length]
        if verbose:
            print(
                f"Modified 'prefix_map' using truncation (length={map_prefix_length})"
            )

    # Generate prefix_recomb column if present
    if "prefix_recomb" in df.columns:
        if prefix_recomb_func is not None:
            # Use custom prefix function for recomb
            df["prefix_recomb"] = df.apply(prefix_recomb_func, axis=1)
            if verbose:
                print("Modified 'prefix_recomb' using custom function")
        elif skip_cycles_recomb is not None:
            # Use skip_cycles_prefix_function for prefix_recomb
            if recomb_prefix_length is not None and skip_cycles_recomb is not None:
                prefix_length = recomb_prefix_length - len(skip_cycles_recomb)
            else:
                prefix_length = recomb_prefix_length
            prefix_func_recomb = create_skip_cycles_prefix_function(
                skip_cycles=skip_cycles_recomb,
                prefix_length=prefix_length,
                column_name="prefix_recomb",
            )
            df["prefix_recomb"] = df.apply(prefix_func_recomb, axis=1)
            if verbose:
                print(
                    f"Modified 'prefix_recomb' using skip_cycles_prefix_function (skip_cycles={skip_cycles_recomb}, length={recomb_prefix_length})"
                )
        elif recomb_prefix_length is not None:
            df["prefix_recomb"] = (
                df["prefix_recomb"].astype(str).str[:recomb_prefix_length]
            )
            if verbose:
                print(
                    f"Modified 'prefix_recomb' using truncation (length={recomb_prefix_length})"
                )

    # Validate prefix generation
    if df["prefix_map"].isna().any():
        warnings.warn(
            "Some prefixes could not be generated. Check your prefix function or barcode data."
        )

    # Handle ampersand genes if requested
    if handle_ampersand_genes and "gene_symbol" in df.columns:
        df = simplify_ampersand_genes(
            df, gene_symbol_col="gene_symbol", verbose=verbose
        )

    # Standardize non-targeting controls with pattern mapping
    if standardize_nontargeting and "gene_symbol" in df.columns:
        df = standardize_nontargeting_controls(
            df,
            nontargeting_patterns=nontargeting_patterns,
            nontargeting_format=nontargeting_format,
            nontargeting_pattern_map=nontargeting_pattern_map,
            verbose=verbose,
        )

    # Add UniProt annotation
    df = add_uniprot_annotation(df, uniprot_data_path, verbose=verbose)

    # Check that genes were successfully annotated (excluding non-targeting controls)
    regular_genes = df[~df["gene_symbol"].str.startswith("nontargeting_", na=False)]
    if len(regular_genes) > 0:
        unannotated_count = regular_genes["uniprot_entry"].isna().sum()
        if unannotated_count > 0:
            unannotated_genes = regular_genes[regular_genes["uniprot_entry"].isna()][
                "gene_symbol"
            ].unique()
            warnings.warn(
                f"{unannotated_count} genes could not be validated against UniProt: "
                f"{list(unannotated_genes)[:5]}{'...' if len(unannotated_genes) > 5 else ''}. "
                f"Consider reviewing gene symbols or removing these entries."
            )

    # Organize columns
    required_cols = ["prefix_map", "gene_symbol", "uniprot_entry"]
    if "gene_id" in df.columns:
        required_cols.insert(
            2, "gene_id"
        )  # Insert gene_id after gene_symbol if present
    if "prefix_recomb" in df.columns:
        required_cols.append("prefix_recomb")

    if keep_extra_cols:
        # Keep additional columns that might be useful
        extra_cols = [col for col in df.columns if col not in required_cols]
        final_cols = required_cols + extra_cols
    else:
        final_cols = required_cols

    df = df[final_cols]

    if verbose:
        print(f"Standardized barcode design table:")
        print(f"  - {len(df)} unique barcodes")
        print(f"  - Columns: {list(df.columns)}")
        if "gene_symbol" in df.columns:
            non_empty_genes = (df["gene_symbol"] != "").sum()
            print(f"  - Gene symbols: {non_empty_genes} non-empty")
            unique_genes = df["gene_symbol"].nunique()
            print(f"  - Unique gene symbols: {unique_genes}")
        annotated_count = df["uniprot_entry"].notna().sum()
        nontargeting_count = (
            df["gene_symbol"].str.startswith("nontargeting_", na=False).sum()
        )
        print(f"  - UniProt validated: {annotated_count}/{len(df)} entries")
        if nontargeting_count > 0:
            print(f"  - Non-targeting controls: {nontargeting_count} standardized")

    return df


def simplify_ampersand_genes(
    df: pd.DataFrame, gene_symbol_col: str = "gene_symbol", verbose: bool = True
) -> pd.DataFrame:
    """Simplify gene names by replacing ampersand-containing gene names with just the first gene in the list.

    This is commonly needed when dealing with dual-targeting guides or multi-gene perturbations
    where the gene name is stored as "GENE1&GENE2" but you want to analyze based on the primary target.

    Args:
        df (pd.DataFrame): The DataFrame containing gene names
        gene_symbol_col (str): The column name containing gene names (default: "gene_symbol")
        verbose (bool): Whether to print information about the splitting

    Returns:
        pd.DataFrame: DataFrame with simplified gene names
    """
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()

    if gene_symbol_col not in result_df.columns:
        warnings.warn(
            f"Column '{gene_symbol_col}' not found. No ampersand processing performed."
        )
        return result_df

    # Find rows with ampersands in the gene symbol column
    mask = result_df[gene_symbol_col].astype(str).str.contains("&", na=False)

    if mask.any():
        n_ampersand = mask.sum()

        # For those rows, replace with just the first gene name before the ampersand
        result_df.loc[mask, gene_symbol_col] = result_df.loc[
            mask, gene_symbol_col
        ].apply(lambda x: x.split("&")[0].strip() if isinstance(x, str) else x)

        if verbose:
            print(f"Simplified {n_ampersand} ampersand-containing gene names:")
            print(f"  - Split 'GENE1&GENE2' → 'GENE1' format")
            print(
                f"  - This represents dual-targeting guides analyzed by primary target"
            )

    return result_df


def load_and_process_uniprot_data(
    uniprot_data_path: str, verbose: bool = True
) -> pd.DataFrame:
    """Load and process UniProt annotation data for gene validation.

    Expands the UniProt data to handle gene synonyms and homologs by creating
    separate rows for each gene name listed in the gene_names column.

    Args:
        uniprot_data_path (str): Path to UniProt TSV file
        verbose (bool): Whether to print processing information

    Returns:
        pd.DataFrame: Processed UniProt data with columns:
            - gene_name: Individual gene names
            - uniprot_entry: UniProt entry ID

    Expected UniProt file format:
        Columns should include: entry, gene_names
        gene_names should contain space-separated gene names/synonyms
    """
    try:
        uniprot_data = pd.read_csv(uniprot_data_path, sep="\t")

        if verbose:
            print(f"Loaded UniProt data: {len(uniprot_data)} entries")

        # Check for required columns
        required_cols = ["entry", "gene_names"]
        missing_cols = [col for col in required_cols if col not in uniprot_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required UniProt columns: {missing_cols}")

        # Function and link columns are optional for basic validation
        if "function" not in uniprot_data.columns:
            uniprot_data["function"] = ""
        if "link" not in uniprot_data.columns:
            uniprot_data["link"] = ""

        # Remove entries without gene names
        initial_count = len(uniprot_data)
        uniprot_data = uniprot_data.dropna(subset=["gene_names"])
        if verbose and len(uniprot_data) < initial_count:
            print(
                f"Removed {initial_count - len(uniprot_data)} entries without gene names"
            )

        # Expand rows to deal with synonyms and homologs
        expanded_rows = []
        for _, row in uniprot_data.iterrows():
            gene_names = str(row["gene_names"]).split()
            for position, gene in enumerate(gene_names):
                expanded_rows.append(
                    {
                        "gene_name": gene,
                        "position": position,  # Position 0 = primary name, 1+ = synonyms
                        "uniprot_entry": row["entry"],
                    }
                )

        expanded_df = pd.DataFrame(expanded_rows)

        # Sort by gene name and position, keep only the first match (primary name preferred)
        uniprot_processed = expanded_df.sort_values(
            ["gene_name", "position"]
        ).drop_duplicates("gene_name", keep="first")

        # Keep only essential columns
        uniprot_processed = uniprot_processed[["gene_name", "uniprot_entry"]]

        if verbose:
            print(f"Processed UniProt data: {len(uniprot_processed)} unique gene names")

        return uniprot_processed

    except Exception as e:
        raise ValueError(
            f"Error processing UniProt data from {uniprot_data_path}: {str(e)}"
        )


def standardize_nontargeting_controls(
    df: pd.DataFrame,
    nontargeting_patterns: List[str] = [
        "nontargeting",
        "sg_nt",
        "non-targeting",
    ],
    nontargeting_format: str = "nontargeting_{prefix}",
    nontargeting_pattern_map: Optional[dict] = None,
    gene_symbol_col: str = "gene_symbol",
    prefix_col: str = "prefix_map",
    verbose: bool = True,
) -> pd.DataFrame:
    """Standardize non-targeting control naming and annotation with support for different control types.

    Identifies non-targeting controls based on patterns in gene symbols and standardizes
    their naming to a consistent format. Supports mapping different patterns to different
    standardized formats (e.g., intergenic_controls -> nontargeting_intergenic).

    Args:
        df (pd.DataFrame): DataFrame with gene symbols and prefixes
        nontargeting_patterns (List[str]): Default patterns to identify non-targeting controls
            Case-insensitive matching against gene symbols
        nontargeting_format (str): Default format string for standardized names
            Use {prefix} placeholder for barcode prefix, {original} for original name
        nontargeting_pattern_map (dict, optional): Mapping of specific patterns to custom formats
            Example: {
                "intergenic": "nontargeting_intergenic_{prefix}",
                "nontargeting": "nontargeting_noncutting_{prefix}"
            }
            If None, uses default nontargeting_format for all patterns
        gene_symbol_col (str): Column containing gene symbols (default: "gene_symbol")
        prefix_col (str): Column containing barcode prefixes (default: "prefix_map")
        verbose (bool): Whether to print processing information

    Returns:
        pd.DataFrame: DataFrame with standardized non-targeting control names

    Examples:
        # Basic usage (backward compatible)
        df = standardize_nontargeting_controls(df)

        # Enhanced usage with pattern mapping
        df = standardize_nontargeting_controls(
            df,
            nontargeting_patterns=["nontargeting", "intergenic_controls"],
            nontargeting_pattern_map={
                "intergenic": "nontargeting_intergenic_{prefix}",
                "nontargeting": "nontargeting_noncutting_{prefix}"
            }
        )
    """
    result_df = df.copy()

    if gene_symbol_col not in result_df.columns:
        warnings.warn(
            f"Column '{gene_symbol_col}' not found. No non-targeting standardization performed."
        )
        return result_df

    if prefix_col not in result_df.columns:
        warnings.warn(
            f"Column '{prefix_col}' not found. Cannot generate standardized non-targeting names."
        )
        return result_df

    # Store original names for reference
    if "original_gene_symbol" not in result_df.columns:
        result_df["original_gene_symbol"] = result_df[gene_symbol_col].copy()

    total_processed = 0
    pattern_counts = {}

    # If pattern_map is provided, process each pattern individually
    if nontargeting_pattern_map is not None:
        for pattern, custom_format in nontargeting_pattern_map.items():
            # Find matches for this specific pattern (case-insensitive)
            pattern_mask = (
                result_df[gene_symbol_col]
                .astype(str)
                .str.contains(pattern, na=False, regex=True, case=False)
            )

            if pattern_mask.any():
                n_matches = pattern_mask.sum()
                pattern_counts[pattern] = n_matches
                total_processed += n_matches

                # Generate standardized names for this pattern
                standardized_names = []
                for _, row in result_df[pattern_mask].iterrows():
                    prefix = str(row[prefix_col])
                    original = str(row[gene_symbol_col])
                    # Format the new name using custom format for this pattern
                    new_name = custom_format.format(prefix=prefix, original=original)
                    standardized_names.append(new_name)

                # Apply standardized names
                result_df.loc[pattern_mask, gene_symbol_col] = standardized_names

                if verbose:
                    print(
                        f"Applied pattern '{pattern}' → '{custom_format}': {n_matches} controls"
                    )

    # Process any remaining patterns not covered by pattern_map using default format
    if nontargeting_patterns:
        # Create a combined pattern for remaining matches, excluding already processed patterns
        remaining_patterns = nontargeting_patterns.copy()

        # Remove patterns already processed via pattern_map
        if nontargeting_pattern_map is not None:
            for processed_pattern in nontargeting_pattern_map.keys():
                # Remove exact matches and partial matches
                remaining_patterns = [
                    p for p in remaining_patterns if processed_pattern not in p.lower()
                ]

        if remaining_patterns:
            # Create regex for remaining patterns
            pattern_regex = "|".join(remaining_patterns)

            # Find matches that haven't been processed yet
            remaining_mask = (
                result_df[gene_symbol_col]
                .astype(str)
                .str.contains(pattern_regex, na=False, regex=True, case=False)
            )

            # Exclude already processed entries (those that have been standardized)
            already_standardized = result_df[gene_symbol_col].str.startswith(
                "nontargeting_", na=False
            )
            remaining_mask = remaining_mask & ~already_standardized

            if remaining_mask.any():
                n_remaining = remaining_mask.sum()
                total_processed += n_remaining

                # Generate standardized names using default format
                standardized_names = []
                for _, row in result_df[remaining_mask].iterrows():
                    prefix = str(row[prefix_col])
                    original = str(row[gene_symbol_col])
                    # Format the new name using default format
                    new_name = nontargeting_format.format(
                        prefix=prefix, original=original
                    )
                    standardized_names.append(new_name)

                # Apply standardized names
                result_df.loc[remaining_mask, gene_symbol_col] = standardized_names

                if verbose:
                    print(
                        f"Applied default format to remaining patterns: {n_remaining} controls"
                    )
                    print(f"  - Patterns: {remaining_patterns}")
                    print(f"  - Format: {nontargeting_format}")

    if verbose and total_processed > 0:
        print(f"\nTotal standardized non-targeting controls: {total_processed}")
        if pattern_counts:
            for pattern, count in pattern_counts.items():
                print(f"  - {pattern}: {count} controls")

        # Show examples
        nontargeting_examples = result_df[
            result_df[gene_symbol_col].str.startswith("nontargeting_", na=False)
        ]
        if len(nontargeting_examples) > 0:
            print(f"\nExample transformations:")
            for i, (_, row) in enumerate(nontargeting_examples.head(3).iterrows()):
                old = row.get("original_gene_symbol", "N/A")
                new = row[gene_symbol_col]
                print(f"    '{old}' → '{new}'")

    return result_df


def add_uniprot_annotation(
    df: pd.DataFrame, uniprot_data_path: str, verbose: bool = True
) -> pd.DataFrame:
    """Add UniProt annotation to a dataframe containing gene symbols.

    Non-targeting controls (identified by gene symbols starting with "nontargeting_")
    are given special annotation instead of being validated against UniProt.

    Args:
        df (pd.DataFrame): DataFrame with gene_symbol column
        uniprot_data_path (str): Path to UniProt TSV file
        verbose (bool): Whether to print processing information

    Returns:
        pd.DataFrame: DataFrame with added uniprot_entry column
    """
    if "gene_symbol" not in df.columns:
        warnings.warn("No gene_symbol column found. Skipping UniProt annotation.")
        return df

    # Load and process UniProt data
    uniprot_data = load_and_process_uniprot_data(uniprot_data_path, verbose=verbose)

    # Store original index to maintain row correspondence
    df_with_idx = df.reset_index(drop=False)
    original_index_name = (
        df_with_idx.columns[0] if "index" in df_with_idx.columns else None
    )

    # Identify non-targeting controls BEFORE merge
    nontargeting_mask = (
        df_with_idx["gene_symbol"].astype(str).str.startswith("nontargeting_", na=False)
    )
    # nontargeting_mask is not used further, but kept for clarity
    # Process all rows together, but handle annotation differently
    df_annotated = df_with_idx.merge(
        uniprot_data[["gene_name", "uniprot_entry"]],
        how="left",
        left_on="gene_symbol",
        right_on="gene_name",
    ).drop(columns="gene_name")

    # Re-identify non-targeting controls in the merged DataFrame
    # (should be the same, but safer to recalculate)
    nontargeting_mask_merged = (
        df_annotated["gene_symbol"]
        .astype(str)
        .str.startswith("nontargeting_", na=False)
    )

    # Set non-targeting controls to special identifier
    df_annotated.loc[nontargeting_mask_merged, "uniprot_entry"] = (
        "NON_TARGETING_CONTROL"
    )

    # Restore original index if it existed
    if original_index_name and original_index_name != "index":
        df_annotated = df_annotated.set_index(original_index_name)
    elif "index" in df_annotated.columns:
        df_annotated = df_annotated.drop(columns="index")

    # Count results for reporting
    n_controls = nontargeting_mask_merged.sum()
    regular_genes = df_annotated[~nontargeting_mask_merged]
    annotated_count = regular_genes["uniprot_entry"].notna().sum()
    unannotated_genes = regular_genes[regular_genes["uniprot_entry"].isna()][
        "gene_symbol"
    ].unique()

    if verbose:
        print(f"UniProt annotation results:")
        print(
            f"  - {annotated_count}/{len(regular_genes)} regular genes successfully annotated"
        )

        if n_controls > 0:
            print(
                f"  - {n_controls} non-targeting controls marked as NON_TARGETING_CONTROL"
            )

        if len(unannotated_genes) > 0:
            print(
                f"  - {len(unannotated_genes)} unique genes without UniProt annotation:"
            )
            print(
                f"    {list(unannotated_genes)[:10]}{'...' if len(unannotated_genes) > 10 else ''}"
            )
            print(f"  - Consider reviewing these genes or updating gene symbols")

    return df_annotated


def validate_gene_symbols(
    df: pd.DataFrame,
    uniprot_data_path: Optional[str] = None,
    gene_symbol_col: str = "gene_symbol",
) -> pd.DataFrame:
    """Validate gene symbols against UniProt database and provide recommendations.

    Args:
        df (pd.DataFrame): DataFrame containing gene symbols
        uniprot_data_path (str, optional): Path to UniProt file for validation
        gene_symbol_col (str): Column containing gene symbols

    Returns:
        pd.DataFrame: Validation report with columns:
            - gene_symbol: Gene symbol
            - validated: Whether gene was found in UniProt
            - recommendation: Suggested action
    """
    if gene_symbol_col not in df.columns:
        raise ValueError(f"Column '{gene_symbol_col}' not found in dataframe")

    unique_genes = df[gene_symbol_col].dropna().unique()

    if uniprot_data_path is not None:
        # Validate against UniProt
        uniprot_data = load_and_process_uniprot_data(uniprot_data_path, verbose=False)
        valid_genes = set(uniprot_data["gene_name"])

        validation_results = []
        for gene in unique_genes:
            if gene == "":
                continue

            is_valid = gene in valid_genes

            if is_valid:
                recommendation = "Gene symbol validated ✓"
            else:
                recommendation = (
                    "Consider checking gene symbol or removing from analysis"
                )

            validation_results.append(
                {
                    "gene_symbol": gene,
                    "validated": is_valid,
                    "recommendation": recommendation,
                }
            )
    else:
        # Basic validation without UniProt
        validation_results = []
        for gene in unique_genes:
            if gene == "":
                continue

            # Check for obvious issues
            has_ampersand = "&" in str(gene)
            is_all_caps = str(gene).isupper()
            has_numbers = any(c.isdigit() for c in str(gene))

            if has_ampersand:
                recommendation = "Contains '&' - consider splitting dual targets"
            elif not is_all_caps:
                recommendation = (
                    "Consider converting to uppercase (standard gene naming)"
                )
            elif has_numbers:
                recommendation = "Contains numbers - verify gene symbol format"
            else:
                recommendation = "Basic format looks reasonable"

            validation_results.append(
                {
                    "gene_symbol": gene,
                    "validated": None,  # Can't validate without UniProt
                    "recommendation": recommendation,
                }
            )

    return pd.DataFrame(validation_results).sort_values("gene_symbol")


def get_barcode_list(
    df_barcode_library: pd.DataFrame,
    use_prefix: bool = True,
    sequencing_order: str = "map_recomb",
) -> List[str]:
    """Extract list of barcodes for mapping validation.

    Args:
        df_barcode_library (pd.DataFrame): Standardized barcode design table
        use_prefix (bool): Whether to return prefixes or full barcodes
        sequencing_order (str): Order of concatenating prefixes. Options:
                                'map_recomb' or 'recomb_map'.

    Returns:
        List[str]: List of barcode sequences
    """
    required_columns = ["prefix_map", "prefix_recomb", "gene_symbol", "uniprot_entry"]

    # Check required columns exist
    missing_cols = [
        col for col in required_columns if col not in df_barcode_library.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if use_prefix:
        if sequencing_order == "map_recomb":
            return (
                df_barcode_library["prefix_map"] + df_barcode_library["prefix_recomb"]
            ).tolist()
        elif sequencing_order == "recomb_map":
            return (
                df_barcode_library["prefix_recomb"] + df_barcode_library["prefix_map"]
            ).tolist()
        else:
            raise ValueError(
                f"Invalid sequencing_order: {sequencing_order}. Must be 'map_recomb' or 'recomb_map'."
            )
    else:
        return df_barcode_library["prefix_map"].tolist()


# Helper functions for common manipulations
def create_dialout_filter(dialout_values: List):
    """Create filter function for dialout values."""

    def filter_func(df):
        return df[df["dialout"].isin(dialout_values)] if "dialout" in df.columns else df

    return filter_func


def create_dynamic_prefix_function(
    prefix_length_col: str = "prefix_length", column_name: str = "prefix_map"
):
    """Create a prefix function that uses a column to determine prefix length for each row.

    Args:
        prefix_length_col (str): Name of column containing prefix lengths
        column_name (str): Name of column containing the barcode sequences to process

    Returns:
        callable: Function that can be used as prefix_map_func or prefix_recomb_func in standardize_barcode_design
    """

    def prefix_func(row):
        if prefix_length_col not in row.index:
            raise ValueError(
                f"Column '{prefix_length_col}' not found in row. "
                f"Available columns: {list(row.index)}"
            )

        if column_name not in row.index:
            raise ValueError(
                f"Column '{column_name}' not found in row. "
                f"Available columns: {list(row.index)}"
            )

        prefix_length = row[prefix_length_col]
        if pd.isna(prefix_length):
            # If prefix_length is NaN, use full sequence
            return row[column_name]

        # Convert to int if it's not already
        prefix_length = int(prefix_length)

        return row[column_name][:prefix_length]

    return prefix_func


def create_skip_cycles_prefix_function(
    skip_cycles, prefix_length: Optional[int] = None, column_name: str = "prefix_map"
):
    """Create a prefix function that skips specified cycles when building prefixes.

    Args:
        skip_cycles (list): List of cycle numbers to skip (1-based, e.g., [1, 5])
        prefix_length (int, optional): Length of prefix to return. If None, returns full prefix.
        column_name (str): Name of column containing the barcode sequences to process

    Returns:
        callable: Function that can be used as prefix_map_func or prefix_recomb_func in standardize_barcode_design
    """

    def prefix_func(row):
        if column_name not in row.index:
            raise ValueError(
                f"Column '{column_name}' not found in row. "
                f"Available columns: {list(row.index)}"
            )

        barcode = row[column_name]

        # Convert 1-based cycle numbers to 0-based indices
        skip_indices = [cycle - 1 for cycle in skip_cycles]

        # Create list of characters, skipping the specified cycles
        prefix_chars = []
        for i, char in enumerate(barcode):
            if i not in skip_indices:
                prefix_chars.append(char)

        # Join the characters to form the prefix
        prefix = "".join(prefix_chars)

        # Truncate the prefix if the length is specified
        prefix = prefix[:prefix_length] if prefix_length is not None else prefix

        return prefix

    return prefix_func
