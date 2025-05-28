"""Standardize barcode design tables for SBS analysis.

This module provides functions to standardize barcode design tables from various sources
into the format expected by the SBS pipeline. Includes gene validation and annotation
functions for quality control.
"""

import pandas as pd
import warnings
from typing import Optional, Union, List, Callable


def standardize_barcode_design(
    df_design: pd.DataFrame,
    barcode_col: str = "sgRNA",
    gene_symbol_col: Optional[str] = "gene_symbol",
    gene_id_col: Optional[str] = None,
    prefix_func: Optional[Callable] = None,
    prefix_length: Optional[int] = None,
    filter_func: Optional[Callable] = None,
    drop_duplicates: bool = True,
    keep_extra_cols: bool = False,
    handle_ampersand_genes: bool = True,
    standardize_nontargeting: bool = True,
    nontargeting_patterns: List[str] = ["nontargeting", "sg_nt", "non-targeting"],
    nontargeting_format: str = "nontargeting_{prefix}",
    uniprot_data_path: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Standardize a barcode design table for use with SBS analysis.

    This function handles the most common manipulations needed to prepare barcode
    design tables for the call_cells function. UniProt validation is required to
    ensure gene symbols are valid.

    Args:
        df_design (pd.DataFrame): Input barcode design table
        barcode_col (str): Name of column containing barcode sequences (default: "sgRNA")
        gene_symbol_col (str, optional): Name of column containing gene symbols.
            If None, will create empty gene_symbol column
        gene_id_col (str, optional): Name of column containing gene IDs.
            Optional - only included if user wants additional gene identifiers
        prefix_func (callable, optional): Custom function to generate prefix.
            Function should take a row and return the prefix string.
            If None, uses simple truncation with prefix_length.
        prefix_length (int, optional): Length of barcode prefix for simple truncation.
            If None, uses full barcode length. Ignored if prefix_func is provided.
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
        uniprot_data_path (str): Path to UniProt annotation file (REQUIRED for gene validation)
        verbose (bool): Whether to print processing information (default: True)

    Returns:
        pd.DataFrame: Standardized barcode design table with columns:
            - sgRNA: barcode sequences
            - gene_symbol: gene symbols
            - gene_id: gene identifiers (optional, only if gene_id_col provided)
            - prefix: barcode prefixes for matching
            - uniprot_entry: UniProt entry ID (REQUIRED)
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
    if barcode_col not in df.columns:
        raise ValueError(
            f"Barcode column '{barcode_col}' not found in design table. "
            f"Available columns: {list(df.columns)}"
        )

    # Apply custom filtering if provided
    if filter_func is not None:
        initial_count = len(df)
        df = filter_func(df)
        if verbose:
            print(f"Applied custom filter: {initial_count} → {len(df)} entries")

    # Rename barcode column to standard name
    if barcode_col != "sgRNA":
        df = df.rename(columns={barcode_col: "sgRNA"})
        if verbose:
            print(f"Renamed '{barcode_col}' column to 'sgRNA'")

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
    # Note: gene_id column is optional, so we don't create an empty one if not provided

    # Remove any rows with missing barcodes
    initial_count = len(df)
    df = df.dropna(subset=["sgRNA"])
    if len(df) < initial_count and verbose:
        print(f"Removed {initial_count - len(df)} entries with missing barcodes")

    # Drop duplicates if requested (before prefix generation to avoid issues)
    if drop_duplicates:
        initial_count = len(df)
        df = df.drop_duplicates(subset=["sgRNA"])
        if len(df) < initial_count and verbose:
            print(f"Removed {initial_count - len(df)} duplicate barcodes")

    # Generate prefix column
    if prefix_func is not None:
        # Use custom prefix function
        df["prefix"] = df.apply(prefix_func, axis=1)
        if verbose:
            print("Generated prefixes using custom function")
    else:
        # Use simple truncation
        if prefix_length is None:
            # Use full barcode length as default
            barcode_lengths = df["sgRNA"].str.len()
            prefix_length = int(barcode_lengths.mode()[0])  # Most common length
            if verbose:
                print(f"Using full barcode length ({prefix_length}) for prefix")

        df["prefix"] = df["sgRNA"].str[:prefix_length]
        if verbose:
            print(f"Generated prefixes using truncation (length={prefix_length})")

    # Validate prefix generation
    if df["prefix"].isna().any():
        warnings.warn(
            "Some prefixes could not be generated. Check your prefix function or barcode data."
        )

    # Handle ampersand genes if requested
    if handle_ampersand_genes and "gene_symbol" in df.columns:
        df = simplify_ampersand_genes(
            df, gene_symbol_col="gene_symbol", verbose=verbose
        )

    # Standardize non-targeting controls if requested (BEFORE UniProt annotation)
    if standardize_nontargeting and "gene_symbol" in df.columns:
        df = standardize_nontargeting_controls(
            df,
            nontargeting_patterns=nontargeting_patterns,
            nontargeting_format=nontargeting_format,
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
    required_cols = ["sgRNA", "gene_symbol", "prefix", "uniprot_entry"]
    if "gene_id" in df.columns:
        required_cols.insert(
            2, "gene_id"
        )  # Insert gene_id after gene_symbol if present

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
    gene_symbol_col: str = "gene_symbol",
    prefix_col: str = "prefix",
    verbose: bool = True,
) -> pd.DataFrame:
    """Standardize non-targeting control naming and annotation.

    Identifies non-targeting controls based on patterns in gene symbols and standardizes
    their naming to a consistent format. Also handles their UniProt annotation specially.

    Args:
        df (pd.DataFrame): DataFrame with gene symbols and prefixes
        nontargeting_patterns (List[str]): Patterns to identify non-targeting controls
            Case-insensitive matching against gene symbols
        nontargeting_format (str): Format string for standardized names
            Use {prefix} placeholder for barcode prefix, {original} for original name
        gene_symbol_col (str): Column containing gene symbols (default: "gene_symbol")
        prefix_col (str): Column containing barcode prefixes (default: "prefix")
        verbose (bool): Whether to print processing information

    Returns:
        pd.DataFrame: DataFrame with standardized non-targeting control names

    Examples:
        # Detect common patterns and rename
        df = standardize_nontargeting_controls(df)

        # Custom patterns and format
        df = standardize_nontargeting_controls(
            df,
            nontargeting_patterns=["neg", "scramble", "control"],
            nontargeting_format="ctrl_{prefix}"
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

    # Create case-insensitive pattern matching - fix the regex construction
    # Join patterns first, then add the case-insensitive flag
    pattern_regex = "|".join(nontargeting_patterns)

    # Find non-targeting controls using case-insensitive matching
    nontargeting_mask = (
        result_df[gene_symbol_col]
        .astype(str)
        .str.contains(pattern_regex, na=False, regex=True, case=False)
    )

    if nontargeting_mask.any():
        n_nontargeting = nontargeting_mask.sum()

        # Store original names for reference
        if "original_gene_symbol" not in result_df.columns:
            result_df["original_gene_symbol"] = result_df[gene_symbol_col].copy()

        # Generate standardized names
        standardized_names = []
        for idx, row in result_df[nontargeting_mask].iterrows():
            prefix = str(row[prefix_col])
            original = str(row[gene_symbol_col])

            # Format the new name
            new_name = nontargeting_format.format(prefix=prefix, original=original)
            standardized_names.append(new_name)

        # Apply standardized names
        result_df.loc[nontargeting_mask, gene_symbol_col] = standardized_names

        if verbose:
            print(f"Standardized {n_nontargeting} non-targeting controls:")
            print(f"  - Detected patterns: {nontargeting_patterns}")
            print(f"  - New format: {nontargeting_format}")
            print(f"  - Examples:")
            for i, (old, new) in enumerate(
                zip(
                    result_df.loc[nontargeting_mask, "original_gene_symbol"].head(3),
                    result_df.loc[nontargeting_mask, gene_symbol_col].head(3),
                )
            ):
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


def get_barcode_list(df_barcode_library: pd.DataFrame, use_prefix: bool = True) -> List[str]:
    """Extract list of barcodes for mapping validation.

    Args:
        df_barcode_library (pd.DataFrame): Standardized barcode design table
        use_prefix (bool): Whether to return prefixes or full barcodes

    Returns:
        List[str]: List of barcode sequences
    """
    required_columns = ["sgRNA", "gene_symbol", "prefix", "uniprot_entry"]

    # Check required columns exist
    missing_cols = [col for col in required_columns if col not in df_barcode_library.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if use_prefix:
        return df_barcode_library["prefix"].tolist()
    else:
        return df_barcode_library["sgRNA"].tolist()


# Helper functions for common manipulations
def create_dialout_filter(dialout_values: List):
    """Create filter function for dialout values."""

    def filter_func(df):
        return df[df["dialout"].isin(dialout_values)] if "dialout" in df.columns else df

    return filter_func


def create_dynamic_prefix_function(prefix_length_col: str = "prefix_length"):
    """Create a prefix function that uses a column to determine prefix length for each row.

    Args:
        prefix_length_col (str): Name of column containing prefix lengths

    Returns:
        callable: Function that can be used as prefix_func in standardize_barcode_design
    """

    def prefix_func(row):
        if prefix_length_col not in row.index:
            raise ValueError(
                f"Column '{prefix_length_col}' not found in row. "
                f"Available columns: {list(row.index)}"
            )

        prefix_length = row[prefix_length_col]
        if pd.isna(prefix_length):
            # If prefix_length is NaN, use full sgRNA
            return row["sgRNA"]

        # Convert to int if it's not already
        prefix_length = int(prefix_length)

        return row["sgRNA"][:prefix_length]

    return prefix_func


def create_concatenation_prefix(
    col1: str, col2: str, col1_slice: Optional[slice] = None
):
    """Create prefix function that concatenates two columns."""

    def prefix_func(row):
        val1 = row[col1]
        if col1_slice is not None:
            val1 = val1[col1_slice]
        return str(val1) + str(row[col2])

    return prefix_func


def create_truncation_prefix(length: int):
    """Create prefix function that truncates sgRNA column."""

    def prefix_func(row):
        return row["sgRNA"][:length]

    return prefix_func
