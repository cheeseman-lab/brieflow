import pandas as pd


def load_hdf_subset(merge_final_fp, n_rows=20000, population_feature="gene_symbol_0"):
    """
    Load a fixed number of random rows from an HDF file without loading entire file into memory.

    Parameters
    ----------
    merge_final_fp : str
        Path to HDF file
    n_rows : int
        Number of rows to get
    population_feature : str
        Column name containing population identifiers

    Returns
    -------
    pd.DataFrame
        Subset of the data with combined blocks
    """
    print(f"Reading first {n_rows:,} rows from {merge_final_fp}")

    # read the first n_rows of the file path
    df = pd.read_hdf(merge_final_fp, stop=n_rows)

    # print the number of unique populations
    print(f"Unique populations: {df[population_feature].nunique()}")

    # print the counts of the well variable
    print(df["well"].value_counts())

    return df


def clean_cell_data(df, population_feature, filter_single_gene=False):
    """
    Clean cell data by removing cells without perturbation assignments and optionally filtering for single-gene cells.

    Args:
        df (pd.DataFrame): Raw dataframe containing cell measurements
        population_feature (str): Column name containing perturbation assignments
        filter_single_gene (bool): If True, only keep cells with mapped_single_gene=True

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Remove cells without perturbation assignments
    clean_df = df[df[population_feature].notna()].copy()
    print(f"Found {len(clean_df)} cells with assigned perturbations")

    if filter_single_gene:
        # Filter for single-gene cells if requested
        clean_df = clean_df[clean_df["mapped_single_gene"] == True]
        print(f"Kept {len(clean_df)} cells with single gene assignments")
    else:
        # Warn about multi-gene cells if not filtering
        multi_gene_cells = len(clean_df[clean_df["mapped_single_gene"] == False])
        if multi_gene_cells > 0:
            print(f"WARNING: {multi_gene_cells} cells have multiple gene assignments")

    return clean_df


def add_filenames():
    return None
