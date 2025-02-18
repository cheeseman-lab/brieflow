"""This module provides functions for loading and formatting data during aggregation.

Functions include:
- Cleaning cell data by removing unassigned or multi-gene cells.

Functions:
    - clean_cell_data: Clean cell data by filtering for valid and optionally single-gene cells.
"""


def clean_cell_data(cell_measurements, population_feature, filter_single_gene=False):
    """Clean cell data by removing cells without perturbation assignments and optionally filtering for single-gene cells.

    Args:
        cell_measurements (pd.DataFrame): Raw dataframe containing cell measurements.
        population_feature (str): Column name containing perturbation assignments.
        filter_single_gene (bool): If True, only keep cells with mapped_single_gene=True.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Remove cells without perturbation assignments
    clean_cell_measurements = cell_measurements[
        cell_measurements[population_feature].notna()
    ].copy()
    print(f"Found {len(clean_cell_measurements)} cells with assigned perturbations")

    if filter_single_gene:
        # Filter for single-gene cells if requested
        clean_cell_measurements = clean_cell_measurements[
            clean_cell_measurements["mapped_single_gene"] == True
        ]
        print(f"Kept {len(clean_cell_measurements)} cells with single gene assignments")
    else:
        # Warn about multi-gene cells if not filtering
        multi_gene_cells = len(
            clean_cell_measurements[
                clean_cell_measurements["mapped_single_gene"] == False
            ]
        )
        if multi_gene_cells > 0:
            print(f"WARNING: {multi_gene_cells} cells have multiple gene assignments")

    return clean_cell_measurements
