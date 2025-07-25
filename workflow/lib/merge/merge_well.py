"""
Well-level merge functions for stitched images instead of individual tiles.
This replaces the tile-by-tile triangle hashing with well-by-well triangle hashing.
"""

import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings
from typing import Tuple, Dict, List

# Import existing functions that we'll reuse
from lib.merge.hash import (
    get_vectors,
    nine_edge_hash,
    get_vc,
    nearest_neighbors,
    gb_apply_parallel,
    extract_rotation,
)
from lib.merge.merge import build_linear_model


def extract_nuclei_positions_from_stitched(
    stitched_image: np.ndarray,
    cell_info_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    well: str,
    tile_shifts: Dict[str, List[int]],
) -> pd.DataFrame:
    """
    Convert tile-based cell positions to stitched well coordinates.

    Args:
        stitched_image: Stitched well image (can be None, not used in current implementation)
        cell_info_df: DataFrame with cell information including i, j coordinates
        metadata_df: Metadata DataFrame with tile information (can be None, not used)
        well: Well identifier
        tile_shifts: Dictionary mapping tile keys to [y, x] shifts

    Returns:
        DataFrame with cell positions in stitched well coordinates
    """
    well_cells = cell_info_df[cell_info_df["well"] == well].copy()

    if len(well_cells) == 0:
        return pd.DataFrame(columns=["well", "cell", "i", "j", "tile"])

    # Convert tile-based coordinates to stitched coordinates
    stitched_positions = []

    for _, cell_row in well_cells.iterrows():
        tile_id = cell_row["tile"]
        tile_key = f"{well}/{tile_id}"

        if tile_key in tile_shifts:
            shift = tile_shifts[tile_key]
            y_shift, x_shift = shift[0], shift[1]

            # Convert tile coordinates to stitched coordinates
            stitched_i = cell_row["i"] + y_shift
            stitched_j = cell_row["j"] + x_shift

            # Determine cell identifier - could be 'cell', 'label', or index
            if "cell" in cell_row:
                cell_id = cell_row["cell"]
            elif "label" in cell_row:
                cell_id = cell_row["label"]
            else:
                cell_id = cell_row.name

            stitched_positions.append(
                {
                    "well": well,
                    "tile": tile_id,
                    "cell": cell_id,
                    "i": stitched_i,
                    "j": stitched_j,
                }
            )

    return pd.DataFrame(stitched_positions)


def hash_stitched_well_locations(well_positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate hashed Delaunay triangulation for stitched well nuclei positions.

    Args:
        well_positions_df: DataFrame with cell positions in stitched coordinates

    Returns:
        DataFrame with triangle hash features
    """
    if len(well_positions_df) < 4:
        print(
            f"Warning: Only {len(well_positions_df)} cells found, need at least 4 for triangulation"
        )
        return pd.DataFrame()

    # Extract coordinates and compute Delaunay triangulation
    coordinates = well_positions_df[["i", "j"]].values

    try:
        v, c = get_vectors(coordinates)

        if len(v) == 0:
            print("No valid triangles generated")
            return pd.DataFrame()

        # Create dataframes for vectors and centers
        df_vectors = pd.DataFrame(v).rename(columns="V_{0}".format)
        df_coords = pd.DataFrame(c).rename(columns="c_{0}".format)

        # Combine and add magnitude
        df_combined = pd.concat([df_vectors, df_coords], axis=1)
        df_result = df_combined.assign(
            magnitude=lambda x: x.eval("(V_0**2 + V_1**2)**0.5")
        )

        # Add well information
        df_result["well"] = well_positions_df["well"].iloc[0]

        return df_result

    except Exception as e:
        print(f"Error computing triangulation: {e}")
        return pd.DataFrame()


def well_level_alignment(
    phenotype_hash: pd.DataFrame,
    sbs_hash: pd.DataFrame,
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    det_range: Tuple[float, float] = (0.8, 1.2),
    score_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Perform alignment between phenotype and SBS stitched wells.

    Args:
        phenotype_hash: Triangle hash features for phenotype well
        sbs_hash: Triangle hash features for SBS well
        phenotype_positions: Cell positions in phenotype well
        sbs_positions: Cell positions in SBS well
        det_range: Acceptable range for transformation determinant
        score_threshold: Minimum score for valid alignment

    Returns:
        DataFrame with alignment parameters
    """
    if len(phenotype_hash) == 0 or len(sbs_hash) == 0:
        print("Empty hash data, cannot perform alignment")
        return pd.DataFrame()

    try:
        # Extract vectors and centers
        V_0, c_0 = get_vc(phenotype_hash)
        V_1, c_1 = get_vc(sbs_hash)

        # Find nearest neighbors between triangle vectors
        i0, i1, distances = nearest_neighbors(V_0, V_1)

        # Filter based on distance threshold
        filt = distances < 0.3  # Triangle matching threshold
        if filt.sum() < 5:
            print(f"Only {filt.sum()} matching triangles found, need at least 5")
            return pd.DataFrame()

        # Get matching triangle centers
        X, Y = c_0[i0[filt]], c_1[i1[filt]]

        # Fit transformation using RANSAC
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = RANSACRegressor(min_samples=5, max_trials=1000)
            model.fit(X, Y)

        rotation = model.estimator_.coef_
        translation = model.estimator_.intercept_

        # Score the transformation
        distances_score = cdist(model.predict(c_0), c_1, metric="sqeuclidean")
        threshold_region = 50
        filt_score = np.sqrt(distances_score.min(axis=0)) < threshold_region
        score = (np.sqrt(distances_score.min(axis=0))[filt_score] < 2).mean()

        # Calculate determinant
        determinant = np.linalg.det(rotation)

        # Create result
        result = pd.DataFrame(
            [
                {
                    "rotation_1": rotation[0] if len(rotation) > 0 else [0, 0],
                    "rotation_2": rotation[1] if len(rotation) > 1 else [0, 0],
                    "translation": translation,
                    "score": score,
                    "determinant": determinant,
                    "well": phenotype_positions["well"].iloc[0]
                    if len(phenotype_positions) > 0
                    else "unknown",
                }
            ]
        )

        return result

    except Exception as e:
        print(f"Alignment failed: {e}")
        return pd.DataFrame()


def merge_stitched_wells(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: pd.Series,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Merge cell positions between phenotype and SBS stitched wells.

    Args:
        phenotype_positions: Cell positions in phenotype well
        sbs_positions: Cell positions in SBS well
        alignment: Alignment parameters (rotation, translation)
        threshold: Maximum distance for cell matching

    Returns:
        DataFrame with merged cell identities
    """
    if len(phenotype_positions) == 0 or len(sbs_positions) == 0:
        return pd.DataFrame(
            columns=[
                "plate",
                "well",
                "tile",
                "cell_0",
                "i_0",
                "j_0",
                "site",
                "cell_1",
                "i_1",
                "j_1",
                "distance",
            ]
        )

    try:
        # Build transformation model
        rotation = np.array([alignment["rotation_1"], alignment["rotation_2"]])
        translation = alignment["translation"]
        model = build_linear_model(rotation, translation)

        # Extract coordinates
        X = phenotype_positions[["i", "j"]].values
        Y = sbs_positions[["i", "j"]].values

        # Apply transformation
        Y_pred = model.predict(X)

        # Calculate distances
        distances = cdist(Y, Y_pred, metric="sqeuclidean")

        # Find nearest neighbors
        ix = distances.argmin(axis=1)
        min_distances = np.sqrt(distances.min(axis=1))

        # Filter by threshold
        filt = min_distances < threshold

        if filt.sum() == 0:
            print("No matches found within threshold")
            return pd.DataFrame(
                columns=[
                    "plate",
                    "well",
                    "tile",
                    "cell_0",
                    "i_0",
                    "j_0",
                    "site",
                    "cell_1",
                    "i_1",
                    "j_1",
                    "distance",
                ]
            )

        # Create merged dataframe
        matched_phenotype = phenotype_positions.iloc[ix[filt]].reset_index(drop=True)
        matched_sbs = sbs_positions[filt].reset_index(drop=True)

        merged_data = []
        for i in range(len(matched_phenotype)):
            merged_data.append(
                {
                    "plate": 1,  # You may need to extract this from your data
                    "well": matched_phenotype.iloc[i]["well"],
                    "tile": matched_phenotype.iloc[i]["tile"],
                    "cell_0": matched_phenotype.iloc[i]["cell"],
                    "i_0": matched_phenotype.iloc[i]["i"],
                    "j_0": matched_phenotype.iloc[i]["j"],
                    "site": matched_sbs.iloc[i]["tile"],  # Using tile as site
                    "cell_1": matched_sbs.iloc[i]["cell"],
                    "i_1": matched_sbs.iloc[i]["i"],
                    "j_1": matched_sbs.iloc[i]["j"],
                    "distance": min_distances[filt][i],
                }
            )

        return pd.DataFrame(merged_data)

    except Exception as e:
        print(f"Merge failed: {e}")
        return pd.DataFrame(
            columns=[
                "plate",
                "well",
                "tile",
                "cell_0",
                "i_0",
                "j_0",
                "site",
                "cell_1",
                "i_1",
                "j_1",
                "distance",
            ]
        )


def well_merge_pipeline(
    phenotype_info: pd.DataFrame,
    sbs_info: pd.DataFrame,
    phenotype_shifts: Dict[str, List[int]],
    sbs_shifts: Dict[str, List[int]],
    well: str,
    det_range: Tuple[float, float] = (0.8, 1.2),
    score_threshold: float = 0.1,
    distance_threshold: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete well-level merge pipeline.

    Args:
        phenotype_info: Phenotype cell information DataFrame
        sbs_info: SBS cell information DataFrame
        phenotype_shifts: Tile shifts for phenotype stitching
        sbs_shifts: Tile shifts for SBS stitching
        well: Well identifier
        det_range: Valid determinant range for alignment
        score_threshold: Minimum alignment score
        distance_threshold: Maximum cell matching distance

    Returns:
        Tuple of (merged_cells_df, alignment_df)
    """
    print(f"Processing well-level merge for well {well}")

    # Extract cell positions in stitched coordinates
    phenotype_positions = extract_nuclei_positions_from_stitched(
        None, phenotype_info, None, well, phenotype_shifts
    )

    sbs_positions = extract_nuclei_positions_from_stitched(
        None, sbs_info, None, well, sbs_shifts
    )

    print(
        f"Found {len(phenotype_positions)} phenotype cells, {len(sbs_positions)} SBS cells"
    )

    if len(phenotype_positions) < 4 or len(sbs_positions) < 4:
        print("Insufficient cells for triangulation")
        return pd.DataFrame(), pd.DataFrame()

    # Generate triangle hashes
    phenotype_hash = hash_stitched_well_locations(phenotype_positions)
    sbs_hash = hash_stitched_well_locations(sbs_positions)

    if len(phenotype_hash) == 0 or len(sbs_hash) == 0:
        print("Failed to generate triangle hashes")
        return pd.DataFrame(), pd.DataFrame()

    # Perform alignment
    alignment_df = well_level_alignment(
        phenotype_hash,
        sbs_hash,
        phenotype_positions,
        sbs_positions,
        det_range,
        score_threshold,
    )

    if len(alignment_df) == 0:
        print("Alignment failed")
        return pd.DataFrame(), pd.DataFrame()

    # Check alignment quality
    alignment = alignment_df.iloc[0]
    if (
        alignment["determinant"] < det_range[0]
        or alignment["determinant"] > det_range[1]
        or alignment["score"] < score_threshold
    ):
        print(
            f"Alignment quality insufficient: det={alignment['determinant']:.3f}, score={alignment['score']:.3f}"
        )
        return pd.DataFrame(), alignment_df

    # Merge cells
    merged_cells = merge_stitched_wells(
        phenotype_positions, sbs_positions, alignment, distance_threshold
    )

    print(f"Successfully merged {len(merged_cells)} cells")

    return merged_cells, alignment_df
