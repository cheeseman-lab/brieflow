import numpy as np
import pandas as pd

from lib.shared.file_utils import validate_dtypes


def _attach_global_pixel_coords(
    merged,
    metadata,
    tile_dimensions,
    suffix,
):
    """Join per-tile stage metadata onto merged cells and compute global pixel coords.

    Args:
        merged: DataFrame with at least (plate, well, tile or site, i_{0/1}, j_{0/1}).
        metadata: DataFrame with x_pos, y_pos, pixel_size_x, pixel_size_y per tile.
        tile_dimensions: (height, width) of the tile in pixels.
        suffix: "0" for phenotype (joins on tile), "1" for sbs (joins on site).

    Returns:
        merged DataFrame with global_i_{suffix}, global_j_{suffix} columns added,
        anchored at the per-(plate, well) min tile-corner.
    """
    if metadata is None or len(metadata) == 0:
        return merged

    # SBS uses "site" in the merged frame to refer to the tile id from the SBS dataset.
    tile_col_in_merged = "tile" if suffix == "0" else "site"
    meta = metadata[
        ["plate", "well", "tile", "x_pos", "y_pos", "pixel_size_x", "pixel_size_y"]
    ].rename(
        columns={
            "tile": tile_col_in_merged,
            "x_pos": f"x_pos_{suffix}",
            "y_pos": f"y_pos_{suffix}",
            "pixel_size_x": f"pixel_size_x_{suffix}",
            "pixel_size_y": f"pixel_size_y_{suffix}",
        }
    )

    merged = merged.merge(meta, how="left", on=["plate", "well", tile_col_in_merged])

    tile_h, tile_w = tile_dimensions

    # Tile top-left corner in stage μm (assuming x_pos / y_pos = tile center).
    corner_x_um = merged[f"x_pos_{suffix}"] - tile_w / 2 * merged[f"pixel_size_x_{suffix}"]
    corner_y_um = merged[f"y_pos_{suffix}"] - tile_h / 2 * merged[f"pixel_size_y_{suffix}"]

    # Per-cell global μm: tile corner + local pixel offset.
    cell_x_um = corner_x_um + merged[f"j_{suffix}"] * merged[f"pixel_size_x_{suffix}"]
    cell_y_um = corner_y_um + merged[f"i_{suffix}"] * merged[f"pixel_size_y_{suffix}"]

    # Anchor each (plate, well) at its top-left tile corner so global pixel coords
    # are non-negative integers comparable within a well.
    well_origin = (
        merged.assign(_corner_x_um=corner_x_um, _corner_y_um=corner_y_um)
        .groupby(["plate", "well"])[["_corner_x_um", "_corner_y_um"]]
        .transform("min")
    )

    merged[f"global_j_{suffix}"] = (
        ((cell_x_um - well_origin["_corner_x_um"]) / merged[f"pixel_size_x_{suffix}"])
        .round()
        .astype("Int32")
    )
    merged[f"global_i_{suffix}"] = (
        ((cell_y_um - well_origin["_corner_y_um"]) / merged[f"pixel_size_y_{suffix}"])
        .round()
        .astype("Int32")
    )

    return merged


# Load deduplicated merge data
merge_deduplicated = validate_dtypes(pd.read_parquet(snakemake.input[0]))

# Extract configuration parameters
approach = snakemake.params.approach

# Load full feature data
cp_phenotype = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Merge full CP data on deduplicated
merged_final = merge_deduplicated.merge(
    cp_phenotype.rename(columns={"label": "cell_0"}),
    how="left",
    on=["plate", "well", "tile", "cell_0"],
)

# For the fast approach, attach phenotype + sbs stage metadata and compute
# global pixel positions per cell (anchored at per-well top-left tile corner).
# The stitch approach already produces global_i_*/global_j_* via stitch_merge.
if approach == "fast":
    phenotype_metadata = pd.read_parquet(snakemake.input.phenotype_metadata)
    sbs_metadata = pd.read_parquet(snakemake.input.sbs_metadata)

    phenotype_dims = snakemake.params.phenotype_dimensions
    sbs_dims = snakemake.params.sbs_dimensions

    if phenotype_dims is None or sbs_dims is None:
        raise ValueError(
            "merge.phenotype_dimensions and merge.sbs_dimensions must be set "
            "in config to compute global pixel coordinates."
        )

    merged_final = _attach_global_pixel_coords(
        merged_final, phenotype_metadata, phenotype_dims, suffix="0"
    )
    merged_final = _attach_global_pixel_coords(
        merged_final, sbs_metadata, sbs_dims, suffix="1"
    )

# Rename coordinate columns to global naming convention only for stitch approach
if approach == "stitch":
    merged_final = merged_final.rename(
        columns={
            "i_0": "global_i_0",
            "j_0": "global_j_0",
            "i_1": "global_i_1",
            "j_1": "global_j_1",
        }
    )

# Save final merged dataset
merged_final.to_parquet(snakemake.output[0])
