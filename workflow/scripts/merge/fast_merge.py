import pandas as pd
import numpy as np

from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import read_parquet, write_parquet
from lib.merge.fast_merge import merge_triangle_hash

# Validate required params
for _param_name in ["det_range", "score", "threshold"]:
    if getattr(snakemake.params, _param_name, None) is None:
        raise ValueError(f"Required config parameter '{_param_name}' is not set")

# Load phenotype and sbs info with cell locations
phenotype_info = validate_dtypes(read_parquet(snakemake.input[0]))
sbs_info = validate_dtypes(read_parquet(snakemake.input[1]))

# Load alignment data
fast_alignment = read_parquet(snakemake.input[2])
fast_alignment["rotation"] = [
    np.array([r1, r2])
    for r1, r2 in zip(fast_alignment["rotation_1"], fast_alignment["rotation_2"])
]
fast_alignment.drop(columns=["rotation_1", "rotation_2"], inplace=True)

# Filter alignment data based on parameters
fast_alignment_filtered = fast_alignment[
    (fast_alignment["determinant"] >= snakemake.params.det_range[0])
    & (fast_alignment["determinant"] <= snakemake.params.det_range[1])
    & (fast_alignment["score"] > snakemake.params.score)
]

print(f"Original tile-by-tile merge approach")
print(f"Total alignments: {len(fast_alignment)}")
print(f"Filtered alignments: {len(fast_alignment_filtered)}")

# Optional polynomial-warp levers — backward compatible: absent keys -> None -> dropped ->
# refine_local_warp falls back to its literal defaults (existing screens unchanged).
warp_kwargs = {
    k: v
    for k, v in {
        "degree": getattr(snakemake.params, "warp_degree", None),
        "iterations": getattr(snakemake.params, "warp_iterations", None),
        "min_correspondences": getattr(snakemake.params, "warp_min_correspondences", None),
    }.items()
    if v is not None
} or None

# Merge cells across well
merge_data = []
for index, alignment_row in fast_alignment_filtered.iterrows():
    # Determine tiles and sites for merging
    phenotype_tile = alignment_row["tile"]
    sbs_site = alignment_row["site"]

    # Filter phenotype and sbs info to the relevant well and tile for merging
    phenotype_info_filtered = phenotype_info[phenotype_info["tile"] == phenotype_tile]
    sbs_info_filtered = sbs_info[sbs_info["tile"] == sbs_site]

    # Merge cells for row of alignment data
    alignment_row_merge = merge_triangle_hash(
        phenotype_info_filtered,
        sbs_info_filtered,
        alignment_row,
        threshold=snakemake.params.threshold,
        local_refinement=getattr(snakemake.params, "local_refinement", None),
        warp_kwargs=warp_kwargs,
    )
    merge_data.append(alignment_row_merge)

# Compile and save merge data
merge_data = pd.concat(merge_data, ignore_index=True)
print(f"Legacy merge completed: {len(merge_data)} cells merged")

write_parquet(merge_data, snakemake.output[0])
