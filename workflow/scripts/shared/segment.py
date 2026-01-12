from lib.shared.io import read_image, save_image
import numpy as np
import pandas as pd

# Load illumination corrected data
aligned_data = read_image(snakemake.input[0])

# Handle Z-dimension: if data is CZYX (4D), reduce to CYX (3D) for segmentation
# Segmentation expects 3D data (CYX)
if aligned_data.ndim == 4:
    print(f"Reducing Z-dimension for segmentation: {aligned_data.shape} -> ", end="")
    # Use max projection along Z axis (axis=1 for CZYX) to match TIFF workflow
    # This should not occur for phenotype if preserve_z=False is set correctly
    aligned_data = aligned_data.max(axis=1)
    print(f"{aligned_data.shape}")

# Get configuration from params
params = snakemake.params.config

# Choose segmentation method based on parameter
method = params.get("segmentation_method", "cellpose")
segment_cells = params.get("segment_cells", True)

if method == "cellpose":
    # Segment cells using cellpose
    from lib.shared.segment_cellpose import segment_cellpose

    if segment_cells:
        nuclei_data, cells_data, counts_df = segment_cellpose(
            data=aligned_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            nuclei_diameter=params["nuclei_diameter"],
            cell_diameter=params["cell_diameter"],
            cyto_model=params["cellpose_model"],
            cellpose_kwargs=dict(
                flow_threshold=params.get("flow_threshold", 0.4),
                cellprob_threshold=params.get("cellprob_threshold", 0),
                nuclei_flow_threshold=params["nuclei_flow_threshold"],
                nuclei_cellprob_threshold=params["nuclei_cellprob_threshold"],
                cell_flow_threshold=params["cell_flow_threshold"],
                cell_cellprob_threshold=params["cell_cellprob_threshold"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=segment_cells,
        )
    else:
        nuclei_data, counts_df = segment_cellpose(
            data=aligned_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            nuclei_diameter=params["nuclei_diameter"],
            cell_diameter=params["cell_diameter"],
            cyto_model=params["cellpose_model"],
            cellpose_kwargs=dict(
                flow_threshold=params.get("flow_threshold", 0.4),
                cellprob_threshold=params.get("cellprob_threshold", 0),
                nuclei_flow_threshold=params["nuclei_flow_threshold"],
                nuclei_cellprob_threshold=params["nuclei_cellprob_threshold"],
                cell_flow_threshold=params["cell_flow_threshold"],
                cell_cellprob_threshold=params["cell_cellprob_threshold"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=segment_cells,
        )
        cells_data = np.zeros_like(nuclei_data)

elif method == "microsam":
    # Segment cells using MicroSAM
    from lib.shared.segment_microsam import segment_microsam

    if segment_cells:
        nuclei_data, cells_data, counts_df = segment_microsam(
            data=aligned_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            model_type=params["microsam_model"],
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=segment_cells,
        )
    else:
        nuclei_data, counts_df = segment_microsam(
            data=aligned_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            model_type=params["microsam_model"],
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=segment_cells,
        )
        cells_data = np.zeros_like(nuclei_data)

elif method == "stardist":
    # Segment cells using StarDist
    from lib.shared.segment_stardist import segment_stardist

    if segment_cells:
        nuclei_data, cells_data, counts_df = segment_stardist(
            data=aligned_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            model_type=params["stardist_model"],
            stardist_kwargs=dict(
                prob_threshold=params.get("prob_threshold", 0.479071),
                nms_threshold=params.get("nms_threshold", 0.3),
                nuclei_prob_threshold=params["nuclei_prob_threshold"],
                nuclei_nms_threshold=params["nuclei_nms_threshold"],
                cell_prob_threshold=params["cell_prob_threshold"],
                cell_nms_threshold=params["cell_nms_threshold"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=segment_cells,
        )
    else:
        nuclei_data, counts_df = segment_stardist(
            data=aligned_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            model_type=params["stardist_model"],
            stardist_kwargs=dict(
                prob_threshold=params.get("prob_threshold", 0.479071),
                nms_threshold=params.get("nms_threshold", 0.3),
                nuclei_prob_threshold=params["nuclei_prob_threshold"],
                nuclei_nms_threshold=params["nuclei_nms_threshold"],
                cell_prob_threshold=params["cell_prob_threshold"],
                cell_nms_threshold=params["cell_nms_threshold"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=segment_cells,
        )
        cells_data = np.zeros_like(nuclei_data)

elif method == "watershed":
    # Segment cells using Watershed
    from lib.shared.segment_watershed import segment_watershed

    if segment_cells:
        nuclei_data, cells_data, counts_df = segment_watershed(
            data=aligned_data,
            nuclei_threshold=params["threshold_dapi"],
            nuclei_area_min=params["nuclei_area_min"],
            nuclei_area_max=params["nuclei_area_max"],
            cell_threshold=params["threshold_cell"],
            cells=True,
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
        )
    else:
        nuclei_data, counts_df = segment_watershed(
            data=aligned_data,
            nuclei_threshold=params["threshold_dapi"],
            nuclei_area_min=params["nuclei_area_min"],
            nuclei_area_max=params["nuclei_area_max"],
            cell_threshold=params["threshold_cell"],
            return_counts=params.get("return_counts", True),
            cells=segment_cells,
        )
        cells_data = np.zeros_like(nuclei_data)
else:
    raise ValueError(
        f"Unknown segmentation method: {method}. Choose one of: cellpose, microsam, stardist, watershed"
    )

# Save segmented nuclei data
save_image(
    nuclei_data,
    snakemake.output[0],
    pixel_size_z=params["pixel_size_z"],
    pixel_size_y=params["pixel_size_y"],
    pixel_size_x=params["pixel_size_x"],
    channel_names=["nuclei"],  # Specific channel name for nuclei
    is_label=True,
)
# Save segmented cells data
save_image(
    cells_data,
    snakemake.output[1],
    pixel_size_z=params["pixel_size_z"],
    pixel_size_y=params["pixel_size_y"],
    pixel_size_x=params["pixel_size_x"],
    channel_names=["cells"],  # Specific channel name for cells
    is_label=True,
)
# Save counts data
counts_df.to_csv(snakemake.output[2], index=False, sep="\t")
