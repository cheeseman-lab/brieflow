from tifffile import imread, imwrite
import numpy as np
import pandas as pd

# Load illumination corrected data
illumination_corrected_data = imread(snakemake.input[0])

# Get configuration from params
params = snakemake.params.config

# Choose segmentation method based on parameter
method = params.get("segmentation_method", "cellpose")
cells_enabled = params.get("cells", True)

if method == "cellpose":
    # Segment cells using cellpose
    from lib.shared.segment_cellpose import segment_cellpose

    if cells_enabled:
        nuclei_data, cells_data, counts_df = segment_cellpose(
            data=illumination_corrected_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            nuclei_diameter=params["nuclei_diameter"],
            cell_diameter=params["cell_diameter"],
            cyto_model=params["cellpose_model"],
            cellpose_kwargs=dict(
                flow_threshold=params["flow_threshold"],
                cellprob_threshold=params["cellprob_threshold"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=True,
        )
    else:
        nuclei_data, counts_df = segment_cellpose(
            data=illumination_corrected_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            nuclei_diameter=params["nuclei_diameter"],
            cell_diameter=params["cell_diameter"],
            cyto_model=params["cellpose_model"],
            cellpose_kwargs=dict(
                flow_threshold=params["flow_threshold"],
                cellprob_threshold=params["cellprob_threshold"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=False,
        )
        cells_data = np.zeros_like(nuclei_data)
        
elif method == "microsam":
    # Segment cells using MicroSAM
    from lib.shared.segment_microsam import segment_microsam

    if cells_enabled:
        nuclei_data, cells_data, counts_df = segment_microsam(
            data=illumination_corrected_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            model_type=params["microsam_model"],
            microsam_kwargs=dict(
                points_per_side=params["points_per_side"],
                points_per_batch=params["points_per_batch"],
                stability_score_thresh=params["stability_score_thresh"],
                pred_iou_thresh=params["pred_iou_thresh"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=True,
        )
    else:
        nuclei_data, counts_df = segment_microsam(
            data=illumination_corrected_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            model_type=params["microsam_model"],
            microsam_kwargs=dict(
                points_per_side=params["points_per_side"],
                points_per_batch=params["points_per_batch"],
                stability_score_thresh=params["stability_score_thresh"],
                pred_iou_thresh=params["pred_iou_thresh"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=False,
        )
        cells_data = np.zeros_like(nuclei_data)
        
elif method == "stardist":
    # Segment cells using StarDist
    from lib.shared.segment_stardist import segment_stardist

    if cells_enabled:
        nuclei_data, cells_data, counts_df = segment_stardist(
            data=illumination_corrected_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            model_type=params["stardist_model"],
            stardist_kwargs=dict(
                prob_thresh=params["prob_thresh"],
                nms_thresh=params["nms_thresh"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=True,
        )
    else:
        nuclei_data, counts_df = segment_stardist(
            data=illumination_corrected_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            model_type=params["stardist_model"],
            stardist_kwargs=dict(
                prob_thresh=params["prob_thresh"],
                nms_thresh=params["nms_thresh"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=False,
        )
        cells_data = np.zeros_like(nuclei_data)
        
elif method == "watershed":
    # Segment cells using Watershed
    from lib.shared.segment_watershed import segment_watershed

    if cells_enabled:
        nuclei_data, cells_data, counts_df = segment_watershed(
            data=illumination_corrected_data,
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
            data=illumination_corrected_data,
            nuclei_threshold=params["threshold_dapi"],
            nuclei_area_min=params["nuclei_area_min"],
            nuclei_area_max=params["nuclei_area_max"],
            cell_threshold=params["threshold_cell"],
            cells=False,
            return_counts=params.get("return_counts", True),
        )
        cells_data = np.zeros_like(nuclei_data)
else:
    raise ValueError(
        f"Unknown segmentation method: {method}. Choose one of: cellpose, microsam, stardist, watershed"
    )

# Save segmented nuclei data
imwrite(snakemake.output[0], nuclei_data)
# Save segmented cells data
imwrite(snakemake.output[1], cells_data)
# Save counts data
counts_df.to_csv(snakemake.output[2], index=False, sep="\t")