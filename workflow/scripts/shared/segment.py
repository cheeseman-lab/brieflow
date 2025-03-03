from tifffile import imread, imwrite
import numpy as np
import pandas as pd

# Load illumination corrected data
illumination_corrected_data = imread(snakemake.input[0])

# Get configuration from params
params = snakemake.params.config

# Choose segmentation method based on parameter
method = params.get("method", "cellpose")

if method == "cellpose":
    # Segment cells using cellpose
    from lib.shared.segment_cellpose import segment_cellpose
    nuclei_data, cells_data, counts_df = segment_cellpose(
        data=illumination_corrected_data,
        dapi_index=params["dapi_index"],
        cyto_index=params["cyto_index"],
        nuclei_diameter=params["nuclei_diameter"],
        cell_diameter=params["cell_diameter"],
        cyto_model=params["cyto_model"],
        cellpose_kwargs=dict(
            flow_threshold=params["flow_threshold"],
            cellprob_threshold=params["cellprob_threshold"],
        ),
        reconcile=params.get("reconcile", False),
        return_counts=params.get("return_counts", True),
        gpu=params.get("gpu", False),
    )
elif method == "microsam":
    # Segment cells using MicroSAM
    from lib.shared.segment_microsam import segment_microsam
    nuclei_data, cells_data, counts_df = segment_microsam(
        data=illumination_corrected_data,
        dapi_index=params["dapi_index"],
        cyto_index=params["cyto_index"],
        model_type=params["microsam_model"],
        microsam_kwargs=dict(
            points_per_side=params.get("points_per_side", 32),
            points_per_batch=params.get("points_per_batch", 64),
            stability_score_thresh=params.get("stability_score_thresh", 0.75),
            pred_iou_thresh=params.get("pred_iou_thresh", 0.75)
        ),
        reconcile=params.get("reconcile", False),
        return_counts=params.get("return_counts", True),
        gpu=params.get("gpu", False),
    )
elif method == "stardist":
    # Segment cells using StarDist
    from lib.shared.segment_stardist import segment_stardist
    nuclei_data, cells_data, counts_df = segment_stardist(
        data=illumination_corrected_data,
        dapi_index=params["dapi_index"],
        cyto_index=params["cyto_index"],
        model_type=params["stardist_model"],
        stardist_kwargs=dict(
            prob_thresh=params["prob_thresh"],
            nms_thresh=params["nms_thresh"],
        ),
        reconcile=params.get("reconcile", False),
        return_counts=params.get("return_counts", True),
        gpu=params.get("gpu", False),
    )
else:
    raise ValueError(f"Unknown segmentation method: {method}. Choose one of: cellpose, microsam, stardist")

# Save segmented nuclei data
imwrite(snakemake.output[0], nuclei_data)
# Save segmented cells data
imwrite(snakemake.output[1], cells_data)
# Save counts data
counts_df.to_csv(snakemake.output[2], index=False, sep="\t")
