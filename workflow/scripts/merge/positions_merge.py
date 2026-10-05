import time

import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import read_parquet, write_parquet
from lib.merge.merge_utils import align_metadata
from lib.merge.positions_merge import filter_tile_metadata, positions_merge
from lib.merge.positions_overlay import (
    positions_image_qc,
    save_figures,
    summarize_image_qc,
    tile_image_paths,
)

for _param_name in ["threshold", "phenotype_dimensions", "sbs_dimensions"]:
    if getattr(snakemake.params, _param_name, None) is None:
        raise ValueError(f"Required config parameter '{_param_name}' is not set")

plate, well = snakemake.params.plate, snakemake.params.well

# Load tile metadata, one row per tile
phenotype_metadata = filter_tile_metadata(
    validate_dtypes(read_parquet(snakemake.input[0])),
    channel=snakemake.params.ph_metadata_channel,
)
sbs_metadata = filter_tile_metadata(
    validate_dtypes(read_parquet(snakemake.input[1])),
    cycle=snakemake.params.sbs_metadata_cycle,
    channel=snakemake.params.sbs_metadata_channel,
)

# Bring the two stage frames together when the screens were acquired on different scopes
alignment_params = {
    "flip_x": snakemake.params.alignment_flip_x,
    "flip_y": snakemake.params.alignment_flip_y,
    "rotate_90": snakemake.params.alignment_rotate_90,
}
if snakemake.params.metadata_align or any(alignment_params.values()):
    phenotype_metadata, sbs_metadata, _ = align_metadata(
        phenotype_metadata,
        sbs_metadata,
        x_col="x_pos",
        y_col="y_pos",
        **alignment_params,
    )

# Load cell centroids
phenotype_info = validate_dtypes(read_parquet(snakemake.input[2]))
sbs_info = validate_dtypes(read_parquet(snakemake.input[3]))

merge_data, merge_qc, placement = positions_merge(
    phenotype_info,
    sbs_info,
    phenotype_metadata,
    sbs_metadata,
    phenotype_dimensions=snakemake.params.phenotype_dimensions,
    sbs_dimensions=snakemake.params.sbs_dimensions,
    threshold=snakemake.params.threshold,
    flipud=snakemake.params.flipud,
    fliplr=snakemake.params.fliplr,
    rot90=snakemake.params.rot90,
    phenotype_pixel_size=snakemake.params.phenotype_pixel_size,
    sbs_pixel_size=snakemake.params.sbs_pixel_size,
)

# Bounded image readouts of the fitted placement: seams, phenotype in SBS space, mosaic
image_records = pd.DataFrame(
    columns=["kind", "tile_a", "tile_b", "residual_px", "colored_fraction"]
)
figures = {}
if placement is not None and snakemake.params.image_qc:
    start = time.time()
    templates = {
        "labels": {
            "phenotype": snakemake.params.phenotype_label_template,
            "sbs": snakemake.params.sbs_label_template,
        },
        "images": {
            "phenotype": snakemake.params.phenotype_image_template,
            "sbs": snakemake.params.sbs_image_template,
        },
    }
    paths = {
        kind: {
            name: tile_image_paths(
                template, placement[name]["tiles"].index, plate, well
            )
            for name, template in by_name.items()
        }
        for kind, by_name in templates.items()
    }
    image_records, figures = positions_image_qc(
        placement,
        paths["labels"],
        paths["images"],
        {
            "phenotype": snakemake.params.phenotype_dapi_index or 0,
            "sbs": snakemake.params.sbs_dapi_index or 0,
        },
        {
            "phenotype": phenotype_info["tile"].value_counts(),
            "sbs": sbs_info["tile"].value_counts(),
        },
    )
    for key, value in summarize_image_qc(image_records).items():
        merge_qc[key] = value
    merge_qc["image_qc_seconds"] = round(time.time() - start, 1)
    if merge_qc["image_qc_warning"].iloc[0] and merge_qc["status"].iloc[0] == "ok":
        merge_qc["status"] = "image_qc_warning"

merge_qc.insert(0, "well", well)
merge_qc.insert(0, "plate", plate)
print(merge_qc.T.to_string(header=False))
if merge_qc["status"].iloc[0] != "ok":
    print(
        f"WARNING: positions merge status is {merge_qc['status'].iloc[0]}; check the QC table"
    )

write_parquet(merge_data, snakemake.output[0])
merge_qc.to_csv(snakemake.output[1], sep="\t", index=False)
image_records.to_csv(snakemake.output[2], sep="\t", index=False)
save_figures(figures, snakemake.output[3:6])
