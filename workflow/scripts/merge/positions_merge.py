from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import read_parquet, write_parquet
from lib.merge.positions_merge import filter_tile_metadata
from lib.merge.positions_overlay import positions_merge_well, save_figures

for _param_name in ["threshold", "phenotype_dimensions", "sbs_dimensions"]:
    if getattr(snakemake.params, _param_name, None) is None:
        raise ValueError(f"Required config parameter '{_param_name}' is not set")

# Load tile metadata, one row per tile, and cell centroids
phenotype_metadata = filter_tile_metadata(
    validate_dtypes(read_parquet(snakemake.input[0])),
    channel=snakemake.params.ph_metadata_channel,
)
sbs_metadata = filter_tile_metadata(
    validate_dtypes(read_parquet(snakemake.input[1])),
    cycle=snakemake.params.sbs_metadata_cycle,
    channel=snakemake.params.sbs_metadata_channel,
)
phenotype_info = validate_dtypes(read_parquet(snakemake.input[2]))
sbs_info = validate_dtypes(read_parquet(snakemake.input[3]))

templates = None
if snakemake.params.image_qc:
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

merge_data, merge_qc, _, image_records, figures = positions_merge_well(
    phenotype_info,
    sbs_info,
    phenotype_metadata,
    sbs_metadata,
    plate=snakemake.params.plate,
    well=snakemake.params.well,
    phenotype_dimensions=snakemake.params.phenotype_dimensions,
    sbs_dimensions=snakemake.params.sbs_dimensions,
    threshold=snakemake.params.threshold,
    flipud=snakemake.params.flipud,
    fliplr=snakemake.params.fliplr,
    rot90=snakemake.params.rot90,
    phenotype_pixel_size=snakemake.params.phenotype_pixel_size,
    sbs_pixel_size=snakemake.params.sbs_pixel_size,
    alignment={
        "metadata_align": snakemake.params.metadata_align,
        "flip_x": snakemake.params.alignment_flip_x,
        "flip_y": snakemake.params.alignment_flip_y,
        "rotate_90": snakemake.params.alignment_rotate_90,
    },
    templates=templates,
    dapi_index={
        "phenotype": snakemake.params.phenotype_dapi_index,
        "sbs": snakemake.params.sbs_dapi_index,
    },
)

# Fit status goes to the log; eval_merge evaluates the merge as for the fast approach
print(merge_qc.T.to_string(header=False))
if merge_qc["status"].iloc[0] != "ok":
    print(f"WARNING: positions merge status is {merge_qc['status'].iloc[0]}")

write_parquet(merge_data, snakemake.output[0])
if templates is not None:
    merge_qc.to_csv(snakemake.output[1], sep="\t", index=False)
    image_records.to_csv(snakemake.output[2], sep="\t", index=False)
    save_figures(figures, snakemake.output[3:6])
