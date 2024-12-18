from tifffile import imread, imwrite

from lib.shared.segment_cellpose import segment_cellpose

# load illumination corrected data
illumination_corrected_data = imread(snakemake.input[0])

# segment cells using cellpose
nuclei_data, cells_data, counts_df = segment_cellpose(
    data=illumination_corrected_data,
    dapi_index=snakemake.params.dapi_index,
    cyto_index=snakemake.params.cyto_index,
    nuclei_diameter=snakemake.params.nuclei_diameter,
    cell_diameter=snakemake.params.cell_diameter,
    cyto_model=snakemake.params.cyto_model,
    cellpose_kwargs=dict(
        flow_threshold=snakemake.params.flow_threshold,
        cellprob_threshold=snakemake.params.cellprob_threshold,
    ),
    return_counts=snakemake.params.return_counts,
    gpu=snakemake.params.gpu,
)

# save segmented nuclei data
imwrite(snakemake.output[0], nuclei_data)

# save segmented cells data
imwrite(snakemake.output[1], cells_data)

# save counts data
counts_df.to_csv(snakemake.output[2], index=False, sep="\t")
