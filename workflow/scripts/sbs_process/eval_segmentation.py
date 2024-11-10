from lib.sbs_process.eval_segmentation import get_segmentation_overview


# Get the segmentation overview
segmentation_overview = get_segmentation_overview(
    snakemake.input.segmentation_stats_paths
)
# Save the segmentation overview
segmentation_overview.to_csv(snakemake.output[0], sep="\t", index=False)
