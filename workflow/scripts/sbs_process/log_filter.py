from lib.sbs_process.log_filter import log_filter
from skimage.io import imsave

# apply log filter
log_filtered = log_filter(
    data=snakemake.input[0],
    skip_index=snakemake.params.skip_index,
)

# Save the aligned data as a .tiff file
imsave(snakemake.output[0], log_filtered)
