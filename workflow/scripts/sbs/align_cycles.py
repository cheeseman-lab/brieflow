import pandas as pd
from tifffile import imread, imwrite

from lib.sbs.align_cycles import align_cycles

# load image data
image_data = [imread(file_path) for file_path in snakemake.input]

# align cycles
aligned_data, metrics = align_cycles(
    image_data,
    channel_order=snakemake.params.channel_names,
    method=snakemake.params.method,
    upsample_factor=snakemake.params.upsample_factor,
    window=snakemake.params.window,
    skip_cycles=snakemake.params.skip_cycles_indices,
    manual_background_cycle=snakemake.params.manual_background_cycle_index,
    manual_channel_mapping=snakemake.params.manual_channel_mapping,
    return_metrics=True,
)

# Save the aligned data as a .tiff file
imwrite(snakemake.output[0], aligned_data)

# Save alignment metrics to TSV (one row per tile)
metrics_df = pd.DataFrame(
    [
        {
            "plate": snakemake.wildcards.plate,
            "well": snakemake.wildcards.well,
            "tile": snakemake.wildcards.tile,
            **metrics,
        }
    ]
)
metrics_df.to_csv(snakemake.output[1], index=False, sep="\t")
