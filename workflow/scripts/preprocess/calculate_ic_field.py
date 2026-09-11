from lib.shared.illumination_correction import calculate_ic_field
from lib.shared.image_io import save_image

# Calculate IC field
ic_field = calculate_ic_field(
    snakemake.input,
    threading=snakemake.params.threading,
    sample_fraction=snakemake.params.sample_fraction,
    random_seed=snakemake.params.random_seed,
)

# Save IC field (supports both TIFF and Zarr based on output path)
save_image(ic_field, snakemake.output[0])
