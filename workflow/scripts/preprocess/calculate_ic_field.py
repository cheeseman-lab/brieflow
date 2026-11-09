from lib.shared.illumination_correction import calculate_ic_field
from lib.shared.image_io import save_image

# zarr SBS tiles hold every cycle; read this job's cycle from each tile image
image_paths = getattr(snakemake.params, "image_paths", None)

# Calculate IC field
ic_field = calculate_ic_field(
    image_paths or snakemake.input,
    threading=snakemake.params.threading,
    sample_fraction=snakemake.params.sample_fraction,
    n_jobs=-1,
    cycle=getattr(snakemake.params, "cycle", None) if image_paths else None,
)

# Save IC field (supports both TIFF and Zarr based on output path)
save_image(ic_field, snakemake.output[0])
