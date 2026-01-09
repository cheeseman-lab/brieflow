from pathlib import Path
from tifffile import imread
from lib.shared.omezarr_writer import write_labels_omezarr

# Read input mask
label_data = imread(snakemake.input[0])

# Determine paths
output_path = Path(snakemake.output[0])
# Expected structure: .../image.ome.zarr/labels/label_name
image_zarr_path = output_path.parent.parent
label_name = output_path.name

# Write labels
write_labels_omezarr(
    label_data=label_data,
    out_path=str(image_zarr_path),
    label_name=label_name,
    axes=snakemake.params.axes,
)
