"""Convert ND2 files to standard Zarr arrays (not OME-Zarr multiscale).

This creates simple Zarr arrays with the same structure as TIFF outputs, suitable for downstream processing.
"""

import zarr
from lib.preprocess.preprocess import convert_to_array, get_data_config

# Get data configuration from rule name
rule_name = snakemake.rule
image_type = "sbs" if "sbs" in rule_name else "phenotype"

data_config = get_data_config(
    image_type, {"preprocess": snakemake.config.get("preprocess", {})}
)

# Load and process image (same as TIFF conversion)
print(f"Loading {len(snakemake.input)} input files")
image_data = convert_to_array(
    snakemake.input,
    data_format=data_config["data_format"],
    data_organization=data_config["image_data_organization"],
    position=snakemake.params.tile
    if data_config["image_data_organization"] == "well"
    else None,
    channel_order_flip=data_config["channel_order_flip"],
    n_z_planes=data_config.get("n_z_planes"),
    verbose=False,
)
print(f"Loaded shape: {image_data.shape}, dtype: {image_data.dtype}")

# Determine optimal chunk shape based on data dimensions
if image_data.ndim == 3:
    # (C, Y, X) - typical after max projection
    # Keep all channels together, chunk spatially
    chunk_y = min(1024, image_data.shape[1])
    chunk_x = min(1024, image_data.shape[2])
    chunks = (image_data.shape[0], chunk_y, chunk_x)
elif image_data.ndim == 4:
    # (C, Z, Y, X) - if Z was preserved
    chunk_z = 1
    chunk_y = min(1024, image_data.shape[2])
    chunk_x = min(1024, image_data.shape[3])
    chunks = (image_data.shape[0], chunk_z, chunk_y, chunk_x)
else:
    chunks = None

print(f"Creating Zarr array at: {snakemake.output[0]}")
print(f"Chunk shape: {chunks}")

# Create Zarr group (compatible with zarr v2 and v3)
root = zarr.open_group(snakemake.output[0], mode='w')

# Configure compression (zstd is fast and has good compression)
try:
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
except Exception:
    # Fallback if blosc not available
    compressor = None
    print("Warning: Blosc compression not available, using default")

# Create dataset (use create_array for zarr v3 compatibility)
try:
    # Zarr v3 API
    zarr_array = root.create_array(
        '0',
        shape=image_data.shape,
        dtype=image_data.dtype,
        chunks=chunks,
        compressor=compressor if compressor else None,
        overwrite=True,
    )
except AttributeError:
    # Zarr v2 API fallback
    zarr_array = root.create_dataset(
        '0',
        shape=image_data.shape,
        dtype=image_data.dtype,
        chunks=chunks,
        compressor=compressor,
        overwrite=True,
    )

# Write data
print("Writing data to Zarr...")
zarr_array[:] = image_data

# Store metadata for compatibility
zarr_array.attrs.update({
    'shape': list(image_data.shape),
    'dtype': str(image_data.dtype),
    'source': 'brieflow_preprocess',
    'format': 'standard_zarr',  # Not OME-Zarr multiscale
    'chunks': list(chunks),
})

# Add root-level metadata
root.attrs.update({
    'version': '1.0',
    'format': 'standard_zarr',
})

print(f"   Standard Zarr array created successfully")
print(f"   Shape: {image_data.shape}")
print(f"   Dtype: {image_data.dtype}")
print(f"   Chunks: {chunks}")
print(f"   Size: {image_data.nbytes / 1024**2:.1f} MB")
