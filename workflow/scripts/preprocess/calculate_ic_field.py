from pathlib import Path
import zarr
from tifffile import imwrite

from lib.shared.illumination_correction import calculate_ic_field

# Calculate IC field
ic_field = calculate_ic_field(
    snakemake.input,
    threading=snakemake.params.threading,
    sample_fraction=snakemake.params.sample_fraction,
)

# Save IC field (supports both TIFF and Zarr based on output path)
output_path = Path(snakemake.output[0])

if output_path.suffix.lower() == '.zarr':
    # Save as standard Zarr array (not OME-Zarr multiscale)
    # Create Zarr group
    root = zarr.open_group(str(output_path), mode='w')
    
    # Determine optimal chunk shape
    if ic_field.ndim == 2:
        chunk_y = min(1024, ic_field.shape[0])
        chunk_x = min(1024, ic_field.shape[1])
        chunks = (chunk_y, chunk_x)
    elif ic_field.ndim == 3:
        chunk_y = min(1024, ic_field.shape[1])
        chunk_x = min(1024, ic_field.shape[2])
        chunks = (ic_field.shape[0], chunk_y, chunk_x)
    else:
        chunks = None
    
    # Configure compression
    try:
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    except Exception:
        compressor = None
    
    # Create dataset
    try:
        # Zarr v3 API
        zarr_array = root.create_array(
            '0',
            shape=ic_field.shape,
            dtype=ic_field.dtype,
            chunks=chunks,
            compressor=compressor if compressor else None,
            overwrite=True,
        )
    except AttributeError:
        # Zarr v2 API fallback
        zarr_array = root.create_dataset(
            '0',
            shape=ic_field.shape,
            dtype=ic_field.dtype,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )
    
    # Write data
    zarr_array[:] = ic_field
    
    # Store metadata
    zarr_array.attrs.update({
        'format': 'standard_zarr',
        'purpose': 'illumination_correction',
    })
else:
    # Save as TIFF
    imwrite(str(output_path), ic_field)
