import inspect
from pathlib import Path

from lib.preprocess.preprocess import nd2_to_omezarr
from lib.shared.omezarr_utils import ensure_omero_channel_colors


params = getattr(snakemake, "params", {})  # noqa: F821 - defined by Snakemake

chunk_shape = tuple(int(value) for value in params.get("chunk_shape", (1, 1024, 1024)))
if len(chunk_shape) not in (3, 4):
    raise ValueError(
        "chunk_shape must have three (C, Y, X) or four (C, Z, Y, X) elements."
    )

output_dir = Path(snakemake.output[0])  # noqa: F821

supports_compressor = "compressor" in inspect.signature(nd2_to_omezarr).parameters
additional_kwargs = {}
if supports_compressor and params.get("compressor") is not None:
    additional_kwargs["compressor"] = params.get("compressor")

result_path = nd2_to_omezarr(
    snakemake.input,  # noqa: F821
    output_dir=output_dir,
    channel_order_flip=params.get("channel_order_flip", False),
    chunk_shape=chunk_shape,
    coarsening_factor=params.get("coarsening_factor", 2),
    max_levels=params.get("max_levels"),
    verbose=params.get("verbose", False),
    **additional_kwargs,
)

# Post-process OMERO metadata to ensure meaningful channel labels and colors.
# - result_path / output_dir points to the root of the written Zarr store.
# - channel_labels comes from the Snakemake rule (e.g. config['sbs']['channel_names']).
channel_labels = params.get("channel_labels")
try:
    ensure_omero_channel_colors(result_path, channel_labels=channel_labels)
except Exception as exc:  # Metadata issues should not break the core conversion
    print(f"Warning: failed to update OMERO channel metadata for {result_path}: {exc}")
