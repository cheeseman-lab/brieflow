from pathlib import Path

from lib.preprocess.preprocess import nd2_to_omezarr


params = getattr(snakemake, "params", {})  # noqa: F821 - defined by Snakemake

chunk_shape = tuple(int(value) for value in params.get("chunk_shape", (1, 512, 512)))
if len(chunk_shape) != 3:
    raise ValueError("chunk_shape must have three elements (C, Y, X).")

nd2_to_omezarr(
    snakemake.input,  # noqa: F821
    output_dir=Path(snakemake.output[0]),  # noqa: F821
    channel_order_flip=params.get("channel_order_flip", False),
    chunk_shape=chunk_shape,
    coarsening_factor=params.get("coarsening_factor", 2),
    max_levels=params.get("max_levels"),
    verbose=params.get("verbose", False),
)
