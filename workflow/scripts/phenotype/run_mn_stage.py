"""Snakemake adapter; all image algorithms live in lib/phenotype/mn_pipeline.py."""

from lib.phenotype.mn_pipeline import execute_stage

execute_stage(
    snakemake.params.stage,
    dict(snakemake.input.items()),
    dict(snakemake.output.items()),
    snakemake.params.settings,
    {key: str(getattr(snakemake.wildcards, key)) for key in ["plate", "well", "tile"]},
)
