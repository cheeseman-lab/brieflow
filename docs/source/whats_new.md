# What's New in 1.5.0

Brieflow 1.5.0 is the release these docs describe.
The full list of changes, with the pull request behind each, is in the [1.5.0 release notes](https://github.com/cheeseman-lab/brieflow/releases/tag/v1.5.0).

## Main changes

- **marimo notebooks.** brieflow-analysis configures each phase in a marimo notebook, `0_preprocess.py` … `7_analyze.py`, with the values you set gathered in {term}`parameter cells <Parameter cell>`. `flow.sh <module>` replaces the numbered run scripts. See [Working with the Notebooks](notebook_editor_setup.md).
- **OME-Zarr output.** `all.image_format: zarr` writes {term}`OME-Zarr` plate stores; `tiff` keeps one file per tile. See [Output formats](3.running_modules.md#output-formats-tiff-and-zarr).
- **Merge.** The `positions` approach matches cells from stage positions; the `stitch` approach is removed.
- **Screen description.** `analysis/screen.yaml` describes the screen, and an optional {term}`per-well annotations <Per-well annotations>` table selects the wells to process and the groups to aggregate.
- **Raw data layout.** Inputs are expected one directory per SBS cycle and per phenotype round; see [Before You Screen](1.before_you_screen.md).
- **Aggregate and cluster.** Grouped aggregation, bootstrap control scopes, and a bootstrap that keeps no null distributions on disk.
- **Python 3.12**, and the small test run in both output formats as the pipeline's test.

## Upgrading from 1.4.x

- Start a new screen from the brieflow-analysis template and follow [Installation and Analysis Setup](2.installation_analysis_setup.md).
- An existing screen keeps running the brieflow commit its submodule pins. To move it to 1.5.0, update the submodule, rebuild the environment, and configure each phase again in the marimo notebooks: the config keys and defaults changed, so a 1.4 `config.yml` is not reused as is.
- The release notes list every renamed or removed config key.

## Upgrading a run started on 1.5.0

- Per-tile SBS `reads`, `cells` and `sbs_info` tables and phenotype `phenotype_cp` tables are now parquet files under `parquets/` instead of TSV files under `tsvs/`; see [Output formats](3.running_modules.md#output-formats-tiff-and-zarr).
- A run started on 1.5.0 does not reuse those TSVs. After the update, Snakemake re-runs the steps that write these tables (`call_reads`, `call_cells`, `extract_sbs_info`, `extract_phenotype_cp`) and the steps downstream of them; images and segmentation masks are reused. Delete the old copies in `sbs/tsvs/` and `phenotype/tsvs/` when the run finishes, since nothing reads them.
- Rows of `merge_final` can come out in a different order from run to run; sort by `plate`, `well`, `tile` and `cell_0` before comparing two runs.
