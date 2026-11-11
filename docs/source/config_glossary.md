# Config Glossary

The brieflow config holds all of the parameters used for a brieflow run.
Each notebook is used to configure the parameter variables, which are then saved to the `config.yml`.
Each analysis requires a specific `config.yml` and the associated files (pool dataframe, cell classification model, etc).
A complete example is the small test's [tests/small_test_analysis/config/config.yml](https://github.com/cheeseman-lab/brieflow/blob/main/tests/small_test_analysis/config/config.yml).
Every parameter is explained in the notebook that sets it; [Notebook Parameters](notebook_parameters.md) maps the important notebook variables to their config keys, and we comment on a few keys here as well:
- `preprocess:sbs_samples_fp`/`preprocess:phenotype_samples_fp`: Path to dataframes with one entry for an SBS/phenotype file's path and the associated metadata (plate, well, tile, etc).
- `sbs:df_barcode_library_fp`: Path to the standardized barcode library that the SBS notebook builds from the raw design table (`DF_DESIGN_FP`): one row per guide with its barcode and target gene.
- `*_combo_fp`: Path to dataframe with wildcards for file processing in a particular module.
Each combination usually corresponds to a one process that needs to be done with Snakemake.
For example, each plate, well, tile combination in `phenotype_combo_fp` corresponds to one raw file conversion during preprocessing.
- `all:image_format`: `tiff` or `zarr`, the output layout for the whole run; absent means `zarr`. See [Output formats](3.running_modules.md#output-formats-tiff-and-zarr).
- `all:zarr_max_levels`: set by `ZARR_MAX_LEVELS` in the preprocess notebook. Number of resolution levels written per converted image, `1` (the default when absent) meaning no pyramid. Each extra level costs disk and conversion time: on the small test, going from 1 to 5 levels added ~16% to the bytes per tile. Only read when `image_format: zarr`.
- `all:transparent_figures`: set by `TRANSPARENT_FIGURES` in the preprocess notebook. Save the PNG plots in `eval/`, `cluster/` and the classifier outputs with a transparent background (for example to import into Illustrator) instead of white. Default `false` (also when absent).
- `merge:positions_image_qc`: with `merge:approach: positions`, also write per-well image readouts to `merge/eval/` (tile-overlap and phenotype-in-SBS overlays, a nuclei mosaic, their residuals and the fit table). Default `false`; the same overlays are in the merge notebook.
- `sbs:combinatorial:ambiguous_call`: set by `AMBIGUOUS_CALL` in the SBS notebook. With the combinatorial `frac` caller, `N` calls a cycle whose dye state is equally near two or more bases `N` (quality 0, left out of `q_min`); `nearest`, the behavior when absent, gives it to the first tied base in code order.
- `sbs:n_wildcard`: set by `N_WILDCARD` in the SBS notebook. `call_cells` maps a barcode with up to this many `N` cycles to the one library barcode that agrees at every other cycle. Default `0` (off) when absent.
- `sbs:recomb_max_n`: set by `RECOMB_MAX_N` in the SBS notebook (`barcode_type: multi`). Up to this many `N` cycles in the recombination barcode are skipped when it is compared with the expected one; with more, the recombination check is left undetermined. Default `0` when absent.
- `sbs:use_align_within_cycle`: set by `USE_ALIGN_WITHIN_CYCLE` in the SBS notebook. Whether `align_sbs` registers the base channels of each cycle to each other before aligning cycles. Default `true` when absent; set `false` to skip the step on data whose channels are already registered within each cycle.
- `phenotype:texture_intensity_range`: `[low, high]` in raw intensity units. Texture (Haralick) features map this range to 0–255 before measuring; absent keeps the conversion over the full integer range, which leaves dim 16-bit channels with a few grey levels and near-zero texture. Use one range for the whole screen (for example from intensity percentiles of a few tiles) so texture stays comparable across cells and wells. Default absent.
