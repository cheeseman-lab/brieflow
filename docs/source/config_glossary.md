# Config Glossary

The brieflow config holds all of the parameters used for a brieflow run.
Each notebook is used to configure the parameter variables, which are then saved to the `config.yml`.
Each analysis requires a specific `config.yml` and the associated files (pool dataframe, cell classification model, etc).
A complete example is the small test's [tests/small_test_analysis/config/config.yml](https://github.com/cheeseman-lab/brieflow/blob/main/tests/small_test_analysis/config/config.yml).
Every parameter is explained in the notebook that sets it; [Notebook Parameters](notebook_parameters.md) maps the important notebook variables to their config keys, and we comment on a few keys here as well:
- `preprocess:sbs_samples_fp`/`preprocess:phenotype_samples_fp`: Path to dataframes with one entry for an SBS/phenotype file's path and the associated metadata (plate, well, tile, etc).
- `sbs:df_design_path`: Path to dataframe with SBS pool design information regarding gene, sgRNA, oligo, etc.
- `*_combo_fp`: Path to dataframe with wildcards for file processing in a particular module.
Each combination usually corresponds to a one process that needs to be done with Snakemake.
For example, each plate, well, tile combination in `phenotype_combo_fp` corresponds to one raw file conversion during preprocessing.
- `all:image_format`: `tiff` or `zarr`, the output layout for the whole run; absent means `tiff`. See [Output formats](3.running_modules.md#output-formats-tiff-and-zarr).
