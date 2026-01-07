# Brieflow

Extensible tool for processing optical pooled screens data.

**Notes**: 
- Read [brieflow.readthedocs.io](https://brieflow.readthedocs.io/) before starting to get a good grasp of brieflow and brieflow-analysis!
- Join the brieflow [Discord](https://discord.gg/yrEh6GP8JJ) to ask questions, share ideas, and get help from other users and developers.

## Definitions

Terms mentioned throughout the code and documentation include:
- **Brieflow library**: Code in [workflow/lib](workflow/lib) used to perform Brieflow processing.
Used with Snakemake to run Brieflow steps.
- **Module**: Used synonymously to refer to larger steps of the Brieflow pipeline.
Example modules: `preprocessing`, `sbs`, `phenotype`.
- **OME-Zarr**: A cloud-optimized, chunked, and compressed image data format used for storing large multidimensional images. All major image checkpoints are now converted to OME-Zarr v0.4 with the `.zarr` file extension.


## Project Structure

Brieflow is built on top of [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html#snakemake).
We follow the [Snakemake structure guidelines](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html) with some exceptions.
The Brieflow project structure is as follows:

```
workflow/
├── lib/ - Brieflow library code used for performing Brieflow processing. Organized into module-specific, shared, and external code.
├── rules/ - Snakemake rule files for each module. Used to organize processses within each module with inputs, outputs, parameters, and script file location.
├── scripts/ - Python script files for processes called by modules. Organized into module-specific and shared code.
├── targets/ - Snakemake files used to define inputs and their mappings for each module. 
└── Snakefile - Main Snakefile used to call modules.
```

Brieflow runs as follows:
- A user configure parameters in Jupyter notebooks to use the Brieflow library code correctly for their data.
- A user runs the main Snakefile with bash scripts (locally or on an HPC).
- The main Snakefile calls module-specific snakemake files with rules for each process.
- Each process rule calls a script.
- Scripts use the Brieflow library code to transform the input files defined in targets into the output files defined in targets.

## Brieflow Setup

Brieflow is usually loaded as a submodule within a Brieflow analysis.
Each Brieflow analysis corresponds to one optical pooled screen.
Look at [brieflow-analysis](https://github.com/cheeseman-lab/brieflow-analysis/) for instructions on using Brieflow as part of a Brieflow analysis.

### Conda Environment

Use the following commands to set up the brieflow Conda environment (~10 min):

```sh
# create and activate brieflow_SCREEN_NAME conda environment
# NOTE: replace brieflow_SCREEN_NAME with the name of your screen to ensure a screen-specific installation
# using this screen-specific installation will refer to library code in ./brieflow/workflow/lib
conda create -n brieflow_SCREEN_NAME -c conda-forge python=3.11 uv pip -y
conda activate brieflow_SCREEN_NAME
# install external packages
uv pip install -r pyproject.toml
# install editable version of brieflow
uv pip install -e .
# install conda-only packages
conda install -c conda-forge micro_sam -y # skip if not using micro-sam for segmentation
```

**Notes:**
- We recommend a screen-specific installation because changes to this particular `./brieflow/workflow/lib` code will live within this specific installation of brieflow, and an explicit name helps keep track of different brieflow installations.
One could also install one version of brieflow that is used across brieflow-analysis repositories.
- For a rule-specific package consider creating a separate conda environment file and using it for the particular rule as described in the [Snakemake integrated package management notes](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html#integrated-package-management).

### Tests

Run the following commands to ensure your Brieflow is set up correctly
This process takes about 14 minutes on our machine.

```sh
# activate brieflow env
conda activate brieflow_SCREEN_NAME
# enter small test analysis dir
cd brieflow/tests/small_test_analysis
# set up small test analysis
python small_test_analysis_setup.py
# run brieflow
sh run_brieflow.sh
# run tests
cd ../../
pytest
```

### HPC Integrations

The steps for running workflows currently include local and Slurm integration.
To use the Slurm integration for Brieflow configure the Slurm resources in [analysis/slurm/config.yaml](analysis/slurm/config.yaml).
The `slurm_partition` and `slurm_account` in `default-resources` need to be configured while the other resource requirements have suggested values.
These can be adjusted as necessary.

**Note**: Other Snakemake HPC integrations can be found in the [Snakemake plugin catalog](https://snakemake.github.io/snakemake-plugin-catalog/index.html#snakemake-plugin-catalog).
Only the `slurm` plugin has been tested. It is important to understand that these plugins assume that the Snakemake scheduler will operate on the head HPC node, and *only the individual jobs* are submitted to the various nodes available to the HPC. Therefore, the Snakefile should be run through bash on the head node (with `slurm` or other HPC configurations). We recommend starting a tmux session for this, especially for larger jobs.

## Example Analysis

The [denali-analysis](https://github.com/cheeseman-lab/denali-analysis) details an example Brieflow run.
We do not include the data necessary for this example analysis in this repo as it is too large.

## Contribution Notes

- Brieflow is still actively under development and we welcome community use/development. 
- File a [GitHub issue](https://github.com/cheeseman-lab/brieflow/issues) to share comments and issues.
- File a GitHub PR to contribute to Brieflow as detailed in the [pull request template](.github/pull_request_template.md).
Read about how to [contribute to a project](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) to understand forks, branches, and PRs.

### Dev Tools

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting code.

### Conventions

We use the following conventions:
- [One sentence per line](https://nick.groenen.me/notes/one-sentence-per-line/) convention for markdown files
- [Google format](format) for function docstrings
- [tsv](https://en.wikipedia.org/wiki/Tab-separated_values#:~:text=Tab%2Dseparated%20values%20(TSV),similar%20to%20comma%2Dseparated%20values.) file format for saving small dataframes that require easy readability
- [parquet](https://www.databricks.com/glossary/what-is-parquet) file format for saving large dataframes
- Data location information (well, tile, cycle, etc) + `__` + type of information (cell features, phenotype info, etc) + `.` + file type. 
Data is stored in its respective analysis directories. 
For example: `brieflow_output/preprocess/metadata/phenotype/P-1_W-A2_T-571__metadata.tsv`
- We use [semantic versioning](https://semver.org/) for brieflow and brieflow-analysis versions.
These two repositories should always have the same version.
Small version changes are only tracked in the [pyproject.toml](pyproject.toml) file of brieflow (not in brieflow-analysis).