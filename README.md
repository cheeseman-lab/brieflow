# Brieflow

Extensible tool for processing optical pooled screens data.

## Definitions

Terms mentioned throughout the code and documentation include:
- **Brieflow library**: Code in [workflow/lib](workflow/lib) used to perform Brieflow processing.
Used with Snakemake to run Brieflow steps.
- **Module**: Used synonymously to refer to larger steps of the Brieflow pipeline.
Example modules: `preprocessing`, `sbs`, `phenotype`.
- **Process**: Refers to a smaller step within a module.
Processes use scripts and Brieflow library code to complete tasks.
Example processes in the preprocessing module: `extract_metadata_sbs`, `convert_sbs`, `calculate_ic_sbs`.


## Project Structure

Brieflow is built on top of [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html#snakemake).
We follow the [Snakemake structure guidelines](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html) with some exceptions.
The Brieflow project structure is as follows:

```
workflow/
├── envs/ - Environment YAML files that describe dependencies for different modules.
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

## Running Brieflow

Brieflow is usually loaded as a submodule within a Brieflow analysis.
Each Brieflow analysis corresponds to one optical pooled screen.
Look at [brieflow-analysis](https://github.com/cheeseman-lab/brieflow-analysis/) for instructions on using Brieflow as part of a Brieflow analysis.

### Set up workflow/configuration Conda environments

**Configuring and running Brieflow requires two separate environments!**

The modules share a base environment (`brieflow_workflows`) and each have their own Conda environments compiled by Snakemake at runtime (in [workflow/envs](workflow/envs)).
All notebooks share a configuration environment (`brieflow_configuration`).

**Note:** If large changes to Brieflow code are expected for a particular screen analysis, we recommend changing the names of the workflow/configuration environments to be screen-specific so development of this code does not affect other Brieflow runs.
Change the name of the workflow and configuration environments in [brieflow_workflows_env.yml](brieflow_workflows_env.yml) and [brieflow_configuration.yml](brieflow_configuration.yml).

#### Set up Brieflow workflows environment

Use the following commands to set up the `brieflow_workflows` Conda environment:

```sh
# create brieflow_workflows conda environment
conda env create --file=brieflow_workflows_env.yml
# activate brieflow_workflows conda environment
conda activate brieflow_workflows
# set conda installation to use strict channel priorities
conda config --set channel_priority strict
```

#### Set up Brieflow configuration environment

Use the following commands to set up the `brieflow_configuration` Conda environment:

```sh
# create brieflow_configuration conda environment
conda env create --file=brieflow_configuration_env.yml
```

### HPC Integrations

The steps for running workflows currently include local and Slurm integration.
To use the Slurm integration for Brieflow configure the Slurm resources in [analysis/slurm/config.yaml](analysis/slurm/config.yaml).
The `slurm_partition` and `slurm_account` in `default-resources` need to be configured while the other resource requirements have suggested values.
These can be adjusted as necessary.

**Note**: Other Snakemake HPC integrations can be found in the [Snakemake plugin catalog](https://snakemake.github.io/snakemake-plugin-catalog/index.html#snakemake-plugin-catalog).
Only the `slurm` plugin has been tested. It is important to understand that these plugins assume that the Snakemake scheduler will operate on the head HPC node, and *only the individual jobs* are submitted to the various nodes available to the HPC. Therefore, the Snakefile should be run through bash on the head node (with `slurm` or other HPC configurations). We recommend starting a tmux session for this, especially for larger jobs.

### Example Analysis

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
For example: `analysis_root/preprocess/metadata/phenotype/P-1_W-A2_T-571__metadata.tsv`