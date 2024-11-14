# Brieflow

Extensible pipeline tool for processing optical pooled screens data.

We are actively moving code from [OpticalPooledScreens](https://github.com/cheeseman-lab/OpticalPooledScreens).
Please check back for updates! 


## Running Example Analyis

Brieflow is set up as a Snakemake workflow with user configuration between steps where necessary. 
Thus, a user must configure parameters between workflow steps with configuration notebooks.
This example analysis details the steps necessary for configuring parameters and running workflows.
While each step's workflow has its own Conda environment (compiled by Snakemake at runtime), the notebooks all share a configuration environment.

### Step 1: Set up workflow/configuration Conda environments

The workflows share a base environment (`brieflow_workflows`) and each have their own Conda environments compiled by Snakemake at runtime (in [workflow/envs](workflow/envs)).
All notebooks share a configuration environment (`brieflow_configuration`).

#### Step 1a: Set up Brieflow workflows environment

Use the following commands to set up the `brieflow_workflows` Conda environment:

```sh
# create brieflow_workflows conda environment
conda env create --file=brieflow_workflows_env.yml
# activate brieflow_workflows conda environment
conda activate brieflow_workflows
# set conda installation to use strict channel priorities
conda config --set channel_priority strict
```

#### Step 1a: Set up Brieflow configuration environment

Use the following commands to set up the `brieflow_configuration` Conda environment:

```sh
# create brieflow_configuration conda environment
conda env create --file=brieflow_configuration_env.yml
```

### Step 2: Run workflow steps

#### Step 2.0: Configure preprocess params

#### Step 2.1: Run preprocessing workflow

#### Step 2.2: Configure SBS process params

#### Step 2.3: Run SBS process workflow

***Note**: Use `brieflow_configuration` Conda environment for each configuration notebook.

### Run entire workflow steps

If the 

```sh
cd example_analysis
sh run_analysis.sh
```

*Note*: We do not include the data necessary for this example analysis in this repo as it is too large.


## Contribution Notes

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting code.

We use the following conventions:
- [One sentence per line](https://nick.groenen.me/notes/one-sentence-per-line/) convention for markdown files
- [Google format](format) for function docstrings
- [tsv](https://en.wikipedia.org/wiki/Tab-separated_values#:~:text=Tab%2Dseparated%20values%20(TSV),similar%20to%20comma%2Dseparated%20values.) file format for saving small dataframes that require easy readability
- [hdf5](https://www.hdfgroup.org/solutions/hdf5/) file format for saving large dataframes
- Data location information (well, tile, cycle, etc) + `__` + type of information (cell features, phenotype info, etc) + `.` + file type. 
Data is stored in its respective analysis directories. 
For example: `analysis_root/preprocess/ic_fields/sbs/A1_T1_C1__ic_field.tiff`
