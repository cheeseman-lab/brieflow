# Brieflow

Extensible pipeline tool for processing optical pooled screens data.

We are actively moving code from [OpticalPooledScreens](https://github.com/cheeseman-lab/OpticalPooledScreens).
Please check back for updates! 

## Definitions

Terms mentioned throughout the code and documentation include:
- **Brieflow library**: Code in [workflow/lib](workflow/lib) used to perform Brieflow processing.
Used with Snakemake to run Brieflow steps.
- **Process/Workflow**: Used synonymously to refer to larger steps of the Brieflow pipeline.
Example processes/workflows: `preprocessing`, `sbs_process`, `phenotype_process`
- **Sub-process**: Refers to a smaller step within a process.
Sub-processes use scripts and Brieflow library code to complete tasks.
Example sub-processes in the preprocessing process: `extract_metadata_sbs`, `convert_sbs`, `calculate_ic_sbs`.


## Project Structure

Brieflow is built on top of [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html#snakemake).
We follow the [Snakemake structure guidelines](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html) with some exceptions.
The Brieflow project structure is as follows:

```
workflow/
├── envs/ - Environment YAML files that describe dependencies for different workflows.
├── lib/ - Brieflow library code used for performing Brieflow processing. 
├── rules/ - Snakemake rule files for each process. Used to organize sub-processses within each process with inputs, outputs, parameters, and script file location.
├── scripts/ - Python script files for sub-processes. Called by processes.
├── targets/ - Snakemake files used to define inputs and their mappings for each process.
└── Snakefile - Main Snakefile used to call processes.
```


## Running An Analyis

Brieflow is set up as a Snakemake workflow with user configuration between steps where necessary. 
Thus, a user must configure parameters between workflow steps with configuration notebooks.
While each step's workflow has its own Conda environment (compiled by Snakemake at runtime), the notebooks all share a configuration environment.

See the steps below to set up the workflow/configuration environments and run your own analysis with Brieflow.
**Note**: The example analysis shows an analyis done with a very small subset of OPS data.

### Set up workflow/configuration Conda environments

The workflows share a base environment (`brieflow_workflows`) and each have their own Conda environments compiled by Snakemake at runtime (in [workflow/envs](workflow/envs)).
All notebooks share a configuration environment (`brieflow_configuration`).

*Note*: We do not include the data necessary for this example analysis in this repo as it is too large.

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

### Analysis Steps

Follow the instructions below to configure paramers and run wokflows.
All of these steps are done in the analysis folder.
Use the following command to enter this folder:
`cd analysis/`.

#### Step 0: Configure preprocess params

Follow the steps in [0.configure_preprocess_params.ipynb](analysis/0.configure_preprocess_params.ipynb) to configure preprocess params.

#### Step 1: Run preprocessing workflow

```sh
conda activate brieflow_workflows
sh 1.run_preprocessing.sh
```

#### Step 2: Configure SBS process params

Follow the steps in [3.configure_phenotype_process_params.ipynb](analysis/3.configure_phenotype_process_params.ipynb) to configure phenotype process params.

#### Step 3: Configure phenotype process params

Follow the steps in [2.configure_sbs_process_params.ipynb](analysis/2.configure_sbs_process_params.ipynb) to configure SBS process params.

#### Step 4: Run SBS/phenotype process workflow

```sh
conda activate brieflow_workflows
sh 4.run_sbs_phenotype_processes.sh
```

***Note**: Use `brieflow_configuration` Conda environment for each configuration notebook.

### Run Entire Analysis

If all parameter configurations are known for the entire Brieflow pipeline, it is possible to run the entire pipeline with the following:

```sh
cd analysis/
sh run_entire_analysis.sh
```

### Example Analysis

The [example analysis](example_analysis) details an example Brieflow run with a small testing set of OPS data.


## Contribution Notes

### Dev Tools

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting code.

### Conventions

We use the following conventions:
- [One sentence per line](https://nick.groenen.me/notes/one-sentence-per-line/) convention for markdown files
- [Google format](format) for function docstrings
- [tsv](https://en.wikipedia.org/wiki/Tab-separated_values#:~:text=Tab%2Dseparated%20values%20(TSV),similar%20to%20comma%2Dseparated%20values.) file format for saving small dataframes that require easy readability
- [hdf5](https://www.hdfgroup.org/solutions/hdf5/) file format for saving large dataframes
- Data location information (well, tile, cycle, etc) + `__` + type of information (cell features, phenotype info, etc) + `.` + file type. 
Data is stored in its respective analysis directories. 
For example: `analysis_root/preprocess/ic_fields/sbs/A1_T1_C1__ic_field.tiff`
