# BrieFlow

Extensible pipeline tool for processing optical pooled screens data.

We are actively moving code from [OpticalPooledScreens](https://github.com/cheeseman-lab/OpticalPooledScreens).
Please check back for updates! 


## Running Example Analyis

### Step 1: Create and activate Conda environment

Use the following commands to create and activate the `brieflow` conda environment:

```sh
conda env create --file=environment.yml
conda activate brieflow
```

### Step 2: Run example analysis

Use the following commands to run the example analysis:

```sh
cd example_analysis
sh run_analysis.sh
```

*Note*: We do not include the data necessary for this example analysis in this repo as it is too large.


## Contribution Notes

- We use the [one sentence per line](https://nick.groenen.me/notes/one-sentence-per-line/) convention for markdown files.
- We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting code.
- We use the [Google format](format) for function docstrings.
