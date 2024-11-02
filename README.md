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

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting code.

We use the following conventions:
- [One sentence per line](https://nick.groenen.me/notes/one-sentence-per-line/) convention for markdown files
- [Google format](format) for function docstrings
- [tsv](https://en.wikipedia.org/wiki/Tab-separated_values#:~:text=Tab%2Dseparated%20values%20(TSV),similar%20to%20comma%2Dseparated%20values.) file format for saving small dataframes that require easy readability
- [hdf5](https://www.hdfgroup.org/solutions/hdf5/) file format for saving large dataframes
- Data location information (well, tile, cycle, etc) + `__` + type of information (cell features, phenotype info, etc) + `.` + file type. 
Data is stored in its respective analysis directories. 
For example: `analysis_root/preprocess/ic_fields/sbs/A1_T1_C1__ic_field.tiff`
