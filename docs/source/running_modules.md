# Running Modules

## Overview

Each large brieflow process is referred to as a "module".
This includes preprocess, SBS, phenotype, merge, aggregate, and cluster.
Each module has its own config notebook (in brieflow-analysis) and targets/rules file (in brieflow).
A configuration notebook is used to configure a module's parameters, which are then used in the targets/rules.
The main `Snakefile` (at `brieflow/workflow/Snakefile`) connects each of these modules, as shown below:


<details>
<summary>View Snakefile code</summary>

```python
include: "targets/preprocess.smk"
include: "rules/preprocess.smk"

if "sbs" in config and len(sbs_wildcard_combos) > 0:

    # Include target and rule files
    include: "targets/sbs.smk"
    include: "rules/sbs.smk"

if "phenotype" in config and len(phenotype_wildcard_combos) > 0:

    # Include target and rule files
    include: "targets/phenotype.smk"
    include: "rules/phenotype.smk"

if "merge" in config:
    MERGE_COMBO_FP = Path(config["merge"]["merge_combo_fp"])
    merge_wildcard_combos = pd.read_csv(MERGE_COMBO_FP, sep="\t")

    # Include target and rule files
    include: "targets/merge.smk"
    include: "rules/merge.smk"

if "aggregate" in config:
    AGGREGATE_COMBO_FP = Path(config["aggregate"]["aggregate_combo_fp"])
    aggregate_wildcard_combos = pd.read_csv(AGGREGATE_COMBO_FP, sep="\t")

    # Include target and rule files
    include: "targets/aggregate.smk"
    include: "rules/aggregate.smk"

if "cluster" in config:
    CLUSTER_COMBO_FP = Path(config["cluster"]["cluster_combo_fp"])
    cluster_wildcard_combos = pd.read_csv(CLUSTER_COMBO_FP, sep="\t")

    # Include target and rule files
    include: "targets/cluster.smk"
    include: "rules/cluster.smk"
```

</details>

## Steps

A usual run of a module looks like:
1) Run the respective notebook in `brieflow-analysis/analysis` to determine parameters, which are then dumped into the config file at `brieflow-analysis/analysis/config/config.yml`.
Ex: run `brieflow-analysis/analysis/0.configure_preprocess_params.ipynb`.
2) Test the run with a dry run with a local run script.
Ex: run `1.run_preprocessing.sh` script with the `-n` snakemake modifier to ensure a dry run.
3) Complete a full run with a slurm run scirpt.
Ex: run `1.run_preprocessing_slurm.sh` script.

## Example Video

TODO: Add video <>

## Notes

- A slurm run's log files are output to `brieflow-analysis/analysis/slurm/slurm_output/main`
- The preprocessing, SBS, and phenotype modules have special slurm files to optimize running by grouping rule jobs and splitting snakemake runs by plate.
- Because of how snakemake generates DAGs, it is usually helpful to restrict the rules loaded for very large runs. When analyzing a large screen, we recommend commenting out the uncessary targets/rules in `brieflow/workflow/Snakefile`. Ex: when running aggregate, only the merge targets and aggregate rules/targets are neccesssary, so we can comment out the other components as shown below.
<details>
<summary>View restricted Snakefile example</summary>

```python
# include: "targets/preprocess.smk"
# include: "rules/preprocess.smk"

# if "sbs" in config and len(sbs_wildcard_combos) > 0:

#     # Include target and rule files
#     include: "targets/sbs.smk"
#     include: "rules/sbs.smk"

# if "phenotype" in config and len(phenotype_wildcard_combos) > 0:

#     # Include target and rule files
#     include: "targets/phenotype.smk"
#     include: "rules/phenotype.smk"

if "merge" in config:
    MERGE_COMBO_FP = Path(config["merge"]["merge_combo_fp"])
    merge_wildcard_combos = pd.read_csv(MERGE_COMBO_FP, sep="\t")

    # Include target and rule files
    include: "targets/merge.smk"
    # include: "rules/merge.smk"

if "aggregate" in config:
    AGGREGATE_COMBO_FP = Path(config["aggregate"]["aggregate_combo_fp"])
    aggregate_wildcard_combos = pd.read_csv(AGGREGATE_COMBO_FP, sep="\t")

    # Include target and rule files
    include: "targets/aggregate.smk"
    include: "rules/aggregate.smk"

# if "cluster" in config:
#     CLUSTER_COMBO_FP = Path(config["cluster"]["cluster_combo_fp"])
#     cluster_wildcard_combos = pd.read_csv(CLUSTER_COMBO_FP, sep="\t")

#     # Include target and rule files
#     include: "targets/cluster.smk"
#     include: "rules/cluster.smk"
```
</details>
^ Make sure to do a dry run to make sure the correct jobs will be run (step 2)! 