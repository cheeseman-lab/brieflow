# Moving from the Jupyter Notebooks

Screens set up from the brieflow-analysis `main` branch use Jupyter notebooks and one pair of run scripts per module.
The `marimo` branch replaces both with marimo notebooks and a single `flow.sh`, and pins the brieflow line that can write OME-Zarr output.
The pipeline steps and most parameter names are the same; this page lists what changes for an existing screen.

## What maps to what

| Jupyter line (`main`) | marimo line (`marimo`) |
|---|---|
| `0.configure_preprocess_params.ipynb` | `0_preprocess.py` |
| `1.run_preprocessing.sh`, `1.run_preprocessing_slurm.sh` | `bash flow.sh preprocess [--backend slurm]` |
| `2.configure_sbs_params.ipynb` | `2_sbs.py` |
| `3.configure_phenotype_params.ipynb` | `3_phenotype.py` |
| `4.run_sbs_phenotype.sh`, `4a.run_sbs_slurm.sh`, `4b.run_phenotype_slurm.sh` | `bash flow.sh sbs phenotype [--backend slurm]` |
| `5.configure_merge_params.ipynb` | `5_merge.py` |
| `6.run_merge.sh`, `6.run_merge_slurm.sh` | `bash flow.sh merge [--backend slurm]` |
| `7.configure_classify_params.ipynb` | `7_classify.py` |
| `8.configure_aggregate_params.ipynb` | `8_aggregate.py` |
| `9.run_aggregate.sh`, `9.run_aggregate_slurm.sh` | `bash flow.sh aggregate [--backend slurm]` |
| `10.configure_cluster_params.ipynb` | `10_cluster.py` |
| `11.run_cluster.sh`, `11.run_cluster_slurm.sh` | `bash flow.sh cluster [--backend slurm]` |
| `12.analyze.ipynb` | `12_analyze.py` |
| `13.run_mozzarellm.sh` | `bash flow.sh mozzarellm [--backend slurm]` |
| `14.run_visualization.sh` | `bash flow.sh viz` |

## What changes

- **Notebooks are Python files.** Open them with `python -m marimo edit` from `analysis/` (see [Working with the Notebooks](notebook_editor_setup.md)). Parameters sit in marked operator-parameter cells, and cells rerun reactively, so there is no run-all-cells step before the config is written.
- **One run script.** `flow.sh` replaces the numbered `.sh` scripts, and dry runs use `--dry-run` instead of editing `-n` in and out of a script. There is no `NUM_PLATES` to set; preprocess, SBS and phenotype run plate by plate with the plate count read from the sample tables.
- **No Snakefile edits.** `flow.sh` switches off the rules of modules you did not name, which replaces commenting out `include:` lines in the `Snakefile` for large screens.
- **Slurm profile.** `analysis/slurm/config.yaml` has the same role, but ships with `slurm_partition` and `slurm_account` blank for you to fill in, a one-day default runtime, and no `--output` in `slurm_extra` (`flow.sh` sets the log directory). `flow.sh` honors its `jobs:` limit. Array jobs are off unless you pass `--arrays`.
- **Output format.** The preprocess notebook sets `IMAGE_FORMAT` (`all.image_format`), and its default is `zarr`. A config with no `image_format` runs in TIFF mode, with the same file names as before. The two modes write to different paths, so switching an existing screen from TIFF to zarr re-runs it from preprocess.
- **`screen.yaml`.** The template is new and more structured (organism, library, SBS and phenotype channels, data location). 12_analyze builds MozzareLLM screen contexts from it, and the visualizer shows it, so transcribe your old file into the new template.
- **Removed settings.** RANSAC in merge is seeded at 0 in code, so `merge.ransac_random_state` is gone; a leftover key is ignored.
- **New options** you may want for an existing screen: per-well annotations and `SPLIT_COL`, grouped aggregation and control scopes, `TVN_BATCH_CORRECTION`, `CLUSTER_CONTROL_KEY`, secondary objects and `PIPELINE_GPU`. See [Running a Screen](3.running_modules.md#module-by-module).

## Moving a screen over

The safest path is to bring in the marimo files, then re-run each notebook with your existing values so the notebooks write a fresh `config/config.yml` in the format the new brieflow expects:

```bash
cd YOUR-SCREEN-REPO
git checkout -b marimo-migration
git remote add template https://github.com/cheeseman-lab/brieflow-analysis.git
git fetch template marimo

# the marimo notebooks, run script, Slurm profile and screen.yaml template
git checkout template/marimo -- analysis/0_preprocess.py analysis/2_sbs.py analysis/3_phenotype.py \
    analysis/5_merge.py analysis/7_classify.py analysis/8_aggregate.py analysis/10_cluster.py \
    analysis/12_analyze.py analysis/flow.sh analysis/slurm/config.yaml
git show template/marimo:analysis/screen.yaml > analysis/screen.new.yaml

# pin the brieflow commit the marimo branch uses, then rebuild the environment
git checkout template/marimo -- brieflow
git submodule update --init --recursive
```

Then:

1. Rebuild the conda environment from the new `brieflow/` as in [Installation](2.installation_analysis_setup.md), since dependencies differ.
2. Copy the values from `analysis/screen.yaml` into `analysis/screen.new.yaml`, then replace the old file with it.
3. Keep a copy of the old `config/config.yml`, open each marimo notebook in order, and set its parameters to the values you used before. Keep `IMAGE_FORMAT = "tiff"` if you hope to reuse existing TIFF outputs; the next step shows whether they are reused.
4. Dry-run each module (`bash flow.sh <module> --dry-run`) and check that finished phases are not scheduled to re-run before running anything.
5. Remove the old `.ipynb` notebooks and `.sh` scripts once the screen runs from the new files.
