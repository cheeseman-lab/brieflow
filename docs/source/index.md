# Brieflow

Brieflow is a [Snakemake](https://snakemake.readthedocs.io) pipeline for analyzing {term}`optical pooled screens <OPS>`: it takes raw microscope files from the sequencing ({term}`SBS`) and {term}`phenotype <Phenotype>` arms of a screen and produces per-cell features matched to perturbations, per-perturbation profiles, and gene clusters.
It is written to run on a Slurm cluster, and also runs on a single machine or a cloud VM.

Brieflow is used together with [brieflow-analysis](https://github.com/cheeseman-lab/brieflow-analysis), a template repository that holds one screen's notebooks, configuration and run script.
These docs are for labs and core facilities setting up a screen, and for existing users moving from the Jupyter notebooks to the marimo notebooks.

## The pipeline at a glance

| Phase | What it does |
|---|---|
| Preprocess | Converts raw `.nd2`, `.ims` or `.tiff` files into per-tile images (TIFF or {term}`OME-Zarr`), extracts metadata and computes {term}`illumination correction fields <IC field>`. |
| SBS | Aligns sequencing cycles, finds spots, calls reads and assigns {term}`barcodes <Barcode>` to segmented cells. |
| Phenotype | Aligns phenotype channels, segments nuclei and cells (and optional {term}`secondary objects <Secondary object>`), and extracts morphology and intensity features. |
| Merge | Matches each phenotype cell to its SBS cell, so every phenotyped cell carries its perturbation. |
| Classify (optional) | Trains a cell classifier (for example interphase versus mitotic) that aggregate applies. |
| Aggregate | Filters and normalizes single-cell features and collapses them into one profile per perturbation, with bootstrap statistics. |
| Cluster | Embeds perturbation profiles with PHATE, clusters them with Leiden, and benchmarks the clusters against known complexes and pathways. |
| Analyze and MozzareLLM | Picks a clustering to study and, optionally, annotates its clusters with a language model. |
| Visualize | A Streamlit app for QC plots, screen metadata and cluster exploration. |

Each phase is configured in a notebook and run with `flow.sh`; [Running a Screen](3.running_modules.md) walks through them in order.

```{tip}
The whole pipeline can also be driven by an agent.
[brieflow-auto](brieflow_auto.md) is a Claude Code plugin that configures each notebook, runs each phase, recovers from common failures and gates each phase on QC, pausing only for decisions a person has to make.
```

## Where to start

- New to brieflow: read [Brieflow and Brieflow Analysis](0.brieflow_brieflow_analysis.md), then [Installation and Analysis Setup](2.installation_analysis_setup.md), and run the small test.
- Setting up a screen: [Installation and Analysis Setup](2.installation_analysis_setup.md), [Working with the Notebooks](notebook_editor_setup.md), then [Running a Screen](3.running_modules.md).
- Looking up a term or a notebook parameter: the [Glossary](glossary.md) and [Notebook Parameters](notebook_parameters.md).

## Citing brieflow

If you use brieflow, please cite the manuscript:

> Di Bernardo M, Kern RS, Mallar A, Nutter-Upham A, Blainey PC, Cheeseman I.
> Brieflow: An Integrated Computational Pipeline for High-Throughput Analysis of Optical Pooled Screening Data.
> bioRxiv (2025). [doi:10.1101/2025.05.26.656231](https://doi.org/10.1101/2025.05.26.656231)

Brieflow is a community project and contributions are welcome; see [Development](5.development.md).

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

0.brieflow_brieflow_analysis.md
1.before_you_screen.md
2.installation_analysis_setup.md
notebook_editor_setup.md
```

```{toctree}
:maxdepth: 2
:caption: Running a screen
:hidden:

3.running_modules.md
4.visualization.md
brieflow_auto.md
example_analyses.md
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

glossary.md
notebook_parameters.md
config_glossary.md
```

```{toctree}
:maxdepth: 2
:caption: Contributing
:hidden:

5.development.md
```
