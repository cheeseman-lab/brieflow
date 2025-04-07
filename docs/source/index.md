# Welcome to Brieflow

```{image} ../../images/brieflow_logo.png
:width: 200px
:height: 200px
:align: center
:alt: Brieflow logo
```

Brieflow is tool written with [Snakemake](http://snakemake.readthedocs.io) that simplifies and automates many of the steps in optical pooled screen (OPS) analysis.
Although written to be easily deployed on a Slurm cluster, it can also be run on other cloud-based or local systems.
We have built Brieflow in tandem with [brieflow-analysis](brieflow_analysis.md) to configure and organize Brieflow runs.

Brieflow currently automates the following OPS tasks:

- **Preprocessing**: Converts raw microscope `.nd2` files into tiled `.tiff` images and extracts associated metadata (e.g. cycle, tile, well).
- **SBS**: Identifies and decodes in situ sequencing barcodes from fluorescence imaging data.
- **Phenotype**: Extracts morphological and intensity-based features for each cell from the imaging data.
- **Merge**: Matches phenotypic features with decoded barcodes across cycles and imaging rounds.
- **Aggregate**: Aggregates single-cell data by perturbation or barcode, producing summary-level datasets.
- **Cluster**: Performs unsupervised clustering to identify patterns or phenotypic signatures across perturbations.

To get started, see the [quickstart guide](quickstart.md)!

Brieflow is a community-driven project, and we welcome contributions from anyone interested in improving the tool :)

```{toctree}
:hidden:
:maxdepth: 2
:caption: Contents:

quickstart.md
brieflow_analysis.md
```