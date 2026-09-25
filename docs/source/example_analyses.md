# Example Analyses

Cheeseman lab screens analyzed with brieflow through at least the cluster phase.
Each repository is a copy of the [brieflow-analysis](https://github.com/cheeseman-lab/brieflow-analysis) template: `analysis/` holds the notebooks, the configuration and `screen.yaml`.

- **Jupyter**: numbered `*.ipynb` notebooks on brieflow `main`, TIFF outputs.
- **marimo**: `0_preprocess.py` … `12_analyze.py` on brieflow 1.5.0 (`zarr3`), {term}`OME-Zarr` outputs.

The first three screens were analyzed with earlier brieflow releases; the marimo screens run brieflow 1.5.0, and Damavand and Jannu were driven end to end by [brieflow-auto](brieflow_auto.md).
Versions are the brieflow commit each repository pins.

| Screen | Cells | Phenotype | Library | Notebooks, output | Brieflow version | Repository |
|---|---|---|---|---|---|---|
| Aconcagua | HeLa | DAPI, tubulin, γH2AX, phalloidin | 5,299 genes, 20,445 guides | Jupyter, TIFF | 1.4.10 (`2e41c7f`) | [aconcagua-analysis](https://github.com/cheeseman-lab/aconcagua-analysis) |
| Etna | HeLa | 9 channels; combinatorial knockouts (CROPseq-multi) | 6,000 constructs: 360 genes, 280 gene pairs | Jupyter, TIFF | 1.2.0 (`23ef5ab`) | [etna-analysis](https://github.com/cheeseman-lab/etna-analysis) |
| Whitney | HeLa | Hoechst, COX4, AGP, ConA | 20,553 genes | Jupyter, TIFF | 1.4.6 (`c043901`) | [whitney-analysis](https://github.com/cheeseman-lab/whitney-analysis) |
| Damavand | HeLa | DAPI, CENP-A, COX IV, WGA | 5,000 essential genes | marimo, OME-Zarr | 1.5.0 (`zarr3` @ `05e75c1`) | private |
| Matterhorn | HeLa | DAPI, microtubules, glycoRNA, cyclin B1 | 5,000 essential genes | marimo, OME-Zarr | 1.5.0 (`zarr3` @ `05e75c1`) | private |
| Kazbek | HeLa | DAPI, ACA, CENP-T, MitoTracker, vimentin, ITGB1, WGA, phalloidin | 5,000 essential genes | marimo, OME-Zarr | 1.5.0 (`zarr3` @ `05e75c1`) | private |
| Jannu | RPE1 | DAPI, tubulin, COX IV, vimentin, WGA, ACA, phalloidin | 5,000 essential genes | marimo, OME-Zarr | 1.5.0 (`zarr3` @ `05e75c1`) | private |
| Snowdon | HeLa | DAPI, ATAC-see, histone marks, RNA Pol II pSer2/pSer5, Ki-67 | ~20,000 genes | marimo, OME-Zarr | 1.5.0 (`zarr3` @ `05e75c1`) | private |

A few screens show specific features:

- **Aconcagua** reanalyzes the genome-scale essential-gene screen of [Funk et al., 2022](https://doi.org/10.1016/j.cell.2022.10.017) and uses the optional classify step (interphase versus mitotic cells) before aggregate.
- **Etna** reanalyzes the combinatorial CROPseq-multi screen of Walton et al. with 15 SBS cycles (a recombination barcode and a mapping barcode); raw images are at [BioImage Archive S-BIAD3248](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD3248).
- **Damavand** has the most cells per guide of these screens (median 515) and is the main development screen.
- **Matterhorn**, **Kazbek** and **Jannu** use olfactory-receptor (OR) genes as controls; **Snowdon** uses intergenic controls.
- **Kazbek** and **Jannu** have two phenotype rounds.

## Where the names come from

Screens are named after mountains.

| Screen | Range | Country | Elevation |
|---|---|---|---|
| Aconcagua | Andes | Argentina | 6,961 m |
| Etna | (volcano) | Italy (Sicily) | 3,357 m |
| Whitney | Sierra Nevada | United States (California) | 4,421 m |
| Damavand | Alborz | Iran | 5,609 m |
| Matterhorn | Pennine Alps | Switzerland / Italy | 4,478 m |
| Kazbek | Caucasus | Georgia | 5,054 m |
| Jannu | Himalaya | Nepal | 7,710 m |
| Snowdon | Snowdonia | United Kingdom (Wales) | 1,085 m |
