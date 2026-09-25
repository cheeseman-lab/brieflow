# Example Analyses

Cheeseman lab screens analyzed with brieflow through at least the cluster phase.
Each repository is a copy of the [brieflow-analysis](https://github.com/cheeseman-lab/brieflow-analysis) template: `analysis/` holds the notebooks, the configuration and `screen.yaml`.

- **Jupyter**: numbered `*.ipynb` notebooks on brieflow `main` (or a branch of it), TIFF outputs.
- **marimo**: `0_preprocess.py` … `12_analyze.py` on brieflow `zarr3`, {term}`OME-Zarr` outputs.

Several screens were analyzed twice: first by hand in Jupyter, then re-run on the marimo line under a new name, which makes them useful for comparing the two lines on the same raw data.
Damavand and Jannu were driven end to end by [brieflow-auto](brieflow_auto.md).

Brieflow versions are the submodule commit each repository pins, relative to `cheeseman-lab/brieflow` on 2026-09-25 (`main` at `83a3158`, `zarr3` at `f03a2c8`).

| Screen | Cells | Phenotype | Library | Notebooks, output | Brieflow version | Repository |
|---|---|---|---|---|---|---|
| Aconcagua | HeLa | DAPI, tubulin, γH2AX, phalloidin | 5,299 genes, 20,445 guides | Jupyter, TIFF | `main` @ `2e41c7f` (`v1.4.10`) | [aconcagua-analysis](https://github.com/cheeseman-lab/aconcagua-analysis) |
| Denali | HeLa | DAPI, CENP-A, COX IV, WGA | 5,000 essential genes | Jupyter, TIFF | fork of `main` @ `ce3280a` (`v1.0.0`) | [denali-analysis](https://github.com/cheeseman-lab/denali-analysis) |
| Damavand (Denali re-run) | HeLa | DAPI, CENP-A, COX IV, WGA | 5,000 essential genes | marimo, OME-Zarr | `zarr3` @ `05e75c1` | private |
| Jebel | RPE1 | DAPI, tubulin, COX IV, vimentin, WGA, ACA, phalloidin | 5,000 essential genes | Jupyter, TIFF | `main` @ `d7e0b19` (`v1.0.0`) | [jebel-analysis](https://github.com/cheeseman-lab/jebel-analysis) |
| Jannu (Jebel re-run) | RPE1 | as Jebel | 5,000 essential genes | marimo, OME-Zarr | `zarr3` @ `05e75c1` | private |
| Kilimanjaro | HeLa | DAPI, ACA, CENP-T, MitoTracker, vimentin, ITGB1, WGA, phalloidin | 5,000 essential genes | Jupyter, TIFF | `main` @ `0dbe52b` (`v1.0.0`) | [kilimanjaro-analysis](https://github.com/cheeseman-lab/kilimanjaro-analysis) |
| Kazbek (Kilimanjaro re-run) | HeLa | as Kilimanjaro | 5,000 essential genes | marimo, OME-Zarr | `zarr3` @ `05e75c1` | private |
| Mayon | HeLa | DAPI, microtubules, glycoRNA, cyclin B1 | 5,000 essential genes | Jupyter, TIFF | `main` @ `d6aecad` (`v1.0.0`) | [mayon-analysis-new](https://github.com/cheeseman-lab/mayon-analysis-new) |
| Matterhorn (Mayon re-run) | HeLa | as Mayon | 5,000 essential genes | marimo, OME-Zarr | `zarr3` @ `05e75c1` | private |
| Shasta | HeLa | DAPI, ATAC-see, histone marks, RNA Pol II pSer2/pSer5, Ki-67 | genome-wide | Jupyter, TIFF | `main` @ `5e51b21` (`v1.0.0`) | [shasta-analysis](https://github.com/cheeseman-lab/shasta-analysis) |
| Snowdon (Shasta re-run) | HeLa | as Shasta | ~20,000 genes | marimo, OME-Zarr | `zarr3` @ `05e75c1` | private |
| Etna | HeLa | 9 channels; combinatorial knockouts (CROPseq-multi) | 6,000 constructs: 360 genes, 280 gene pairs | Jupyter, TIFF | `v1.2.0` (`23ef5ab`) | [etna-analysis](https://github.com/cheeseman-lab/etna-analysis) |
| Whitney | HeLa | Hoechst, COX4, AGP, ConA | 20,553 genes | Jupyter, TIFF | `v1.4.6` (`c043901`) | [whitney-analysis](https://github.com/cheeseman-lab/whitney-analysis) |
| Baker | HeLa | DAPI, CENP-A, COX IV, WGA | 400-gene pilot | Jupyter, TIFF | branch of `main` @ `318765f` (`v1.0.0`) | [baker-analysis](https://github.com/cheeseman-lab/baker-analysis) |
| Cotopaxi | RPE1 | DAPI, tubulin, γH2AX, phalloidin | 400-gene pilot | Jupyter, TIFF | branch of `main` @ `8e04e8b` (`v1.0.0`) | [cotopaxi-analysis](https://github.com/cheeseman-lab/cotopaxi-analysis) |

A few screens show specific features:

- **Aconcagua** reanalyzes the genome-scale essential-gene screen of [Funk et al., 2022](https://doi.org/10.1016/j.cell.2022.10.017) and uses the optional classify step (interphase versus mitotic cells) before aggregate.
- **Etna** reanalyzes the combinatorial CROPseq-multi screen of Walton et al. with 15 SBS cycles (a recombination barcode and a mapping barcode); raw images are at [BioImage Archive S-BIAD3248](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD3248).
- **Mayon** starts SBS reads at cycle 2 because cycle 1 misaligned; **Kilimanjaro** segments on a later SBS cycle because the first was unusable.
- **Jebel** and **Kilimanjaro** have two phenotype rounds.
- **Baker** and **Cotopaxi** are single-plate pilots, small enough to read end to end.
