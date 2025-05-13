# Before You Screen

## Resources

We highly recommend reviewing the following materials before screening:

- Feldman, D., Funk, L., Le, A. et al. Pooled genetic perturbation screens with image‑based phenotypes. *Nat Protoc* 17, 476–512 (2022). [doi.org/10.1038/s41596-021-00653-8](https://doi.org/10.1038/s41596-021-00653-8)
- Walton, R. T., Singh, A., Blainey, P. C. et al. Pooled genetic screens with image‑based profiling. *Mol Syst Biol* 18, e10768 (2022). [doi.org/10.15252/msb.202110768](https://doi.org/10.15252/msb.202110768)
- Feldman, D., Singh, A., Schmid‑Burgk, J. L. et al. Optical pooled screens in human cells. *Cell* 179, 787–799.e17 (2019). [doi.org/10.1016/j.cell.2019.09.016](https://doi.org/10.1016/j.cell.2019.09.016)
- Blainey Lab [OPS Protocols](https://blainey.mit.edu/protocols/).
**Note**: Ignore the GitHub Repository as this is outdated!

## Data Creation SOP

## 1. Data Structure

The OPS data must follow a hierarchical structure:

```
screen_name/
└── plate_1/
├── input_sbs/
│ ├── c1/
│ ├── c2/
│ └── ... (cycles of sequencing data)
└── input_ph/
├── round_1/
├── round_2/
└── ... (rounds of phenotyping data)
├── plate_2/
└── ... (plates of pooled screening data)
```

## 2. File Naming Conventions

**Note:** The Nikon microscope may generate subfolders with dates that are required for its output structure. Including these as subfolders of the rounds or cycles directories is not a problem. The same concept is relevant to if you are testing different magnifications or image acquisition approaches—these should be in subfolders of the round or cycles directories.

### 2.1 SBS (Sequencing) Files

- Files must be located within cycle directories (`c1`, `c2`, etc.)
- Naming format should include plate number, well ID, and tile number information
- Channel information should be included if each ND2 image is for an individual channel. Otherwise, channel information can be included, but is not necessary.

**Example SBS filename:** `screen_name/plate_1/c2/P001_Wells-A3_Points-214.nd2`

### 2.2 Phenotype Files

- Files must be located within round directories (`round_1`, `round_2`, etc.)
- Naming format should include plate number, well ID, and tile number information
- Channel information should be included if each ND2 image is for an individual channel. Otherwise, channel information can be included, but is not necessary.

**Example Phenotype filename:** `screen_name/plate_1/input_ph/round_1/P001_Wells-B1_Points-585.nd2`

## 3. Alignment Verification

**Critical Quality Control Step:** Check alignment between:
- Rounds of phenotyping data (`round_1` vs. `round_2` vs. …)  
- Cycles of sequencing data for each cycle (`c1` vs. `c2` vs. …)

Alignment verification should be performed before proceeding with downstream analysis. Downstream analysis will be impossible with large spatial shifts in alignment.

## 4. Screen Name and Metadata Tracking

Please inform Matteo Di Bernardo (mdiberna@wi.mit.edu) when you are beginning to acquire a screen. He will provide an appropriate unique screen name and a metadata tracking guide that will be associated with your screen and facilitate downstream analysis.
