# OME-Zarr Conversion Flow Diagram

## Current Test Run (tmp_snake_run)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Configuration Files                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  sbs_samples.tsv:                                              │
│    ✓ 1 entry → Well6, tile=0, cycle=0                         │
│                                                                 │
│  sbs_combo.tsv:                                                │
│    ✓ 1 combination → plate=1, well=Well6, tile=0, cycle=0     │
│                                                                 │
│  phenotype_samples.tsv:                                        │
│    ✓ 1 entry → Well6, tile=0                                  │
│                                                                 │
│  phenotype_combo.tsv:                                          │
│    ✓ 1 combination → plate=1, well=Well6, tile=0              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Source ND2 Files                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  /Users/clairepeterson/projects/ops_data/                      │
│    Round1_HDGFL2-488_TGN46-568/                               │
│      ✓ Well6_Point6_0000_...Seq0000.nd2 (44 MB)               │
│        [4 channels, 2400×2400 pixels]                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Snakemake Processing                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Rule: convert_sbs_omezarr                                     │
│    Input:  Well6_Point6_0000_...Seq0000.nd2                   │
│    Process: nd2_to_omezarr()                                   │
│      • Read ND2 → numpy array (CYX format)                     │
│      • Build pyramid (2 levels, 2× coarsening)                 │
│      • Write Zarr chunks (512×512)                             │
│    Output: P-1_W-Well6_T-0_C-0__image.ome.zarr                │
│                                                                 │
│  Rule: convert_phenotype_omezarr                               │
│    Input:  Well6_Point6_0000_...Seq0000.nd2                   │
│    Process: nd2_to_omezarr()                                   │
│      • Read ND2 → numpy array (CYX format)                     │
│      • Build pyramid (2 levels, 2× coarsening)                 │
│      • Write Zarr chunks (512×512)                             │
│    Output: P-1_W-Well6_T-0__image.ome.zarr                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Output OME-Zarr                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  tmp_snake_run/output/preprocess/omezarr/                     │
│                                                                 │
│    sbs/                                                        │
│      ✓ P-1_W-Well6_T-0_C-0__image.ome.zarr/                   │
│          scale0/ [4, 2400, 2400]                               │
│          scale1/ [4, 1200, 1200]                               │
│          .zattrs (OME-NGFF metadata)                           │
│          .zgroup                                                │
│                                                                 │
│    phenotype/                                                  │
│      ✓ P-1_W-Well6_T-0__image.ome.zarr/                       │
│          scale0/ [4, 2400, 2400]                               │
│          scale1/ [4, 1200, 1200]                               │
│          .zattrs (OME-NGFF metadata)                           │
│          .zgroup                                                │
│                                                                 │
│  Result: 2 OME-Zarr directories ✓                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Comparison: small_test_data vs Actual Test

```
┌───────────────────────────────┬───────────────────────────────┐
│      small_test_data          │    Actual Test (ops_data)     │
│    (NOT being used)           │      (Currently running)      │
├───────────────────────────────┼───────────────────────────────┤
│                               │                               │
│  Location:                    │  Location:                    │
│  tests/small_test_analysis/   │  /Users/.../ops_data/         │
│    small_test_data/           │    Round1.../                 │
│                               │                               │
│  SBS Files: 66 ND2 files      │  SBS Files: 16 ND2 files      │
│    • C1-C11 (11 cycles)       │    • Seq0000-Seq0015          │
│    • Wells: A1, A2            │    • Well6_Point6 only        │
│    • Points: 000, 032, 002    │                               │
│                               │                               │
│  Phenotype: 6 ND2 files       │  Phenotype: Same files        │
│    • Wells: A1, A2            │    • Well6_Point6 only        │
│    • Points: 005, 141, 002    │                               │
│                               │                               │
│  Config entries: NONE         │  Config entries: 2            │
│  Expected output: 0           │  Expected output: 2           │
│  Actual output: 0 ✓           │  Actual output: 2 ✓          │
│                               │                               │
│  Status: Available for future │  Status: PROCESSED ✓          │
│          testing              │                               │
│                               │                               │
└───────────────────────────────┴───────────────────────────────┘
```

---

## How to Process small_test_data

If you want to process ALL 72 files in small_test_data:

```
Step 1: Create sbs_samples.tsv with 66 rows
┌────────────────────────────────────────────────────────────┐
│ sample_fp                             plate  well  tile cy │
├────────────────────────────────────────────────────────────┤
│ .../P001_SBS_10x_C1_Wells-A1_Pt-000   1     A1    0    1  │
│ .../P001_SBS_10x_C1_Wells-A1_Pt-032   1     A1    32   1  │
│ .../P001_SBS_10x_C1_Wells-A1_Pt-002   1     A1    2    1  │
│ .../P001_SBS_10x_C1_Wells-A2_Pt-000   1     A2    0    1  │
│ ... (62 more rows)                                         │
└────────────────────────────────────────────────────────────┘

Step 2: Create sbs_combo.tsv with 66 rows
┌──────────────────────────┐
│ plate  well  tile  cycle │
├──────────────────────────┤
│ 1      A1    0     1     │
│ 1      A1    32    1     │
│ 1      A1    2     1     │
│ 1      A1    0     2     │
│ ... (62 more rows)       │
└──────────────────────────┘

Step 3: Run Snakemake
$ snakemake --cores all --until all_preprocess

Step 4: Expected Output
┌─────────────────────────────────────────────┐
│ preprocess/omezarr/sbs/                     │
│   P-1_W-A1_T-0_C-1__image.ome.zarr         │
│   P-1_W-A1_T-32_C-1__image.ome.zarr        │
│   P-1_W-A1_T-2_C-1__image.ome.zarr         │
│   P-1_W-A1_T-0_C-2__image.ome.zarr         │
│   ... (62 more directories)                 │
│                                              │
│ Total: 66 OME-Zarr directories              │
└─────────────────────────────────────────────┘

(Similar process for phenotype → 6 directories)
```

---

## Key Insight

```
┌─────────────────────────────────────────────────────────────┐
│                     THE TRUTH                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  The number of OME-Zarr output directories is determined   │
│  by the CONFIGURATION FILES, not by what files exist on     │
│  disk.                                                      │
│                                                             │
│  • Configuration says: "process 2 samples"                 │
│  • System processes: 2 samples                             │
│  • Output created: 2 OME-Zarr directories                  │
│                                                             │
│  ✓ This is CORRECT behavior                                │
│                                                             │
│  The fact that small_test_data has 72 files is irrelevant  │
│  because those files are not referenced in the config.      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

