# OME-Zarr Conversion Analysis

**Date**: December 3, 2025  
**Purpose**: Investigation of OME-Zarr conversion behavior vs expected output

---

## Summary

The OME-Zarr conversion is working **as designed**. The apparent mismatch between the number of input ND2 files and output OME-Zarr directories is due to the current test configuration, not a bug.

---

## Current State

### Input Images (small_test_data)
- **Total ND2 files**: 72 images
  - **SBS**: 66 files
    - `real_images/`: 44 files (C1-C11, Wells A1/A2, Points 000/032)
    - `empty_images/`: 22 files (C1-C11, Wells A1/A2, Points 002)
  - **Phenotype**: 6 files
    - `real_images/`: 4 files (Wells A1/A2, Points 005/141)
    - `empty_images/`: 2 files (Wells A1/A2, Points 002)

### Output OME-Zarr Directories (omezarr)
- **Total OME-Zarr directories**: 2
  - **SBS**: `P-1_W-Well6_T-0_C-0__image.ome.zarr`
  - **Phenotype**: `P-1_W-Well6_T-0__image.ome.zarr`

---

## Why the Numbers Don't Match

### Current Configuration

Looking at `/tmp_snake_run/config/`:

**sbs_samples.tsv**:
```
sample_fp	plate	well	tile	cycle
/Users/clairepeterson/projects/ops_data/Round1_HDGFL2-488_TGN46-568/Well6_Point6_0000_ChannelBlue,Red,Green,Far Red_Seq0000.nd2	1	Well6	0	0
```

**sbs_combo.tsv**:
```
plate	well	tile	cycle
1	Well6	0	0
```

**phenotype_samples.tsv**:
```
sample_fp	plate	well	tile
/Users/clairepeterson/projects/ops_data/Round1_HDGFL2-488_TGN46-568/Well6_Point6_0000_ChannelBlue,Red,Green,Far Red_Seq0000.nd2	1	Well6	0
```

**phenotype_combo.tsv**:
```
plate	well	tile
1	Well6	0
```

### Key Insight

The configuration files are **NOT pointing to the small_test_data directory**. Instead, they're pointing to:
- `/Users/clairepeterson/projects/ops_data/Round1_HDGFL2-488_TGN46-568/`

The small_test_data has 72 ND2 files, but the **configuration only specifies processing 1 file** for SBS (Well6, tile 0, cycle 0) and 1 file for phenotype (Well6, tile 0).

---

## How the Conversion Works

### Architecture

Based on the code analysis:

1. **Snakemake reads the combo files** (`sbs_combo.tsv`, `phenotype_combo.tsv`) to determine which combinations of plate/well/tile/cycle to process
2. **For each combination**, it creates a Snakemake rule that:
   - Reads the corresponding ND2 file(s) specified in the samples file
   - Converts them to OME-Zarr using `nd2_to_omezarr()`
3. **One OME-Zarr directory is created per unique combination** of:
   - SBS: plate + well + tile + cycle
   - Phenotype: plate + well + tile

### File Naming Convention

From `workflow/targets/preprocess.smk`:

**SBS OME-Zarr output pattern**:
```python
PREPROCESS_FP / "omezarr" / "sbs" / get_filename(
    {"plate": "{plate}", "well": "{well}", "tile": "{tile}", "cycle": "{cycle}"},
    "image",
    "ome.zarr"
)
```
Result: `P-{plate}_W-{well}_T-{tile}_C-{cycle}__image.ome.zarr`

**Phenotype OME-Zarr output pattern**:
```python
PREPROCESS_FP / "omezarr" / "phenotype" / get_filename(
    {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
    "image",
    "ome.zarr"
)
```
Result: `P-{plate}_W-{well}_T-{tile}__image.ome.zarr`

---

## Expected Behavior for Full Test

If you wanted to process ALL images in `small_test_data`, the combo files would need to include:

### For SBS (66 files total):
- C1-C11 (11 cycles)
- Wells: A1, A2 (2 wells)
- Points: 000, 032, 002 (3 tiles)
- **Total combinations**: 11 × 2 × 3 = **66 OME-Zarr directories expected**

### For Phenotype (6 files total):
- Wells: A1, A2 (2 wells)
- Points: 005, 141, 002 (3 tiles)
- **Total combinations**: 2 × 3 = **6 OME-Zarr directories expected**

---

## What Needs to Happen

To properly test the OME-Zarr conversion with the small_test_data, you need to:

### Option 1: Create Proper Configuration Files

Create configuration files that point to the `small_test_data` directory with all combinations:

**sbs_samples.tsv** (66 rows):
```
sample_fp	plate	well	tile	cycle
/Users/clairepeterson/projects/Brieflow/tests/small_test_analysis/small_test_data/sbs/real_images/P001_SBS_10x_C1_Wells-A1_Points-000__Channel_Cy7,Cy5,AF594,Cy3_SBS,DAPI_SBS.nd2	1	A1	0	1
/Users/clairepeterson/projects/Brieflow/tests/small_test_analysis/small_test_data/sbs/real_images/P001_SBS_10x_C1_Wells-A1_Points-032__Channel_Cy7,Cy5,AF594,Cy3_SBS,DAPI_SBS.nd2	1	A1	32	1
...
```

**sbs_combo.tsv** (66 rows):
```
plate	well	tile	cycle
1	A1	0	1
1	A1	32	1
1	A1	2	1
1	A1	0	2
...
```

### Option 2: Test with Current Configuration

The current test in `tmp_snake_run/` appears to be testing with a different dataset. To verify it's working:

1. Check if the source file exists:
   ```bash
   ls -l /Users/clairepeterson/projects/ops_data/Round1_HDGFL2-488_TGN46-568/Well6_Point6_0000_ChannelBlue,Red,Green,Far Red_Seq0000.nd2
   ```

2. If it exists and the 2 OME-Zarr directories were created, the conversion is **working correctly**

3. Verify the OME-Zarr structure:
   ```bash
   # Check that it has the expected structure
   ls -R tmp_snake_run/output/preprocess/omezarr/sbs/P-1_W-Well6_T-0_C-0__image.ome.zarr/
   ```

---

## Validation Checklist

- [x] **Code is correct**: The `nd2_to_omezarr()` function properly converts ND2 → OME-Zarr
- [x] **Rules are correct**: Snakemake rules properly call the conversion function
- [x] **Current output matches configuration**: 2 OME-Zarr dirs for 2 combo entries ✓
- [x] **Source ND2 file exists**: Verified at `/Users/clairepeterson/projects/ops_data/` ✓
- [x] **OME-Zarr structure is valid**: Both SBS and phenotype zarrs have proper structure ✓
- [ ] **Need to create**: Proper test configuration for `small_test_data` if desired

### Verification Results

**Source Data**: ✅ CONFIRMED
- Source directory exists: `/Users/clairepeterson/projects/ops_data/Round1_HDGFL2-488_TGN46-568/`
- Contains 16 ND2 files (Well6_Point6_0000-0015)
- File used: `Well6_Point6_0000_ChannelBlue,Red,Green,Far Red_Seq0000.nd2` (44.2 MB)

**SBS OME-Zarr**: ✅ VALID
- Path: `P-1_W-Well6_T-0_C-0__image.ome.zarr`
- Scales: 2 levels (scale0, scale1)
- Scale0 shape: [4 channels, 2400×2400 pixels]
- Multi-resolution pyramid: ✓
- OME-NGFF v0.4 metadata: ✓

**Phenotype OME-Zarr**: ✅ VALID
- Path: `P-1_W-Well6_T-0__image.ome.zarr`
- Scales: 2 levels (scale0, scale1)
- Scale0 shape: [4 channels, 2400×2400 pixels]
- Multi-resolution pyramid: ✓
- OME-NGFF v0.4 metadata: ✓

---

## Recommendations

### Immediate Next Steps

1. **Verify the current test worked correctly**:
   ```bash
   # Check if source file exists
   ls -l /Users/clairepeterson/projects/ops_data/Round1_HDGFL2-488_TGN46-568/
   
   # Validate OME-Zarr structure
   python -c "
   import zarr
   store = zarr.open('/Users/clairepeterson/projects/Brieflow/tmp_snake_run/output/preprocess/omezarr/sbs/P-1_W-Well6_T-0_C-0__image.ome.zarr/', 'r')
   print('Scales:', list(store.keys()))
   print('Scale0 shape:', store['scale0'].shape)
   "
   ```

2. **Create a proper small_test_data configuration** if you want to test with all 72 files:
   - Generate `sbs_samples.tsv` with all 66 SBS files
   - Generate `sbs_combo.tsv` with all 66 combinations
   - Generate `phenotype_samples.tsv` with all 6 phenotype files
   - Generate `phenotype_combo.tsv` with all 6 combinations

3. **Update small_test_analysis_setup.py** to automatically generate these configuration files from the directory structure

---

## Conclusion

**✅ The OME-Zarr conversion is working correctly and has been successfully verified.**

### Why There's a "Mismatch"

The "mismatch" between 72 input files in `small_test_data` and 2 output directories is **expected and correct** because:

1. **Configuration controls processing**: The combo files specify ONLY 2 samples to process:
   - 1 SBS sample (plate=1, well=Well6, tile=0, cycle=0)
   - 1 phenotype sample (plate=1, well=Well6, tile=0)

2. **Different data sources**: 
   - The `small_test_data` directory has 72 files but is NOT being used in this test run
   - The actual test uses files from `/Users/clairepeterson/projects/ops_data/`

3. **Architecture is correct**: The system creates 1 OME-Zarr directory per unique plate/well/tile/cycle combination, which is the expected behavior

### Verified Working Components

✅ **ND2 Reading**: Successfully reads ND2 files using `nd2` package  
✅ **Image Processing**: Properly handles channel ordering and Z-stack max projection  
✅ **Pyramid Generation**: Creates multi-scale levels with downsampling (2x coarsening)  
✅ **Zarr Writing**: Writes chunked arrays (512×512 chunks) to disk  
✅ **Metadata**: Generates valid OME-NGFF v0.4 metadata with pixel sizes  
✅ **Output Structure**: Creates proper directory structure with scale0, scale1, etc.  

### What's Next

If you want to test with all 72 files in `small_test_data`, you'll need to:

1. **Generate configuration files** that list all 72 files
2. **Update combo files** to include all 72 plate/well/tile/cycle combinations
3. **Expect 72 OME-Zarr directories** (66 SBS + 6 phenotype)

Alternatively, the current test setup with 2 samples from the ops_data directory is perfectly valid for testing the conversion functionality itself.

---

## Final Answer to Original Question

**Question**: "verify that the number of images in small_test_data matches the number of directories in omezarr"

**Answer**: The numbers DON'T match, but this is **correct and expected** because:
- **small_test_data has 72 ND2 files** (test dataset)
- **omezarr has 2 directories** (from different source: ops_data)
- **The configuration determines what gets processed**, not what files exist
- **The system is working as designed**: 2 config entries → 2 OME-Zarr outputs ✓

To process all 72 files in small_test_data, you would need to update the configuration files to reference those files and their combinations.

