# CellProfiler Emulator Feature Documentation

This document describes the phenotype features extracted by the `cp_emulator.py` module, which replicates CellProfiler's feature extraction capabilities.

## Overview

The module extracts features in the following categories:
1. **Intensity Features** - Pixel intensity statistics
2. **Intensity Distribution Features** - Radial distribution of intensity
3. **Texture Features** - Haralick and PFTAS texture descriptors
4. **Shape Features** - Morphological measurements
5. **Correlation/Colocalization Features** - Multi-channel relationships
6. **Neighbor Features** - Spatial relationships between objects

---

## Intensity Features

Extracted from `intensity_features_multichannel`. These measure pixel intensity statistics within each segmented region.

| Feature | Description | Formula/Method |
|---------|-------------|----------------|
| `int` | Integrated intensity | Sum of all pixel intensities in the region |
| `mean` | Mean intensity | Average pixel intensity |
| `std` | Standard deviation | Standard deviation of pixel intensities |
| `max` | Maximum intensity | Maximum pixel value in region |
| `min` | Minimum intensity | Minimum pixel value in region |
| `lower_quartile` | 25th percentile | `np.percentile(pixels, 25)` |
| `median` | Median intensity | 50th percentile of pixel intensities |
| `mad` | Median absolute deviation | `median(|Xi - median(X)|)` |
| `upper_quartile` | 75th percentile | `np.percentile(pixels, 75)` |
| `mass_displacement` | Distance between geometric and intensity-weighted centroids | `sqrt((centroid - weighted_centroid)^2)` |
| `center_mass_r`, `center_mass_c` | Intensity-weighted centroid coordinates | Row and column of weighted centroid |
| `max_location_r`, `max_location_c` | Location of maximum intensity pixel | Row and column coordinates |

### Edge Intensity Features

Computed on boundary pixels only (using inner boundary with connectivity=2):

| Feature | Description |
|---------|-------------|
| `int_edge` | Sum of edge pixel intensities |
| `mean_edge` | Mean of edge pixel intensities |
| `std_edge` | Standard deviation of edge pixels |
| `max_edge` | Maximum edge pixel intensity |
| `min_edge` | Minimum edge pixel intensity |

---

## Intensity Distribution Features

Extracted from `intensity_distribution_features_multichannel`. These analyze how intensity is distributed radially from the object center.

The object is divided into **4 concentric bins** (rings) from center to edge. The center point is defined as the point farthest from any edge (maximum of distance transform).

| Feature | Description |
|---------|-------------|
| `frac_at_d_0` to `frac_at_d_3` | Fraction of total intensity in each radial bin (0=innermost, 3=outermost) |
| `mean_frac_0` to `mean_frac_3` | Mean fractional intensity per pixel in each bin: `frac_at_d / frac_pixels_at_d` |
| `radial_cv_0` to `radial_cv_3` | Coefficient of variation across 8 radial wedges for each bin |

### Weighted Hu Moments

| Feature | Description |
|---------|-------------|
| `weighted_hu_moments_0` to `weighted_hu_moments_6` | Seven Hu moment invariants computed from intensity-weighted image moments |

---

## Texture Features

### PFTAS (Parameter-Free Threshold Adjacency Statistics)

54 features extracted using the Mahotas implementation. PFTAS is computed on three binary threshold images and their complements, with 9 statistics each.

| Feature | Description |
|---------|-------------|
| `pftas_0` to `pftas_53` | PFTAS texture statistics |

Reference: Coelho LP et al. (2007) "Structured Literature Image Finder", BMC Bioinformatics 8:110

### Haralick Texture Features

13 features computed from gray-level co-occurrence matrices (GLCM) at distance=5 pixels, averaged across 4 directions.

| Feature | Description |
|---------|-------------|
| `haralick_5_0` | Angular Second Moment (Energy) |
| `haralick_5_1` | Contrast |
| `haralick_5_2` | Correlation |
| `haralick_5_3` | Sum of Squares: Variance |
| `haralick_5_4` | Inverse Difference Moment (Homogeneity) |
| `haralick_5_5` | Sum Average |
| `haralick_5_6` | Sum Variance |
| `haralick_5_7` | Sum Entropy |
| `haralick_5_8` | Entropy |
| `haralick_5_9` | Difference Variance |
| `haralick_5_10` | Difference Entropy |
| `haralick_5_11` | Information Measure of Correlation 1 |
| `haralick_5_12` | Information Measure of Correlation 2 |

Reference: Haralick RM et al. (1973) IEEE Trans. SMC 3(6):610-621

---

## Shape Features

Extracted from `shape_features`. These are morphological measurements independent of intensity.

| Feature | Description | Formula/Method |
|---------|-------------|----------------|
| `area` | Number of pixels in region | Count of pixels |
| `perimeter` | Boundary length | Approximated contour length |
| `convex_area` | Area of convex hull | Pixels in convex hull |
| `form_factor` | Circularity/isoperimetric quotient | `4 * pi * area / perimeter^2` |
| `solidity` | Proportion of convex hull filled | `area / convex_area` |
| `extent` | Proportion of bounding box filled | `area / bounding_box_area` |
| `euler_number` | Topological descriptor | `objects - holes` |
| `centroid_r`, `centroid_c` | Geometric center coordinates | |
| `eccentricity` | Ellipse eccentricity | 0 (circle) to 1 (line) |
| `major_axis` | Length of major axis of fitted ellipse | |
| `minor_axis` | Length of minor axis of fitted ellipse | |
| `orientation` | Angle of major axis from horizontal | Radians |
| `compactness` | Variance of radial distribution normalized by area | `2 * pi * (M02 + M20) / area^2` |

### Radius Features

Computed from the distance transform of the object:

| Feature | Description |
|---------|-------------|
| `max_radius` | Maximum distance from any interior point to edge |
| `median_radius` | Median distance to edge |
| `mean_radius` | Mean distance to edge |

### Feret Diameter Features

| Feature | Description |
|---------|-------------|
| `min_feret_diameter` | Minimum caliper diameter (rotating calipers algorithm) |
| `max_feret_diameter` | Maximum caliper diameter (longest distance between hull vertices) |
| `min_feret_r0`, `min_feret_c0`, `min_feret_r1`, `min_feret_c1` | Endpoints of minimum Feret diameter |
| `max_feret_r0`, `max_feret_c0`, `max_feret_r1`, `max_feret_c1` | Endpoints of maximum Feret diameter |

### Hu Moments

| Feature | Description |
|---------|-------------|
| `hu_moments_0` to `hu_moments_6` | Seven Hu moment invariants (rotation, scale, translation invariant) |

### Zernike Moments

30 Zernike polynomial coefficients computed up to degree 9, normalized by the minimum enclosing circle:

| Feature | Description |
|---------|-------------|
| `zernike_0_0` to `zernike_9_9` | Zernike moment magnitudes (indexed by radial and azimuthal order) |

---

## Correlation/Colocalization Features

Extracted from `correlation_features_multichannel`. These measure relationships between intensity channels.

For each pair of channels (first, second):

| Feature | Description | Formula |
|---------|-------------|---------|
| `correlation_{first}_{second}` | Pearson correlation coefficient | Standard correlation of masked pixel intensities |
| `lstsq_slope_{first}_{second}` | Least squares regression slope | Slope of `second = m * first + b` |

### Colocalization Metrics

Based on Manders coefficients and related measures. Threshold determined by Otsu's method.

| Feature | Description |
|---------|-------------|
| `overlap_{first}_{second}` | Overlap coefficient | `sum(A*B) / sqrt(sum(A^2) * sum(B^2))` |
| `K_{first}_{second}` | Colocalization coefficient K1 | `sum(A*B) / sum(A^2)` |
| `K_{second}_{first}` | Colocalization coefficient K2 | `sum(A*B) / sum(B^2)` |
| `manders_{first}_{second}` | Manders M1 coefficient | Fraction of A intensity colocalizing with B |
| `manders_{second}_{first}` | Manders M2 coefficient | Fraction of B intensity colocalizing with A |
| `rwc_{first}_{second}` | Rank-weighted colocalization 1 | RWC with rank-based intensity weighting |
| `rwc_{second}_{first}` | Rank-weighted colocalization 2 | RWC with rank-based intensity weighting |

Reference: Singan et al. (2011) BMC Bioinformatics 12:407

---

## Neighbor Features

Extracted separately using `neighbor_measurements()`. These measure spatial relationships between segmented objects.

### Distance-based neighbors (computed at distance=1 pixel by default)

| Feature | Description |
|---------|-------------|
| `number_neighbors_{d}` | Count of neighboring objects within distance d |
| `percent_touching_{d}` | Fraction of perimeter touching other objects |

### Closest object features

| Feature | Description |
|---------|-------------|
| `first_neighbor_distance` | Distance to centroid of closest neighbor |
| `second_neighbor_distance` | Distance to centroid of second-closest neighbor |
| `angle_between_neighbors` | Angle (degrees) formed by first and second neighbors with object as vertex |

---

## Foci Features

When foci detection is enabled, the following features are extracted per foci channel:

| Feature | Description |
|---------|-------------|
| `{channel}_foci_count` | Number of detected foci within the cell |
| `{channel}_foci_area` | Total area (pixels) of all foci in the cell |

Foci detection uses:
1. White tophat filter (radius=3) to enhance small bright spots
2. Laplacian of Gaussian for blob detection
3. Thresholding (default=10)
4. Watershed segmentation to separate touching foci
5. Optional removal of border-touching foci

---

## Configuration Parameters

### Granularity (not enabled by default)

These parameters control granularity spectrum computation (computationally expensive):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRANULARITY_BACKGROUND` | 10 | Radius for background estimation |
| `GRANULARITY_BACKGROUND_DOWNSAMPLE` | 1 | Downsampling for background |
| `GRANULARITY_DOWNSAMPLE` | 1 | Downsampling for spectrum |
| `GRANULARITY_LENGTH` | 16 | Number of spectrum components |

### Other Constants

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EDGE_CONNECTIVITY` | 2 | Connectivity for edge detection (includes diagonal) |
| `ZERNIKE_DEGREE` | 9 | Maximum degree for Zernike polynomials |

---

## References

- Haralick RM, Shanmugam K, Dinstein I. (1973) "Textural Features for Image Classification" IEEE Trans. SMC 3(6):610-621
- Coelho LP et al. (2007) "Structured Literature Image Finder" BMC Bioinformatics 8:110
- Singan et al. (2011) "Dual channel rank-based intensity weighting for quantitative co-localization" BMC Bioinformatics 12:407
- Maragos P. (1989) "Pattern spectrum and multiscale shape representation" IEEE PAMI 11(7):701-716
- CellProfiler documentation: https://cellprofiler-manual.s3.amazonaws.com/
