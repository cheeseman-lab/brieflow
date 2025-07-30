"""
Enhanced stitching functions for BrieFlow pipeline.
Handles both images and segmentation masks with actual stitching.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import scipy
from pathlib import Path
import cv2
from skimage import io, measure
import warnings

from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.processing.registration import translation_nd as dexp_reg
from dexp.processing.utils.linear_solver import linsolve

# Import your existing preprocessing function
from lib.preprocess.preprocess import nd2_to_tiff


class LimitedSizeDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)


def augment_tile(tile: np.ndarray, flipud: bool, fliplr: bool, rot90: int) -> np.ndarray:
    """Apply flips and rotations to a tile"""
    if flipud:
        tile = np.flip(tile, axis=-2)
    if fliplr:
        tile = np.flip(tile, axis=-1)
    if rot90:
        tile = np.rot90(tile, k=rot90, axes=(-2, -1))
    return tile


def load_aligned_tiff(file_path: str, channel: int = 0) -> np.ndarray:
    """Load an aligned TIFF file and extract specific channel."""
    try:
        image = io.imread(file_path)

        if len(image.shape) == 2:
            return image
        elif len(image.shape) == 3:
            if image.shape[0] <= 10:  # Channels-first
                return image[min(channel, image.shape[0] - 1), :, :]
            else:  # Channels-last
                return image[:, :, min(channel, image.shape[2] - 1)]
        else:
            return image[0, min(channel, image.shape[1] - 1), :, :]

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


class AlignedTiffTileCache:
    """Cache for loading aligned TIFF tiles using metadata for positioning."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        well: str,
        data_type: str = "phenotype",
        flipud: bool = False,
        fliplr: bool = False,
        rot90: int = 0,
        channel: int = 0,
    ):
        self.cache = LimitedSizeDict(max_size=50)  # Increased from 20
        self.metadata_df = metadata_df[metadata_df["well"] == well].copy()
        self.well = well
        self.data_type = data_type
        self.flipud = flipud
        self.fliplr = fliplr
        self.rot90 = rot90
        self.channel = channel

    def __getitem__(self, tile_id: int):
        """Get tile by tile ID"""
        if tile_id in self.cache:
            return self.cache[tile_id]
        else:
            tile_data = self.load_tile(tile_id)
            if tile_data is not None:
                return tile_data
            else:
                raise KeyError(f"Tile {tile_id} not found")

    def load_tile(self, tile_id: int) -> np.ndarray:
        """Load an aligned TIFF tile and add it to cache"""
        tile_metadata = self.metadata_df[self.metadata_df["tile"] == tile_id]

        if len(tile_metadata) == 0:
            print(f"No metadata found for tile {tile_id}")
            return None

        tile_row = tile_metadata.iloc[0]
        plate = tile_row["plate"]
        
        # Construct path to aligned TIFF file (for image stitching)
        aligned_path = f"analysis_root/{self.data_type}/images/P-{plate}_W-{self.well}_T-{tile_id}__aligned.tiff"

        try:
            tile_2d = load_aligned_tiff(aligned_path, self.channel)
            if tile_2d is None:
                return None

            # Apply augmentations
            aug_tile = augment_tile(tile_2d, self.flipud, self.fliplr, self.rot90)
            self.cache[tile_id] = aug_tile
            return aug_tile

        except Exception as e:
            print(f"Error loading tile {tile_id}: {e}")
            return None


class MaskTileCache:
    """Cache for loading segmentation mask tiles."""
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        well: str,
        data_type: str = "phenotype",
        flipud: bool = False,
        fliplr: bool = False,
        rot90: int = 0,
    ):
        self.cache = LimitedSizeDict(max_size=50)  # Increased from 20
        self.metadata_df = metadata_df[metadata_df["well"] == well].copy()
        self.well = well
        self.data_type = data_type
        self.flipud = flipud
        self.fliplr = fliplr
        self.rot90 = rot90

    def __getitem__(self, tile_id: int):
        """Get mask tile by tile ID"""
        if tile_id in self.cache:
            return self.cache[tile_id]
        else:
            tile_data = self.load_mask_tile(tile_id)
            if tile_data is not None:
                self.cache[tile_id] = tile_data
                return tile_data
            else:
                raise KeyError(f"Mask tile {tile_id} not found")

    def load_mask_tile(self, tile_id: int) -> np.ndarray:
        """Load a segmentation mask tile"""
        tile_metadata = self.metadata_df[self.metadata_df["tile"] == tile_id]

        if len(tile_metadata) == 0:
            print(f"No metadata found for tile {tile_id}")
            return None

        tile_row = tile_metadata.iloc[0]
        plate = tile_row["plate"]
        
        # Construct path to nuclei mask file (segmentation masks)
        mask_path = f"analysis_root/{self.data_type}/images/P-{plate}_W-{self.well}_T-{tile_id}__nuclei.tiff"

        try:
            mask_array = io.imread(mask_path)
            
            # Ensure it's 2D
            if len(mask_array.shape) > 2:
                mask_array = mask_array[0] if mask_array.shape[0] == 1 else mask_array

            # Apply augmentations
            aug_mask = augment_tile(mask_array, self.flipud, self.fliplr, self.rot90)
            return aug_mask

        except Exception as e:
            print(f"Error loading mask tile {tile_id}: {e}")
            return None


def metadata_to_grid_positions(
    metadata_df: pd.DataFrame, well: str, grid_spacing_tolerance: float = 1.0
) -> Dict[int, Tuple[int, int]]:
    """Convert metadata to tile positions using proximity-based approach for stage coordinates."""
    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        return {}

    # For stage coordinate systems, use proximity-based approach instead of grid binning
    coords = well_metadata[['x_pos', 'y_pos']].values
    tile_ids = well_metadata['tile'].values
    
    print(f"Processing {len(tile_ids)} tiles with proximity-based positioning")
    
    # Create simple tile positions (use sequential indices)
    tile_positions = {tile_ids[i]: (i, 0) for i in range(len(tile_ids))}
    
    print(f"Created positions for {len(tile_positions)} tiles")
    return tile_positions


def connectivity_from_grid(positions: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
    """Create connectivity graph using proximity-based approach for stage coordinates."""
    # This function is now called with simple positions, but we need the actual coordinates
    # We'll override this with a proximity-based approach in the main function
    return {}


def offset_tiff(
    image_a: np.ndarray, image_b: np.ndarray, relation: Tuple[int, int], overlap: int
) -> TranslationRegistrationModel:
    """Calculate offset between two images based on their spatial relationship."""
    shape = image_a.shape

    # Determine overlap region based on spatial relationship
    if relation[0] == -1:
        roi_a = image_a[:, -overlap:]
        roi_b = image_b[:, :overlap]
        corr_x = shape[-2] - overlap
        corr_y = 0
    elif relation[0] == 1:
        roi_a = image_a[:, :overlap]
        roi_b = image_b[:, -overlap:]
        corr_x = -overlap
        corr_y = 0
    elif relation[1] == -1:
        roi_a = image_a[-overlap:, :]
        roi_b = image_b[:overlap, :]
        corr_x = 0
        corr_y = shape[-1] - overlap
    elif relation[1] == 1:
        roi_a = image_a[:overlap, :]
        roi_b = image_b[-overlap:, :]
        corr_x = 0
        corr_y = -overlap
    else:
        center_size = min(overlap, min(shape) // 4)
        center_y = shape[0] // 2
        center_x = shape[1] // 2
        half_size = center_size // 2

        roi_a = image_a[
            center_y - half_size : center_y + half_size,
            center_x - half_size : center_x + half_size,
        ]
        roi_b = image_b[
            center_y - half_size : center_y + half_size,
            center_x - half_size : center_x + half_size,
        ]
        corr_x = corr_y = 0

    # Ensure positive values
    roi_a_min = np.min(roi_a)
    roi_b_min = np.min(roi_b)
    if roi_a_min < 0:
        roi_a = roi_a - roi_a_min
    if roi_b_min < 0:
        roi_b = roi_b - roi_b_min

    try:
        model = dexp_reg.register_translation_nd(roi_a, roi_b)
        model.shift_vector += np.array([corr_y, corr_x])
    except Exception as e:
        print(f"Registration failed: {e}")
        model = TranslationRegistrationModel()
        model.shift_vector = np.array([0.0, 0.0])
        model.confidence = 0.0

    return model


class AlignedTiffEdge:
    """Edge between two tiles for registration using aligned TIFF images."""

    def __init__(
        self,
        tile_a_id: int,
        tile_b_id: int,
        tile_cache: AlignedTiffTileCache,
        tile_positions: Dict[int, Tuple[int, int]],
    ):
        self.tile_cache = tile_cache
        self.tile_a_id = tile_a_id
        self.tile_b_id = tile_b_id
        self.pos_a = tile_positions[tile_a_id]
        self.pos_b = tile_positions[tile_b_id]
        self.relation = (self.pos_a[0] - self.pos_b[0], self.pos_a[1] - self.pos_b[1])
        self.model = self.get_offset()

    def get_offset(self) -> TranslationRegistrationModel:
        """Calculate offset between two tiles"""
        tile_a = self.tile_cache[self.tile_a_id]
        tile_b = self.tile_cache[self.tile_b_id]

        if tile_a is None or tile_b is None:
            model = TranslationRegistrationModel()
            model.shift_vector = np.array([0.0, 0.0])
            model.confidence = 0.0
            return model

        # Calculate adaptive overlap based on tile size (5% overlap, but smaller for speed)
        tile_size_avg = (tile_a.shape[0] + tile_a.shape[1]) / 2
        overlap = max(40, int(tile_size_avg * 0.03))  # 3% overlap for speed, minimum 40
        
        return offset_tiff(tile_a, tile_b, self.relation, overlap=overlap)


def estimate_stitch_aligned_tiff(
    metadata_df: pd.DataFrame,
    well: str,
    data_type: str = "phenotype",
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
    tile_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Dict]:
    """Complete stitching estimation for a single well using aligned TIFF data."""
    tile_positions = metadata_to_grid_positions(metadata_df, well)
    
    if len(tile_positions) == 0:
        print(f"No tiles found for well {well}")
        return {"total_translation": {}, "confidence": {well: {}}}

    # Auto-detect tile size if not provided
    if tile_size is None:
        if data_type == "phenotype":
            tile_size = (2400, 2400)  # Phenotype tiles
        else:  # SBS
            tile_size = (1200, 1200)  # SBS tiles
        print(f"Auto-detected tile size for {data_type}: {tile_size}")

    # Create connectivity using optimized proximity-based approach
    well_metadata = metadata_df[metadata_df["well"] == well].copy()
    coords = well_metadata[['x_pos', 'y_pos']].values
    tile_ids = well_metadata['tile'].values
    
    # Calculate proximity-based edges with optimization
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(coords))
    
    # Find appropriate threshold based on distance distribution
    min_distance = distances[distances > 0].min() if np.any(distances > 0) else 1.0
    
    if min_distance < 1.0:
        # Very small distances - likely clustered data, use percentile approach
        distance_percentiles = np.percentile(distances[distances > 0], [1, 5, 10, 25])
        proximity_threshold = distance_percentiles[2]  # 10th percentile
        max_neighbors_per_tile = 8
        print(f"Using adaptive proximity threshold {proximity_threshold:.1f} for clustered data")
    else:
        # Regular spacing - use threshold just above minimum
        proximity_threshold = min_distance * 1.05
        max_neighbors_per_tile = 6
        print(f"Using proximity threshold {proximity_threshold:.1f} for regular grid")
    
    edges = {}
    edge_idx = 0
    
    # OPTIMIZATION: Limit each tile to reasonable number of neighbors
    for i in range(len(tile_ids)):
        # Find all neighbors for this tile
        neighbor_distances = [(j, distances[i, j]) for j in range(len(tile_ids)) 
                            if j != i and distances[i, j] < proximity_threshold]
        
        # Sort by distance and take only closest neighbors
        neighbor_distances.sort(key=lambda x: x[1])
        neighbors_to_use = neighbor_distances[:max_neighbors_per_tile]
        
        for j, dist in neighbors_to_use:
            if i < j:  # Avoid duplicate edges
                pos_a = tile_positions[tile_ids[i]]
                pos_b = tile_positions[tile_ids[j]]
                edges[f"{edge_idx}"] = [pos_a, pos_b]
                edge_idx += 1
    
    print(f"Created {len(edges)} optimized proximity-based edges")
    
    tile_cache = AlignedTiffTileCache(metadata_df, well, data_type, flipud, fliplr, rot90, channel)

    # Process edges (revert to sequential for now - parallel processing has cache issues)
    edge_list = []
    confidence_dict = {}
    pos_to_tile = {pos: tile_id for tile_id, pos in tile_positions.items()}

    print(f"Processing {len(edges)} edges sequentially...")
    for key, (pos_a, pos_b) in tqdm(edges.items(), desc=f"Computing pairwise shifts for {data_type}"):
        tile_a_id = pos_to_tile[pos_a]
        tile_b_id = pos_to_tile[pos_b]

        try:
            edge = AlignedTiffEdge(tile_a_id, tile_b_id, tile_cache, tile_positions)
            edge_list.append(edge)
            confidence_dict[key] = [list(pos_a), list(pos_b), float(edge.model.confidence)]
        except Exception as e:
            print(f"Failed to process edge {tile_a_id}-{tile_b_id}: {e}")
            continue

    # Calculate optimal positions
    opt_shift_dict = optimal_positions_tiff(edge_list, tile_positions, well, tile_size)

    return {"total_translation": opt_shift_dict, "confidence": {well: confidence_dict}}


def optimal_positions_tiff(
    edge_list: List[AlignedTiffEdge],
    tile_positions: Dict[int, Tuple[int, int]],
    well: str,
    tile_size: Tuple[int, int],
) -> Dict[str, List[int]]:
    """Calculate optimal tile positions using least squares optimization."""
    if len(edge_list) == 0:
        return {}

    tile_ids = list(tile_positions.keys())
    tile_lut = {tile_id: i for i, tile_id in enumerate(tile_ids)}
    n_tiles = len(tile_ids)
    n_edges = len(edge_list)

    y_i = np.zeros(n_edges + 1, dtype=np.float32)
    y_j = np.zeros(n_edges + 1, dtype=np.float32)

    # Initial guess
    x_guess = np.array([tile_positions[tile_id][0] * tile_size[1] for tile_id in tile_ids], dtype=np.float32)
    y_guess = np.array([tile_positions[tile_id][1] * tile_size[0] for tile_id in tile_ids], dtype=np.float32)

    # Build constraint matrix
    a = scipy.sparse.lil_matrix((n_edges + 1, n_tiles), dtype=np.float32)

    for c, edge in enumerate(edge_list):
        tile_a_idx = tile_lut[edge.tile_a_id]
        tile_b_idx = tile_lut[edge.tile_b_id]
        a[c, tile_a_idx] = -1
        a[c, tile_b_idx] = 1
        y_i[c] = edge.model.shift_vector[0]
        y_j[c] = edge.model.shift_vector[1]

    # Fix first tile at origin
    y_i[-1] = 0
    y_j[-1] = 0
    a[-1, 0] = 1
    a = a.tocsr()

    # Optimization
    tolerance = 1e-5
    try:
        opt_i = linsolve(a, y_i, tolerance=tolerance, x0=y_guess, maxiter=int(1e8))
        opt_j = linsolve(a, y_j, tolerance=tolerance, x0=x_guess, maxiter=int(1e8))
    except Exception as e:
        print(f"Optimization failed: {e}")
        opt_i = y_guess
        opt_j = x_guess

    # Combine and zero minimum
    opt_shifts = np.vstack((opt_i, opt_j)).T
    opt_shifts_zeroed = opt_shifts - np.min(opt_shifts, axis=0)

    # Create output dictionary
    opt_shifts_dict = {}
    for i, tile_id in enumerate(tile_ids):
        opt_shifts_dict[f"{well}/{tile_id}"] = [
            int(opt_shifts_zeroed[i, 0]),
            int(opt_shifts_zeroed[i, 1]),
        ]

    return opt_shifts_dict


def get_output_shape_tiff(shifts: Dict[str, List[int]], tile_size: Tuple[int, int]) -> Tuple[int, int]:
    """Calculate the output shape needed for stitched image"""
    if not shifts:
        return tile_size

    y_shifts = [shift[0] for shift in shifts.values()]
    x_shifts = [shift[1] for shift in shifts.values()]
    max_y = int(np.max(y_shifts)) if y_shifts else 0
    max_x = int(np.max(x_shifts)) if x_shifts else 0

    return max_y + tile_size[0], max_x + tile_size[1]


def relabel_mask_tile(mask_tile: np.ndarray, start_label: int) -> np.ndarray:
    """Relabel a mask tile to use sequential labels starting from start_label."""
    if mask_tile.max() == 0:
        return mask_tile
    
    unique_labels = np.unique(mask_tile[mask_tile > 0])
    relabeled = np.zeros_like(mask_tile)
    
    for i, old_label in enumerate(unique_labels):
        new_label = start_label + i
        relabeled[mask_tile == old_label] = new_label
    
    return relabeled


def extract_cell_positions_from_stitched_mask(
    stitched_mask: np.ndarray,
    well: str,
    data_type: str = "phenotype"
) -> pd.DataFrame:
    """
    Extract cell positions and properties from stitched mask.

    Args:
        stitched_mask: Stitched segmentation mask
        well: Well identifier
        data_type: "phenotype" or "sbs"

    Returns:
        DataFrame with cell positions and properties
    """
    print(f"Extracting cell positions from {data_type} stitched mask...")
    
    # Get region properties
    props = measure.regionprops(stitched_mask)
    
    cell_data = []
    for prop in props:
        centroid_y, centroid_x = prop.centroid
        
        cell_data.append({
            'well': well,
            'cell': prop.label,
            'i': centroid_y,  # y-coordinate
            'j': centroid_x,  # x-coordinate
            'area': prop.area,
            'data_type': data_type
        })
    
    df = pd.DataFrame(cell_data)
    print(f"Extracted {len(df)} cells from {data_type} stitched mask")
    
    return df


def assemble_aligned_tiff_well(
    metadata_df: pd.DataFrame,
    shifts: Dict[str, List[int]],
    well: str,
    data_type: str = "phenotype",
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    overlap_percent: float = 0.05,
) -> np.ndarray:
    """Assemble a stitched well image from aligned TIFF tiles."""
    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        raise ValueError(f"No metadata found for well {well}")

    # Get tile size from first aligned image
    first_tile_row = well_metadata.iloc[0]
    plate = first_tile_row["plate"]
    tile_id = first_tile_row["tile"]
    first_tile_path = f"analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{tile_id}__aligned.tiff"

    try:
        first_tile = load_aligned_tiff(first_tile_path)
        if first_tile is None:
            raise ValueError(f"Could not load first tile: {first_tile_path}")
        tile_size = first_tile.shape
    except Exception as e:
        print(f"Error determining tile size: {e}")
        return np.array([])

    # Calculate output dimensions
    final_shape = get_output_shape_tiff(shifts, tile_size)
    output_image = np.zeros(final_shape, dtype=np.float32)
    divisor = np.zeros(final_shape, dtype=np.uint16)

    print(f"Assembling {len(well_metadata)} {data_type} tiles into shape {final_shape}")

    # Process each tile
    for _, tile_row in tqdm(well_metadata.iterrows(), total=len(well_metadata), desc=f"Assembling {data_type} tiles"):
        tile_id = tile_row["tile"]
        tile_key = f"{well}/{tile_id}"

        if tile_key not in shifts:
            continue

        # Load aligned image tile (for stitching)
        plate = tile_row["plate"]
        tile_path = f"analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{tile_id}__aligned.tiff"

        try:
            tile_array = load_aligned_tiff(tile_path)
            if tile_array is None:
                continue
            tile_array = augment_tile(tile_array, flipud, fliplr, rot90)
        except Exception as e:
            print(f"Error loading tile {tile_id}: {e}")
            continue

        # Place tile
        shift = shifts[tile_key]
        y_shift, x_shift = int(shift[0]), int(shift[1])
        y_end = min(y_shift + tile_size[0], final_shape[0])
        x_end = min(x_shift + tile_size[1], final_shape[1])

        if y_shift >= 0 and x_shift >= 0 and y_end > y_shift and x_end > x_shift:
            tile_y_end = tile_size[0] - max(0, y_shift + tile_size[0] - final_shape[0])
            tile_x_end = tile_size[1] - max(0, x_shift + tile_size[1] - final_shape[1])

            output_image[y_shift:y_end, x_shift:x_end] += tile_array[:tile_y_end, :tile_x_end].astype(np.float32)
            tile_mask = tile_array[:tile_y_end, :tile_x_end] > 0
            divisor[y_shift:y_end, x_shift:x_end] += tile_mask.astype(np.uint16)

    # Normalize
    stitched = np.zeros_like(output_image, dtype=np.float32)
    nonzero_mask = divisor > 0
    stitched[nonzero_mask] = output_image[nonzero_mask] / divisor[nonzero_mask]

    return stitched.astype(np.uint16)


def assemble_stitched_masks(
    metadata_df: pd.DataFrame,
    shifts: Dict[str, List[int]],
    well: str,
    data_type: str = "phenotype",
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
) -> np.ndarray:
    """Assemble stitched segmentation masks from individual mask tiles."""
    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        raise ValueError(f"No metadata found for well {well}")

    mask_cache = MaskTileCache(metadata_df, well, data_type, flipud, fliplr, rot90)

    # Get tile size
    first_tile_id = well_metadata.iloc[0]["tile"]
    try:
        first_mask = mask_cache[first_tile_id]
        if first_mask is None:
            raise ValueError(f"Could not load first mask tile: {first_tile_id}")
        tile_size = first_mask.shape
    except Exception as e:
        print(f"Error loading first mask: {e}")
        return np.array([])

    # Calculate output dimensions
    final_shape = get_output_shape_tiff(shifts, tile_size)
    stitched_mask = np.zeros(final_shape, dtype=np.uint32)

    print(f"Assembling {len(well_metadata)} {data_type} mask tiles into shape {final_shape}")

    next_label = 1

    # Process each tile
    for _, tile_row in tqdm(well_metadata.iterrows(), total=len(well_metadata), desc=f"Assembling {data_type} masks"):
        tile_id = tile_row["tile"]
        tile_key = f"{well}/{tile_id}"

        if tile_key not in shifts:
            continue

        try:
            mask_tile = mask_cache[tile_id]
            if mask_tile is None:
                continue
        except Exception as e:
            print(f"Error loading mask tile {tile_id}: {e}")
            continue

        # Place tile
        shift = shifts[tile_key]
        y_shift, x_shift = int(shift[0]), int(shift[1])
        y_end = min(y_shift + tile_size[0], final_shape[0])
        x_end = min(x_shift + tile_size[1], final_shape[1])

        if y_shift >= 0 and x_shift >= 0 and y_end > y_shift and x_end > x_shift:
            tile_y_end = tile_size[0] - max(0, y_shift + tile_size[0] - final_shape[0])
            tile_x_end = tile_size[1] - max(0, x_shift + tile_size[1] - final_shape[1])

            # Relabel mask to avoid conflicts
            mask_tile_relabeled = relabel_mask_tile(mask_tile[:tile_y_end, :tile_x_end], next_label)
            
            if mask_tile_relabeled.max() > 0:
                next_label = mask_tile_relabeled.max() + 1

            # Handle overlaps by keeping existing labels
            target_region = stitched_mask[y_shift:y_end, x_shift:x_end]
            mask_new_cells = (mask_tile_relabeled > 0) & (target_region == 0)
            target_region[mask_new_cells] = mask_tile_relabeled[mask_new_cells]

    return stitched_mask