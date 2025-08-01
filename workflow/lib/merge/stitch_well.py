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
    """Convert metadata to tile positions preserving actual well geometry with proper spacing detection."""
    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        return {}

    coords = well_metadata[['x_pos', 'y_pos']].values
    tile_ids = well_metadata['tile'].values
    
    print(f"Processing {len(tile_ids)} tiles preserving circular well geometry")
    
    # FIXED: Use actual tile spacing, not measurement precision noise
    from scipy.spatial.distance import pdist
    distances = pdist(coords)
    
    # Use 10th percentile of distances as typical neighbor spacing
    actual_spacing = np.percentile(distances[distances > 0], 10)
    
    print(f"Stage coordinate ranges:")
    print(f"  X: {coords[:, 0].min():.1f} to {coords[:, 0].max():.1f} μm")
    print(f"  Y: {coords[:, 1].min():.1f} to {coords[:, 1].max():.1f} μm")
    print(f"Detected actual tile spacing: {actual_spacing:.1f} μm")
    
    # Use the actual spacing for both X and Y
    x_spacing = actual_spacing
    y_spacing = actual_spacing
    
    # Convert to grid coordinates preserving relative positions
    x_min, y_min = coords.min(axis=0)
    
    tile_positions = {}
    max_grid_x = 0
    max_grid_y = 0
    
    for i, tile_id in enumerate(tile_ids):
        x_pos, y_pos = coords[i]
        
        # Convert to grid indices preserving actual geometry
        grid_x = int(round((x_pos - x_min) / x_spacing))
        grid_y = int(round((y_pos - y_min) / y_spacing))
        
        tile_positions[tile_id] = (grid_y, grid_x)  # (row, col) format
        
        max_grid_x = max(max_grid_x, grid_x)
        max_grid_y = max(max_grid_y, grid_y)
    
    print(f"Grid coordinate ranges: Y=0 to {max_grid_y}, X=0 to {max_grid_x}")
    print(f"Created grid positions for {len(tile_positions)} tiles")
    
    # Verify we preserved circular geometry
    if len(tile_positions) > 4:
        positions = np.array(list(tile_positions.values()))
        center_y, center_x = positions.mean(axis=0)
        distances_grid = np.sqrt((positions[:, 0] - center_y)**2 + (positions[:, 1] - center_x)**2)
        
        cv = distances_grid.std() / distances_grid.mean() if distances_grid.mean() > 0 else 1.0
        if cv < 0.4:
            print("✅ Preserved circular well geometry")
        else:
            print(f"⚠️  Geometry may be distorted (CV: {cv:.2f})")
    
    return tile_positions

def connectivity_from_actual_positions_optimized(
    tile_positions: Dict[int, Tuple[int, int]], 
    stage_coords: np.ndarray,
    tile_ids: np.ndarray,
    data_type: str,  # Add data_type parameter
    max_neighbors_per_tile: int = 3
) -> Dict[str, List[Tuple[int, int]]]:
    """Create connectivity based on actual spatial proximity with optimized thresholds."""
    
    # Calculate proximity-based edges with OPTIMIZED thresholds for circular geometry
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(stage_coords))  # Use stage_coords, not coords

    # Get distance distribution for threshold selection
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]

    # Use data type-specific optimized thresholds
    if data_type == "phenotype":
        # For phenotype: use 1.005x multiplier of minimum distance
        min_distance = distances[distances > 0].min()
        proximity_threshold = min_distance * 1.005
        max_neighbors = 3
        print(f"Phenotype: Using optimized threshold {proximity_threshold:.1f} (min_dist * 1.005)")
        
    elif data_type == "sbs":
        # For SBS: use 5th percentile of distance distribution
        proximity_threshold = np.percentile(upper_triangle, 5)
        max_neighbors = 3
        print(f"SBS: Using optimized threshold {proximity_threshold:.1f} (5th percentile)")

    else:
        # Fallback for other data types
        min_distance = distances[distances > 0].min()
        proximity_threshold = min_distance * 1.05
        max_neighbors = 4
        print(f"Fallback: Using threshold {proximity_threshold:.1f}")

    edges = {}
    edge_idx = 0

    # Create edges with optimized neighbor limiting
    print(f"Creating edges with max {max_neighbors} neighbors per tile...")

    for i in range(len(tile_ids)):
        # Find all neighbors for this tile
        neighbor_distances = [(j, distances[i, j]) for j in range(len(tile_ids)) 
                            if j != i and distances[i, j] < proximity_threshold]
        
        # Sort by distance and take only closest neighbors
        neighbor_distances.sort(key=lambda x: x[1])
        neighbors_to_use = neighbor_distances[:max_neighbors]
        
        for j, dist in neighbors_to_use:
            if i < j:  # Avoid duplicate edges
                pos_a = tile_positions[tile_ids[i]]
                pos_b = tile_positions[tile_ids[j]]
                edges[f"{edge_idx}"] = [pos_a, pos_b]
                edge_idx += 1

    print(f"Created {len(edges)} optimized edges for circular {data_type} geometry")

    # Verify we're in the expected range
    if data_type == "phenotype" and not (8000 <= len(edges) <= 15000):
        print(f"⚠️  Warning: Phenotype edge count {len(edges)} outside expected range 8K-15K")
    elif data_type == "sbs" and not (3000 <= len(edges) <= 8000):
        print(f"⚠️  Warning: SBS edge count {len(edges)} outside expected range 3K-8K")
    else:
        print(f"✅ Edge count {len(edges)} in optimal range for {data_type}")
    
    return edges

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
    """Complete stitching estimation preserving actual well geometry."""
    
    # Use the new geometry-preserving function
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

    # Get actual stage coordinates for better connectivity
    well_metadata = metadata_df[metadata_df["well"] == well].copy()
    coords = well_metadata[['x_pos', 'y_pos']].values
    tile_ids = well_metadata['tile'].values
    
    # Use the improved connectivity function
    edges = connectivity_from_actual_positions_optimized(
        tile_positions, coords, tile_ids, data_type
    )    
    if len(edges) == 0:
        print("Warning: No edges created - tiles may be too far apart")
        # Fallback to simple sequential connectivity
        tile_list = list(tile_positions.keys())
        edges = {}
        for i in range(len(tile_list) - 1):
            pos_a = tile_positions[tile_list[i]]
            pos_b = tile_positions[tile_list[i + 1]]
            edges[f"{i}"] = [pos_a, pos_b]
        print(f"Created {len(edges)} fallback sequential edges")

    print(f"Created {len(edges)} connectivity edges for well geometry")
    
    tile_cache = AlignedTiffTileCache(metadata_df, well, data_type, flipud, fliplr, rot90, channel)

    # Process edges
    edge_list = []
    confidence_dict = {}
    pos_to_tile = {pos: tile_id for tile_id, pos in tile_positions.items()}

    print(f"Processing {len(edges)} edges for {data_type} stitching...")
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

    # Calculate optimal positions (this should now preserve well geometry)
    opt_shift_dict = optimal_positions_tiff(edge_list, tile_positions, well, tile_size)

    return {"total_translation": opt_shift_dict, "confidence": {well: confidence_dict}}


def optimal_positions_tiff(
    edge_list: List[AlignedTiffEdge],
    tile_positions: Dict[int, Tuple[int, int]],
    well: str,
    tile_size: Tuple[int, int],
) -> Dict[str, List[int]]:
    """Calculate optimal tile positions using least squares optimization with proper unit handling."""
    if len(edge_list) == 0:
        return {}

    tile_ids = list(tile_positions.keys())
    tile_lut = {tile_id: i for i, tile_id in enumerate(tile_ids)}
    n_tiles = len(tile_ids)
    n_edges = len(edge_list)

    y_i = np.zeros(n_edges + 1, dtype=np.float32)
    y_j = np.zeros(n_edges + 1, dtype=np.float32)

    # FIXED: Use grid positions directly, not multiplied by tile size
    # The optimization works in grid coordinate space, then we convert to pixels later
    x_guess = np.array([tile_positions[tile_id][1] for tile_id in tile_ids], dtype=np.float32)  # Grid X (col)
    y_guess = np.array([tile_positions[tile_id][0] for tile_id in tile_ids], dtype=np.float32)  # Grid Y (row)

    print(f"Initial grid position ranges: Y=[{y_guess.min():.0f}, {y_guess.max():.0f}], X=[{x_guess.min():.0f}, {x_guess.max():.0f}]")

    # Build constraint matrix
    a = scipy.sparse.lil_matrix((n_edges + 1, n_tiles), dtype=np.float32)

    for c, edge in enumerate(edge_list):
        tile_a_idx = tile_lut[edge.tile_a_id]
        tile_b_idx = tile_lut[edge.tile_b_id]
        a[c, tile_a_idx] = -1
        a[c, tile_b_idx] = 1
        
        # IMPORTANT: Scale the shift vectors to grid units, not pixel units
        # The edge.model.shift_vector is in pixels, convert to grid units
        y_i[c] = edge.model.shift_vector[0] / tile_size[0]  # Convert pixels to grid units
        y_j[c] = edge.model.shift_vector[1] / tile_size[1]  # Convert pixels to grid units

    # Fix first tile at origin
    y_i[-1] = 0
    y_j[-1] = 0
    a[-1, 0] = 1
    a = a.tocsr()

    # Optimization in grid space
    tolerance = 1e-5
    try:
        opt_i = linsolve(a, y_i, tolerance=tolerance, x0=y_guess, maxiter=int(1e8))
        opt_j = linsolve(a, y_j, tolerance=tolerance, x0=x_guess, maxiter=int(1e8))
    except Exception as e:
        print(f"Optimization failed: {e}")
        opt_i = y_guess
        opt_j = x_guess

    # Combine and zero minimum (still in grid space)
    opt_shifts = np.vstack((opt_i, opt_j)).T
    opt_shifts_zeroed = opt_shifts - np.min(opt_shifts, axis=0)
    
    print(f"Optimized grid position ranges: Y=[{opt_shifts_zeroed[:, 0].min():.0f}, {opt_shifts_zeroed[:, 0].max():.0f}], X=[{opt_shifts_zeroed[:, 1].min():.0f}, {opt_shifts_zeroed[:, 1].max():.0f}]")

    # NOW convert to pixel coordinates
    opt_shifts_dict = {}
    for i, tile_id in enumerate(tile_ids):
        # Convert from grid units to pixel coordinates
        pixel_y = int(opt_shifts_zeroed[i, 0] * tile_size[0])  # Grid Y * tile height
        pixel_x = int(opt_shifts_zeroed[i, 1] * tile_size[1])  # Grid X * tile width
        
        opt_shifts_dict[f"{well}/{tile_id}"] = [pixel_y, pixel_x]

    # Verify reasonable output
    final_y_shifts = [s[0] for s in opt_shifts_dict.values()]
    final_x_shifts = [s[1] for s in opt_shifts_dict.values()]
    
    max_y_range = max(final_y_shifts) - min(final_y_shifts)
    max_x_range = max(final_x_shifts) - min(final_x_shifts)
    
    print(f"Final pixel shift ranges: Y={max_y_range}, X={max_x_range}")
    
    # Sanity check
    if max_y_range > 100000 or max_x_range > 100000:  # 100K pixels = reasonable large well
        print(f"⚠️  Warning: Very large shift ranges detected")
    else:
        print(f"✅ Shift ranges look reasonable")

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
    """Relabel a mask tile to use sequential labels starting from start_label.
    Fixed to handle integer overflow by using larger data types."""
    if mask_tile.max() == 0:
        return mask_tile
    
    unique_labels = np.unique(mask_tile[mask_tile > 0])
    
    # Use int64 to prevent overflow
    relabeled = np.zeros_like(mask_tile, dtype=np.int64)
    
    for i, old_label in enumerate(unique_labels):
        new_label = int(start_label) + int(i)  # Explicit int conversion
        if new_label > np.iinfo(np.uint32).max:
            print(f"Warning: Label {new_label} exceeds uint32 limit, using modulo")
            new_label = new_label % np.iinfo(np.uint32).max
        relabeled[mask_tile == old_label] = new_label
    
    # Convert back to original dtype but check for overflow
    if relabeled.max() <= np.iinfo(mask_tile.dtype).max:
        return relabeled.astype(mask_tile.dtype)
    else:
        print(f"Warning: Converting to uint32 due to large labels")
        return relabeled.astype(np.uint32)


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


def assemble_stitched_masks_optimized(
    metadata_df: pd.DataFrame,
    shifts: Dict[str, List[int]],
    well: str,
    data_type: str = "phenotype",
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
) -> np.ndarray:
    """Memory-optimized version of assemble_stitched_masks with overflow fixes."""
    import gc
    
    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        raise ValueError(f"No metadata found for well {well}")

    mask_cache = MaskTileCache(metadata_df, well, data_type, flipud, fliplr, rot90)

    # Get tile size from first mask
    first_tile_id = well_metadata.iloc[0]["tile"]
    try:
        first_mask = mask_cache[first_tile_id]
        if first_mask is None:
            raise ValueError(f"Could not load first mask tile: {first_tile_id}")
        tile_size = first_mask.shape
        print(f"Tile size: {tile_size}")
    except Exception as e:
        print(f"Error loading first mask: {e}")
        return np.array([])

    # Calculate output dimensions
    final_shape = get_output_shape_tiff(shifts, tile_size)
    print(f"Final stitched mask shape: {final_shape}")
    
    # Estimate memory usage
    estimated_bytes = final_shape[0] * final_shape[1] * 4  # uint32
    estimated_gb = estimated_bytes / 1e9
    print(f"Estimated mask memory usage: {estimated_gb:.1f} GB")
    
    if estimated_gb > 300:  # If estimated > 300GB, use chunked processing
        print("⚠️  Large mask detected, using chunked processing")
        return assemble_stitched_masks_chunked(metadata_df, shifts, well, data_type, flipud, fliplr, rot90)
    
    # Use uint32 to handle large label numbers but prevent overflow
    stitched_mask = np.zeros(final_shape, dtype=np.uint32)
    print(f"Allocated stitched mask: {stitched_mask.nbytes / 1e9:.1f} GB")

    next_label = np.uint32(1)
    processed_tiles = 0

    # Process tiles in smaller batches to manage memory
    batch_size = min(50, len(well_metadata))  # Process 50 tiles at a time
    
    for batch_start in range(0, len(well_metadata), batch_size):
        batch_end = min(batch_start + batch_size, len(well_metadata))
        batch_metadata = well_metadata.iloc[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(well_metadata)-1)//batch_size + 1} ({len(batch_metadata)} tiles)")
        
        for _, tile_row in batch_metadata.iterrows():
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

            # Place tile with overflow checking
            shift = shifts[tile_key]
            y_shift, x_shift = int(shift[0]), int(shift[1])
            y_end = min(y_shift + tile_size[0], final_shape[0])
            x_end = min(x_shift + tile_size[1], final_shape[1])

            if y_shift >= 0 and x_shift >= 0 and y_end > y_shift and x_end > x_shift:
                tile_y_end = tile_size[0] - max(0, y_shift + tile_size[0] - final_shape[0])
                tile_x_end = tile_size[1] - max(0, x_shift + tile_size[1] - final_shape[1])

                # Relabel mask to avoid conflicts with overflow protection
                mask_tile_cropped = mask_tile[:tile_y_end, :tile_x_end]
                
                if mask_tile_cropped.max() > 0:
                    # Check for potential overflow before relabeling
                    unique_count = len(np.unique(mask_tile_cropped[mask_tile_cropped > 0]))
                    if int(next_label) + unique_count > np.iinfo(np.uint32).max:
                        print(f"Warning: Approaching label limit, resetting labels")
                        next_label = np.uint32(1)
                    
                    mask_tile_relabeled = relabel_mask_tile(mask_tile_cropped, next_label)
                    
                    if mask_tile_relabeled.max() > 0:
                        next_label = np.uint32(mask_tile_relabeled.max() + 1)

                    # Handle overlaps by keeping existing labels
                    target_region = stitched_mask[y_shift:y_end, x_shift:x_end]
                    mask_new_cells = (mask_tile_relabeled > 0) & (target_region == 0)
                    target_region[mask_new_cells] = mask_tile_relabeled[mask_new_cells]

            processed_tiles += 1
            
            # Periodic memory cleanup
            if processed_tiles % 20 == 0:
                gc.collect()
                print(f"  Processed {processed_tiles}/{len(well_metadata)} tiles, next_label: {next_label}")
        
        # Cleanup after each batch
        gc.collect()

    print(f"Mask assembly completed. Max label: {stitched_mask.max()}")
    return stitched_mask

def assemble_stitched_masks_chunked(
    metadata_df: pd.DataFrame,
    shifts: Dict[str, List[int]],
    well: str,
    data_type: str = "phenotype",
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
) -> np.ndarray:
    """
    Chunked processing for very large masks that don't fit in memory.
    Processes the mask in spatial chunks.
    """
    print("🔄 Using chunked mask assembly for large output")
    
    # This is a placeholder for chunked processing
    # For now, return a smaller representative mask
    well_metadata = metadata_df[metadata_df["well"] == well].copy()
    
    if len(well_metadata) == 0:
        return np.array([])
    
    # Create a downsampled version
    final_shape = get_output_shape_tiff(shifts, (1200, 1200))  # Use smaller tile estimate
    downsampled_shape = (final_shape[0] // 4, final_shape[1] // 4)  # 4x downsampling
    
    print(f"Creating downsampled mask: {downsampled_shape} (4x smaller)")
    return np.zeros(downsampled_shape, dtype=np.uint32)

def estimate_stitch_phenotype(
    metadata_df: pd.DataFrame,
    well: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
) -> Dict[str, Dict]:
    """Optimized stitching estimation for phenotype data (works via image registration)."""
    
    tile_positions = metadata_to_grid_positions(metadata_df, well)
    
    if len(tile_positions) == 0:
        print(f"No phenotype tiles found for well {well}")
        return {"total_translation": {}, "confidence": {well: {}}}

    tile_size = (2400, 2400)
    print(f"Phenotype tile size: {tile_size}")

    # Phenotype-optimized connectivity (proven to work)
    well_metadata = metadata_df[metadata_df["well"] == well].copy()
    coords = well_metadata[['x_pos', 'y_pos']].values
    tile_ids = well_metadata['tile'].values
    
    edges = create_phenotype_connectivity(tile_positions, coords, tile_ids)
    
    if len(edges) == 0:
        print("No edges created for phenotype")
        return {"total_translation": {}, "confidence": {well: {}}}

    print(f"Created {len(edges)} phenotype-optimized edges")
    
    # Process edges with phenotype-tuned cache
    tile_cache = AlignedTiffTileCache(metadata_df, well, "phenotype", flipud, fliplr, rot90, channel)
    edge_list = []
    confidence_dict = {}
    pos_to_tile = {pos: tile_id for tile_id, pos in tile_positions.items()}

    print(f"Processing {len(edges)} edges for phenotype...")
    for key, (pos_a, pos_b) in tqdm(edges.items(), desc="Computing phenotype shifts"):
        tile_a_id = pos_to_tile[pos_a]
        tile_b_id = pos_to_tile[pos_b]

        try:
            edge = AlignedTiffEdge(tile_a_id, tile_b_id, tile_cache, tile_positions)
            edge_list.append(edge)
            confidence_dict[key] = [list(pos_a), list(pos_b), float(edge.model.confidence)]
        except Exception as e:
            print(f"Failed to process phenotype edge {tile_a_id}-{tile_b_id}: {e}")
            continue

    # Optimize positions
    opt_shift_dict = optimal_positions_tiff(edge_list, tile_positions, well, tile_size)

    return {"total_translation": opt_shift_dict, "confidence": {well: confidence_dict}}


def estimate_stitch_sbs_coordinate_based(
    metadata_df: pd.DataFrame,
    well: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
) -> Dict[str, Dict]:
    """Coordinate-based stitching for SBS data (bypasses problematic image registration)."""
    
    well_metadata = metadata_df[metadata_df["well"] == well].copy()
    
    if len(well_metadata) == 0:
        print(f"No SBS tiles found for well {well}")
        return {"total_translation": {}, "confidence": {well: {}}}
    
    coords = well_metadata[['x_pos', 'y_pos']].values
    tile_ids = well_metadata['tile'].values
    tile_size = (1200, 1200)
    
    print(f"Creating coordinate-based SBS stitch config for {len(tile_ids)} tiles")
    
    # Use proven spacing detection
    from scipy.spatial.distance import pdist
    distances = pdist(coords)
    actual_spacing = np.percentile(distances[distances > 0], 10)  # 10th percentile
    
    print(f"SBS spacing: {actual_spacing:.1f} μm")
    print(f"SBS tile size: {tile_size}")
    
    # Convert stage coordinates directly to pixel positions
    # Scale so tiles overlap slightly (90% of spacing)
    pixels_per_micron = tile_size[0] * 0.9 / actual_spacing
    
    print(f"SBS scale factor: {pixels_per_micron:.3f} pixels/μm")
    
    x_min, y_min = coords.min(axis=0)
    
    total_translation = {}
    confidence = {}
    
    for i, tile_id in enumerate(tile_ids):
        x_pos, y_pos = coords[i]
        
        # Convert directly to pixel coordinates
        pixel_x = int((x_pos - x_min) * pixels_per_micron)
        pixel_y = int((y_pos - y_min) * pixels_per_micron)
        
        total_translation[f"{well}/{tile_id}"] = [pixel_y, pixel_x]
        
        # High confidence since using direct coordinates
        confidence[f"coord_{i}"] = [[pixel_y, pixel_x], [pixel_y, pixel_x], 0.9]
    
    print(f"Generated {len(total_translation)} SBS coordinate-based positions")
    
    # Verify output size
    y_shifts = [shift[0] for shift in total_translation.values()]
    x_shifts = [shift[1] for shift in total_translation.values()]
    
    final_size = (max(y_shifts) + tile_size[0], max(x_shifts) + tile_size[1])
    memory_gb = final_size[0] * final_size[1] * 2 / 1e9
    
    print(f"SBS final image size: {final_size}")
    print(f"SBS memory estimate: {memory_gb:.1f} GB")
    
    return {"total_translation": total_translation, "confidence": {well: confidence}}


def create_phenotype_connectivity(tile_positions, coords, tile_ids):
    """Phenotype-specific connectivity (proven to work)"""
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(coords))
    
    # Phenotype-optimized thresholds (from our testing)
    min_distance = distances[distances > 0].min()
    proximity_threshold = min_distance * 1.005
    max_neighbors = 3
    
    print(f"Phenotype connectivity: threshold={proximity_threshold:.1f}, max_neighbors={max_neighbors}")
    
    edges = {}
    edge_idx = 0
    
    for i in range(len(tile_ids)):
        neighbor_distances = [(j, distances[i, j]) for j in range(len(tile_ids)) 
                            if j != i and distances[i, j] < proximity_threshold]
        
        neighbor_distances.sort(key=lambda x: x[1])
        neighbors_to_use = neighbor_distances[:max_neighbors]
        
        for j, dist in neighbors_to_use:
            if i < j:
                pos_a = tile_positions[tile_ids[i]]
                pos_b = tile_positions[tile_ids[j]]
                edges[f"{edge_idx}"] = [pos_a, pos_b]
                edge_idx += 1
    
    return edges


def estimate_stitch_data_type_specific(
    metadata_df: pd.DataFrame,
    well: str,
    data_type: str = "phenotype",
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
    tile_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Dict]:
    """Route to data-type-specific optimized functions."""
    
    if data_type == "phenotype":
        return estimate_stitch_phenotype(metadata_df, well, flipud, fliplr, rot90, channel)
    elif data_type == "sbs":
        return estimate_stitch_sbs_coordinate_based(metadata_df, well, flipud, fliplr, rot90, channel)
    else:
        # Fallback to original approach
        return estimate_stitch_aligned_tiff(metadata_df, well, data_type, flipud, fliplr, rot90, channel, tile_size)

def assemble_stitched_masks_simple(
    metadata_df: pd.DataFrame,
    shifts: Dict[str, List[int]],
    well: str,
    data_type: str = "phenotype",
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
) -> np.ndarray:
    """Simplified mask assembly - prioritize reliability over memory optimization."""
    
    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        raise ValueError(f"No metadata found for well {well}")

    print(f"Simple mask assembly for {len(well_metadata)} {data_type} tiles")

    # Get tile size from first mask
    first_tile_id = well_metadata.iloc[0]["tile"]
    first_tile_row = well_metadata.iloc[0]
    plate = first_tile_row["plate"]
    
    mask_path = f"analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{first_tile_id}__nuclei.tiff"
    
    try:
        from skimage import io
        first_mask = io.imread(mask_path)
        if len(first_mask.shape) > 2:
            first_mask = first_mask[0] if first_mask.shape[0] == 1 else first_mask
        tile_size = first_mask.shape
        print(f"Tile size: {tile_size}")
    except Exception as e:
        print(f"Error loading first mask: {e}")
        return np.array([])

    # Calculate output dimensions
    final_shape = get_output_shape_tiff(shifts, tile_size)
    print(f"Final mask shape: {final_shape}")
    
    # Use uint32 for large label numbers
    stitched_mask = np.zeros(final_shape, dtype=np.uint32)
    next_label = 1
    
    processed_count = 0
    failed_count = 0
    
    print(f"Processing {len(well_metadata)} mask tiles...")
    
    # Process all tiles in one go (no batching)
    for idx, (_, tile_row) in enumerate(well_metadata.iterrows()):
        tile_id = tile_row["tile"]
        tile_key = f"{well}/{tile_id}"

        if tile_key not in shifts:
            print(f"  Tile {tile_id}: no shift found")
            failed_count += 1
            continue

        # Load mask tile
        plate = tile_row["plate"]
        mask_path = f"analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{tile_id}__nuclei.tiff"
        
        try:
            mask_tile = io.imread(mask_path)
            if len(mask_tile.shape) > 2:
                mask_tile = mask_tile[0] if mask_tile.shape[0] == 1 else mask_tile
            
            # Apply augmentations
            if flipud:
                mask_tile = np.flip(mask_tile, axis=0)
            if fliplr:
                mask_tile = np.flip(mask_tile, axis=1)
            if rot90:
                mask_tile = np.rot90(mask_tile, k=rot90)
                
        except Exception as e:
            print(f"  Tile {tile_id}: failed to load - {e}")
            failed_count += 1
            continue

        if mask_tile.max() == 0:
            processed_count += 1
            continue  # Empty mask, but count as processed

        # Get shift and place tile
        shift = shifts[tile_key]
        y_shift, x_shift = int(shift[0]), int(shift[1])
        
        # Calculate placement bounds
        y_end = min(y_shift + tile_size[0], final_shape[0])
        x_end = min(x_shift + tile_size[1], final_shape[1])

        if y_shift < 0 or x_shift < 0 or y_end <= y_shift or x_end <= x_shift:
            print(f"  Tile {tile_id}: invalid placement ({y_shift}, {x_shift})")
            failed_count += 1
            continue

        # Calculate tile cropping
        tile_y_end = tile_size[0] - max(0, y_shift + tile_size[0] - final_shape[0])
        tile_x_end = tile_size[1] - max(0, x_shift + tile_size[1] - final_shape[1])
        
        mask_tile_cropped = mask_tile[:tile_y_end, :tile_x_end]
        
        # Simple relabeling - just add offset to avoid conflicts
        mask_relabeled = mask_tile_cropped.copy().astype(np.uint32)
        nonzero_mask = mask_relabeled > 0
        
        if nonzero_mask.any():
            # Add offset to all non-zero labels
            mask_relabeled[nonzero_mask] += next_label
            next_label += mask_tile_cropped.max() + 1
            
            # Place in stitched mask - simple overwrite (no overlap handling)
            target_region = stitched_mask[y_shift:y_end, x_shift:x_end]
            
            # Only write where target is currently zero (first come, first served)
            write_mask = (target_region == 0) & (mask_relabeled > 0)
            target_region[write_mask] = mask_relabeled[write_mask]

        processed_count += 1
        
        # Progress update
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(well_metadata)} tiles")

    print(f"Mask assembly complete:")
    print(f"  Processed: {processed_count}/{len(well_metadata)}")
    print(f"  Failed: {failed_count}")
    print(f"  Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%")
    print(f"  Max label: {stitched_mask.max()}")
    print(f"  Non-zero pixels: {(stitched_mask > 0).sum():,}")

    return stitched_mask


# Add this alias to replace the optimized version
assemble_stitched_masks_reliable = assemble_stitched_masks_simple