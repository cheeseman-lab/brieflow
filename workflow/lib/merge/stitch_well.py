"""
Adapted stitching functions to work with TIFF files and metadata instead of OME-ZARR.
These functions integrate with the existing BrieFlow pipeline structure.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Dict, Tuple
from tqdm import tqdm
import scipy
from pathlib import Path
import cv2
from skimage import io

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


class TiffTileCache:
    """
    Cache for loading TIFF tiles from ND2 files using metadata for positioning.
    Replaces the OME-ZARR based TileCache.
    """

    def __init__(self, metadata_df: pd.DataFrame, well: str, flipud: bool = False, 
                 fliplr: bool = False, rot90: int = 0, channel: int = 0):
        self.cache = LimitedSizeDict(max_size=20)
        self.metadata_df = metadata_df[metadata_df['well'] == well].copy()
        self.well = well
        self.flipud = flipud
        self.fliplr = fliplr
        self.rot90 = rot90
        self.channel = channel  # Which channel to use for registration

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
        """Load a tile from ND2 file and add it to cache"""
        tile_metadata = self.metadata_df[self.metadata_df['tile'] == tile_id]
        
        if len(tile_metadata) == 0:
            print(f"No metadata found for tile {tile_id}")
            return None
            
        tile_row = tile_metadata.iloc[0]
        nd2_path = tile_row['filename']
        
        try:
            # Load the ND2 file and convert to TIFF format
            tile_array = nd2_to_tiff(
                files=nd2_path,
                channel_order_flip=False,
                verbose=False,
                z_handling="max_projection"  # or use your preferred z_handling
            )
            
            # Extract the specific channel for registration
            if len(tile_array.shape) > 2:
                tile_2d = tile_array[self.channel, :, :]
            else:
                tile_2d = tile_array
                
            # Apply augmentations
            aug_tile = augment_tile(tile_2d, self.flipud, self.fliplr, self.rot90)
            
            # Cache the result
            self.cache[tile_id] = aug_tile
            return aug_tile
            
        except Exception as e:
            print(f"Error loading tile {tile_id}: {e}")
            return None


class TiffEdge:
    """
    Edge between two tiles for registration, adapted for TIFF pipeline.
    """

    def __init__(self, tile_a_id: int, tile_b_id: int, tile_cache: TiffTileCache, 
                 tile_positions: Dict[int, Tuple[int, int]]):
        self.tile_cache = tile_cache
        self.tile_a_id = tile_a_id
        self.tile_b_id = tile_b_id
        
        # Get grid positions
        self.pos_a = tile_positions[tile_a_id]
        self.pos_b = tile_positions[tile_b_id]
        
        # Calculate relationship between tiles
        self.relation = (self.pos_a[0] - self.pos_b[0], self.pos_a[1] - self.pos_b[1])
        
        # Calculate the offset
        self.model = self.get_offset()

    def get_offset(self) -> TranslationRegistrationModel:
        """Calculate offset between two tiles"""
        tile_a = self.tile_cache[self.tile_a_id]
        tile_b = self.tile_cache[self.tile_b_id]
        
        if tile_a is None or tile_b is None:
            # Return a dummy model with zero shift if tiles can't be loaded
            model = TranslationRegistrationModel()
            model.shift_vector = np.array([0.0, 0.0])
            model.confidence = 0.0
            return model
            
        return offset_tiff(tile_a, tile_b, self.relation, overlap=150)


def augment_tile(tile: np.ndarray, flipud: bool, fliplr: bool, rot90: int) -> np.ndarray:
    """Apply flips and rotations to a tile"""
    if flipud:
        tile = np.flip(tile, axis=-2)
    if fliplr:
        tile = np.flip(tile, axis=-1)
    if rot90:
        tile = np.rot90(tile, k=rot90, axes=(-2, -1))
    return tile


def offset_tiff(image_a: np.ndarray, image_b: np.ndarray, relation: Tuple[int, int], 
                overlap: int) -> TranslationRegistrationModel:
    """
    Calculate offset between two images based on their spatial relationship.
    
    Args:
        image_a: First image
        image_b: Second image  
        relation: Tuple indicating relative position (tile_a_pos - tile_b_pos)
        overlap: Estimated overlap in pixels
    """
    shape = image_a.shape
    
    # Determine overlap region based on spatial relationship
    if relation[0] == -1:
        # tile_b is to the right of tile_a
        roi_a = image_a[:, -overlap:]
        roi_b = image_b[:, :overlap]
        corr_x = shape[-2] - overlap
        corr_y = 0
    elif relation[0] == 1:
        # tile_b is to the left of tile_a  
        roi_a = image_a[:, :overlap]
        roi_b = image_b[:, -overlap:]
        corr_x = -overlap
        corr_y = 0
    elif relation[1] == -1:
        # tile_b is below tile_a
        roi_a = image_a[-overlap:, :]
        roi_b = image_b[:overlap, :]
        corr_x = 0
        corr_y = shape[-1] - overlap
    elif relation[1] == 1:
        # tile_b is above tile_a
        roi_a = image_a[:overlap, :]
        roi_b = image_b[-overlap:, :]
        corr_x = 0
        corr_y = -overlap
    else:
        # No direct adjacency, use center regions
        center_size = min(overlap, min(shape) // 4)
        center_y = shape[0] // 2
        center_x = shape[1] // 2
        half_size = center_size // 2
        
        roi_a = image_a[center_y-half_size:center_y+half_size, 
                       center_x-half_size:center_x+half_size]
        roi_b = image_b[center_y-half_size:center_y+half_size,
                       center_x-half_size:center_x+half_size]
        corr_x = corr_y = 0

    # Ensure positive values for registration
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


def connectivity_from_grid(positions: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Create connectivity graph for grid positions.
    Returns edges between neighboring tiles.
    """
    position_set = set(positions)
    edges = {}
    edge_idx = 0
    
    # Define neighbor directions (right, down)
    directions = [(0, 1), (1, 0)]
    
    for pos in positions:
        x, y = pos
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if neighbor in position_set:
                edges[f"{edge_idx}"] = [pos, neighbor]
                edge_idx += 1
                
    return edges


def metadata_to_grid_positions(metadata_df: pd.DataFrame, well: str) -> Dict[int, Tuple[int, int]]:
    """
    Convert metadata to grid positions for tiles in a well with continuous stage coordinates.
    """
    return metadata_to_grid_positions_continuous(metadata_df, well, grid_spacing_tolerance=50.0)

def metadata_to_grid_positions_continuous(metadata_df: pd.DataFrame, well: str, 
                                          grid_spacing_tolerance: float = 50.0) -> Dict[int, Tuple[int, int]]:
    """
    Convert metadata to grid positions for tiles in a well with continuous stage coordinates.
    
    This version handles snaking patterns where each tile has slightly different coordinates
    by binning them into a regular grid based on spacing.
    
    Args:
        metadata_df: DataFrame with tile metadata including x_pos, y_pos
        well: Well identifier
        grid_spacing_tolerance: Tolerance for considering positions as being in the same grid row/column
        
    Returns:
        Dictionary mapping tile_id to (grid_x, grid_y) positions
    """
    well_metadata = metadata_df[metadata_df['well'] == well].copy()
    
    if len(well_metadata) == 0:
        return {}
    
    # Sort by y first (rows), then by x (columns within rows)
    well_metadata = well_metadata.sort_values(['y_pos', 'x_pos'])
    
    x_positions = well_metadata['x_pos'].values
    y_positions = well_metadata['y_pos'].values
    
    # Estimate grid spacing by looking at position differences
    x_diffs = np.diff(np.sort(x_positions))
    y_diffs = np.diff(np.sort(y_positions))
    
    # Remove very small differences (likely measurement noise)
    x_diffs = x_diffs[x_diffs > grid_spacing_tolerance]
    y_diffs = y_diffs[y_diffs > grid_spacing_tolerance]
    
    if len(x_diffs) == 0 or len(y_diffs) == 0:
        print(f"Warning: Could not determine grid spacing for well {well}")
        return {}
    
    # Estimate typical spacing (use median to be robust to outliers)
    x_spacing = np.median(x_diffs)
    y_spacing = np.median(y_diffs)
    
    print(f"Estimated grid spacing: x={x_spacing:.1f}, y={y_spacing:.1f}")
    
    # Create grid by binning positions
    x_min, x_max = x_positions.min(), x_positions.max()
    y_min, y_max = y_positions.min(), y_positions.max()
    
    # Create grid bins
    x_bins = np.arange(x_min - x_spacing/2, x_max + x_spacing, x_spacing)
    y_bins = np.arange(y_min - y_spacing/2, y_max + y_spacing, y_spacing)
    
    # Assign each tile to a grid position
    tile_positions = {}
    
    for _, row in well_metadata.iterrows():
        tile_id = row['tile']
        x_pos = row['x_pos']
        y_pos = row['y_pos']
        
        # Find which grid bin each position belongs to
        grid_x = np.digitize(x_pos, x_bins) - 1
        grid_y = np.digitize(y_pos, y_bins) - 1
        
        # Ensure grid positions are non-negative
        grid_x = max(0, grid_x)
        grid_y = max(0, grid_y)
        
        tile_positions[tile_id] = (grid_x, grid_y)
    
    print(f"Created grid positions for {len(tile_positions)} tiles")
    print(f"Grid dimensions: {max(pos[0] for pos in tile_positions.values()) + 1} x {max(pos[1] for pos in tile_positions.values()) + 1}")
    
    return tile_positions


def pairwise_shifts_tiff(metadata_df: pd.DataFrame, well: str, flipud: bool = False, 
                        fliplr: bool = False, rot90: int = 0, channel: int = 0) -> Tuple[List[TiffEdge], Dict]:
    """
    Calculate pairwise shifts between adjacent tiles using TIFF data.
    
    Args:
        metadata_df: DataFrame with tile metadata
        well: Well identifier
        flipud: Whether to flip tiles up-down
        fliplr: Whether to flip tiles left-right  
        rot90: Number of 90-degree rotations
        channel: Which channel to use for registration
        
    Returns:
        Tuple of (edge_list, confidence_dict)
    """
    # Get grid positions for tiles
    tile_positions = metadata_to_grid_positions(metadata_df, well)
    
    if len(tile_positions) == 0:
        print(f"No tiles found for well {well}")
        return [], {}
    
    # Create connectivity graph
    grid_positions = list(tile_positions.values())
    edges = connectivity_from_grid(grid_positions)
    
    # Create tile cache
    tile_cache = TiffTileCache(metadata_df, well, flipud, fliplr, rot90, channel)
    
    # Process each edge
    edge_list = []
    confidence_dict = {}
    
    # Create reverse lookup for positions to tile IDs
    pos_to_tile = {pos: tile_id for tile_id, pos in tile_positions.items()}
    
    for key, (pos_a, pos_b) in tqdm(edges.items(), desc="Computing pairwise shifts"):
        tile_a_id = pos_to_tile[pos_a]
        tile_b_id = pos_to_tile[pos_b]
        
        try:
            edge = TiffEdge(tile_a_id, tile_b_id, tile_cache, tile_positions)
            edge_list.append(edge)
            
            confidence_dict[key] = [
                list(pos_a),
                list(pos_b), 
                float(edge.model.confidence)
            ]
        except Exception as e:
            print(f"Failed to process edge {tile_a_id}-{tile_b_id}: {e}")
            continue
    
    return edge_list, confidence_dict


def optimal_positions_tiff(edge_list: List[TiffEdge], tile_positions: Dict[int, Tuple[int, int]], 
                          well: str, tile_size: Tuple[int, int]) -> Dict[str, List[int]]:
    """
    Calculate optimal tile positions using least squares optimization.
    
    Args:
        edge_list: List of edges with calculated shifts
        tile_positions: Mapping of tile_id to grid positions
        well: Well identifier
        tile_size: Size of tiles (height, width) in pixels
        
    Returns:
        Dictionary mapping tile identifiers to optimal [y, x] positions
    """
    if len(edge_list) == 0:
        print("No edges provided for optimization")
        return {}
    
    # Create tile lookup table
    tile_ids = list(tile_positions.keys())
    tile_lut = {tile_id: i for i, tile_id in enumerate(tile_ids)}
    
    # Initialize arrays for optimization
    n_tiles = len(tile_ids)
    n_edges = len(edge_list)
    
    y_i = np.zeros(n_edges + 1, dtype=np.float32)
    y_j = np.zeros(n_edges + 1, dtype=np.float32)
    
    # Initial guess based on grid positions
    x_guess = np.array([tile_positions[tile_id][0] * tile_size[1] for tile_id in tile_ids], dtype=np.float32)
    y_guess = np.array([tile_positions[tile_id][1] * tile_size[0] for tile_id in tile_ids], dtype=np.float32)
    
    # Build constraint matrix
    a = scipy.sparse.lil_matrix((n_edges + 1, n_tiles), dtype=np.float32)
    
    for c, edge in enumerate(edge_list):
        tile_a_idx = tile_lut[edge.tile_a_id]
        tile_b_idx = tile_lut[edge.tile_b_id]
        
        a[c, tile_a_idx] = -1
        a[c, tile_b_idx] = 1
        
        y_i[c] = edge.model.shift_vector[0]  # y shift
        y_j[c] = edge.model.shift_vector[1]  # x shift
    
    # Add constraint to fix first tile at origin
    y_i[-1] = 0
    y_j[-1] = 0
    a[-1, 0] = 1
    
    a = a.tocsr()
    
    # Optimization parameters
    tolerance = 1e-5
    order_error = 1
    order_reg = 1
    alpha_reg = 0
    maxiter = int(1e8)
    
    print("Optimizing tile positions...")
    
    try:
        opt_i = linsolve(
            a, y_i,
            tolerance=tolerance,
            order_error=order_error,
            order_reg=order_reg,
            alpha_reg=alpha_reg,
            x0=y_guess,
            maxiter=maxiter,
        )
        
        opt_j = linsolve(
            a, y_j,
            tolerance=tolerance,
            order_error=order_error,
            order_reg=order_reg,
            alpha_reg=alpha_reg,
            x0=x_guess,
            maxiter=maxiter,
        )
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fall back to grid positions
        opt_i = y_guess
        opt_j = x_guess
    
    # Combine and zero the minimum values
    opt_shifts = np.vstack((opt_i, opt_j)).T
    opt_shifts_zeroed = opt_shifts - np.min(opt_shifts, axis=0)
    
    # Create output dictionary
    opt_shifts_dict = {}
    for i, tile_id in enumerate(tile_ids):
        # Format: "well/tile_id": [y_position, x_position]
        opt_shifts_dict[f"{well}/{tile_id}"] = [int(opt_shifts_zeroed[i, 0]), int(opt_shifts_zeroed[i, 1])]
    
    return opt_shifts_dict


def estimate_stitch_tiff(metadata_df: pd.DataFrame, well: str, flipud: bool = False,
                        fliplr: bool = False, rot90: int = 0, channel: int = 0,
                        tile_size: Tuple[int, int] = (2048, 2048)) -> Dict[str, Dict]:
    """
    Complete stitching estimation for a single well using TIFF data.
    
    Args:
        metadata_df: DataFrame with tile metadata
        well: Well identifier
        flipud: Whether to flip tiles up-down
        fliplr: Whether to flip tiles left-right
        rot90: Number of 90-degree rotations  
        channel: Which channel to use for registration
        tile_size: Size of tiles (height, width) in pixels
        
    Returns:
        Dictionary with total_translation and confidence information
    """
    # Get tile positions
    tile_positions = metadata_to_grid_positions(metadata_df, well)
    
    # Calculate pairwise shifts
    edge_list, confidence_dict = pairwise_shifts_tiff(
        metadata_df, well, flipud, fliplr, rot90, channel
    )
    
    # Calculate optimal positions  
    opt_shift_dict = optimal_positions_tiff(edge_list, tile_positions, well, tile_size)
    
    return {
        "total_translation": opt_shift_dict,
        "confidence": {well: confidence_dict}
    }


def get_output_shape_tiff(shifts: Dict[str, List[int]], tile_size: Tuple[int, int]) -> Tuple[int, int]:
    """Calculate the output shape needed for stitched image"""
    if not shifts:
        return tile_size
        
    y_shifts = [shift[0] for shift in shifts.values()]
    x_shifts = [shift[1] for shift in shifts.values()]
    
    max_y = int(np.max(y_shifts)) if y_shifts else 0
    max_x = int(np.max(x_shifts)) if x_shifts else 0
    
    return max_y + tile_size[0], max_x + tile_size[1]


def assemble_tiff_well(metadata_df: pd.DataFrame, shifts: Dict[str, List[int]], 
                       well: str, flipud: bool = False, fliplr: bool = False, 
                       rot90: int = 0, overlap_percent: float = 0.1) -> np.ndarray:
    """
    Assemble a stitched well image from individual tiles using calculated shifts.
    
    Args:
        metadata_df: DataFrame with tile metadata including file paths
        shifts: Dictionary mapping tile identifiers to [y, x] shift positions
        well: Well identifier
        flipud: Whether to flip tiles up-down
        fliplr: Whether to flip tiles left-right
        rot90: Number of 90-degree rotations
        overlap_percent: Percentage of tile that overlaps (for blending)
        
    Returns:
        Stitched image as numpy array with shape (C, Y, X)
    """
    well_metadata = metadata_df[metadata_df['well'] == well].copy()
    
    if len(well_metadata) == 0:
        raise ValueError(f"No metadata found for well {well}")
    
    # Get tile size from first tile
    first_tile_path = well_metadata.iloc[0]['filename']
    try:
        first_tile_array = nd2_to_tiff(
            files=first_tile_path,
            channel_order_flip=False,
            verbose=False,
            z_handling="max_projection"
        )
        tile_shape = first_tile_array.shape
        if len(tile_shape) == 2:
            tile_shape = (1, tile_shape[0], tile_shape[1])
        tile_size = tile_shape[-2:]
        n_channels = tile_shape[0] if len(tile_shape) > 2 else 1
    except Exception as e:
        print(f"Error loading first tile to determine shape: {e}")
        return np.array([])
    
    # Calculate output dimensions
    final_shape_yx = get_output_shape_tiff(shifts, tile_size)
    final_shape = (n_channels,) + final_shape_yx
    
    # Initialize output arrays
    output_image = np.zeros(final_shape, dtype=np.float32)
    divisor = np.zeros(final_shape, dtype=np.uint16)
    
    print(f"Assembling {len(well_metadata)} tiles into shape {final_shape}")
    
    # Process each tile
    for _, tile_row in tqdm(well_metadata.iterrows(), total=len(well_metadata), desc="Assembling tiles"):
        tile_id = tile_row['tile']
        tile_key = f"{well}/{tile_id}"
        
        if tile_key not in shifts:
            print(f"Warning: No shift found for tile {tile_key}, skipping")
            continue
            
        # Load tile
        try:
            tile_array = nd2_to_tiff(
                files=tile_row['filename'],
                channel_order_flip=False,
                verbose=False,
                z_handling="max_projection"
            )
            
            # Ensure 3D array (C, Y, X)
            if len(tile_array.shape) == 2:
                tile_array = np.expand_dims(tile_array, axis=0)
            elif len(tile_array.shape) > 3:
                # Take first timepoint if 4D+
                tile_array = tile_array[0] if tile_array.shape[0] == 1 else tile_array
                
            # Apply augmentations to each channel
            for c in range(tile_array.shape[0]):
                tile_array[c] = augment_tile(tile_array[c], flipud, fliplr, rot90)
                
        except Exception as e:
            print(f"Error loading tile {tile_id}: {e}")
            continue
        
        # Get shift position
        shift = shifts[tile_key]
        y_shift, x_shift = int(shift[0]), int(shift[1])
        
        # Place tile in output image
        y_end = y_shift + tile_size[0]
        x_end = x_shift + tile_size[1]
        
        # Ensure we don't exceed output bounds
        y_end = min(y_end, final_shape_yx[0])
        x_end = min(x_end, final_shape_yx[1])
        
        # Calculate actual tile slice to use
        tile_y_end = tile_size[0] - max(0, y_shift + tile_size[0] - final_shape_yx[0])
        tile_x_end = tile_size[1] - max(0, x_shift + tile_size[1] - final_shape_yx[1])
        
        if y_shift >= 0 and x_shift >= 0 and y_end > y_shift and x_end > x_shift:
            # Add tile to output
            output_image[:, y_shift:y_end, x_shift:x_end] += tile_array[:, :tile_y_end, :tile_x_end].astype(np.float32)
            
            # Create mask for non-zero pixels
            tile_mask = tile_array[:, :tile_y_end, :tile_x_end] > 0
            divisor[:, y_shift:y_end, x_shift:x_end] += tile_mask.astype(np.uint16)
    
    # Normalize by overlap
    stitched = np.zeros_like(output_image, dtype=np.float32)
    
    # Avoid division by zero
    nonzero_mask = divisor > 0
    stitched[nonzero_mask] = output_image[nonzero_mask] / divisor[nonzero_mask]
    
    return stitched.astype(np.uint16)

def load_aligned_tiff(file_path: str, channel: int = 0) -> np.ndarray:
    """
    Load an aligned TIFF file and extract specific channel.
    
    Args:
        file_path: Path to the aligned TIFF file
        channel: Which channel to extract (for registration)
        
    Returns:
        2D numpy array of the image
    """
    try:
        # Load the TIFF file
        image = io.imread(file_path)
        
        # Handle different image dimensions
        if len(image.shape) == 2:
            # Already 2D grayscale
            return image
        elif len(image.shape) == 3:
            # Multi-channel image - extract specified channel
            if image.shape[0] <= 10:  # Assume channels-first format
                return image[min(channel, image.shape[0]-1), :, :]
            else:  # Assume channels-last format
                return image[:, :, min(channel, image.shape[2]-1)]
        else:
            # 4D or higher - take first timepoint and specified channel
            return image[0, min(channel, image.shape[1]-1), :, :]
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


class AlignedTiffTileCache:
    """
    Cache for loading aligned TIFF tiles using metadata for positioning.
    Replaces the ND2-based TileCache for aligned images.
    """

    def __init__(self, metadata_df: pd.DataFrame, well: str, data_type: str = "phenotype",
                 flipud: bool = False, fliplr: bool = False, rot90: int = 0, channel: int = 0):
        self.cache = LimitedSizeDict(max_size=20)
        self.metadata_df = metadata_df[metadata_df['well'] == well].copy()
        self.well = well
        self.data_type = data_type  # "phenotype" or "sbs"
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
        tile_metadata = self.metadata_df[self.metadata_df['tile'] == tile_id]
        
        if len(tile_metadata) == 0:
            print(f"No metadata found for tile {tile_id}")
            return None
            
        tile_row = tile_metadata.iloc[0]
        
        # Construct path to aligned TIFF file
        # Pattern: analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{tile}__aligned.tiff
        plate = tile_row['plate']
        aligned_path = f"analysis_root/{self.data_type}/images/P-{plate}_W-{self.well}_T-{tile_id}__aligned.tiff"
        
        try:
            # Load the aligned TIFF file
            tile_2d = load_aligned_tiff(aligned_path, self.channel)
            
            if tile_2d is None:
                return None
                
            # Apply augmentations
            aug_tile = augment_tile(tile_2d, self.flipud, self.fliplr, self.rot90)
            
            # Cache the result
            self.cache[tile_id] = aug_tile
            return aug_tile
            
        except Exception as e:
            print(f"Error loading tile {tile_id}: {e}")
            return None


class AlignedTiffEdge:
    """
    Edge between two tiles for registration using aligned TIFF images.
    """

    def __init__(self, tile_a_id: int, tile_b_id: int, tile_cache: AlignedTiffTileCache, 
                 tile_positions: Dict[int, Tuple[int, int]]):
        self.tile_cache = tile_cache
        self.tile_a_id = tile_a_id
        self.tile_b_id = tile_b_id
        
        # Get grid positions
        self.pos_a = tile_positions[tile_a_id]
        self.pos_b = tile_positions[tile_b_id]
        
        # Calculate relationship between tiles
        self.relation = (self.pos_a[0] - self.pos_b[0], self.pos_a[1] - self.pos_b[1])
        
        # Calculate the offset
        self.model = self.get_offset()

    def get_offset(self) -> TranslationRegistrationModel:
        """Calculate offset between two tiles"""
        tile_a = self.tile_cache[self.tile_a_id]
        tile_b = self.tile_cache[self.tile_b_id]
        
        if tile_a is None or tile_b is None:
            # Return a dummy model with zero shift if tiles can't be loaded
            model = TranslationRegistrationModel()
            model.shift_vector = np.array([0.0, 0.0])
            model.confidence = 0.0
            return model
            
        return offset_tiff(tile_a, tile_b, self.relation, overlap=150)


def pairwise_shifts_aligned_tiff(metadata_df: pd.DataFrame, well: str, data_type: str = "phenotype",
                                flipud: bool = False, fliplr: bool = False, rot90: int = 0, 
                                channel: int = 0) -> Tuple[List[AlignedTiffEdge], Dict]:
    """
    Calculate pairwise shifts between adjacent tiles using aligned TIFF data.
    
    Args:
        metadata_df: DataFrame with tile metadata
        well: Well identifier
        data_type: "phenotype" or "sbs"
        flipud: Whether to flip tiles up-down
        fliplr: Whether to flip tiles left-right  
        rot90: Number of 90-degree rotations
        channel: Which channel to use for registration
        
    Returns:
        Tuple of (edge_list, confidence_dict)
    """
    # Get grid positions for tiles
    tile_positions = metadata_to_grid_positions(metadata_df, well)
    
    if len(tile_positions) == 0:
        print(f"No tiles found for well {well}")
        return [], {}
    
    # Create connectivity graph
    grid_positions = list(tile_positions.values())
    edges = connectivity_from_grid(grid_positions)
    
    # Create tile cache for aligned TIFF files
    tile_cache = AlignedTiffTileCache(metadata_df, well, data_type, flipud, fliplr, rot90, channel)
    
    # Process each edge
    edge_list = []
    confidence_dict = {}
    
    # Create reverse lookup for positions to tile IDs
    pos_to_tile = {pos: tile_id for tile_id, pos in tile_positions.items()}
    
    for key, (pos_a, pos_b) in tqdm(edges.items(), desc=f"Computing pairwise shifts for {data_type}"):
        tile_a_id = pos_to_tile[pos_a]
        tile_b_id = pos_to_tile[pos_b]
        
        try:
            edge = AlignedTiffEdge(tile_a_id, tile_b_id, tile_cache, tile_positions)
            edge_list.append(edge)
            
            confidence_dict[key] = [
                list(pos_a),
                list(pos_b), 
                float(edge.model.confidence)
            ]
        except Exception as e:
            print(f"Failed to process edge {tile_a_id}-{tile_b_id}: {e}")
            continue
    
    return edge_list, confidence_dict


def estimate_stitch_aligned_tiff(metadata_df: pd.DataFrame, well: str, data_type: str = "phenotype",
                                flipud: bool = False, fliplr: bool = False, rot90: int = 0, 
                                channel: int = 0, tile_size: Tuple[int, int] = (2048, 2048)) -> Dict[str, Dict]:
    """
    Complete stitching estimation for a single well using aligned TIFF data.
    
    Args:
        metadata_df: DataFrame with tile metadata
        well: Well identifier
        data_type: "phenotype" or "sbs"
        flipud: Whether to flip tiles up-down
        fliplr: Whether to flip tiles left-right
        rot90: Number of 90-degree rotations  
        channel: Which channel to use for registration
        tile_size: Size of tiles (height, width) in pixels
        
    Returns:
        Dictionary with total_translation and confidence information
    """
    # Get tile positions
    tile_positions = metadata_to_grid_positions(metadata_df, well)
    
    # Calculate pairwise shifts
    edge_list, confidence_dict = pairwise_shifts_aligned_tiff(
        metadata_df, well, data_type, flipud, fliplr, rot90, channel
    )
    
    # Calculate optimal positions  
    opt_shift_dict = optimal_positions_tiff(edge_list, tile_positions, well, tile_size)
    
    return {
        "total_translation": opt_shift_dict,
        "confidence": {well: confidence_dict}
    }


def assemble_aligned_tiff_well(metadata_df: pd.DataFrame, shifts: Dict[str, List[int]], 
                              well: str, data_type: str = "phenotype", flipud: bool = False, 
                              fliplr: bool = False, rot90: int = 0, 
                              overlap_percent: float = 0.1) -> np.ndarray:
    """
    Assemble a stitched well image from aligned TIFF tiles using calculated shifts.
    
    Args:
        metadata_df: DataFrame with tile metadata
        shifts: Dictionary mapping tile identifiers to [y, x] shift positions
        well: Well identifier
        data_type: "phenotype" or "sbs" 
        flipud: Whether to flip tiles up-down
        fliplr: Whether to flip tiles left-right
        rot90: Number of 90-degree rotations
        overlap_percent: Percentage of tile that overlaps (for blending)
        
    Returns:
        Stitched image as numpy array
    """
    well_metadata = metadata_df[metadata_df['well'] == well].copy()
    
    if len(well_metadata) == 0:
        raise ValueError(f"No metadata found for well {well}")
    
    # Get tile size from first tile
    first_tile_row = well_metadata.iloc[0]
    plate = first_tile_row['plate']
    tile_id = first_tile_row['tile']
    first_tile_path = f"analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{tile_id}__aligned.tiff"
    
    try:
        first_tile = load_aligned_tiff(first_tile_path)
        if first_tile is None:
            raise ValueError(f"Could not load first tile: {first_tile_path}")
        tile_size = first_tile.shape
    except Exception as e:
        print(f"Error loading first tile to determine shape: {e}")
        return np.array([])
    
    # Calculate output dimensions
    final_shape_yx = get_output_shape_tiff(shifts, tile_size)
    final_shape = final_shape_yx  # 2D output
    
    # Initialize output arrays
    output_image = np.zeros(final_shape, dtype=np.float32)
    divisor = np.zeros(final_shape, dtype=np.uint16)
    
    print(f"Assembling {len(well_metadata)} {data_type} tiles into shape {final_shape}")
    
    # Process each tile
    for _, tile_row in tqdm(well_metadata.iterrows(), total=len(well_metadata), 
                           desc=f"Assembling {data_type} tiles"):
        tile_id = tile_row['tile']
        tile_key = f"{well}/{tile_id}"
        
        if tile_key not in shifts:
            print(f"Warning: No shift found for tile {tile_key}, skipping")
            continue
            
        # Load tile
        plate = tile_row['plate']
        tile_path = f"analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{tile_id}__aligned.tiff"
        
        try:
            tile_array = load_aligned_tiff(tile_path)
            
            if tile_array is None:
                print(f"Warning: Could not load tile {tile_id}")
                continue
                
            # Apply augmentations
            tile_array = augment_tile(tile_array, flipud, fliplr, rot90)
                
        except Exception as e:
            print(f"Error loading tile {tile_id}: {e}")
            continue
        
        # Get shift position
        shift = shifts[tile_key]
        y_shift, x_shift = int(shift[0]), int(shift[1])
        
        # Place tile in output image
        y_end = y_shift + tile_size[0]
        x_end = x_shift + tile_size[1]
        
        # Ensure we don't exceed output bounds
        y_end = min(y_end, final_shape[0])
        x_end = min(x_end, final_shape[1])
        
        # Calculate actual tile slice to use
        tile_y_end = tile_size[0] - max(0, y_shift + tile_size[0] - final_shape[0])
        tile_x_end = tile_size[1] - max(0, x_shift + tile_size[1] - final_shape[1])
        
        if y_shift >= 0 and x_shift >= 0 and y_end > y_shift and x_end > x_shift:
            # Add tile to output
            output_image[y_shift:y_end, x_shift:x_end] += tile_array[:tile_y_end, :tile_x_end].astype(np.float32)
            
            # Create mask for non-zero pixels
            tile_mask = tile_array[:tile_y_end, :tile_x_end] > 0
            divisor[y_shift:y_end, x_shift:x_end] += tile_mask.astype(np.uint16)
    
    # Normalize by overlap
    stitched = np.zeros_like(output_image, dtype=np.float32)
    
    # Avoid division by zero
    nonzero_mask = divisor > 0
    stitched[nonzero_mask] = output_image[nonzero_mask] / divisor[nonzero_mask]
    
    return stitched.astype(np.uint16)