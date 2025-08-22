"""
Enhanced stitching functions for BrieFlow pipeline.
Handles both images and segmentation masks with actual stitching.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Union
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

from lib.preprocess.preprocess import nd2_to_tiff


def extract_cell_positions_from_stitched_mask(
    stitched_mask: np.ndarray,
    well: str,
    plate: str,
    data_type: str = "phenotype",
    metadata_df: pd.DataFrame = None,
    shifts: Dict[str, List[int]] = None,
    tile_size: Tuple[int, int] = None,
    cell_id_mapping: Dict[int, Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    ENHANCED: Extract cell positions using preserved cell ID mapping.
    Now requires cell_id_mapping for proper functionality and includes plate parameter.
    """

    print(f"🔍 EXTRACTING CELL POSITIONS: {data_type} well {well}, plate {plate}")

    # Get region properties from stitched mask
    props = measure.regionprops(stitched_mask)

    if len(props) == 0:
        print("⚠️  No cells found in stitched mask")
        return pd.DataFrame(
            columns=[
                "well",
                "plate",
                "cell",
                "tile",
                "label",
                "stitched_cell_id",
                "original_cell_id",
                "original_tile_id",
                "i",
                "j",
                "area",
                "data_type",
                "tile_i",
                "tile_j",
                "mapping_method",
                "stage_x",
                "stage_y",
                "tile_row",
                "tile_col",
            ]
        )

    print(f"Found {len(props)} cells in stitched mask")

    # Require the essential parameters
    if cell_id_mapping is None:
        raise ValueError("cell_id_mapping is required for preserved cell ID extraction")
    if metadata_df is None:
        raise ValueError("metadata_df is required for tile information")
    if shifts is None:
        raise ValueError("shifts is required for tile positioning")

    # Auto-detect tile size if not provided
    if tile_size is None:
        tile_size = (2400, 2400) if data_type == "phenotype" else (1200, 1200)

    print("✅ Using preserved cell ID mapping")

    # Create metadata lookup
    well_metadata = metadata_df[metadata_df["well"] == well].copy()
    tile_metadata_lookup = {}
    tile_shift_lookup = {}

    for _, tile_row in well_metadata.iterrows():
        tile_id = tile_row["tile"]
        tile_metadata_lookup[tile_id] = tile_row.to_dict()

        tile_key = f"{well}/{tile_id}"
        if tile_key in shifts:
            tile_shift_lookup[tile_id] = shifts[tile_key]

    # Extract cell information using preserved mapping
    cell_data = []
    direct_mappings = 0
    missing_mappings = 0

    for prop in props:
        global_centroid_y, global_centroid_x = prop.centroid
        stitched_label = prop.label

        # Base cell information - ALWAYS include plate
        cell_info = {
            "well": well,
            "plate": str(plate),  # Ensure plate is string
            "stitched_cell_id": stitched_label,
            "i": global_centroid_y,  # Global stitched coordinates
            "j": global_centroid_x,  # Global stitched coordinates
            "area": prop.area,
            "data_type": data_type,
        }

        # Use preserved mapping to get original tile and cell ID
        if stitched_label in cell_id_mapping:
            original_tile_id, original_cell_id = cell_id_mapping[stitched_label]
            mapping_method = "preserved_mapping"
            direct_mappings += 1

            # Calculate relative position within original tile
            if original_tile_id in tile_shift_lookup:
                tile_shift = tile_shift_lookup[original_tile_id]
                relative_y = global_centroid_y - tile_shift[0]
                relative_x = global_centroid_x - tile_shift[1]
            else:
                relative_y = -1
                relative_x = -1

            # Get tile metadata
            tile_metadata = tile_metadata_lookup.get(original_tile_id, {})

            # STANDARDIZED: Always create both 'cell' and 'label' columns consistently
            cell_info.update(
                {
                    "cell": original_cell_id, 
                    "tile": original_tile_id,  
                    "tile_i": relative_y,
                    "tile_j": relative_x,
                    "mapping_method": mapping_method,
                    "stage_x": tile_metadata.get("x_pos", np.nan),
                    "stage_y": tile_metadata.get("y_pos", np.nan),
                    "tile_row": tile_metadata.get("tile_row", -1),
                    "tile_col": tile_metadata.get("tile_col", -1),
                }
            )

        cell_data.append(cell_info)

    df = pd.DataFrame(cell_data)

    # FINAL VALIDATION: Ensure required columns exist and have correct types
    if len(df) > 0:
        # Ensure integer types for ID columns (use nullable integer type)
        for col in ['cell', 'tile', 'original_cell_id', 'original_tile_id', 'stitched_cell_id', 'label']:
            if col in df.columns:
                # Handle NaN values for integer columns
                df[col] = df[col].fillna(-1).astype('Int64')  # Nullable integer type
        
        # Ensure plate is string
        df['plate'] = df['plate'].astype(str)
        
        # Ensure well is string  
        df['well'] = df['well'].astype(str)
        
        # Ensure data_type is string
        df['data_type'] = df['data_type'].astype(str)

    print(f"✅ EXTRACTION COMPLETE:")
    print(f"  Total cells: {len(df)}")
    print(f"  Direct mappings: {direct_mappings}")
    print(f"  Missing mappings: {missing_mappings}")
    print(f"  Success rate: {direct_mappings / len(df) * 100:.1f}%")

    # Show tile distribution
    if len(df) > 0 and "original_tile_id" in df.columns:
        tile_counts = df["original_tile_id"].value_counts()
        valid_tiles = tile_counts[tile_counts.index.notna() & (tile_counts.index != -1)]
        if len(valid_tiles) > 0:
            print(
                f"  Cells per tile: min={valid_tiles.min()}, max={valid_tiles.max()}, mean={valid_tiles.mean():.1f}"
            )
            print(f"  Active tiles: {len(valid_tiles)}")

    # Validate that cell and label are consistent for preserved mappings
    if len(df) > 0:
        preserved_rows = df[df['mapping_method'] == 'preserved_mapping']
        if len(preserved_rows) > 0:
            if (preserved_rows['cell'] == preserved_rows['label']).all():
                print(f"✅ Cell/label consistency verified for {len(preserved_rows)} preserved mappings")
            else:
                print(f"⚠️  Warning: Cell/label mismatch detected in preserved mappings")

    # Show column summary
    if len(df) > 0:
        print(f"📊 Column Summary:")
        print(f"  Plates: {df['plate'].nunique()} unique")
        print(f"  Wells: {df['well'].nunique()} unique") 
        print(f"  Tiles: {df['tile'].nunique()} unique")
        print(f"  Cell ID range: {df['cell'].min()} - {df['cell'].max()}")
        print(f"  Label range: {df['label'].min()} - {df['label'].max()}")

    return df


def find_original_tile_from_position(
    global_i, global_j, metadata_df, shifts, tile_size
):
    """
    Find which original tile a position in the stitched image came from

    Parameters:
    -----------
    global_i, global_j : float
        Coordinates in the stitched image
    metadata_df : pd.DataFrame
        Metadata for tiles with tile information
    shifts : dict
        Translation shifts for each tile {tile_id: (shift_x, shift_y)}
    tile_size : tuple
        (height, width) of individual tiles

    Returns:
    --------
    dict with tile, site, tile_i, tile_j information
    """

    tile_height, tile_width = tile_size
    min_distance = float("inf")
    best_tile_info = {"tile": 1, "site": 1, "tile_i": 0, "tile_j": 0}

    # First pass: look for exact containment
    for _, tile_row in metadata_df.iterrows():
        tile_id = tile_row["tile"]

        # Get the shift for this tile
        if tile_id not in shifts:
            continue

        shift_x, shift_y = shifts[tile_id]

        # Calculate tile boundaries in stitched coordinates
        tile_start_j = shift_x  # x shift corresponds to j (column)
        tile_start_i = shift_y  # y shift corresponds to i (row)
        tile_end_j = tile_start_j + tile_width
        tile_end_i = tile_start_i + tile_height

        # Check if point is within this tile (with small tolerance for boundaries)
        tolerance = 5  # pixels
        if (
            tile_start_i - tolerance <= global_i <= tile_end_i + tolerance
            and tile_start_j - tolerance <= global_j <= tile_end_j + tolerance
        ):
            return {
                "tile": int(tile_id),
                "site": int(tile_id),  # Use tile as site for SBS compatibility
                "tile_i": tile_row.get("tile_i", 0),
                "tile_j": tile_row.get("tile_j", 0),
            }

    # Second pass: if no exact match, find closest tile center
    for _, tile_row in metadata_df.iterrows():
        tile_id = tile_row["tile"]
        if tile_id not in shifts:
            continue

        shift_x, shift_y = shifts[tile_id]
        tile_center_j = shift_x + tile_width / 2
        tile_center_i = shift_y + tile_height / 2

        distance = np.sqrt(
            (global_i - tile_center_i) ** 2 + (global_j - tile_center_j) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            best_tile_info = {
                "tile": int(tile_id),
                "site": int(tile_id),
                "tile_i": tile_row.get("tile_i", 0),
                "tile_j": tile_row.get("tile_j", 0),
            }

    return best_tile_info


class LimitedSizeDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)


def augment_tile(
    tile: np.ndarray, flipud: bool, fliplr: bool, rot90: int
) -> np.ndarray:
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

    coords = well_metadata[["x_pos", "y_pos"]].values
    tile_ids = well_metadata["tile"].values

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
        distances_grid = np.sqrt(
            (positions[:, 0] - center_y) ** 2 + (positions[:, 1] - center_x) ** 2
        )

        cv = (
            distances_grid.std() / distances_grid.mean()
            if distances_grid.mean() > 0
            else 1.0
        )
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
    max_neighbors_per_tile: int = 3,
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
        print(
            f"Phenotype: Using optimized threshold {proximity_threshold:.1f} (min_dist * 1.005)"
        )

    elif data_type == "sbs":
        # For SBS: use 5th percentile of distance distribution
        proximity_threshold = np.percentile(upper_triangle, 5)
        max_neighbors = 3
        print(
            f"SBS: Using optimized threshold {proximity_threshold:.1f} (5th percentile)"
        )

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
        neighbor_distances = [
            (j, distances[i, j])
            for j in range(len(tile_ids))
            if j != i and distances[i, j] < proximity_threshold
        ]

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
        print(
            f"⚠️  Warning: Phenotype edge count {len(edges)} outside expected range 8K-15K"
        )
    elif data_type == "sbs" and not (3000 <= len(edges) <= 8000):
        print(f"⚠️  Warning: SBS edge count {len(edges)} outside expected range 3K-8K")
    else:
        print(f"✅ Edge count {len(edges)} in optimal range for {data_type}")

    return edges


def connectivity_from_grid(
    positions: List[Tuple[int, int]],
) -> Dict[str, List[Tuple[int, int]]]:
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
    coords = well_metadata[["x_pos", "y_pos"]].values
    tile_ids = well_metadata["tile"].values

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

    tile_cache = AlignedTiffTileCache(
        metadata_df, well, data_type, flipud, fliplr, rot90, channel
    )

    # Process edges
    edge_list = []
    confidence_dict = {}
    pos_to_tile = {pos: tile_id for tile_id, pos in tile_positions.items()}

    print(f"Processing {len(edges)} edges for {data_type} stitching...")
    for key, (pos_a, pos_b) in tqdm(
        edges.items(), desc=f"Computing pairwise shifts for {data_type}"
    ):
        tile_a_id = pos_to_tile[pos_a]
        tile_b_id = pos_to_tile[pos_b]

        try:
            edge = AlignedTiffEdge(tile_a_id, tile_b_id, tile_cache, tile_positions)
            edge_list.append(edge)
            confidence_dict[key] = [
                list(pos_a),
                list(pos_b),
                float(edge.model.confidence),
            ]
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
    x_guess = np.array(
        [tile_positions[tile_id][1] for tile_id in tile_ids], dtype=np.float32
    )  # Grid X (col)
    y_guess = np.array(
        [tile_positions[tile_id][0] for tile_id in tile_ids], dtype=np.float32
    )  # Grid Y (row)

    print(
        f"Initial grid position ranges: Y=[{y_guess.min():.0f}, {y_guess.max():.0f}], X=[{x_guess.min():.0f}, {x_guess.max():.0f}]"
    )

    # Build constraint matrix
    a = scipy.sparse.lil_matrix((n_edges + 1, n_tiles), dtype=np.float32)

    for c, edge in enumerate(edge_list):
        tile_a_idx = tile_lut[edge.tile_a_id]
        tile_b_idx = tile_lut[edge.tile_b_id]
        a[c, tile_a_idx] = -1
        a[c, tile_b_idx] = 1

        # IMPORTANT: Scale the shift vectors to grid units, not pixel units
        # The edge.model.shift_vector is in pixels, convert to grid units
        y_i[c] = (
            edge.model.shift_vector[0] / tile_size[0]
        )  # Convert pixels to grid units
        y_j[c] = (
            edge.model.shift_vector[1] / tile_size[1]
        )  # Convert pixels to grid units

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

    print(
        f"Optimized grid position ranges: Y=[{opt_shifts_zeroed[:, 0].min():.0f}, {opt_shifts_zeroed[:, 0].max():.0f}], X=[{opt_shifts_zeroed[:, 1].min():.0f}, {opt_shifts_zeroed[:, 1].max():.0f}]"
    )

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
    if (
        max_y_range > 100000 or max_x_range > 100000
    ):  # 100K pixels = reasonable large well
        print(f"⚠️  Warning: Very large shift ranges detected")
    else:
        print(f"✅ Shift ranges look reasonable")

    return opt_shifts_dict


def get_output_shape_tiff(
    shifts: Dict[str, List[int]], tile_size: Tuple[int, int]
) -> Tuple[int, int]:
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
    first_tile_path = (
        f"analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{tile_id}__aligned.tiff"
    )

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
    for _, tile_row in tqdm(
        well_metadata.iterrows(),
        total=len(well_metadata),
        desc=f"Assembling {data_type} tiles",
    ):
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

            output_image[y_shift:y_end, x_shift:x_end] += tile_array[
                :tile_y_end, :tile_x_end
            ].astype(np.float32)
            tile_mask = tile_array[:tile_y_end, :tile_x_end] > 0
            divisor[y_shift:y_end, x_shift:x_end] += tile_mask.astype(np.uint16)

    # Normalize
    stitched = np.zeros_like(output_image, dtype=np.float32)
    nonzero_mask = divisor > 0
    stitched[nonzero_mask] = output_image[nonzero_mask] / divisor[nonzero_mask]

    return stitched.astype(np.uint16)


def assemble_stitched_masks_simple(
    metadata_df,
    shifts,
    well,
    data_type,
    flipud=False,
    fliplr=False,
    rot90=0,
    return_cell_mapping=False,
):
    """
    FIXED: Assemble stitched masks while preserving original cell IDs.
    Key fix: Create mapping BEFORE relabeling, so original IDs are preserved.
    """

    print(f"🔧 MASK ASSEMBLY: {data_type} well {well}")
    print(f"📊 Input: {len(metadata_df)} tiles, {len(shifts)} shifts")

    # Calculate canvas size
    tile_size = (2400, 2400) if data_type == "phenotype" else (1200, 1200)
    tile_height, tile_width = tile_size

    max_x = max_y = 0
    for tile_id, (shift_x, shift_y) in shifts.items():
        max_x = max(max_x, shift_x + tile_width)
        max_y = max(max_y, shift_y + tile_height)

    print(f"Canvas size: {max_y} x {max_x}")

    # Initialize stitched mask
    stitched_mask = np.zeros((max_y, max_x), dtype=np.uint32)

    # FIXED: Comprehensive cell tracking with preserved IDs
    cell_id_mapping = {}  # global_id -> (tile_id, original_cell_id)
    next_global_id = 1

    # Process each tile
    for _, row in metadata_df.iterrows():
        tile_id = row["tile"]
        tile_key = f"{well}/{tile_id}"

        if tile_key not in shifts:
            print(f"⚠️  No shift found for tile {tile_id}")
            continue

        # Load tile mask
        mask_path = f"analysis_root/{data_type}/images/P-{row['plate']}_W-{well}_T-{tile_id}__nuclei.tiff"

        if not Path(mask_path).exists():
            print(f"⚠️  Mask not found: {mask_path}")
            continue

        try:
            tile_mask = io.imread(mask_path)

            # Apply transformations
            if flipud:
                tile_mask = np.flipud(tile_mask)
            if fliplr:
                tile_mask = np.fliplr(tile_mask)
            if rot90 > 0:
                tile_mask = np.rot90(tile_mask, rot90)

            # FIXED: Get unique cell IDs BEFORE any modification
            unique_labels = np.unique(tile_mask)
            original_cell_ids = unique_labels[unique_labels > 0]  # Exclude background

            if len(original_cell_ids) == 0:
                print(f"Tile {tile_id}: No cells found")
                continue

            print(
                f"Tile {tile_id}: Found {len(original_cell_ids)} cells "
                f"(original IDs: {original_cell_ids.min()}-{original_cell_ids.max()})"
            )

            # FIXED: Create mapping FIRST, before any relabeling
            tile_registry = {}
            for original_id in original_cell_ids:
                global_id = next_global_id

                # Store both directions of mapping
                cell_id_mapping[global_id] = (tile_id, int(original_id))
                tile_registry[int(original_id)] = global_id

                next_global_id += 1

            # FIXED: NOW create relabeled mask using the mapping
            relabeled_mask = np.zeros_like(tile_mask, dtype=np.uint32)
            for original_id, global_id in tile_registry.items():
                relabeled_mask[tile_mask == original_id] = global_id

            # Place the relabeled tile in the stitched mask
            shift_y, shift_x = shifts[tile_key]
            end_y = min(shift_y + tile_height, max_y)
            end_x = min(shift_x + tile_width, max_x)

            # Extract regions
            mask_region = stitched_mask[shift_y:end_y, shift_x:end_x]
            tile_region = relabeled_mask[: end_y - shift_y, : end_x - shift_x]

            # Place ALL cells (no overlap checking needed since edge nuclei are removed)
            mask_region[tile_region > 0] = tile_region[tile_region > 0]

            # DEBUG: Check if all cells made it (count unique labels, not pixels)
            placed_labels = np.unique(tile_region[tile_region > 0])
            placed_cells = len(placed_labels)
            expected_cells = len(original_cell_ids)
            if placed_cells != expected_cells:
                print(
                    f"⚠️  Tile {tile_id}: Expected {expected_cells}, placed {placed_cells}"
                )

            print(
                f"✅ Tile {tile_id}: {len(original_cell_ids)} cells placed "
                f"(global IDs: {min(tile_registry.values())}-{max(tile_registry.values())})"
            )

        except Exception as e:
            print(f"❌ Error processing tile {tile_id}: {e}")
            continue

    # Final statistics with integrity check
    final_labels = np.unique(stitched_mask)
    final_cells = final_labels[final_labels > 0]

    print(f"🎉 STITCHING COMPLETE:")
    print(f"  Final cell count: {len(final_cells):,}")
    print(f"  Max global ID: {final_cells.max() if len(final_cells) > 0 else 0}")
    print(f"  Mapping entries: {len(cell_id_mapping)}")

    # FIXED: Verify mapping integrity
    mapped_globals = set(cell_id_mapping.keys())
    mask_globals = set(final_cells)
    missing_mappings = mask_globals - mapped_globals
    extra_mappings = mapped_globals - mask_globals

    if not missing_mappings and not extra_mappings:
        print(f"✅ Perfect mapping integrity: {len(cell_id_mapping)} mappings")
    else:
        if missing_mappings:
            print(f"⚠️  Missing mappings for {len(missing_mappings)} cells")
        if extra_mappings:
            print(f"⚠️  Extra mappings for {len(extra_mappings)} cells")

    if return_cell_mapping:
        return stitched_mask, cell_id_mapping
    else:
        return stitched_mask


def extract_cell_positions_from_stitched_mask(
    stitched_mask: np.ndarray,
    well: str,
    plate: str,
    data_type: str = "phenotype",
    metadata_df: pd.DataFrame = None,
    shifts: Dict[str, List[int]] = None,
    tile_size: Tuple[int, int] = None,
    cell_id_mapping: Dict[int, Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    ENHANCED: Extract cell positions using preserved cell ID mapping.
    Now requires cell_id_mapping for proper functionality.
    """

    print(f"🔍 EXTRACTING CELL POSITIONS: {data_type} well {well}")

    # Get region properties from stitched mask
    props = measure.regionprops(stitched_mask)

    if len(props) == 0:
        print("⚠️  No cells found in stitched mask")
        return pd.DataFrame(
            columns=[
                "well",
                "cell",
                "stitched_cell_id",
                "original_cell_id",
                "original_tile_id",
                "i",
                "j",
                "area",
                "data_type",
                "tile_i",
                "tile_j",
                "mapping_method",
            ]
        )

    print(f"Found {len(props)} cells in stitched mask")

    # Require the essential parameters
    if cell_id_mapping is None:
        raise ValueError("cell_id_mapping is required for preserved cell ID extraction")
    if metadata_df is None:
        raise ValueError("metadata_df is required for tile information")
    if shifts is None:
        raise ValueError("shifts is required for tile positioning")

    # Auto-detect tile size if not provided
    if tile_size is None:
        tile_size = (2400, 2400) if data_type == "phenotype" else (1200, 1200)

    print("✅ Using preserved cell ID mapping")

    # Create metadata lookup
    well_metadata = metadata_df[metadata_df["well"] == well].copy()
    tile_metadata_lookup = {}
    tile_shift_lookup = {}

    for _, tile_row in well_metadata.iterrows():
        tile_id = tile_row["tile"]
        tile_metadata_lookup[tile_id] = tile_row.to_dict()

        tile_key = f"{well}/{tile_id}"
        if tile_key in shifts:
            tile_shift_lookup[tile_id] = shifts[tile_key]

    # Extract cell information using preserved mapping
    cell_data = []
    direct_mappings = 0
    missing_mappings = 0

    for prop in props:
        global_centroid_y, global_centroid_x = prop.centroid
        stitched_label = prop.label

        # Base cell information
        cell_info = {
            "well": well,
            "plate": plate,
            "stitched_cell_id": stitched_label,
            "i": global_centroid_y,  # Global stitched coordinates
            "j": global_centroid_x,  # Global stitched coordinates
            "area": prop.area,
            "data_type": data_type,
        }

        # Use preserved mapping to get original tile and cell ID
        if stitched_label in cell_id_mapping:
            original_tile_id, original_cell_id = cell_id_mapping[stitched_label]
            mapping_method = "preserved_mapping"
            direct_mappings += 1

            # Calculate relative position within original tile
            if original_tile_id in tile_shift_lookup:
                tile_shift = tile_shift_lookup[original_tile_id]
                relative_y = global_centroid_y - tile_shift[0]
                relative_x = global_centroid_x - tile_shift[1]
            else:
                relative_y = -1
                relative_x = -1

            # Get tile metadata
            tile_metadata = tile_metadata_lookup.get(original_tile_id, {})

            # Complete cell information with preserved IDs
            cell_column = "label" if data_type == "phenotype" else "cell"

            cell_info.update(
                {
                    cell_column: original_cell_id,
                    "original_cell_id": original_cell_id,
                    "tile": original_tile_id, 
                    "tile_i": relative_y,
                    "tile_j": relative_x,
                    "mapping_method": mapping_method,
                    "stage_x": tile_metadata.get("x_pos", np.nan),
                    "stage_y": tile_metadata.get("y_pos", np.nan),
                    "tile_row": tile_metadata.get("tile_row", -1),
                    "tile_col": tile_metadata.get("tile_col", -1),
                }
            )

        else:
            # This shouldn't happen with the fixed approach
            print(f"⚠️  Missing mapping for stitched label {stitched_label}")
            missing_mappings += 1

            cell_info.update(
                {
                    "cell": stitched_label,  # Fallback
                    "original_cell_id": None,
                    "original_tile_id": None,
                    "tile": -1,
                    "tile_i": -1,
                    "tile_j": -1,
                    "mapping_method": "missing_mapping",
                    "stage_x": np.nan,
                    "stage_y": np.nan,
                    "tile_row": -1,
                    "tile_col": -1,
                }
            )

        cell_data.append(cell_info)

    df = pd.DataFrame(cell_data)

    print(f"✅ EXTRACTION COMPLETE:")
    print(f"  Total cells: {len(df)}")
    print(f"  Direct mappings: {direct_mappings}")
    print(f"  Missing mappings: {missing_mappings}")
    print(f"  Success rate: {direct_mappings / len(df) * 100:.1f}%")

    # Show tile distribution
    if len(df) > 0 and "original_tile_id" in df.columns:
        tile_counts = df["original_tile_id"].value_counts()
        valid_tiles = tile_counts[tile_counts.index.notna() & (tile_counts.index != -1)]
        if len(valid_tiles) > 0:
            print(
                f"  Cells per tile: min={valid_tiles.min()}, max={valid_tiles.max()}, mean={valid_tiles.mean():.1f}"
            )
            print(f"  Active tiles: {len(valid_tiles)}")

    return df


# Add this alias to replace the optimized version
assemble_stitched_masks_reliable = assemble_stitched_masks_simple


def create_tile_arrangement_qc_plot(
    cell_positions_df: pd.DataFrame, output_path: str, data_type: str = "phenotype"
):
    """Create QC plot showing tile arrangement and cell distribution."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{data_type.title()} Tile Arrangement QC", fontsize=16)

    # Use the correct column for preserved tile mapping
    tile_column = (
        "original_tile_id"
        if "original_tile_id" in cell_positions_df.columns
        else "tile"
    )

    # Plot 1: Cell positions colored by tile
    ax1 = axes[0, 0]
    if tile_column in cell_positions_df.columns:
        scatter = ax1.scatter(
            cell_positions_df["j"],
            cell_positions_df["i"],
            c=cell_positions_df[tile_column],
            cmap="tab20",
            s=0.1,
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax1, label="Original Tile ID")
    ax1.set_title("Cell Positions by Original Tile")
    ax1.set_xlabel("X Position (pixels)")
    ax1.set_ylabel("Y Position (pixels)")
    ax1.invert_yaxis()

    # Plot 2: Stage coordinates with tile numbers
    ax2 = axes[0, 1]
    if (
        "stage_x" in cell_positions_df.columns
        and "stage_y" in cell_positions_df.columns
    ):
        tile_info = (
            cell_positions_df.groupby(tile_column)
            .agg({"stage_x": "first", "stage_y": "first"})
            .reset_index()
        )

        ax2.scatter(tile_info["stage_x"], tile_info["stage_y"], s=50)
        for _, row in tile_info.iterrows():
            if not pd.isna(row["stage_x"]) and not np.isnan(row["stage_x"]):
                ax2.annotate(
                    f"{int(row[tile_column])}",
                    (row["stage_x"], row["stage_y"]),
                    fontsize=8,
                    ha="center",
                )
    ax2.set_title("Tile Arrangement (Stage Coordinates)")
    ax2.set_xlabel("Stage X (μm)")
    ax2.set_ylabel("Stage Y (μm)")

    # Plot 3: Cells per tile histogram
    ax3 = axes[1, 0]
    if tile_column in cell_positions_df.columns:
        tile_counts = cell_positions_df[tile_column].value_counts()
        ax3.hist(tile_counts.values, bins=50, alpha=0.7)
        ax3.axvline(
            tile_counts.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {tile_counts.mean():.1f}",
        )
        ax3.legend()
    ax3.set_title("Distribution of Cells per Original Tile")
    ax3.set_xlabel("Cells per Tile")
    ax3.set_ylabel("Number of Tiles")

    # Plot 4: Tile boundaries overlay
    ax4 = axes[1, 1]
    if "tile_i" in cell_positions_df.columns and "tile_j" in cell_positions_df.columns:
        ax4.scatter(
            cell_positions_df["tile_j"],
            cell_positions_df["tile_i"],
            c=cell_positions_df[tile_column],
            cmap="tab20",
            s=0.1,
            alpha=0.7,
        )
    ax4.set_title("Relative Positions within Original Tiles")
    ax4.set_xlabel("Tile-relative X")
    ax4.set_ylabel("Tile-relative Y")
    ax4.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"QC plot saved to: {output_path}")


def verify_cell_id_preservation(
    enhanced_positions: pd.DataFrame,
    original_data_path: str,
    well: str,
    data_type: str = "phenotype",
) -> None:
    """
    Verify the quality of cell ID preservation by comparing with original data.
    Call this after running the enhanced pipeline to check quality.
    """
    print(f"\n=== VERIFYING CELL ID PRESERVATION ===")
    print(f"Well: {well}, Data type: {data_type}")

    try:
        # Load original data
        if Path(original_data_path).exists():
            original_data = pd.read_parquet(original_data_path)
            print(f"✅ Loaded original data: {len(original_data)} cells")
        else:
            print(f"❌ Original data not found: {original_data_path}")
            return

        # Overall statistics
        total_stitched = len(enhanced_positions)
        direct_mapped = len(
            enhanced_positions[enhanced_positions["mapping_method"] == "direct_mapping"]
        )
        position_estimated = len(
            enhanced_positions[
                enhanced_positions["mapping_method"] == "position_estimate"
            ]
        )

        print(f"\n📊 Overall Statistics:")
        print(f"  Original cells: {len(original_data)}")
        print(f"  Stitched cells: {total_stitched}")
        print(
            f"  Direct mappings: {direct_mapped} ({direct_mapped / total_stitched * 100:.1f}%)"
        )
        print(
            f"  Position estimates: {position_estimated} ({position_estimated / total_stitched * 100:.1f}%)"
        )

        # Per-tile analysis
        print(f"\n🔍 Per-Tile Analysis:")
        tile_comparison = []

        for tile_id in enhanced_positions["original_tile_id"].unique():
            if pd.isna(tile_id) or tile_id == -1:
                continue

            tile_id = int(tile_id)

            # Count in stitched data
            stitched_count = len(
                enhanced_positions[enhanced_positions["original_tile_id"] == tile_id]
            )

            # Count in original data
            original_count = len(original_data[original_data["tile"] == tile_id])

            recovery_rate = stitched_count / original_count if original_count > 0 else 0

            tile_comparison.append(
                {
                    "tile": tile_id,
                    "original": original_count,
                    "stitched": stitched_count,
                    "recovery": recovery_rate,
                }
            )

        if tile_comparison:
            tile_df = pd.DataFrame(tile_comparison)
            mean_recovery = tile_df["recovery"].mean()
            good_tiles = len(tile_df[tile_df["recovery"] > 0.8])

            print(f"  Mean recovery rate: {mean_recovery:.3f}")
            print(f"  Tiles with >80% recovery: {good_tiles}/{len(tile_df)}")
            print(
                f"  Total original cells accounted: {tile_df['stitched'].sum()}/{tile_df['original'].sum()}"
            )

            # Show worst performing tiles
            worst_tiles = tile_df.nsmallest(3, "recovery")
            if len(worst_tiles) > 0:
                print(f"\n⚠️  Tiles needing attention:")
                for _, row in worst_tiles.iterrows():
                    print(
                        f"    Tile {row['tile']}: {row['stitched']}/{row['original']} ({row['recovery']:.3f})"
                    )

        print(f"\n✅ Verification complete!")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()


def estimate_stitch_sbs_coordinate_based(
    metadata_df: pd.DataFrame,
    well: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
) -> Dict[str, Dict]:
    """Coordinate-based stitching for SBS data with correct scaling."""

    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        print(f"No SBS tiles found for well {well}")
        return {"total_translation": {}, "confidence": {well: {}}}

    coords = well_metadata[["x_pos", "y_pos"]].values
    tile_ids = well_metadata["tile"].values
    tile_size = (1200, 1200)  # SBS tile size in pixels

    print(f"Creating coordinate-based SBS stitch config for {len(tile_ids)} tiles")

    # Use proven spacing detection
    from scipy.spatial.distance import pdist

    distances = pdist(coords)
    actual_spacing = np.percentile(distances[distances > 0], 10)  # 10th percentile

    print(f"SBS detected spacing: {actual_spacing:.1f} μm")
    print(f"SBS tile size: {tile_size} pixels")

    # FIXED: Use pixel size from metadata
    if (
        "pixel_size_x" in well_metadata.columns
        and "pixel_size_y" in well_metadata.columns
    ):
        # Use pixel size from metadata (in μm per pixel)
        pixel_size_um = well_metadata["pixel_size_x"].iloc[0]  # μm per pixel
        pixels_per_micron = 1.0 / pixel_size_um  # pixels per μm
        print(f"SBS pixel size from metadata: {pixel_size_um:.6f} μm/pixel")
        print(f"SBS pixels per micron: {pixels_per_micron:.4f}")

        # Verify pixel_size_y matches pixel_size_x
        pixel_size_y = well_metadata["pixel_size_y"].iloc[0]
        if abs(pixel_size_um - pixel_size_y) > 1e-6:
            print(
                f"⚠️  Warning: pixel_size_x ({pixel_size_um:.6f}) != pixel_size_y ({pixel_size_y:.6f})"
            )
    else:
        print(
            "⚠️  pixel_size_x/y not found in metadata, falling back to calculated values"
        )
        # Fallback: SBS specs: 1560 μm field of view, 1200 pixels -> 0.7692 pixels/μm
        sbs_field_of_view_um = 1560.0  # μm
        pixels_per_micron = tile_size[0] / sbs_field_of_view_um
        print(f"SBS fallback - field of view: {sbs_field_of_view_um} μm")
        print(f"SBS fallback - pixels per micron: {pixels_per_micron:.4f}")

    x_min, y_min = coords.min(axis=0)

    total_translation = {}
    confidence = {}

    for i, tile_id in enumerate(tile_ids):
        x_pos, y_pos = coords[i]

        # Convert directly to pixel coordinates using correct scale
        pixel_x = int((x_pos - x_min) * pixels_per_micron)
        pixel_y = int((y_pos - y_min) * pixels_per_micron)

        total_translation[f"{well}/{tile_id}"] = [pixel_y, pixel_x]

        # High confidence since using direct coordinates
        confidence[f"coord_{i}"] = [[pixel_y, pixel_x], [pixel_y, pixel_x], 0.9]

    print(f"Generated {len(total_translation)} SBS coordinate-based positions")

    # Verify output size and spacing
    y_shifts = [shift[0] for shift in total_translation.values()]
    x_shifts = [shift[1] for shift in total_translation.values()]

    if len(y_shifts) > 1:
        pixel_spacings = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                stage_dist = np.sqrt(
                    (coords[i][0] - coords[j][0]) ** 2
                    + (coords[i][1] - coords[j][1]) ** 2
                )
                pixel_dist = np.sqrt(
                    (y_shifts[i] - y_shifts[j]) ** 2 + (x_shifts[i] - x_shifts[j]) ** 2
                )
                if stage_dist > 0:
                    pixel_spacings.append(pixel_dist / stage_dist)

        if pixel_spacings:
            avg_pixel_spacing = np.mean(pixel_spacings)
            print(
                f"Verification - Average pixel spacing ratio: {avg_pixel_spacing:.4f} pixels/μm"
            )

            # Check for expected tile overlap
            expected_tile_spacing_pixels = actual_spacing * pixels_per_micron
            actual_avg_spacing = np.mean(
                [
                    np.sqrt(
                        (y_shifts[i] - y_shifts[j]) ** 2
                        + (x_shifts[i] - x_shifts[j]) ** 2
                    )
                    for i in range(len(y_shifts))
                    for j in range(i + 1, len(y_shifts))
                ]
            )

            if actual_avg_spacing > 0:
                overlap_percent = (
                    (tile_size[0] - actual_avg_spacing) / tile_size[0] * 100
                )
                print(f"SBS tile overlap: {overlap_percent:.1f}%")
                if overlap_percent < 0:
                    print("⚠️  Warning: Negative overlap detected - tiles may have gaps")
                elif overlap_percent > 50:
                    print(
                        "⚠️  Warning: Very high overlap detected - may indicate scaling issues"
                    )

    final_size = (max(y_shifts) + tile_size[0], max(x_shifts) + tile_size[1])
    memory_gb = final_size[0] * final_size[1] * 2 / 1e9

    print(f"SBS final image size: {final_size}")
    print(f"SBS memory estimate: {memory_gb:.1f} GB")

    return {"total_translation": total_translation, "confidence": {well: confidence}}


def estimate_stitch_phenotype_coordinate_based(
    metadata_df: pd.DataFrame,
    well: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
) -> Dict[str, Dict]:
    """Coordinate-based stitching for phenotype data with correct scaling."""

    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        print(f"No phenotype tiles found for well {well}")
        return {"total_translation": {}, "confidence": {well: {}}}

    coords = well_metadata[["x_pos", "y_pos"]].values
    tile_ids = well_metadata["tile"].values
    tile_size = (2400, 2400)  # Phenotype tile size in pixels

    print(
        f"Creating coordinate-based phenotype stitch config for {len(tile_ids)} tiles"
    )

    # Use proven spacing detection
    from scipy.spatial.distance import pdist

    distances = pdist(coords)
    actual_spacing = np.percentile(distances[distances > 0], 10)  # 10th percentile

    print(f"Phenotype detected spacing: {actual_spacing:.1f} μm")
    print(f"Phenotype tile size: {tile_size} pixels")

    # FIXED: Use pixel size from metadata
    if (
        "pixel_size_x" in well_metadata.columns
        and "pixel_size_y" in well_metadata.columns
    ):
        # Use pixel size from metadata (in μm per pixel)
        pixel_size_um = well_metadata["pixel_size_x"].iloc[0]  # μm per pixel
        pixels_per_micron = 1.0 / pixel_size_um  # pixels per μm
        print(f"Phenotype pixel size from metadata: {pixel_size_um:.6f} μm/pixel")
        print(f"Phenotype pixels per micron: {pixels_per_micron:.4f}")

        # Verify pixel_size_y matches pixel_size_x
        pixel_size_y = well_metadata["pixel_size_y"].iloc[0]
        if abs(pixel_size_um - pixel_size_y) > 1e-6:
            print(
                f"⚠️  Warning: pixel_size_x ({pixel_size_um:.6f}) != pixel_size_y ({pixel_size_y:.6f})"
            )
    else:
        print(
            "⚠️  pixel_size_x/y not found in metadata, falling back to calculated values"
        )
        # Fallback: Phenotype specs: 260 μm field of view, 2400 pixels -> 9.2308 pixels/μm
        phenotype_field_of_view_um = 260.0  # μm
        pixels_per_micron = tile_size[0] / phenotype_field_of_view_um
        print(f"Phenotype fallback - field of view: {phenotype_field_of_view_um} μm")
        print(f"Phenotype fallback - pixels per micron: {pixels_per_micron:.4f}")

    x_min, y_min = coords.min(axis=0)

    total_translation = {}
    confidence = {}

    for i, tile_id in enumerate(tile_ids):
        x_pos, y_pos = coords[i]

        # Convert directly to pixel coordinates using correct scale
        pixel_x = int((x_pos - x_min) * pixels_per_micron)
        pixel_y = int((y_pos - y_min) * pixels_per_micron)

        total_translation[f"{well}/{tile_id}"] = [pixel_y, pixel_x]

        # High confidence since using direct coordinates
        confidence[f"coord_{i}"] = [[pixel_y, pixel_x], [pixel_y, pixel_x], 0.9]

    print(f"Generated {len(total_translation)} phenotype coordinate-based positions")

    # Verify output size and spacing
    y_shifts = [shift[0] for shift in total_translation.values()]
    x_shifts = [shift[1] for shift in total_translation.values()]

    if len(y_shifts) > 1:
        pixel_spacings = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                stage_dist = np.sqrt(
                    (coords[i][0] - coords[j][0]) ** 2
                    + (coords[i][1] - coords[j][1]) ** 2
                )
                pixel_dist = np.sqrt(
                    (y_shifts[i] - y_shifts[j]) ** 2 + (x_shifts[i] - x_shifts[j]) ** 2
                )
                if stage_dist > 0:
                    pixel_spacings.append(pixel_dist / stage_dist)

        if pixel_spacings:
            avg_pixel_spacing = np.mean(pixel_spacings)
            print(
                f"Verification - Average pixel spacing ratio: {avg_pixel_spacing:.4f} pixels/μm"
            )

            # Check for expected tile overlap
            expected_tile_spacing_pixels = actual_spacing * pixels_per_micron
            actual_avg_spacing = np.mean(
                [
                    np.sqrt(
                        (y_shifts[i] - y_shifts[j]) ** 2
                        + (x_shifts[i] - x_shifts[j]) ** 2
                    )
                    for i in range(len(y_shifts))
                    for j in range(i + 1, len(y_shifts))
                ]
            )

            if actual_avg_spacing > 0:
                overlap_percent = (
                    (tile_size[0] - actual_avg_spacing) / tile_size[0] * 100
                )
                print(f"Phenotype tile overlap: {overlap_percent:.1f}%")
                if overlap_percent < 0:
                    print("⚠️  Warning: Negative overlap detected - tiles may have gaps")
                elif overlap_percent > 50:
                    print(
                        "⚠️  Warning: Very high overlap detected - may indicate scaling issues"
                    )

    final_size = (max(y_shifts) + tile_size[0], max(x_shifts) + tile_size[1])
    memory_gb = final_size[0] * final_size[1] * 2 / 1e9

    print(f"Phenotype final image size: {final_size}")
    print(f"Phenotype memory estimate: {memory_gb:.1f} GB")

    return {"total_translation": total_translation, "confidence": {well: confidence}}


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
    """Route to data-type-specific optimized functions with corrected coordinate scaling."""

    if data_type == "phenotype":
        return estimate_stitch_phenotype_coordinate_based(
            metadata_df, well, flipud, fliplr, rot90, channel
        )
    elif data_type == "sbs":
        return estimate_stitch_sbs_coordinate_based(
            metadata_df, well, flipud, fliplr, rot90, channel
        )
    else:
        # Fallback to original approach
        return estimate_stitch_aligned_tiff(
            metadata_df, well, data_type, flipud, fliplr, rot90, channel, tile_size
        )
