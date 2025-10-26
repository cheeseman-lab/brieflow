"""Well-stitching functions for images and segmentation masks.

This module provides comprehensive functionality for stitching microscopy tiles
into complete well images, including image registration, mask assembly with
preserved cell IDs, and cell position extraction.

The registration functionality is based on the dexp library from Royer Lab:
https://github.com/royerlab/dexp
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import scipy
from pathlib import Path
from skimage import io, measure
from scipy.spatial.distance import pdist, squareform

from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.processing.registration import translation_nd as dexp_reg
from dexp.processing.utils.linear_solver import linsolve


def assemble_aligned_tiff_well(
    tile_files: Dict[int, str],
    shifts: Dict[str, List[int]],
    well: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
) -> np.ndarray:
    """Assemble a stitched well image from aligned TIFF tiles.

    This function loads individual tile images and assembles them into a complete
    well image using the calculated shift positions. Overlapping regions are
    blended using weighted averaging. Tile size is automatically detected.

    Parameters
    ----------
    tile_files : Dict[int, str]
        Mapping from tile_id to file path for aligned TIFFs
    shifts : Dict[str, List[int]]
        Tile shift positions as {well/tile_id: [y, x]}
    well : str
        Well identifier
    flipud : bool, default False
        Whether to flip tiles vertically
    fliplr : bool, default False
        Whether to flip tiles horizontally
    rot90 : int, default 0
        Number of 90-degree rotations
    channel : int, default 0
        Channel to extract from multi-channel images

    Returns:
    -------
    np.ndarray
        Assembled stitched image
    """
    if len(tile_files) == 0:
        raise ValueError(f"No tile files provided for well {well}")

    # Auto-detect tile size from first tile
    first_tile_id = next(iter(tile_files.keys()))
    first_tile = load_aligned_tiff(tile_files[first_tile_id], channel)
    if first_tile is None:
        raise ValueError(f"Could not load first tile")

    first_tile = augment_tile(first_tile, flipud, fliplr, rot90)
    tile_size = first_tile.shape
    print(f"Detected tile size: {tile_size}")

    # Calculate output dimensions and initialize arrays
    final_shape = get_output_shape_tiff(shifts, tile_size)
    output_image = np.zeros(final_shape, dtype=np.float32)
    divisor = np.zeros(final_shape, dtype=np.uint16)

    print(f"Assembling {len(tile_files)} tiles into shape {final_shape}")

    # Process each tile
    for tile_id, tile_path in tqdm(tile_files.items(), desc="Assembling tiles"):
        tile_key = f"{well}/{tile_id}"
        if tile_key not in shifts:
            continue

        # Load and transform tile
        tile_array = load_aligned_tiff(tile_path, channel)
        if tile_array is None:
            continue
        tile_array = augment_tile(tile_array, flipud, fliplr, rot90)

        # Place tile in output image
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

    # Normalize overlapping regions
    stitched = np.zeros_like(output_image, dtype=np.float32)
    nonzero_mask = divisor > 0
    stitched[nonzero_mask] = output_image[nonzero_mask] / divisor[nonzero_mask]

    return stitched.astype(np.uint16)


def assemble_stitched_masks(
    mask_files: Dict[int, str],
    shifts: Dict[str, List[int]],
    well: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    return_cell_mapping: bool = False,
):
    """Assemble stitched masks while preserving original cell IDs.

    This function loads individual mask tiles and assembles them into a complete
    well mask, maintaining a mapping between stitched cell IDs and their original
    tile and cell identifiers. Tile size is automatically detected.

    Parameters
    ----------
    mask_files : Dict[int, str]
        Mapping from tile_id to file path for mask files
    shifts : Dict[str, List[int]]
        Tile shift positions as {well/tile_id: [y, x]}
    well : str
        Well identifier
    flipud : bool, default False
        Whether to flip tiles vertically
    fliplr : bool, default False
        Whether to flip tiles horizontally
    rot90 : int, default 0
        Number of 90-degree rotations
    return_cell_mapping : bool, default False
        Whether to return cell ID mapping dictionary

    Returns:
    -------
    np.ndarray or Tuple[np.ndarray, Dict]
        Stitched mask array, and optionally cell ID mapping dictionary
    """
    print(f"MASK ASSEMBLY: well {well}")
    print(f"Input: {len(mask_files)} tiles, {len(shifts)} shifts")

    # Auto-detect tile size from first mask
    first_tile_id = next(iter(mask_files.keys()))
    first_mask = io.imread(mask_files[first_tile_id])
    if len(first_mask.shape) > 2:
        first_mask = first_mask[0] if first_mask.shape[0] == 1 else first_mask
    first_mask = augment_tile(first_mask, flipud, fliplr, rot90)
    tile_height, tile_width = first_mask.shape
    print(f"Detected tile size: {tile_height} x {tile_width}")

    # Calculate canvas size
    max_x = max_y = 0
    for tile_key, (shift_y, shift_x) in shifts.items():
        max_x = max(max_x, shift_x + tile_width)
        max_y = max(max_y, shift_y + tile_height)
    print(f"Canvas size: {max_y} x {max_x}")

    # Initialize stitched mask and cell mapping
    stitched_mask = np.zeros((max_y, max_x), dtype=np.uint32)
    cell_id_mapping = {}
    next_global_id = 1
    processed_tiles = 0
    tiles_with_no_cells = []
    total_cells_placed = 0

    # Process each tile
    for tile_id, mask_path in mask_files.items():
        tile_key = f"{well}/{tile_id}"
        if tile_key not in shifts or not Path(mask_path).exists():
            continue

        # Load and transform mask
        tile_mask = io.imread(mask_path)
        if flipud:
            tile_mask = np.flipud(tile_mask)
        if fliplr:
            tile_mask = np.fliplr(tile_mask)
        if rot90 > 0:
            tile_mask = np.rot90(tile_mask, rot90)

        # Get unique cell IDs
        original_cell_ids = np.unique(tile_mask)
        original_cell_ids = original_cell_ids[original_cell_ids > 0]
        processed_tiles += 1

        if len(original_cell_ids) == 0:
            tiles_with_no_cells.append(tile_id)
            continue

        # Create mapping and relabel
        tile_registry = {}
        for original_id in original_cell_ids:
            global_id = next_global_id
            cell_id_mapping[global_id] = (tile_id, int(original_id))
            tile_registry[int(original_id)] = global_id
            next_global_id += 1

        relabeled_mask = np.zeros_like(tile_mask, dtype=np.uint32)
        for original_id, global_id in tile_registry.items():
            relabeled_mask[tile_mask == original_id] = global_id

        # Place in stitched mask
        shift_y, shift_x = shifts[tile_key]
        end_y = min(shift_y + tile_height, max_y)
        end_x = min(shift_x + tile_width, max_x)

        mask_region = stitched_mask[shift_y:end_y, shift_x:end_x]
        tile_region = relabeled_mask[: end_y - shift_y, : end_x - shift_x]
        mask_region[tile_region > 0] = tile_region[tile_region > 0]

        total_cells_placed += len(original_cell_ids)

    # Report statistics
    if tiles_with_no_cells:
        print(f"Tiles with no cells found: {tiles_with_no_cells}")

    final_cells = np.unique(stitched_mask)
    final_cells = final_cells[final_cells > 0]

    print(f"STITCHING COMPLETE:")
    print(f"  Processed tiles: {processed_tiles}")
    print(f"  Final cell count: {len(final_cells):,}")
    print(f"  Cells placed: {total_cells_placed:,}")

    if return_cell_mapping:
        return stitched_mask, cell_id_mapping
    return stitched_mask


def extract_cell_positions_from_stitched_mask(
    stitched_mask: np.ndarray,
    well: str,
    plate: str,
    tile_metadata: pd.DataFrame,
    shifts: Dict[str, List[int]],
    cell_id_mapping: Dict[int, Tuple[int, int]],
    data_type: str = "phenotype",
) -> pd.DataFrame:
    """Extract cell positions using preserved cell ID mapping.

    This function analyzes the stitched mask to extract cell positions and
    properties, using the preserved mapping to maintain connections to original
    tile and cell identifiers.

    Parameters
    ----------
    stitched_mask : np.ndarray
        Stitched segmentation mask
    well : str
        Well identifier
    plate : str
        Plate identifier
    tile_metadata : pd.DataFrame
        Metadata containing tile information (must include tile_id column)
    shifts : Dict[str, List[int]]
        Tile shift positions as {well/tile_id: [y, x]}
    cell_id_mapping : Dict[int, Tuple[int, int]]
        Mapping from stitched cell ID to (tile_id, original_cell_id)
    data_type : str, default "phenotype"
        Type of data being processed

    Returns:
    -------
    pd.DataFrame
        Cell positions with preserved ID mapping and metadata
    """
    print(f"EXTRACTING CELL POSITIONS: {data_type} well {well}, plate {plate}")

    # Get region properties
    props = measure.regionprops(stitched_mask)

    if len(props) == 0:
        print("No cells found in stitched mask")
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

    # Create lookup dictionaries
    tile_metadata_lookup = tile_metadata.set_index("tile").to_dict("index")
    tile_shift_lookup = {
        tile_id: shifts[f"{well}/{tile_id}"]
        for tile_id in tile_metadata["tile"].unique()
        if f"{well}/{tile_id}" in shifts
    }

    # Extract cell information
    cell_data = []
    missing_mappings = 0

    for prop in props:
        cell_info = create_cell_info(
            prop,
            well,
            plate,
            data_type,
            cell_id_mapping,
            tile_metadata_lookup,
            tile_shift_lookup,
        )

        if cell_info["mapping_method"] == "missing_mapping":
            missing_mappings += 1

        cell_data.append(cell_info)

    df = pd.DataFrame(cell_data)

    # Ensure correct data types
    if len(df) > 0:
        for col in [
            "cell",
            "tile",
            "original_cell_id",
            "original_tile_id",
            "stitched_cell_id",
            "label",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(-1).astype("Int64")

        df["plate"] = df["plate"].astype(str)
        df["well"] = df["well"].astype(str)
        df["data_type"] = df["data_type"].astype(str)

    print(f"EXTRACTION COMPLETE:")
    print(f"  Total cells: {len(df)}")
    print(f"  Mapped cells: {len(df) - missing_mappings}")
    if missing_mappings > 0:
        print(f"  Missing mappings: {missing_mappings}")
    print(f"  Success rate: {(len(df) - missing_mappings) / len(df) * 100:.1f}%")

    return df


def metadata_to_grid_positions(
    metadata_df: pd.DataFrame, well: str, grid_spacing_tolerance: float = 1.0
) -> Dict[int, Tuple[int, int]]:
    """Convert stage coordinate metadata to grid positions preserving well geometry.

    This function converts continuous stage coordinates to discrete grid positions
    while preserving the spatial relationships and circular geometry of wells.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata containing stage coordinates
    well : str
        Well identifier to process
    grid_spacing_tolerance : float, default 1.0
        Tolerance for grid spacing detection (currently unused)

    Returns:
    -------
    Dict[int, Tuple[int, int]]
        Mapping from tile_id to (grid_row, grid_col) positions
    """
    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        return {}

    coords = well_metadata[["x_pos", "y_pos"]].values
    tile_ids = well_metadata["tile"].values

    print(f"Processing {len(tile_ids)} tiles preserving circular well geometry")

    # Determine actual tile spacing from coordinate distribution
    distances = pdist(coords)
    actual_spacing = np.percentile(distances[distances > 0], 10)

    print(f"Stage coordinate ranges:")
    print(f"  X: {coords[:, 0].min():.1f} to {coords[:, 0].max():.1f} μm")
    print(f"  Y: {coords[:, 1].min():.1f} to {coords[:, 1].max():.1f} μm")
    print(f"Detected tile spacing: {actual_spacing:.1f} μm")

    # Convert to grid coordinates preserving relative positions
    x_min, y_min = coords.min(axis=0)
    x_spacing = y_spacing = actual_spacing

    tile_positions = {}
    max_grid_x = max_grid_y = 0

    for i, tile_id in enumerate(tile_ids):
        x_pos, y_pos = coords[i]
        grid_x = int(round((x_pos - x_min) / x_spacing))
        grid_y = int(round((y_pos - y_min) / y_spacing))
        tile_positions[tile_id] = (grid_y, grid_x)
        max_grid_x = max(max_grid_x, grid_x)
        max_grid_y = max(max_grid_y, grid_y)

    print(f"Grid coordinate ranges: Y=0 to {max_grid_y}, X=0 to {max_grid_x}")
    print(f"Created grid positions for {len(tile_positions)} tiles")

    # Verify circular geometry preservation
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
        print(
            "✅ Preserved circular well geometry"
            if cv < 0.4
            else f"⚠️  Geometry may be distorted (CV: {cv:.2f})"
        )

    return tile_positions


def connectivity_from_actual_positions(
    tile_positions: Dict[int, Tuple[int, int]],
    stage_coords: np.ndarray,
    tile_ids: np.ndarray,
    data_type: str,
    max_neighbors_per_tile: int = 3,
) -> Dict[str, List[Tuple[int, int]]]:
    """Create connectivity graph based on spatial proximity with data-type-specific thresholds.

    Parameters
    ----------
    tile_positions : Dict[int, Tuple[int, int]]
        Mapping from tile_id to grid positions
    stage_coords : np.ndarray
        Stage coordinates for tiles
    tile_ids : np.ndarray
        Array of tile identifiers
    data_type : str
        Data type ('phenotype' or 'sbs') for threshold optimization
    max_neighbors_per_tile : int, default 3
        Maximum number of neighbors per tile

    Returns:
    -------
    Dict[str, List[Tuple[int, int]]]
        Dictionary of edges mapping edge_id to [pos_a, pos_b]
    """
    # Configuration for different data types
    CONNECTIVITY_CONFIG = {
        "phenotype": {
            "threshold_multiplier": 1.005,
            "max_neighbors": 3,
            "expected_edges": (8000, 15000),
            "description": "Using threshold = min_dist * 1.005",
        },
        "sbs": {
            "percentile": 5,
            "max_neighbors": 3,
            "expected_edges": (3000, 8000),
            "description": "Using threshold = 5th percentile",
        },
    }

    config = CONNECTIVITY_CONFIG.get(
        data_type,
        {
            "threshold_multiplier": 1.05,
            "max_neighbors": 4,
            "expected_edges": None,
            "description": "Using fallback threshold",
        },
    )

    # Calculate proximity-based edges
    distances = squareform(pdist(stage_coords))
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]

    # Determine threshold
    if "percentile" in config:
        proximity_threshold = np.percentile(upper_triangle, config["percentile"])
    else:
        min_distance = distances[distances > 0].min()
        proximity_threshold = min_distance * config["threshold_multiplier"]

    max_neighbors = config["max_neighbors"]
    print(
        f"{data_type.capitalize()}: {config['description']} = {proximity_threshold:.1f}"
    )

    edges = {}
    edge_idx = 0

    print(f"Creating edges with max {max_neighbors} neighbors per tile...")

    for i in range(len(tile_ids)):
        neighbor_distances = [
            (j, distances[i, j])
            for j in range(len(tile_ids))
            if j != i and distances[i, j] < proximity_threshold
        ]

        neighbor_distances.sort(key=lambda x: x[1])
        neighbors_to_use = neighbor_distances[:max_neighbors]

        for j, dist in neighbors_to_use:
            if i < j:
                pos_a = tile_positions[tile_ids[i]]
                pos_b = tile_positions[tile_ids[j]]
                edges[f"{edge_idx}"] = [pos_a, pos_b]
                edge_idx += 1

    print(f"Created {len(edges)} edges for circular {data_type} geometry")

    # Verify edge count
    expected = config.get("expected_edges")
    if expected and not (expected[0] <= len(edges) <= expected[1]):
        print(
            f"Warning: {data_type.capitalize()} edge count {len(edges)} outside expected range {expected[0]}-{expected[1]}"
        )
    else:
        print(f"Edge count {len(edges)} in optimal range for {data_type}")

    return edges


def optimal_positions_tiff(
    edge_list: List,
    tile_positions: Dict[int, Tuple[int, int]],
    well: str,
) -> Dict[str, List[int]]:
    """Calculate optimal tile positions using least squares optimization.

    This function solves for the optimal tile positions that minimize registration
    errors across all tile pairs using sparse linear least squares.

    Parameters
    ----------
    edge_list : List
        List of edges with registration results (must have tile_a_id, tile_b_id, and model attributes)
    tile_positions : Dict[int, Tuple[int, int]]
        Initial grid positions for tiles
    well : str
        Well identifier

    Returns:
    -------
    Dict[str, List[int]]
        Optimized tile positions as {well/tile_id: [y_pixel, x_pixel]}
    """
    if len(edge_list) == 0:
        return {}

    tile_ids = list(tile_positions.keys())
    tile_lut = {tile_id: i for i, tile_id in enumerate(tile_ids)}
    n_tiles = len(tile_ids)
    n_edges = len(edge_list)

    # Auto-detect tile size from first edge
    first_edge = edge_list[0]
    tile_a = first_edge.tile_cache[first_edge.tile_a_id]
    tile_size = tile_a.shape

    print(f"Detected tile size for optimization: {tile_size}")

    y_i = np.zeros(n_edges + 1, dtype=np.float32)
    y_j = np.zeros(n_edges + 1, dtype=np.float32)

    # Use grid positions as initial guess
    x_guess = np.array(
        [tile_positions[tile_id][1] for tile_id in tile_ids], dtype=np.float32
    )
    y_guess = np.array(
        [tile_positions[tile_id][0] for tile_id in tile_ids], dtype=np.float32
    )

    print(
        f"Initial grid position ranges: Y=[{y_guess.min():.0f}, {y_guess.max():.0f}], "
        f"X=[{x_guess.min():.0f}, {x_guess.max():.0f}]"
    )

    # Build constraint matrix
    a = scipy.sparse.lil_matrix((n_edges + 1, n_tiles), dtype=np.float32)

    for c, edge in enumerate(edge_list):
        tile_a_idx = tile_lut[edge.tile_a_id]
        tile_b_idx = tile_lut[edge.tile_b_id]
        a[c, tile_a_idx] = -1
        a[c, tile_b_idx] = 1

        # Convert shift vectors from pixels to grid units
        y_i[c] = edge.model.shift_vector[0] / tile_size[0]
        y_j[c] = edge.model.shift_vector[1] / tile_size[1]

    # Fix first tile at origin
    y_i[-1] = y_j[-1] = 0
    a[-1, 0] = 1
    a = a.tocsr()

    # Solve optimization in grid space
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

    print(
        f"Optimized grid position ranges: "
        f"Y=[{opt_shifts_zeroed[:, 0].min():.0f}, {opt_shifts_zeroed[:, 0].max():.0f}], "
        f"X=[{opt_shifts_zeroed[:, 1].min():.0f}, {opt_shifts_zeroed[:, 1].max():.0f}]"
    )

    # Convert to pixel coordinates
    opt_shifts_dict = {}
    for i, tile_id in enumerate(tile_ids):
        pixel_y = int(opt_shifts_zeroed[i, 0] * tile_size[0])
        pixel_x = int(opt_shifts_zeroed[i, 1] * tile_size[1])
        opt_shifts_dict[f"{well}/{tile_id}"] = [pixel_y, pixel_x]

    # Verify reasonable output
    final_y_shifts = [s[0] for s in opt_shifts_dict.values()]
    final_x_shifts = [s[1] for s in opt_shifts_dict.values()]
    max_y_range = max(final_y_shifts) - min(final_y_shifts)
    max_x_range = max(final_x_shifts) - min(final_x_shifts)

    print(f"Final pixel shift ranges: Y={max_y_range}, X={max_x_range}")

    if max_y_range > 100000 or max_x_range > 100000:
        print(f"Warning: Very large shift ranges detected")
    else:
        print(f"Shift ranges look reasonable")

    return opt_shifts_dict


class LimitedSizeDict(OrderedDict):
    """A dictionary that maintains a maximum size by removing oldest items."""

    def __init__(self, max_size):
        """Initialize LimitedSizeDict with maximum size.

        Parameters
        ----------
        max_size : int
            Maximum number of items to store in the dictionary
        """
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        """Set item and remove oldest if size exceeds maximum.

        Parameters
        ----------
        key : Any
            Dictionary key
        value : Any
            Dictionary value
        """
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)


class TileCache:
    """Generic cache for loading tiles with transformations."""

    def __init__(self, tile_files, flipud=False, fliplr=False, rot90=0, channel=None):
        """Initialize TileCache with file paths and transformation parameters.

        Parameters
        ----------
        tile_files : Dict[int, str]
            Mapping from tile_id to file paths
        flipud : bool, default False
            Whether to flip tiles vertically
        fliplr : bool, default False
            Whether to flip tiles horizontally
        rot90 : int, default 0
            Number of 90-degree rotations
        channel : int or None, default None
            Channel to extract for multi-channel images
        """
        self.cache = LimitedSizeDict(max_size=50)
        self.tile_files = tile_files
        self.flipud = flipud
        self.fliplr = fliplr
        self.rot90 = rot90
        self.channel = channel

    def __getitem__(self, tile_id):
        """Get tile from cache or load if not cached.

        Parameters
        ----------
        tile_id : int
            Tile identifier

        Returns:
        -------
        np.ndarray
            Tile array

        Raises:
        ------
        KeyError
            If tile cannot be found or loaded
        """
        if tile_id in self.cache:
            return self.cache[tile_id]
        tile_data = self.load_tile(tile_id)
        if tile_data is not None:
            return tile_data
        raise KeyError(f"Tile {tile_id} not found")

    def load_tile(self, tile_id):
        """Override in subclass."""
        raise NotImplementedError


class AlignedTiffTileCache(TileCache):
    """Cache for loading aligned TIFF tiles."""

    def load_tile(self, tile_id):
        """Load and cache an aligned TIFF tile.

        Parameters
        ----------
        tile_id : int
            Tile identifier

        Returns:
        -------
        np.ndarray or None
            Loaded and transformed tile array, or None if loading fails
        """
        if tile_id not in self.tile_files:
            print(f"No file path found for tile {tile_id}")
            return None

        try:
            tile_2d = load_aligned_tiff(self.tile_files[tile_id], self.channel)
            if tile_2d is None:
                return None
            aug_tile = augment_tile(tile_2d, self.flipud, self.fliplr, self.rot90)
            self.cache[tile_id] = aug_tile
            return aug_tile
        except Exception as e:
            print(f"Error loading tile {tile_id}: {e}")
            return None


class MaskTileCache(TileCache):
    """Cache for loading segmentation mask tiles."""

    def load_tile(self, tile_id):
        """Load and cache a segmentation mask tile.

        Parameters
        ----------
        tile_id : int
            Tile identifier

        Returns:
        -------
        np.ndarray or None
            Loaded and transformed mask array, or None if loading fails
        """
        if tile_id not in self.tile_files:
            print(f"No file path found for mask tile {tile_id}")
            return None

        try:
            mask_array = io.imread(self.tile_files[tile_id])
            if len(mask_array.shape) > 2:
                mask_array = mask_array[0] if mask_array.shape[0] == 1 else mask_array
            aug_mask = augment_tile(mask_array, self.flipud, self.fliplr, self.rot90)
            self.cache[tile_id] = aug_mask
            return aug_mask
        except Exception as e:
            print(f"Error loading mask tile {tile_id}: {e}")
            return None


class AlignedTiffEdge:
    """Edge between two tiles for registration using aligned TIFF images.

    Parameters
    ----------
    tile_a_id : int
        First tile identifier
    tile_b_id : int
        Second tile identifier
    tile_cache : TileCache
        Cache object for loading tiles
    tile_positions : Dict[int, Tuple[int, int]]
        Mapping from tile_id to grid positions
    overlap_fraction : float, default 0.03
        Fraction of tile size to use for overlap region
    """

    def __init__(
        self,
        tile_a_id: int,
        tile_b_id: int,
        tile_cache,
        tile_positions: Dict[int, Tuple[int, int]],
        overlap_fraction: float = 0.03,
    ):
        """Initialize edge between two tiles for registration."""
        self.tile_cache = tile_cache
        self.tile_a_id = tile_a_id
        self.tile_b_id = tile_b_id
        self.pos_a = tile_positions[tile_a_id]
        self.pos_b = tile_positions[tile_b_id]
        self.relation = (self.pos_a[0] - self.pos_b[0], self.pos_a[1] - self.pos_b[1])
        self.overlap_fraction = overlap_fraction
        self.model = self.get_offset()

    def get_offset(self) -> TranslationRegistrationModel:
        """Calculate offset between two tiles using image registration.

        Returns:
        -------
        TranslationRegistrationModel
            Registration model containing shift vector and confidence
        """
        tile_a = self.tile_cache[self.tile_a_id]
        tile_b = self.tile_cache[self.tile_b_id]

        if tile_a is None or tile_b is None:
            model = TranslationRegistrationModel()
            model.shift_vector = np.array([0.0, 0.0])
            model.confidence = 0.0
            return model

        tile_size_avg = (tile_a.shape[0] + tile_a.shape[1]) / 2
        overlap = int(tile_size_avg * self.overlap_fraction)

        return offset_tiff(tile_a, tile_b, self.relation, overlap=overlap)


def augment_tile(
    tile: np.ndarray, flipud: bool, fliplr: bool, rot90: int
) -> np.ndarray:
    """Apply geometric transformations to a tile.

    Parameters
    ----------
    tile : np.ndarray
        Input tile array
    flipud : bool
        Whether to flip tile vertically
    fliplr : bool
        Whether to flip tile horizontally
    rot90 : int
        Number of 90-degree rotations to apply

    Returns:
    -------
    np.ndarray
        Transformed tile array
    """
    if flipud:
        tile = np.flip(tile, axis=-2)
    if fliplr:
        tile = np.flip(tile, axis=-1)
    if rot90:
        tile = np.rot90(tile, k=rot90, axes=(-2, -1))
    return tile


def load_aligned_tiff(file_path: str, channel: int = 0) -> Optional[np.ndarray]:
    """Load an aligned TIFF file and extract specific channel.

    Parameters
    ----------
    file_path : str
        Path to TIFF file
    channel : int, default 0
        Channel index to extract from multi-channel images

    Returns:
    -------
    Optional[np.ndarray]
        Loaded image array, or None if loading fails
    """
    try:
        image = io.imread(file_path)

        if len(image.shape) == 2:
            return image
        elif len(image.shape) == 3:
            return image[min(channel, image.shape[0] - 1), :, :]
        else:
            raise ValueError(f"Unexpected image dimensions: {image.shape}")

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def offset_tiff(
    image_a: np.ndarray, image_b: np.ndarray, relation: Tuple[int, int], overlap: int
) -> TranslationRegistrationModel:
    """Calculate offset between two images based on their spatial relationship.

    Parameters
    ----------
    image_a : np.ndarray
        First image array
    image_b : np.ndarray
        Second image array
    relation : Tuple[int, int]
        Spatial relationship between tiles (row_diff, col_diff)
    overlap : int
        Overlap region size in pixels

    Returns:
    -------
    TranslationRegistrationModel
        Registration model containing shift vector and confidence
    """
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


def get_output_shape_tiff(
    shifts: Dict[str, List[int]], tile_size: Tuple[int, int]
) -> Tuple[int, int]:
    """Calculate the output shape needed for stitched image.

    Parameters
    ----------
    shifts : Dict[str, List[int]]
        Tile shift positions as {well/tile_id: [y, x]}
    tile_size : Tuple[int, int]
        Size of individual tiles (height, width)

    Returns:
    -------
    Tuple[int, int]
        Required output dimensions (height, width)
    """
    if not shifts:
        return tile_size

    y_shifts = [shift[0] for shift in shifts.values()]
    x_shifts = [shift[1] for shift in shifts.values()]
    max_y = int(np.max(y_shifts)) if y_shifts else 0
    max_x = int(np.max(x_shifts)) if x_shifts else 0

    return max_y + tile_size[0], max_x + tile_size[1]


def create_cell_info(
    prop,
    well: str,
    plate: str,
    data_type: str,
    cell_id_mapping: Dict[int, Tuple[int, int]],
    tile_metadata_lookup: Dict,
    tile_shift_lookup: Dict,
) -> Dict:
    """Create cell info dictionary from region properties.

    Parameters
    ----------
    prop : RegionProperties
        Scikit-image region properties object
    well : str
        Well identifier
    plate : str
        Plate identifier
    data_type : str
        Type of data being processed
    cell_id_mapping : Dict[int, Tuple[int, int]]
        Mapping from stitched cell ID to (tile_id, original_cell_id)
    tile_metadata_lookup : Dict
        Dictionary mapping tile_id to metadata
    tile_shift_lookup : Dict
        Dictionary mapping tile_id to shift positions

    Returns:
    -------
    Dict
        Cell information dictionary with all relevant metadata
    """
    stitched_label = prop.label

    # Base info
    info = {
        "well": well,
        "plate": str(plate),
        "stitched_cell_id": stitched_label,
        "i": prop.centroid[0],
        "j": prop.centroid[1],
        "area": prop.area,
        "data_type": data_type,
    }

    # Add mapping info
    if stitched_label in cell_id_mapping:
        original_tile_id, original_cell_id = cell_id_mapping[stitched_label]
        tile_shift = tile_shift_lookup.get(original_tile_id, [0, 0])
        tile_meta = tile_metadata_lookup.get(original_tile_id, {})

        info.update(
            {
                "cell": original_cell_id,
                "tile": original_tile_id,
                "label": original_cell_id,
                "original_cell_id": original_cell_id,
                "original_tile_id": original_tile_id,
                "tile_i": info["i"] - tile_shift[0],
                "tile_j": info["j"] - tile_shift[1],
                "mapping_method": "preserved_mapping",
                "stage_x": tile_meta.get("x_pos", np.nan),
                "stage_y": tile_meta.get("y_pos", np.nan),
                "tile_row": tile_meta.get("tile_row", -1),
                "tile_col": tile_meta.get("tile_col", -1),
            }
        )
    else:
        info.update(
            {
                "cell": stitched_label,
                "tile": -1,
                "label": stitched_label,
                "original_cell_id": None,
                "original_tile_id": None,
                "tile_i": -1,
                "tile_j": -1,
                "mapping_method": "missing_mapping",
                "stage_x": np.nan,
                "stage_y": np.nan,
                "tile_row": -1,
                "tile_col": -1,
            }
        )

    return info
