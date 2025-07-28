import pandas as pd
import numpy as np
import json
from pathlib import Path
from tifffile import imread
from lib.sbs.global_parameter_search import (
    global_parameter_search_snakemake,
    get_global_best_parameters,
    visualize_global_results
)
from lib.sbs.barcode_cycle_utils import compute_barcode_cycle_positions
from lib.sbs.standardize_barcode_design import get_barcode_list

# Load barcode library
df_barcode_library = pd.read_csv(snakemake.params.df_barcode_library_fp, sep='\t')

# Get barcodes list
barcodes = get_barcode_list(df_barcode_library, sequencing_order="map_recomb")  # Adjust as needed

# Get available tiles from input files
available_tiles = []
for aligned_path in snakemake.input.aligned_images:
    # Extract tile identifier from file path
    path_parts = Path(aligned_path).stem.split('_')
    # Look for 'tile' in the path parts and extract the tile name
    for i, part in enumerate(path_parts):
        if part == 'tile' and i + 1 < len(path_parts):
            tile_name = f"tile_{path_parts[i + 1]}"
            available_tiles.append(tile_name)
            break
    else:
        # Fallback: use the last part of the stem as tile identifier
        tile_name = f"tile_{path_parts[-1]}" if path_parts else f"tile_{len(available_tiles):03d}"
        available_tiles.append(tile_name)

print(f"Found {len(available_tiles)} available tiles: {available_tiles[:5]}...")

# Create mapping from tile names to file paths
aligned_images_map = {tile: path for tile, path in zip(available_tiles, snakemake.input.aligned_images)}
nuclei_masks_map = {tile: path for tile, path in zip(available_tiles, snakemake.input.nuclei_masks)}
cell_masks_map = {tile: path for tile, path in zip(available_tiles, snakemake.input.cell_masks)}

# Create SBS outputs structure for the function
sbs_outputs = {
    "align_sbs": aligned_images_map,
    "segment_sbs": [nuclei_masks_map, cell_masks_map]
}

# Extract plate and well from wildcards
plate = snakemake.wildcards.plate
well = snakemake.wildcards.well

# Prepare wildcards for automated parameter search
wildcards_dict = dict(snakemake.wildcards)

# Run global parameter search with modified function for direct file access
def modified_global_parameter_search():
    """Modified version that works directly with file paths."""
    from lib.sbs.automated_parameter_search import automated_parameter_search
    
    # Sample tiles
    import random
    if snakemake.params.random_seed is not None:
        random.seed(snakemake.params.random_seed)
    
    total_tiles = len(available_tiles)
    n_sample = int(total_tiles * snakemake.params.sample_fraction)
    n_sample = max(snakemake.params.min_tiles, min(n_sample, snakemake.params.max_tiles, total_tiles))
    
    selected_tiles = random.sample(available_tiles, n_sample)
    print(f"Sampling {n_sample} tiles from {total_tiles}: {selected_tiles}")
    
    all_results = []
    all_cells = []
    tile_metadata = []
    
    for i, tile_name in enumerate(selected_tiles):
        print(f"\nProcessing tile {i+1}/{len(selected_tiles)}: {tile_name}")
        
        # Load tile data directly
        try:
            aligned_images = imread(aligned_images_map[tile_name])
            if snakemake.params.use_cells_for_segmentation:
                segmentation_mask = imread(cell_masks_map[tile_name])
            else:
                segmentation_mask = imread(nuclei_masks_map[tile_name])
                
        except Exception as e:
            print(f"Failed to load data for {tile_name}: {e}")
            tile_metadata.append({
                'tile_name': tile_name,
                'status': 'load_failed',
                'n_cells': 0,
                'n_combinations': 0
            })
            continue
        
        # Run parameter search on this tile
        try:
            results_df, cells_df = automated_parameter_search(
                aligned_images=aligned_images,
                nuclei_mask=segmentation_mask,
                barcodes=barcodes,
                df_barcode_library=df_barcode_library,
                wildcards=wildcards_dict,
                bases=list(snakemake.params.bases),
                extra_channel_indices=snakemake.params.extra_channel_indices,
                # Parameter search ranges
                peak_width_range=snakemake.params.peak_width_range,
                threshold_range=snakemake.params.threshold_range,
                # Processing parameters
                max_filter_width=snakemake.params.max_filter_width,
                call_reads_method=snakemake.params.call_reads_method,
                map_start=snakemake.params.map_start,
                map_end=snakemake.params.map_end,
                recomb_start=snakemake.params.recomb_start,
                recomb_end=snakemake.params.recomb_end,
                map_col=snakemake.params.map_col,
                recomb_col=snakemake.params.recomb_col,
                recomb_filter_col=snakemake.params.recomb_filter_col,
                q_min=snakemake.params.q_min,
                error_correct=snakemake.params.error_correct,
                recomb_q_thresh=snakemake.params.recomb_q_thresh,
                verbose=snakemake.params.verbose,
            )
            
            # Add tile identifier to results
            if results_df is not None and len(results_df) > 0:
                results_df['tile_name'] = tile_name
                all_results.append(results_df)
                
                # Track metadata
                n_cells = len(np.unique(segmentation_mask)) - 1 if segmentation_mask is not None else 0
                tile_metadata.append({
                    'tile_name': tile_name,
                    'status': 'success',
                    'n_cells': n_cells,
                    'n_combinations': len(results_df),
                    'n_successful': len(results_df[results_df['status'] == 'success'])
                })
            
            # Add tile identifier to cells data
            if cells_df is not None and len(cells_df) > 0:
                cells_df['tile_name'] = tile_name
                all_cells.append(cells_df)
                
        except Exception as e:
            print(f"Error processing tile {tile_name}: {e}")
            tile_metadata.append({
                'tile_name': tile_name,
                'status': 'processing_failed',
                'n_cells': 0,
                'n_combinations': 0
            })
            continue
    
    # Combine results
    global_results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    global_cells_df = pd.concat(all_cells, ignore_index=True) if all_cells else None
    
    # Create metadata
    metadata = {
        'plate': plate,
        'well': well,
        'n_tiles_available': len(available_tiles),
        'n_tiles_sampled': len(selected_tiles),
        'n_tiles_successful': len([m for m in tile_metadata if m['status'] == 'success']),
        'sample_fraction': snakemake.params.sample_fraction,
        'random_seed': snakemake.params.random_seed,
        'tile_metadata': tile_metadata,
        'total_cells': global_results_df['total_cells'].sum() if len(global_results_df) > 0 else 0,
        'total_combinations_tested': len(global_results_df) if len(global_results_df) > 0 else 0
    }
    
    return global_results_df, global_cells_df, metadata

# Run the parameter search
global_results_df, global_cells_df, metadata = modified_global_parameter_search()

# Get globally optimal parameters
best_params = get_global_best_parameters(
    global_results_df,
    global_cells_df,
    priority=snakemake.params.priority,
    verbose=snakemake.params.verbose
)

# Save optimal parameters
if best_params is not None:
    optimal_params_df = pd.DataFrame([{
        'peak_width': int(best_params['peak_width']),
        'threshold_reads': int(best_params['threshold_reads']),
        'fraction_one_barcode': best_params['fraction_one_barcode'],
        'fraction_any_barcode': best_params['fraction_any_barcode'],
        'total_reads': int(best_params['total_reads']),
        'priority_used': snakemake.params.priority,
        'n_tiles_sampled': metadata['n_tiles_sampled'],
        'n_tiles_successful': metadata['n_tiles_successful'],
        'plate': plate,
        'well': well,
    }])
else:
    # Fallback to defaults if no good parameters found
    optimal_params_df = pd.DataFrame([{
        'peak_width': snakemake.params.peak_width_range[0],
        'threshold_reads': snakemake.params.threshold_range[0],
        'fraction_one_barcode': 0.0,
        'fraction_any_barcode': 0.0,
        'total_reads': 0,
        'priority_used': snakemake.params.priority,
        'n_tiles_sampled': metadata['n_tiles_sampled'],
        'n_tiles_successful': metadata['n_tiles_successful'],
        'plate': plate,
        'well': well,
        'note': 'No successful parameter combinations found - using defaults'
    }])

# Save outputs
optimal_params_df.to_csv(snakemake.output[0], index=False, sep='\t')

# Save global results
if len(global_results_df) > 0:
    global_results_df.to_csv(snakemake.output[1], index=False, sep='\t')
else:
    # Create empty file with headers
    pd.DataFrame(columns=['peak_width', 'threshold_reads', 'status']).to_csv(
        snakemake.output[1], index=False, sep='\t'
    )

# Save global cells data
if global_cells_df is not None and len(global_cells_df) > 0:
    global_cells_df.to_parquet(snakemake.output[2])
else:
    # Create empty parquet file
    pd.DataFrame().to_parquet(snakemake.output[2])

# Save metadata
with open(snakemake.output[3], 'w') as f:
    json.dump(metadata, f, indent=2, default=str)