from lib.shared.io import read_image, save_image
from lib.shared.illumination_correction import apply_ic_field
import numpy as np
import os
import re

# Load aligned image data (Cycles, C, Z, Y, X)
# Assuming align_sbs output is (Cycle, C, Z, Y, X) or (Cycle, C, Y, X)
aligned_image_data = read_image(snakemake.input[0])

# Prepare a list to hold corrected cycles
corrected_cycles_list = []

# Map IC field paths to their corresponding cycle
ic_fields_map = {}
for cycle_ic_path in snakemake.input[1:]:
    # Extract cycle from the filename, e.g., 'P-1_W-A1_C-1__ic_field.zarr' -> '1'
    match = re.search(r'_C-(\d+)__ic_field.zarr', os.path.basename(cycle_ic_path))
    if match:
        cycle_name = match.group(1)
        ic_fields_map[cycle_name] = read_image(cycle_ic_path)
    else:
        print(f"WARNING: Could not parse cycle from IC field path: {cycle_ic_path}", file=sys.stderr)


# Get the list of all cycles from snakemake params
all_cycles_names = snakemake.params.all_cycles

# Iterate through each cycle of aligned_image_data by its position (index)
# Assuming the cycles in aligned_image_data are ordered corresponding to all_cycles_names
for cycle_array_idx, cycle_data in enumerate(aligned_image_data):
    # Get the cycle name (e.g., '1', '2', '3'...) corresponding to this array index
    # This assumes a direct mapping between array index and cycle name from all_cycles_names
    current_cycle_name = all_cycles_names[cycle_array_idx]

    # Get the corresponding illumination correction field
    ic_field = ic_fields_map.get(current_cycle_name)

    if ic_field is None:
        # If IC field is not found for a cycle, skip correction for this cycle
        print(f"WARNING: No IC field found for cycle {current_cycle_name}. Skipping correction for this cycle.", file=sys.stderr)
        corrected_cycles_list.append(cycle_data) # Add uncorrected data
        continue

    # Apply illumination correction for this cycle
    corrected_cycle_data = apply_ic_field(cycle_data, correction=ic_field)
    corrected_cycles_list.append(corrected_cycle_data)

# Stack the corrected cycles back together along the cycle axis
corrected_image_data = np.stack(corrected_cycles_list, axis=0)


# Save corrected image data
save_image(
    corrected_image_data,
    snakemake.output[0],
    pixel_size_z=snakemake.params.pixel_size_z,
    pixel_size_y=snakemake.params.pixel_size_y,
    pixel_size_x=snakemake.params.pixel_size_x,
    channel_names=snakemake.params.channel_names,
)