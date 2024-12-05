"""Initialize parameter searches."""

from snakemake.utils import Paramspace
import pandas as pd
from itertools import product
from pathlib import Path

from lib.shared.file_utils import get_filename

def initialize_segment_paramsearch(config, wells, tiles):
    """Initialize parameter search for segmentation.

    Args:
        config (dict): Configuration dictionary.
        wells (list): List of wells.
        tiles (list): List of tiles.

    Returns:
        dict: Updated configuration dictionary.
        Paramspace: Nuclei parameter space.
        Paramspace: Cell parameter space.
    """

    if config['sbs_process'].get('mode') != 'segment_paramsearch':
        return config, None, None
        
    if 'paramsearch' not in config['sbs_process']:
        base_nuclei = config['sbs_process']['nuclei_diameter']
        base_cell = config['sbs_process']['cell_diameter']
        nuclei_diameters = [base_nuclei - 2, base_nuclei, base_nuclei + 2]
        cell_diameters = [base_cell - 2, base_cell, base_cell + 2]
    else:
        nuclei_diameters = config['sbs_process']['paramsearch']['nuclei_diameter']
        cell_diameters = config['sbs_process']['paramsearch']['cell_diameter']
    
    SBS_PROCESS_FP = Path(config['all']['root_fp']) / 'sbs_process'

    # Create the output pattern using {wildcards} in the info_type
    return config, nuclei_diameters, cell_diameters, {
        "segment_paramsearch": [
            SBS_PROCESS_FP / "paramsearch" / "images" / get_filename(
                {"well": "{well}", "tile": "{tile}"},
                f"paramsearch_nd{{nuclei_diameter}}_cd{{cell_diameter}}_{suffix}",
                "tiff"
            )
            for suffix in ["nuclei", "cells"]
        ] + [
            SBS_PROCESS_FP / "paramsearch" / "tsvs" / get_filename(
                {"well": "{well}", "tile": "{tile}"},
                "paramsearch_nd{nuclei_diameter}_cd{cell_diameter}_segmentation_stats",
                "tsv"
            )
        ]
    }
