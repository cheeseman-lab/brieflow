"""Initialize parameter searches."""

from snakemake.utils import Paramspace
import pandas as pd
from itertools import product
from pathlib import Path

from lib.shared.file_utils import get_filename

def initialize_segment_sbs_paramsearch(config):
    """Initialize parameter search for sbs segmentation.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Updated configuration dictionary.
        list: Nuclei diameters.
        list: Cell diameters.
        list: Flow thresholds.
        list: Cell probabilities.
        dict: Output patterns.

    """

    if config['sbs_process'].get('mode') != 'segment_sbs_paramsearch':
        return config, None, None, None, None
        
    if 'paramsearch' not in config['sbs_process']:
        base_nuclei = config['sbs_process']['nuclei_diameter']
        base_cell = config['sbs_process']['cell_diameter']
        nuclei_diameters = [base_nuclei - 2, base_nuclei, base_nuclei + 2]
        cell_diameters = [base_cell - 2, base_cell, base_cell + 2]
        flow_thresholds = [0.2, 0.4, 0.6]
        cell_probs = [-4, -2, 0, 2, 4]
    else:
        nuclei_diameters = config['sbs_process']['paramsearch']['nuclei_diameter']
        cell_diameters = config['sbs_process']['paramsearch']['cell_diameter']
        flow_thresholds = config['sbs_process']['paramsearch']['flow_threshold']
        cell_probs = config['sbs_process']['paramsearch']['cellprob_threshold']
    
    SBS_PROCESS_FP = Path(config['all']['root_fp']) / 'sbs_process'

    # Create the output pattern using {wildcards} in the info_type
    return config, nuclei_diameters, cell_diameters, flow_thresholds, cell_probs, {
        "segment_sbs_paramsearch": [
            SBS_PROCESS_FP / "paramsearch" / "images" / get_filename(
                {"well": "{well}", "tile": "{tile}"},
                f"paramsearch_nd{{nuclei_diameter}}_cd{{cell_diameter}}_ft{{flow_threshold}}_cp{{cellprob_threshold}}_{suffix}",
                "tiff"
            )
            for suffix in ["nuclei", "cells"]
        ] + [
            SBS_PROCESS_FP / "paramsearch" / "tsvs" / get_filename(
                {"well": "{well}", "tile": "{tile}"},
                "paramsearch_nd{nuclei_diameter}_cd{cell_diameter}_ft{flow_threshold}_cp{cellprob_threshold}_segmentation_stats",
                "tsv"
            )
        ],
        "summarize_segment_sbs_paramsearch": [
            SBS_PROCESS_FP / "paramsearch" / "summary" / "segmentation_summary.tsv"
        ]       
    }


def initialize_segment_phenotype_paramsearch(config):
    """Initialize parameter search for phenotype segmentation.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Updated configuration dictionary.
        list: Nuclei diameters.
        list: Cell diameters.
        list: Flow thresholds.
        list: Cell probabilities.
        dict: Output patterns.

    """

    if config['phenotype_process'].get('mode') != 'segment_phenotype_paramsearch':
        return config, None, None, None, None
        
    if 'paramsearch' not in config['phenotype_process']:
        base_nuclei = config['phenotype_process']['nuclei_diameter']
        base_cell = config['phenotype_process']['cell_diameter']
        nuclei_diameters = [base_nuclei - 5, base_nuclei, base_nuclei + 5]
        cell_diameters = [base_cell - 5, base_cell, base_cell + 5]
        flow_thresholds = [0.2, 0.4, 0.6]
        cell_probs = [-4, -2, 0, 2, 4]
    else:
        nuclei_diameters = config['phenotype_process']['paramsearch']['nuclei_diameter']
        cell_diameters = config['phenotype_process']['paramsearch']['cell_diameter']
        flow_thresholds = config['phenotype_process']['paramsearch']['flow_threshold']
        cell_probs = config['phenotype_process']['paramsearch']['cellprob_threshold']
    
    PHENOTYPE_PROCESS_FP = Path(config['all']['root_fp']) / 'phenotype_process'

    # Create the output pattern using {wildcards} in the info_type
    return config, nuclei_diameters, cell_diameters, flow_thresholds, cell_probs, {
        "segment_phenotype_paramsearch": [
            PHENOTYPE_PROCESS_FP / "paramsearch" / "images" / get_filename(
                {"well": "{well}", "tile": "{tile}"},
                f"paramsearch_nd{{nuclei_diameter}}_cd{{cell_diameter}}_ft{{flow_threshold}}_cp{{cellprob_threshold}}_{suffix}",
                "tiff"
            )
            for suffix in ["nuclei", "cells"]
        ] + [
            PHENOTYPE_PROCESS_FP / "paramsearch" / "tsvs" / get_filename(
                {"well": "{well}", "tile": "{tile}"},
                "paramsearch_nd{nuclei_diameter}_cd{cell_diameter}_ft{flow_threshold}_cp{cellprob_threshold}_segmentation_stats",
                "tsv"
            )
        ],
        "summarize_segment_phenotype_paramsearch": [
            PHENOTYPE_PROCESS_FP / "paramsearch" / "summary" / "segmentation_summary.tsv"
        ]       
    }


def initialize_mapping_sbs_paramsearch(config):
    """Initialize parameter search for sbs mapping.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Updated configuration dictionary.
        list: Peak thresholds.
        list: Q min thresholds.
        dict: Output patterns.

    """
    
    if config['sbs_process'].get('mode') != 'mapping_sbs_paramsearch':
        return config, None, None, None
        
    if 'paramsearch' not in config['sbs_process']:
        threshold_peaks = [200, 300, 400]
        q_mins = [0.7, 0.8, 0.9, 1]
    else:
        threshold_peaks = config['sbs_process']['paramsearch']['threshold_peaks']
        q_mins = config['sbs_process']['paramsearch']['q_mins']
    
    SBS_PROCESS_FP = Path(config['all']['root_fp']) / 'sbs_process'
    
    return config, threshold_peaks, q_mins, {
        "extract_bases_sbs_paramsearch": [
            SBS_PROCESS_FP / "paramsearch" / "tsvs" / get_filename(
                {"well": "{well}", "tile": "{tile}"}, 
                "bases_tp{threshold_peaks}", 
                "tsv"
            )
        ],
        "call_reads_sbs_paramsearch": [
            SBS_PROCESS_FP / "paramsearch" / "tsvs" / get_filename(
                {"well": "{well}", "tile": "{tile}"}, 
                "reads_tp{threshold_peaks}", 
                "tsv"
            )
        ],
        "call_cells_sbs_paramsearch": [
            SBS_PROCESS_FP / "paramsearch" / "tsvs" / get_filename(
                {"well": "{well}", "tile": "{tile}"}, 
                "cells_tp{threshold_peaks}_qm{q_min}", 
                "tsv"
            )
        ],
        "summarize_mapping_sbs_paramsearch": [
            SBS_PROCESS_FP / "paramsearch" / "summary" / "mapping_summary.tsv"
        ]
    }