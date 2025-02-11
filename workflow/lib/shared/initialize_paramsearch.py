"""Initialize parameter searches."""

from snakemake.utils import Paramspace
import pandas as pd
from itertools import product
from pathlib import Path

from lib.shared.file_utils import get_filename


def initialize_segment_sbs_paramsearch(config):
    """Initialize parameter search for sbs segmentation by setting up config structure.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Updated configuration dictionary.
    """
    if config["sbsmode") != "segment_sbs_paramsearch":
        return config

    if "paramsearch" not in config["sbs"]:
        # Set default parameter search ranges if not specified
        base_nuclei = config["sbs"]["nuclei_diameter"]
        base_cell = config["sbs"]["cell_diameter"]

        config["sbs"]["paramsearch"] = {
            "nuclei_diameter": [base_nuclei - 2, base_nuclei, base_nuclei + 2],
            "cell_diameter": [base_cell - 2, base_cell, base_cell + 2],
            "flow_threshold": [0.2, 0.4, 0.6],
            "cellprob_threshold": [-4, -2, 0, 2, 4],
        }

    return config


def initialize_segment_phenotype_paramsearch(config):
    """Initialize parameter search for phenotype segmentation by setting up config structure.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Updated configuration dictionary.
    """
    if config["phenotype_process"].get("mode") != "segment_phenotype_paramsearch":
        return config

    if "paramsearch" not in config["phenotype_process"]:
        # Set default parameter search ranges if not specified
        base_nuclei = config["phenotype_process"]["nuclei_diameter"]
        base_cell = config["phenotype_process"]["cell_diameter"]

        config["phenotype_process"]["paramsearch"] = {
            "nuclei_diameter": [base_nuclei - 5, base_nuclei, base_nuclei + 5],
            "cell_diameter": [base_cell - 5, base_cell, base_cell + 5],
            "flow_threshold": [0.2, 0.4, 0.6],
            "cellprob_threshold": [-4, -2, 0, 2, 4],
        }

    return config
