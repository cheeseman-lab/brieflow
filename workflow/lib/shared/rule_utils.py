"""Helper functions for using Snakemake rules for use with Brieflow."""

from pathlib import Path
import re
import glob
from typing import Dict, List, Any, Union
import pandas as pd

from lib.shared.file_utils import parse_filename


def get_alignment_params(wildcards, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get alignment parameters for a specific plate.

    Args:
        wildcards (snakemake.Wildcards): Snakemake wildcards object.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Alignment parameters for the specified plate.

    Raises:
        ValueError: If no alignment configuration is found for the specified plate.
    """
    # First check if we have plate-specific alignments
    if "alignments" in config["phenotype"]:
        plate_id = int(wildcards.plate)
        plate_config = config["phenotype"]["alignments"].get(plate_id)

        if not plate_config:
            raise ValueError(
                f"No alignment configuration found for plate {plate_id}. "
                f"Available plates: {list(config['phenotype']['alignments'].keys())}"
            )

        # Return appropriate config based on whether it's multi-step or not
        if "steps" in plate_config:
            # Multi-step alignment - include all parameters
            multi_step_params = {
                "align": True,
                "multi_step": True,
                "steps": plate_config["steps"],
                "window": plate_config.get("window", 2),
                "upsample_factor": plate_config.get("upsample_factor", 2),
            }

            # Add custom channel offsets if they exist
            if "custom_channel_offsets" in plate_config:
                multi_step_params["custom_channel_offsets"] = plate_config[
                    "custom_channel_offsets"
                ]

            return multi_step_params

        # Check if this is custom-offsets-only (no automatic alignment)
        has_automatic = "target" in plate_config and "source" in plate_config

        if not has_automatic:
            # Custom offsets only, no automatic alignment
            custom_only_params = {
                "align": False,
                "multi_step": False,
            }

            if "custom_channel_offsets" in plate_config:
                custom_only_params["custom_channel_offsets"] = plate_config[
                    "custom_channel_offsets"
                ]

            return custom_only_params

        # Create base alignment params (single-step with automatic alignment)
        alignment_params = {
            "align": True,
            "multi_step": False,
            "target": plate_config["target"],
            "source": plate_config["source"],
            "riders": plate_config.get("riders", []),
            "remove_channel": plate_config.get("remove_channel", False),
            "upsample_factor": plate_config.get("upsample_factor", 2),
            "window": plate_config.get("window", 2),
        }

        # Add custom channel offsets if they exist
        if "custom_channel_offsets" in plate_config:
            alignment_params["custom_channel_offsets"] = plate_config[
                "custom_channel_offsets"
            ]

        return alignment_params

    # If no plate-specific alignments, use global config
    base_params = {
        "align": config["phenotype"].get("align", False),
        "multi_step": False,
        "target": config["phenotype"].get("target"),
        "source": config["phenotype"].get("source"),
        "riders": config["phenotype"].get("riders", []),
        "remove_channel": config["phenotype"].get("remove_channel", False),
        "upsample_factor": config["phenotype"].get("upsample_factor", 2),
        "window": config["phenotype"].get("window", 2),
    }

    # Add global custom channel offsets if they exist
    if "custom_channel_offsets" in config["phenotype"]:
        base_params["custom_channel_offsets"] = config["phenotype"][
            "custom_channel_offsets"
        ]

    return base_params


def get_spot_detection_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get spot detection parameters.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Spot detection parameters for SBS processing.

    Raises:
        ValueError: If an unknown spot detection method is specified.
    """
    # Get module config
    module_config = config["sbs"]

    # Get spot detection method, default to standard if not specified
    method = module_config.get("spot_detection_method", "standard")

    # Common parameters for all methods
    params = {
        "method": method,
    }

    # Method-specific parameters
    if method == "standard":
        # No additional parameters needed for standard method
        pass
    elif method == "spotiflow":
        params.update(
            {
                "spotiflow_model": module_config.get("spotiflow_model", "general"),
                "spotiflow_threshold": module_config.get("spotiflow_threshold", 0.3),
                "spotiflow_cycle_index": module_config.get("spotiflow_cycle_index", 0),
                "spotiflow_min_distance": module_config.get(
                    "spotiflow_min_distance", 1
                ),
                "remove_index": module_config.get("spotiflow_remove_index", 0),
            }
        )
    else:
        raise ValueError(
            f"Unknown spot detection method: {method}. Choose one of: standard, spotiflow"
        )

    return params


def get_segmentation_params(module: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get segmentation parameters for a specific module.

    Args:
        module (str): Module name, either "sbs" or "phenotype".
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Segmentation parameters for the specified module.

    Raises:
        ValueError: If an unknown segmentation method is specified.
    """
    module_config = config[module]

    # Get segmentation method, default to cellpose if not specified
    segmentation_method = module_config.get("segmentation_method", "cellpose")

    # Common parameters for all methods
    params = {
        "segmentation_method": segmentation_method,
        "dapi_index": module_config.get("dapi_index"),
        "cyto_index": module_config.get("cyto_index"),
        "reconcile": module_config.get("reconcile", False),
        "return_counts": module_config.get("return_counts", True),
        "gpu": module_config.get("gpu", False),
        "segment_cells": module_config.get("segment_cells", True),
    }

    # Method-specific parameters
    if segmentation_method == "cellpose":
        params.update(
            {
                "cellpose_model": module_config.get("cellpose_model", "cyto3"),
                "helper_index": module_config.get("helper_index"),
                "nuclei_diameter": module_config.get("nuclei_diameter"),
                "cell_diameter": module_config.get("cell_diameter"),
                "flow_threshold": module_config.get("flow_threshold", 0.4),
                "cellprob_threshold": module_config.get("cellprob_threshold", 0),
                "nuclei_flow_threshold": module_config.get(
                    "nuclei_flow_threshold", module_config.get("flow_threshold", 0.4)
                ),
                "nuclei_cellprob_threshold": module_config.get(
                    "nuclei_cellprob_threshold",
                    module_config.get("cellprob_threshold", 0),
                ),
                "cell_flow_threshold": module_config.get(
                    "cell_flow_threshold", module_config.get("flow_threshold", 0.4)
                ),
                "cell_cellprob_threshold": module_config.get(
                    "cell_cellprob_threshold",
                    module_config.get("cellprob_threshold", 0),
                ),
            }
        )
    elif segmentation_method == "microsam":
        params.update(
            {
                "microsam_model": module_config.get("microsam_model", "vit_b_lm"),
            }
        )
    elif segmentation_method == "stardist":
        params.update(
            {
                "stardist_model": module_config.get(
                    "stardist_model", "2D_versatile_fluo"
                ),
                "nuclei_prob_threshold": module_config.get(
                    "nuclei_prob_threshold", 0.479071
                ),
                "nuclei_nms_threshold": module_config.get("nuclei_nms_threshold", 0.3),
                "cell_prob_threshold": module_config.get(
                    "cell_prob_threshold", 0.479071
                ),
                "cell_nms_threshold": module_config.get("cell_nms_threshold", 0.3),
            }
        )
    elif segmentation_method == "watershed":
        params.update(
            {
                "threshold_dapi": module_config.get("threshold_dapi", 4260),
                "nuclei_area_min": module_config.get("nuclei_area_min", 45),
                "nuclei_area_max": module_config.get("nuclei_area_max", 450),
                "threshold_cell": module_config.get("threshold_cell", 1300),
            }
        )
    else:
        raise ValueError(
            f"Unknown segmentation method: {segmentation_method}. Choose one of: cellpose, microsam, stardist, watershed"
        )

    return params


def get_montage_inputs(
    montage_data_checkpoint,
    montage_output_template,
    montage_overlay_template,
    channels,
    cell_class,
):
    """Generate montage input file paths based on checkpoint data and output template.

    Args:
        montage_data_checkpoint (object): Checkpoint object containing output directory information.
        montage_output_template (str): Template string for generating output file paths.
        montage_overlay_template (str): Template string for generating overlay file paths.
        channels (list): List of channels to include in the output file paths.
        cell_class (str): Cell class for which the montage is being generated.

    Returns:
        list: List of generated output file paths for each channel.
    """
    # Resolve the checkpoint output directory using .get()
    checkpoint_output = Path(
        montage_data_checkpoint.get(cell_class=cell_class).output[0]
    )

    # Get actual existing files
    montage_data_files = list(checkpoint_output.glob("*.tsv"))

    # Extract the gene_sgrna parts and make output paths for each channel
    output_files = []
    for montage_data_file in montage_data_files:
        # parse gene, sgrna from filename
        match = re.match(r".*G-(.+?)_SG-(.+?)__montage_data.*", montage_data_file.name)
        gene = match.group(1)
        sgrna = match.group(2)

        for channel in channels:
            # Generate the output file path using the template
            output_file = str(montage_output_template).format(
                gene=gene, sgrna=sgrna, channel=channel, cell_class=cell_class
            )

            # Append the output file path to the list
            output_files.append(output_file)

    # Add the overlay file path
    overlay_file = str(montage_overlay_template).format(
        gene=gene, sgrna=sgrna, cell_class=cell_class
    )
    output_files.append(overlay_file)

    return output_files


def get_bootstrap_inputs(
    checkpoint,
    construct_nulls_pattern: Union[str, Path],
    construct_pvals_pattern: Union[str, Path],
    gene_nulls_pattern: Union[str, Path],
    gene_pvals_pattern: Union[str, Path],
    cell_class: str,
    channel_combo: str,
) -> List[str]:
    """Get all bootstrap inputs for completion flag.

    Args:
        checkpoint: Checkpoint object containing bootstrap data directory information.
        construct_nulls_pattern (Union[str, Path]): Template string for construct null files.
        construct_pvals_pattern (Union[str, Path]): Template string for construct p-value files.
        gene_nulls_pattern (Union[str, Path]): Template string for gene null files.
        gene_pvals_pattern (Union[str, Path]): Template string for gene p-value files.
        cell_class (str): Cell class for bootstrap analysis.
        channel_combo (str): Channel combination for bootstrap analysis.

    Returns:
        List[str]: List of all bootstrap output file paths for both constructs and genes.
    """
    # Get all construct data files from checkpoint
    bootstrap_data_dir = checkpoint.get(
        cell_class=cell_class, channel_combo=channel_combo
    ).output[0]

    construct_files = glob.glob(f"{bootstrap_data_dir}/*__construct_data.tsv")

    outputs = []
    genes_seen = set()

    for construct_file in construct_files:
        # Extract gene__construct from filename
        construct_filename = Path(construct_file).stem.replace("__construct_data", "")

        # Parse using double underscore separator: gene__construct
        if "__" in construct_filename:
            parts = construct_filename.split("__")
            gene = parts[0]  # Full gene name (can contain underscores)
            construct = parts[1]  # Construct ID
        else:
            # Fallback: read from file
            try:
                construct_data = pd.read_csv(construct_file, sep="\t")
                gene = construct_data["gene"].iloc[0]
                construct = construct_data["construct_id"].iloc[0]
            except:
                continue

        # Add construct outputs
        outputs.extend(
            [
                str(construct_nulls_pattern).format(
                    cell_class=cell_class,
                    channel_combo=channel_combo,
                    gene=gene,
                    construct=construct,
                ),
                str(construct_pvals_pattern).format(
                    cell_class=cell_class,
                    channel_combo=channel_combo,
                    gene=gene,
                    construct=construct,
                ),
            ]
        )

        # Add gene outputs (avoid duplicates)
        if gene not in genes_seen:
            genes_seen.add(gene)
            outputs.extend(
                [
                    str(gene_nulls_pattern).format(
                        cell_class=cell_class, channel_combo=channel_combo, gene=gene
                    ),
                    str(gene_pvals_pattern).format(
                        cell_class=cell_class, channel_combo=channel_combo, gene=gene
                    ),
                ]
            )

    return outputs


def get_bootstrap_construct_outputs(
    checkpoint,
    construct_nulls_pattern: Union[str, Path],
    construct_pvals_pattern: Union[str, Path],
    cell_class: str,
    channel_combo: str,
) -> List[str]:
    """Get all construct bootstrap outputs for completion flag.

    Args:
        checkpoint: Checkpoint object containing bootstrap data directory information.
        construct_nulls_pattern (Union[str, Path]): Template string for construct null
            distribution files with format placeholders for cell_class, channel_combo,
            gene, and construct.
        construct_pvals_pattern (Union[str, Path]): Template string for construct p-value
            files with format placeholders for cell_class, channel_combo, gene, and construct.
        cell_class (str): Cell class identifier for bootstrap analysis (e.g., 'live', 'dead').
        channel_combo (str): Channel combination identifier for bootstrap analysis
            (e.g., 'dapi_tubulin', 'all_channels').

    Returns:
        List[str]: List of all construct bootstrap output file paths, including both
            null distribution files and p-value files for each construct found in
            the checkpoint directory.
    """
    # Get all construct data files from checkpoint
    bootstrap_data_dir = checkpoint.get(
        cell_class=cell_class, channel_combo=channel_combo
    ).output[0]

    construct_files = glob.glob(f"{bootstrap_data_dir}/*__construct_data.tsv")

    outputs = []

    for construct_file in construct_files:
        # Extract gene__construct from filename
        construct_filename = Path(construct_file).stem.replace("__construct_data", "")

        # Parse using double underscore separator: gene__construct
        if "__" in construct_filename:
            parts = construct_filename.split("__")
            gene = parts[0]  # Full gene name (can contain underscores)
            construct = parts[1]  # Construct ID
        else:
            # Fallback: read from file
            try:
                construct_data = pd.read_csv(construct_file, sep="\t")
                gene = construct_data["gene"].iloc[0]
                construct = construct_data["construct_id"].iloc[0]
            except:
                continue

        # Add construct outputs
        outputs.extend(
            [
                str(construct_nulls_pattern).format(
                    cell_class=cell_class,
                    channel_combo=channel_combo,
                    gene=gene,
                    construct=construct,
                ),
                str(construct_pvals_pattern).format(
                    cell_class=cell_class,
                    channel_combo=channel_combo,
                    gene=gene,
                    construct=construct,
                ),
            ]
        )

    return outputs


def get_call_cells_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract call_cells parameters based on barcode_type.

    Returns only the parameters needed for the call_cells rule based on
    whether barcode_type is "simple" or "multi". This avoids manual
    parameter enumeration in Snakemake rules.

    Args:
        config: Full Snakemake config dictionary

    Returns:
        Dictionary with parameters for call_cells rule
    """
    sbs_config = config["sbs"]
    barcode_type = sbs_config.get("barcode_type", "simple")

    # Common parameters for both modes
    params = {
        "barcode_type": barcode_type,
        "df_barcode_library_fp": sbs_config["df_barcode_library_fp"],
        "q_min": sbs_config.get("q_min", 0),
        "error_correct": sbs_config.get("error_correct", False),
        "sort_calls": sbs_config.get("sort_calls", "peak"),
        "max_distance": sbs_config.get("max_distance", 2),
        "barcode_info_cols": sbs_config.get("barcode_info_cols", None),
    }

    if barcode_type == "simple":
        # Single-barcode parameters
        params.update(
            {
                "barcode_col": sbs_config.get("barcode_col", "sgRNA"),
                "prefix_col": sbs_config.get("prefix_col", None),
                # Multi-barcode params explicitly None for clarity
                "map_start": None,
                "map_end": None,
                "recomb_start": None,
                "recomb_end": None,
            }
        )
    elif barcode_type == "multi":
        # Multi-barcode parameters
        params.update(
            {
                "barcode_col": None,  # Not used in multi mode
                "prefix_col": None,  # Not used in multi mode
                "map_start": sbs_config.get("map_start"),
                "map_end": sbs_config.get("map_end"),
                "map_col": sbs_config.get("map_col", "prefix_map"),
                "recomb_start": sbs_config.get("recomb_start"),
                "recomb_end": sbs_config.get("recomb_end"),
                "recomb_col": sbs_config.get("recomb_col", "prefix_recomb"),
                "recomb_filter_col": sbs_config.get("recomb_filter_col", None),
                "recomb_q_thresh": sbs_config.get("recomb_q_thresh", 0.1),
            }
        )
    else:
        raise ValueError(
            f"Invalid barcode_type: '{barcode_type}'. Must be 'simple' or 'multi'"
        )

    return params
