"""Helper functions for using Snakemake rules for use with Brieflow."""

from pathlib import Path

from lib.shared.file_utils import parse_filename


def get_alignment_params(wildcards, config):
    """Get alignment parameters for a specific plate.

    Args:
        wildcards (snakemake.Wildcards): Snakemake wildcards object.
        config (dict): Configuration dictionary.

    Returns:
        dict: Alignment parameters for the specified plate.
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
            return {"align": True, "multi_step": True, "steps": plate_config["steps"]}

        # Create base alignment params
        alignment_params = {
            "align": True,
            "multi_step": False,
            "target": plate_config["target"],
            "source": plate_config["source"],
            "riders": plate_config["riders"],
            "remove_channel": plate_config["remove_channel"],
        }

        # Add custom alignment parameters if they exist
        if "custom_align" in plate_config:
            alignment_params["custom_align"] = True
            alignment_params["custom_channels"] = plate_config.get(
                "custom_channels", []
            )
            alignment_params["custom_offset_yx"] = plate_config.get(
                "custom_offset_yx", (0, 0)
            )
        else:
            alignment_params["custom_align"] = False

        return alignment_params

    # If no plate-specific alignments, use global config
    base_params = {
        "align": config["phenotype"].get("align", False),
        "multi_step": False,
        "target": config["phenotype"].get("target"),
        "source": config["phenotype"].get("source"),
        "riders": config["phenotype"].get("riders", []),
        "remove_channel": config["phenotype"].get("remove_channel", False),
    }

    # Add global custom alignment parameters if they exist
    if "custom_align" in config["phenotype"]:
        base_params["custom_align"] = config["phenotype"].get("custom_align", False)
        base_params["custom_channels"] = config["phenotype"].get("custom_channels", [])
        base_params["custom_offset_yx"] = config["phenotype"].get(
            "custom_offset_yx", (0, 0)
        )
    else:
        base_params["custom_align"] = False

    return base_params


def get_spot_detection_params(config):
    """Get spot detection parameters.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Spot detection parameters for SBS processing.
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
            }
        )
    else:
        raise ValueError(
            f"Unknown spot detection method: {method}. Choose one of: standard, spotiflow"
        )

    return params


def get_segmentation_params(module, config):
    """Get segmentation parameters for a specific module.

    Args:
        module (str): Module name, either "sbs" or "phenotype".
        config (dict): Configuration dictionary.

    Returns:
        dict: Segmentation parameters for the specified module.
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
    montage_data_checkpoint, montage_output_template, channels, cell_class
):
    """Generate montage input file paths based on checkpoint data and output template.

    Args:
        montage_data_checkpoint (object): Checkpoint object containing output directory information.
        montage_output_template (str): Template string for generating output file paths.
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
        file_metadata = parse_filename(montage_data_file)[0]
        gene = file_metadata["gene"]
        sgrna = file_metadata["sgrna"]

        for channel in channels:
            output_file = str(montage_output_template).format(
                gene=gene, sgrna=sgrna, channel=channel, cell_class=cell_class
            )
            output_files.append(output_file)

    return output_files
