"""Utility functions for handling files during preprocessing."""

import pandas as pd
from typing import List, Union, Dict, Any


def get_sample_fps(
    samples_df: pd.DataFrame,
    plate: Union[int, str] = None,
    well: Union[int, str] = None,
    tile: Union[int, str] = None,
    cycle: Union[int, str] = None,
    channel: Union[int, str] = None,
    round_order: List[Union[int, str]] = None,
    channel_order: List[Union[int, str]] = None,
    verbose: bool = False,
) -> Union[str, List[str]]:
    """Filters the samples DataFrame and ensures consistent channel and round order.

    Args:
        samples_df (pd.DataFrame): DataFrame containing sample data.
        plate (Union[int, str], optional): Plate number to filter by. Defaults to None.
        well (Union[int, str], optional): Well identifier to filter by. Defaults to None.
        tile (Union[int, str], optional): Tile number to filter by. For well organization, set to None. Defaults to None.
        cycle (Union[int, str], optional): Cycle number to filter by. Defaults to None.
        channel (Union[int, str], optional): Channel to filter by. Defaults to None.
        round_order (List[Union[int, str]], optional): Order of rounds to return. Defaults to None.
        channel_order (List[Union[int, str]], optional): Order of channels. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        Union[str, List[str]]: Either a single filepath or ordered list of filepaths
    """
    filtered_df = samples_df
    if plate is not None:
        filtered_df = filtered_df[filtered_df["plate"].astype(str) == str(plate)]
    if well is not None:
        filtered_df = filtered_df[filtered_df["well"].astype(str) == str(well)]

    if tile is not None:
        filtered_df = filtered_df[filtered_df["tile"].astype(str) == str(tile)]

    if cycle is not None:
        filtered_df = filtered_df[filtered_df["cycle"].astype(str) == str(cycle)]
    if channel is not None:
        filtered_df = filtered_df[filtered_df["channel"].astype(str) == str(channel)]

    if round_order is not None:
        # Filter to only include specified rounds
        filtered_df = filtered_df[filtered_df["round"].isin(round_order)]

        # If no data after filtering, return results based on available rounds
        if len(filtered_df) == 0:
            print(
                f"No data found for specified rounds {round_order}. Using available rounds."
            )
            filtered_df = samples_df
            if plate is not None:
                filtered_df = filtered_df[
                    filtered_df["plate"].astype(str) == str(plate)
                ]
            if well is not None:
                filtered_df = filtered_df[filtered_df["well"].astype(str) == str(well)]
            # KEY CHANGE: Only filter by tile if tile is provided
            if tile is not None:
                filtered_df = filtered_df[filtered_df["tile"].astype(str) == str(tile)]
            if cycle is not None:
                filtered_df = filtered_df[
                    filtered_df["cycle"].astype(str) == str(cycle)
                ]
            if channel is not None:
                filtered_df = filtered_df[
                    filtered_df["channel"].astype(str) == str(channel)
                ]

        # Create dictionary mapping round to DataFrame rows
        round_groups = {
            round_num: group for round_num, group in filtered_df.groupby("round")
        }

        # Initialize lists to store files and channels
        all_files = []
        final_channel_order = []

        # Get available rounds that exist in the data
        available_rounds = sorted(round_groups.keys())

        # Process each available round
        for round_num in available_rounds:
            round_df = round_groups[round_num]

            # If channel order is specified, get files in that order for this round
            if "channel" in round_df.columns and channel_order is not None:
                # Create mapping of available channels to files for this round
                channel_to_file = dict(zip(round_df["channel"], round_df["sample_fp"]))
                # Add files for each requested channel if available in this round
                for channel in channel_order:
                    if channel in channel_to_file:
                        all_files.append(channel_to_file[channel])
                        final_channel_order.append(f"Round {round_num}: {channel}")
            else:
                # If no channel order, just take the first file from this round
                all_files.append(round_df["sample_fp"].iloc[0])
                if "channel" in round_df.columns:
                    final_channel_order.append(
                        f"Round {round_num}: {round_df['channel'].iloc[0]}"
                    )
                else:
                    final_channel_order.append(f"Round {round_num}")

        if verbose:
            print("\nFinal channel order:")
            for chan in final_channel_order:
                print(f"  {chan}")

        return all_files

    # If no rounds specified but we have channels and channel order
    if "channel" in filtered_df.columns and channel_order is not None:
        channel_to_file = dict(zip(filtered_df["channel"], filtered_df["sample_fp"]))
        return [
            channel_to_file[channel]
            for channel in channel_order
            if channel in channel_to_file
        ]

    # Otherwise return single file path
    return filtered_df["sample_fp"].iloc[0]


def get_metadata_wildcard_combos(
    samples_df: pd.DataFrame, metadata_samples_df: pd.DataFrame
) -> pd.DataFrame:
    """Get wildcard combinations for metadata extraction based on available files.

    Args:
        samples_df (pd.DataFrame): DataFrame with image file paths
        metadata_samples_df (pd.DataFrame): DataFrame with metadata file paths

    Returns:
        pd.DataFrame: DataFrame with wildcard combinations for metadata extraction jobs
    """
    if not metadata_samples_df.empty:
        # Use metadata file structure - this determines the job granularity
        metadata_columns = [
            col for col in metadata_samples_df.columns if col != "sample_fp"
        ]
        return metadata_samples_df[metadata_columns].drop_duplicates().astype(str)
    else:
        # Use image file structure for metadata extraction
        return samples_df.drop(columns=["sample_fp"]).drop_duplicates().astype(str)


def get_output_pattern(wildcard_combos: pd.DataFrame) -> Dict[str, str]:
    """Get output pattern from wildcard combinations.

    Args:
        wildcard_combos (pd.DataFrame): DataFrame with wildcard combinations

    Returns:
        Dict[str, str]: Dictionary mapping column names to wildcard patterns
    """
    if len(wildcard_combos) == 0:
        return {"plate": "{plate}"}  # Fallback
    return {col: f"{{{col}}}" for col in wildcard_combos.columns}


def get_inputs_for_metadata_extraction(
    image_type: str,
    config: Dict[str, Any],
    samples_df: pd.DataFrame,
    metadata_samples_df: pd.DataFrame,
    wildcards,
) -> Dict[str, List[str]]:
    """Get appropriate inputs for metadata extraction rules.

    Args:
        image_type (str): 'sbs' or 'phenotype'
        config (Dict[str, Any]): Configuration dictionary
        samples_df (pd.DataFrame): DataFrame with image file paths
        metadata_samples_df (pd.DataFrame): DataFrame with metadata file paths
        wildcards: Snakemake wildcards object

    Returns:
        Dict[str, List[str]]: Dictionary with 'samples' and 'metadata' keys containing file lists
    """
    from lib.preprocess.preprocess import get_data_config

    data_config = get_data_config(image_type, config)

    # Build filter arguments from available wildcards
    filter_args = {}
    for attr in ["plate", "well", "tile", "cycle", "round"]:
        if hasattr(wildcards, attr):
            filter_args[attr] = getattr(wildcards, attr)

    inputs = {"samples": [], "metadata": []}

    if not metadata_samples_df.empty:
        # Use external metadata files - no need for sample files
        metadata_filter_args = {
            k: v for k, v in filter_args.items() if k in metadata_samples_df.columns
        }
        inputs["metadata"] = [
            get_sample_fps(metadata_samples_df, **metadata_filter_args)
        ]
    else:
        # Extract from image files - no metadata files needed
        sample_filter_args = filter_args.copy()
        if data_config["image_data_organization"] == "well":
            # Remove tile for well organization
            sample_filter_args.pop("tile", None)

        # Add channel order if specified
        if f"{image_type}_channel_order" in config.get("preprocess", {}):
            sample_filter_args["channel_order"] = config["preprocess"][
                f"{image_type}_channel_order"
            ]

        inputs["samples"] = get_sample_fps(samples_df, **sample_filter_args)
        if isinstance(inputs["samples"], str):
            inputs["samples"] = [inputs["samples"]]

    return inputs


def get_tile_count_from_well(
    samples_df: pd.DataFrame,
    plate: Union[int, str] = None,
    well: Union[int, str] = None,
    cycle: Union[int, str] = None,
    round_order: List[Union[int, str]] = None,
    channel_order: List[Union[int, str]] = None,
    verbose: bool = False,
) -> int:
    """Get the number of tiles in a well-based ND2 file.

    Args:
        samples_df (pd.DataFrame): DataFrame containing sample data
        plate (Union[int, str], optional): Plate number to filter by. Defaults to None.
        well (Union[int, str], optional): Well identifier to filter by. Defaults to None.
        cycle (Union[int, str], optional): Cycle number to filter by (for SBS). Defaults to None.
        round_order (List[Union[int, str]], optional): Round order to filter by (for phenotype). Defaults to None.
        channel_order (List[Union[int, str]], optional): Channel order to use. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        int: Number of tiles in the well
    """
    # Get a sample file
    sample_file = get_sample_fps(
        samples_df,
        plate=plate,
        well=well,
        cycle=cycle,
        round_order=round_order,
        channel_order=channel_order,
        verbose=verbose,
    )

    # Import here to avoid circular imports
    from lib.preprocess.preprocess import convert_nd2_to_array_well

    # Get tile count from the ND2 file
    _, tile_count = convert_nd2_to_array_well(
        sample_file, position=0, return_tiles=True, verbose=verbose
    )

    return tile_count
