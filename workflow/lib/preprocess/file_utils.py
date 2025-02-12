"""Utility functions for handling files during preprocessing."""

import pandas as pd
from typing import List, Union


def get_sample_fps(
    samples_df: pd.DataFrame,
    plate: int = None,
    well: str = None,
    tile: int = None,
    cycle: int = None,
    channel: str = None,
    round_order: List[int] = None,
    channel_order: List[str] = None,
) -> Union[str, List[str]]:
    """Filters the samples DataFrame and ensures consistent channel and round order.

    Args:
        samples_df (pd.DataFrame): DataFrame containing sample data.
        plate (int, optional): Plate number to filter by. Defaults to None.
        well (str, optional): Well identifier to filter by. Defaults to None.
        tile (int, optional): Tile number to filter by. Defaults to None.
        cycle (int, optional): Cycle number to filter by. Defaults to None.
        channel (str, optional): Channel to filter by. Defaults to None.
        round_order (List[int], optional): Order of rounds to return. Defaults to None.
        channel_order (List[str], optional): Order of channels. Defaults to None.

    Returns:
        Union[str, List[str]]: Either a single filepath or ordered list of filepaths
    """
    filtered_df = samples_df

    if plate is not None:
        filtered_df = filtered_df[filtered_df["plate"] == int(plate)]

    if well is not None:
        filtered_df = filtered_df[filtered_df["well"] == well]

    if tile is not None:
        filtered_df = filtered_df[filtered_df["tile"] == int(tile)]

    if cycle is not None:
        filtered_df = filtered_df[filtered_df["cycle"] == int(cycle)]

    if channel is not None:
        filtered_df = filtered_df[filtered_df["channel"] == channel]

    if round_order is not None:
        # Filter to only include specified rounds
        filtered_df = filtered_df[filtered_df["round"].isin(round_order)]
        # Create dictionary mapping round to DataFrame rows
        round_groups = {
            round_num: group for round_num, group in filtered_df.groupby("round")
        }

        # Initialize list to store file paths for each round
        round_files = []

        # Process each round in the specified order
        for round_num in round_order:
            if round_num not in round_groups:
                raise ValueError(f"Round {round_num} not found in data")

            round_df = round_groups[round_num]

            # If we have channels and channel order is specified
            if "channel" in round_df.columns and channel_order is not None:
                channel_to_file = dict(zip(round_df["channel"], round_df["sample_fp"]))
                round_files.extend(
                    [channel_to_file[channel] for channel in channel_order]
                )
            else:
                round_files.append(round_df["sample_fp"].iloc[0])

        return round_files

    # If no rounds specified but we have channels and channel order
    if "channel" in filtered_df.columns and channel_order is not None:
        channel_to_file = dict(zip(filtered_df["channel"], filtered_df["sample_fp"]))
        return [channel_to_file[channel] for channel in channel_order]

    # Otherwise return single file path
    return filtered_df["sample_fp"].iloc[0]
