"""Utility functions for handling files during preprocessing."""

import pandas as pd
from typing import List, Union


def get_sample_fps(
    samples_df: pd.DataFrame, 
    well: str = None, 
    tile: int = None, 
    cycle: int = None,
    channel_order: List[str] = ["DAPI", "GFP", "CY3", "CY5", "AF750"]  # Define default order
) -> Union[str, List[str]]:
    """Filters the samples DataFrame and ensures consistent channel order.

    Args:
        samples_df (pd.DataFrame): DataFrame containing sample data.
        well (str, optional): Well identifier to filter by.
        tile (int, optional): Tile number to filter by.
        cycle (int, optional): Cycle number to filter by.
        channel_order (List[str], optional): Order of channels. Defaults to ["DAPI", "GFP", "CY3", "CY5", "AF750"].

    Returns:
        Union[str, List[str]]: Either a single filepath or ordered list of filepaths
    """
    filtered_df = samples_df

    if well is not None:
        filtered_df = filtered_df[filtered_df["well"] == well]

    if tile is not None:
        filtered_df = filtered_df[filtered_df["tile"] == int(tile)]

    if cycle is not None:
        filtered_df = filtered_df[filtered_df["cycle"] == int(cycle)]
    
    # If we have a channel column, return list of files in specified order
    if "channel" in filtered_df.columns:
        # Create dictionary mapping channel to file path
        channel_to_file = dict(zip(filtered_df["channel"], filtered_df["sample_fp"]))
        
        # Return files in specified order
        return [channel_to_file[channel] for channel in channel_order]
    
    # Otherwise return single file path (original behavior)
    return filtered_df["sample_fp"].iloc[0]