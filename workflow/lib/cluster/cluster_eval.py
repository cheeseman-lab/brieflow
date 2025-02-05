from pathlib import Path

import pandas as pd

from lib.shared.file_utils import parse_filename


def aggregate_global_metrics(global_metrics_files: list) -> pd.DataFrame:
    """Aggregate global metrics from multiple files into a unified DataFrame.

    Args:
        global_metrics_files (list): List of file paths to global metrics TSV files.

    Returns:
        pd.DataFrame: A DataFrame with one row per file, containing columns for dataset,
                      channel combo, and each metric.
    """
    combined_data = []

    for global_metrics_fp in global_metrics_files:
        # Extract dataset and channel combo
        metadata = parse_filename(global_metrics_fp)[0]
        dataset = metadata["dataset"]
        channel_combo = Path(global_metrics_fp).parent.parent.name

        # Read metrics and convert to a dictionary
        metrics = (
            pd.read_csv(global_metrics_fp, sep="\t")
            .set_index("metric")["value"]
            .to_dict()
        )

        # Append row data
        combined_data.append(
            {"dataset": dataset, "channel_combo": channel_combo, **metrics}
        )

    # Create the final DataFrame
    return pd.DataFrame(combined_data)
