from pathlib import Path

# from lib.sbs_process import eval_segmentation
from lib.shared.file_utils import parse_filename


def segementation_overview(segmentation_stats_paths):
    for segmentation_stats_path in segmentation_stats_paths:
        segmentation_filename = Path(segmentation_stats_path).name
        print(segmentation_filename)
        data_location, _, _ = parse_filename(segmentation_filename)
        print(data_location)


segementation_overview(snakemake.input.segmentation_stats_paths)
