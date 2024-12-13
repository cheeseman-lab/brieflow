import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from lib.shared.eval_segmentation import evaluate_segmentation_paramsearch


def get_file(filepath):
    """Read TSV and add parameters from filename."""
    try:
        # Extract parameters from filename
        path = Path(filepath)
        prefix, params_part = path.stem.split('__')
        
        # Extract well and tile from prefix
        prefix_parts = prefix.split('_')
        well = next(p.replace('W', '') for p in prefix_parts if p.startswith('W'))
        tile = next(p.replace('T', '') for p in prefix_parts if p.startswith('T'))
        
        # Extract other parameters from the second part
        params = params_part.split('_')
        nuclei_diameter = float(next(p.replace('nd', '') for p in params if p.startswith('nd')))
        cell_diameter = float(next(p.replace('cd', '') for p in params if p.startswith('cd')))
        flow_threshold = float(next(p.replace('ft', '') for p in params if p.startswith('ft')))
        cellprob_threshold = float(next(p.replace('cp', '') for p in params if p.startswith('cp')))
        
        # Read the data
        df = pd.read_csv(filepath, sep="\t")
        
        # Add parameter columns before the rest of the columns
        df.insert(0, 'well', well)
        df.insert(1, 'tile', tile)
        df.insert(2, 'nuclei_diameter', nuclei_diameter)
        df.insert(3, 'cell_diameter', cell_diameter)
        df.insert(4, 'flow_threshold', flow_threshold)
        df.insert(5, 'cellprob_threshold', cellprob_threshold)
        df['path'] = path

        return df
    except pd.errors.EmptyDataError:
        pass


# Get input, output, and parameters from Snakemake
input_files = snakemake.input
output_full = snakemake.output[0]
output_grouped = snakemake.output[1]
output_summary = snakemake.output[2]
output_panel = snakemake.output[3]
output_type = getattr(snakemake.params, "output_type", "tsv")
segmentation_process = snakemake.params.segmentation_process
default_cell_diameter = snakemake.params.cell_diameter
default_nuclei_diameter = snakemake.params.nuclei_diameter
default_cellprob_threshold = snakemake.params.cellprob_threshold
default_flow_threshold = snakemake.params.flow_threshold
threads = snakemake.threads
# Set process-specific parameters
if segmentation_process == "sbs_process":
    prepare_cellpose_kwargs = {
        'dapi_index': snakemake.params.dapi_index,
        'cyto_index': snakemake.params.cyto_index,
    }
    channel_cmaps = None
elif segmentation_process == "phenotype_process":
    channel_cmaps = snakemake.params.channel_cmaps
    prepare_cellpose_kwargs = None
else:
    raise ValueError(f"Unsupported segmentation process: {segmentation_process}")

# Load and concatenate the data
arr_reads = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in input_files)
df_reads = pd.concat(arr_reads)

# Evaluate segmentation parameters
grouped_stats, summary_text, seg_panel = evaluate_segmentation_paramsearch(
    df_reads,
    segmentation_process=segmentation_process,
    default_cell_diameter=default_cell_diameter,
    default_nuclei_diameter=default_nuclei_diameter,
    default_cellprob_threshold=default_cellprob_threshold,
    default_flow_threshold=default_flow_threshold,
    channel_cmaps=channel_cmaps,
    prepare_cellpose_kwargs=prepare_cellpose_kwargs
)

# Save the concatenated data based on output_type
if output_type == "hdf":
    df_reads.to_hdf(output_full, "x", mode="w")
elif output_type == "tsv":
    df_reads.to_csv(output_full, sep="\t", index=False)
else:
    raise ValueError(f"Unsupported output type: {output_type}")

# Save the parameter evaluation outputs
grouped_stats.to_csv(output_grouped, sep='\t')
with open(output_summary, 'w') as f:
    f.write(summary_text)

# Save the visualization panel if it was created
if seg_panel is not None:
    seg_panel.savefig(output_panel)