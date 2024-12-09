import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path


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
        return df
    except pd.errors.EmptyDataError:
        pass


# Get input, output, and threads from Snakemake
input_files = snakemake.input
output_file = snakemake.output[0]
output_type = getattr(snakemake.params, "output_type", "tsv")
threads = snakemake.threads

# Load, concatenate, and save the data
arr_reads = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in input_files)
df_reads = pd.concat(arr_reads)

# Save the data based on output_type
if output_type == "hdf":
    df_reads.to_hdf(output_file, "x", mode="w")
elif output_type == "tsv":
    df_reads.to_csv(output_file, sep="\t", index=False)
else:
    raise ValueError(f"Unsupported output type: {output_type}")