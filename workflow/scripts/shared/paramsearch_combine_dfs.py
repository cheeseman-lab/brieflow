import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path


def get_file(filepath):
    """Read TSV and add parameters from filename."""
    try:
        # Extract parameters from filename
        path = Path(filepath)
        params = path.stem.split('__')[1].split('_')
        
        # Extract nuclei and cell diameters using nd and cd prefixes
        nuclei_diameter = float([p for p in params if 'nd' in p][0].replace('nd', ''))
        cell_diameter = float([p for p in params if 'cd' in p][0].replace('cd', ''))
        flow_threshold = float([p for p in params if 'ft' in p][0].replace('ft', ''))
        cellprob_threshold = float([p for p in params if 'cp' in p][0].replace('cp', ''))
        
        # Read the data
        df = pd.read_csv(filepath, sep="\t")
        
        # Add parameter columns before the rest of the columns
        df.insert(0, 'nuclei_diameter', nuclei_diameter)
        df.insert(1, 'cell_diameter', cell_diameter)
        df.insert(2, 'flow_threshold', flow_threshold)
        df.insert(3, 'cellprob_threshold', cellprob_threshold)                
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