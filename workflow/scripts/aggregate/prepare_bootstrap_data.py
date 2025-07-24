from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from lib.aggregate.bootstrap import parse_gene_construct_mapping
from lib.shared.file_utils import get_filename

# Create output directory
output_dir = Path(snakemake.output[0])
output_dir.mkdir(parents=True, exist_ok=True)

# Load construct features array to discover available constructs
print("Loading construct features array...")
construct_features_arr = np.load(snakemake.input[1], allow_pickle=True)

# Extract construct IDs (first column)
construct_ids = np.unique(construct_features_arr[:, 0]).tolist()
print(f"Found {len(construct_ids)} unique constructs")

# Parse gene-construct mapping
gene_construct_mapping = parse_gene_construct_mapping(construct_ids)
print(f"Found {len(gene_construct_mapping)} genes")

# Create construct data files for each construct
print(f"Saving {len(construct_ids)} construct data files to {output_dir}")
print(f"Using {multiprocessing.cpu_count()} CPUs")

def write_construct_data(construct_id):
    """Write construct data file for a single construct."""
    print(f"Processing construct: {construct_id}")
    
    # Create a simple metadata file for the construct
    # This will be used by the bootstrap_construct rule to identify the construct
    construct_data = pd.DataFrame({
        'construct_id': [construct_id],
        'gene': [construct_id.split('.')[0] if '.' in construct_id else construct_id]
    })
    
    # Save construct data file
    output_file = output_dir / get_filename(
        {"construct": construct_id}, 
        "construct_data", 
        "csv"
    )
    construct_data.to_csv(output_file, index=False)

# Process all constructs in parallel
with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    executor.map(write_construct_data, construct_ids)

print(f"Bootstrap data preparation complete! Created files for {len(construct_ids)} constructs")