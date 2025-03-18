from tifffile import imread
import pandas as pd

from lib.phenotype.extract_phenotype_vacuoles import extract_phenotype_vacuoles

# Load inputs
data_phenotype = imread(snakemake.input[0])
vacuole_masks = imread(snakemake.input[1])

# Load cell vacuole table and separate the combined dataframes
combined_df = pd.read_csv(snakemake.input[2], sep="\t")

# Extract the two dataframes based on table_type column
vacuole_cell_mapping_df = combined_df[combined_df['table_type'] == 'vacuole_cell_mapping'].copy()
cell_summary_df = combined_df[combined_df['table_type'] == 'cell_summary'].copy()

# Remove the 'table_type' column and rename the columns back to original
# Strip the prefixes from column names
vacuole_cell_mapping_cols = {col: col.replace('vacuole_mapping_', '') 
                            for col in vacuole_cell_mapping_df.columns 
                            if col.startswith('vacuole_mapping_')}
cell_summary_cols = {col: col.replace('cell_summary_', '') 
                    for col in cell_summary_df.columns 
                    if col.startswith('cell_summary_')}

vacuole_cell_mapping_df = vacuole_cell_mapping_df.rename(columns=vacuole_cell_mapping_cols).drop(columns=['table_type'])
cell_summary_df = cell_summary_df.rename(columns=cell_summary_cols).drop(columns=['table_type'])

# Recreate the original dictionary structure
cell_vacuole_table = {
    'vacuole_cell_mapping': vacuole_cell_mapping_df,
    'cell_summary': cell_summary_df
}

# Extract vacuole phenotype features
vacuole_phenotype = extract_phenotype_vacuoles(
    data_phenotype=data_phenotype,
    vacuoles=vacuole_masks,
    vacuole_cell_mapping_df=cell_vacuole_table['vacuole_cell_mapping'],
    wildcards=snakemake.wildcards,
    foci_channel=snakemake.params.foci_channel,
    channel_names=snakemake.params.channel_names
)

# Save results
vacuole_phenotype.to_csv(snakemake.output[0], sep="\t", index=False)