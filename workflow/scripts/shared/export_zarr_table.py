import pandas as pd
from lib.io.omezarr_writer import write_table_zarr

# Read input table
# Can be parquet, tsv, csv
input_path = str(snakemake.input[0])
if input_path.endswith(".parquet"):
    df = pd.read_parquet(input_path)
elif input_path.endswith(".tsv"):
    df = pd.read_csv(input_path, sep="\t")
else:
    df = pd.read_csv(input_path)

# Write to Zarr
write_table_zarr(
    df=df,
    out_path=str(snakemake.output[0])
)

