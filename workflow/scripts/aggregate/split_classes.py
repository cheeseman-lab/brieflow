import pyarrow.dataset as ds
from lib.aggregate.cell_classification import CellClassifier

# ADD ANY IMPORTS NECESSARY FOR CLASSIFIER
from sklearn.preprocessing import RobustScaler
import numpy as np

# Load classifier
classifier = CellClassifier.load(snakemake.params.classifier_path)

# Load merge data using PyArrow dataset
print("Loading cell data")
cell_data = ds.dataset(snakemake.input[0], format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()
print("Loaded cell data")

if snakemake.params.cell_class == "all":
    cell_data.to_parquet(snakemake.output[0], index=False)
else:
    cell_data = classifier.classify_cells(cell_data, snakemake.params.first_feature)
    cell_data = cell_data[cell_data["class"] == snakemake.params.cell_class]
    cell_data.to_parquet(snakemake.output[0], index=False)
