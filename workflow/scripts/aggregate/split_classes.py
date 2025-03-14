import pandas as pd
from lib.aggregate.cell_classification import CellClassifier

# ADD ANY IMPORTS NECESSARY FOR CLASSIFIER
from sklearn.preprocessing import RobustScaler
import numpy as np

# Load classifier
classifier = CellClassifier.load(snakemake.params.classifier_path)

# Load merge data
import pyarrow.dataset as ds

# Load merge data using PyArrow dataset
print("Loading merge data")
merge_data = ds.dataset(snakemake.input.merge_data_paths, format="parquet")

# You can control parallelism with the num_threads parameter
merge_data = merge_data.to_table(use_threads=True, memory_pool=None).to_pandas()
print(merge_data.head())
print(merge_data.shape)

if snakemake.params.dataset == "all":
    merge_data.to_parquet(snakemake.output[0])
else:
    