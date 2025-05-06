import streamlit as st
import pandas as pd
import os
import uuid
import yaml
from src.filesystem import FileSystem
from src.rendering import VisualizationRenderer
from src.filtering import create_filter_radio, apply_filter

st.set_page_config(
    page_title="Quality Control - Brieflow Analysis",
    page_icon=":microscope:",
    layout="wide",
)

st.title("Quality Control")
st.markdown("Review the quality control metrics from the brieflow pipeline")

@st.cache_data
def load_data(root_dir):
    global filtered_df
    # Apply filters directly during file discovery for better performance
    # Filter for dir_level_0 in ['phenotype', 'merge', 'sbs', 'aggregate'] and dir_level_1 == 'eval'
    files = FileSystem.find_files(
        root_dir, 
        include_any=['phenotype', 'merge', 'sbs', 'aggregate'],
        include_all=['eval'],
        extensions=['png', 'tsv']
    )
    filtered_df = FileSystem.extract_features(root_dir, files)
    return filtered_df




# Configuration
CONFIG_PATH = os.getenv('BRIEFLOW_CONFIG_PATH', '../../analysis/config/config.yml')
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Extract the analysis_root from the config
# repo root is two dirs up from the config file.
analysis_repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(CONFIG_PATH)), '..', 'analysis'))
analysis_root = config['all']['root_fp']
ANALYSIS_ROOT_DIR = os.path.abspath(os.path.join(analysis_repo_root, analysis_root))
st.write(f"Analysis root dir: {ANALYSIS_ROOT_DIR}")

# Load the data
filtered_df = load_data(ANALYSIS_ROOT_DIR)

st.sidebar.title("Filters")

# Create filters using the helper function
selected_dir_level_0 = create_filter_radio(filtered_df, 'dir_level_0', st.sidebar, "Phase")
filtered_df = apply_filter(filtered_df, 'dir_level_0', selected_dir_level_0)

# dir_level_1 is always "eval", so we don't need to create a filter

selected_dir_level_2 = create_filter_radio(filtered_df, 'dir_level_2', st.sidebar, "Subgroup")
filtered_df = apply_filter(filtered_df, 'dir_level_2', selected_dir_level_2)

selected_plate = create_filter_radio(filtered_df, 'plate_id', st.sidebar, "Plate")
filtered_df = apply_filter(filtered_df, 'plate_id', selected_plate)

selected_well = create_filter_radio(filtered_df, 'well_id', st.sidebar, "Well")
filtered_df = apply_filter(filtered_df, 'well_id', selected_well)

selected_metric = create_filter_radio(filtered_df, 'metric_name', st.sidebar, "Metric")
filtered_df = apply_filter(filtered_df, 'metric_name', selected_metric)

# Display the filtered dataframe (debug)
# st.dataframe(filtered_df)

VisualizationRenderer.display_plots_and_tables(filtered_df, ANALYSIS_ROOT_DIR)
