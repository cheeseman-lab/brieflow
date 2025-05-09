import os
import glob

import streamlit as st
from src.filesystem import FileSystem
from src.rendering import VisualizationRenderer
from src.filtering import create_filter_radio, apply_filter
from src.config import BRIEFLOW_OUTPUT_PATH

st.set_page_config(
    page_title="Quality Control - Brieflow Analysis",
    page_icon=":microscope:",
    layout="wide",
)

def find_eval_files(root_dir):
    png_files = glob.glob(os.path.join(root_dir, "*", "eval", "*.png"))
    tsv_files = glob.glob(os.path.join(root_dir, "*", "eval", "*.tsv"))
    return png_files + tsv_files



@st.cache_data
def load_data(root_dir):
    global filtered_df
    files = find_eval_files(root_dir)
    filtered_df = FileSystem.extract_features(root_dir, files)
    return filtered_df


# Create filters using the helper function
def apply_all_filters(df, sidebar):
    """Apply all filters in sequence and return the filtered dataframe."""
    filters = [
        ("dir_level_0", "Phase"),
        # Intentionally omitting dir_level_1
        ("dir_level_2", "Subgroup"),
        ("plate_id", "Plate"),
        ("well_id", "Well"),
        ("metric_name", "Metric"),
    ]

    filtered_df = df.copy()
    selected_values = {}

    for column, label in filters:
        selected_value = create_filter_radio(filtered_df, column, sidebar, label)
        filtered_df = apply_filter(filtered_df, column, selected_value)
        selected_values[column] = selected_value

    return filtered_df, selected_values


st.title("Quality Control")
st.markdown("Review the quality control metrics from the brieflow pipeline")

# Load the data
filtered_df = load_data(BRIEFLOW_OUTPUT_PATH)

st.sidebar.title("Filters")
filtered_df, selected_values = apply_all_filters(filtered_df, st.sidebar)

VisualizationRenderer.display_plots_and_tables(filtered_df, BRIEFLOW_OUTPUT_PATH)
