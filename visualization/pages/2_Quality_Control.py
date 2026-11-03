import os
import glob

import streamlit as st
from src.filesystem import FileSystem
from src.rendering import VisualizationRenderer
from src.filtering import create_filter_radio, apply_filter
from src.config import BRIEFLOW_OUTPUT_PATH
from src.theme import empty_state, page_setup, sidebar_filters_header

page_setup(
    "Quality Control",
    "🔬",
    "Evaluation plots and tables each pipeline module writes to its `eval/` directory.",
)


def find_eval_files(root_dir):
    pattern = os.path.join(root_dir, "*", "eval", "**", "*")
    all_files = glob.glob(pattern, recursive=True)
    return [f for f in all_files if f.endswith(".png") or f.endswith(".tsv")]


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
        # Create a unique key for each filter based on the column name
        key = f"filter_{column}"
        selected_value = create_filter_radio(
            filtered_df, column, sidebar, label, key=key
        )
        filtered_df = apply_filter(filtered_df, column, selected_value)
        selected_values[column] = selected_value

    return filtered_df, selected_values


# Load the data
eval_df = load_data(BRIEFLOW_OUTPUT_PATH)

if eval_df.empty:
    empty_state(
        "No quality control files found.",
        "Each module writes its plots and tables to `<module>/eval/` as it completes.",
    )
    st.stop()

sidebar_filters_header("Narrow the plots shown on the right.")
filtered_df, selected_values = apply_all_filters(eval_df, st.sidebar)

metric_col, plate_col, well_col = st.columns(3)
metric_col.metric("Metrics shown", f"{filtered_df['metric_name'].nunique():,}")
plate_col.metric("Plates", f"{filtered_df['plate_id'].dropna().nunique():,}")
well_col.metric("Wells", f"{filtered_df['well_id'].dropna().nunique():,}")
st.divider()

VisualizationRenderer.display_plots_and_tables(filtered_df, BRIEFLOW_OUTPUT_PATH)
