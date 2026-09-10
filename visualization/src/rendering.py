import os
import streamlit as st
import sys
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from workflow.lib.shared.file_utils import parse_filename
from src.config import STATIC_ASSET_URL_ROOT, STATIC_ASSET_PATH
from src.filesystem import parse_nested_location, read_table

# Plots per row before wrapping, so a phase with many wells stays a grid
COLUMNS_PER_ROW = 3


class VisualizationRenderer:
    @staticmethod
    def display_plots_and_tables(filtered_df, root_dir):
        # Check if the root directory exists
        if not os.path.exists(root_dir):
            st.error(f"Analysis root directory does not exist: {root_dir}")
            return

        if filtered_df.empty:
            st.info("No quality control files match the selected filters.")
            return

        # Group by directory and basename
        grouped = filtered_df.groupby(["dir", "basename"])
        # Iterate through each group
        for group_index, ((dir_name, base_name), group_df) in enumerate(grouped):
            with st.container():
                attrs, metric_name, _ = parse_filename(base_name)
                # Zarr mode encodes the location in the directories, not the filename.
                if not attrs:
                    attrs = parse_nested_location(
                        os.path.join(dir_name, os.path.basename(base_name))
                    )
                metric_title = metric_name.replace("_", " ").title()
                attr_parts = [
                    f"{k.replace('_', ' ').title()}: {v}" for k, v in attrs.items()
                ]
                if group_index:
                    st.divider()
                st.subheader(metric_title)
                if attr_parts:
                    st.caption(" · ".join(attr_parts))

                # Count only the items we'll actually display
                has_png = any(r["ext"] == "png" for _, r in group_df.iterrows())
                display_items = [
                    row
                    for _, row in group_df.iterrows()
                    if row["ext"] == "png" or (row["ext"] == "tsv" and not has_png)
                ]
                if not display_items:
                    continue

                # Create columns based on actual display items
                cols = st.columns(min(COLUMNS_PER_ROW, len(display_items)))

                for idx, row in enumerate(display_items):
                    col_idx = idx % len(cols)
                    with cols[col_idx]:
                        # Check if this group has both PNG and TSV
                        has_tsv = any(r["ext"] == "tsv" for _, r in group_df.iterrows())

                        if row["ext"] == "png":
                            # Always show PNG if it exists
                            try:
                                st.image(
                                    os.path.join(root_dir, row["file_path"]),
                                    caption=f"{row['metric_name']} — well {row['well_id']}",
                                    use_container_width=True,
                                )
                            except Exception as e:
                                st.error(f"Could not load image: {row['file_path']}")
                                st.error(str(e))

                            # If there's a corresponding TSV, add download link
                            if has_tsv:
                                tsv_row = group_df[group_df["ext"] == "tsv"].iloc[0]
                                tsv_path = os.path.join(root_dir, tsv_row["file_path"])
                                if STATIC_ASSET_URL_ROOT and STATIC_ASSET_PATH:
                                    # Use nginx-served static files when configured
                                    relative_path = tsv_path.replace(
                                        STATIC_ASSET_PATH, ""
                                    )
                                    static_url = (
                                        f"{STATIC_ASSET_URL_ROOT}{relative_path}"
                                    )
                                    st.markdown(f"[Download TSV data]({static_url})")
                                else:
                                    # Fall back to direct download when running locally
                                    with open(tsv_path, "rb") as f:
                                        st.download_button(
                                            label="Download TSV data",
                                            data=f,
                                            file_name=os.path.basename(tsv_path),
                                            key=f"download_{str(uuid.uuid4())}",
                                            use_container_width=True,
                                        )

                        elif row["ext"] == "tsv" and not has_png:
                            # Only show TSV if there's no PNG
                            try:
                                tsv_data = read_table(
                                    os.path.join(root_dir, row["file_path"])
                                )
                                st.dataframe(
                                    tsv_data,
                                    hide_index=True,
                                    use_container_width=True,
                                )
                            except Exception as e:
                                st.error(f"Error reading TSV file: {e}")
