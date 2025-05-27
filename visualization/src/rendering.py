import os
import pandas as pd
import streamlit as st
import sys
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from workflow.lib.shared.file_utils import parse_filename
from src.config import STATIC_ASSET_URL_ROOT, STATIC_ASSET_PATH


class VisualizationRenderer:
    @staticmethod
    def display_plots_and_tables(filtered_df, root_dir):
        # Check if the root directory exists
        if not os.path.exists(root_dir):
            st.error(f"Analysis root directory does not exist: {root_dir}")
            return

        if filtered_df.empty:
            st.warning("No data found matching the selected filters.")
            return

        # Group by directory and basename
        grouped = filtered_df.groupby(["dir", "basename"])
        # Iterate through each group
        for (dir_name, base_name), group_df in grouped:
            with st.container():
                attrs, metric_name, _ = parse_filename(base_name)
                metric_title = metric_name.replace("_", " ").title()
                attr_parts = [
                    f"{k.replace('_', ' ').title()}: {v}" for k, v in attrs.items()
                ]
                title = f"{metric_title} - " + ", ".join(attr_parts)
                st.markdown(f"### {title}")

                # Count only the items we'll actually display
                display_items = []
                for _, row in group_df.iterrows():
                    has_png = any(r["ext"] == "png" for _, r in group_df.iterrows())
                    if row["ext"] == "png" or (row["ext"] == "tsv" and not has_png):
                        display_items.append(row)

                # Create columns based on actual display items
                cols = st.columns(min(3, len(display_items)))

                for idx, row in enumerate(display_items):
                    col_idx = idx % len(cols)
                    with cols[col_idx]:
                        # Check if this group has both PNG and TSV
                        has_png = any(r["ext"] == "png" for _, r in group_df.iterrows())
                        has_tsv = any(r["ext"] == "tsv" for _, r in group_df.iterrows())

                        if row["ext"] == "png":
                            # Always show PNG if it exists
                            try:
                                st.image(
                                    os.path.join(root_dir, row["file_path"]),
                                    caption=f"{row['metric_name']} - {row['well_id']}",
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
                                    relative_path = tsv_path.replace(STATIC_ASSET_PATH, "")
                                    static_url = f"{STATIC_ASSET_URL_ROOT}{relative_path}"
                                    st.markdown(f"[Download TSV data]({static_url})")
                                else:
                                    # Fall back to direct download when running locally
                                    with open(tsv_path, "rb") as f:
                                        st.download_button(
                                            label="Download TSV data",
                                            data=f,
                                            file_name=os.path.basename(tsv_path),
                                            key=f"download_{str(uuid.uuid4())}",
                                        )

                        elif row["ext"] == "tsv" and not has_png:
                            # Only show TSV if there's no PNG
                            try:
                                tsv_data = pd.read_csv(
                                    os.path.join(root_dir, row["file_path"]), sep="\t"
                                )
                                st.dataframe(tsv_data)
                            except Exception as e:
                                st.error(f"Error reading TSV file: {e}")

                st.markdown("---")  # Add a separator between groups
