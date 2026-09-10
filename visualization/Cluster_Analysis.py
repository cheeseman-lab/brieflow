import streamlit as st
import uuid

st.set_page_config(
    page_title="Cluster Analysis - Brieflow Analysis",
    layout="wide",
)

import anndata as ad
import pandas as pd
import glob
import os
import json
import sys

import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np

from src.config import get_image_format, load_config
from src.filesystem import FileSystem, read_zarr_channel_names
from src.filtering import create_filter_radio, apply_filter
from src.config import BRIEFLOW_OUTPUT_PATH, STATIC_ASSET_URL_ROOT, STATIC_ASSET_PATH

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from workflow.lib.shared.image_io import read_image

# =====================
# CONSTANTS
CLUSTER_ROOT = os.path.join(BRIEFLOW_OUTPUT_PATH, "cluster")
IMAGE_FORMAT = get_image_format()

# One h5ad per channel_combo/[compartment_combo/]cell_class carries every resolution,
# so name the levels by how many there are rather than by fixed position.
CLUSTER_DIR_LEVELS = {
    2: ["channel_combo", "cell_class"],
    3: ["channel_combo", "compartment_combo", "cell_class"],
}

# obs holds one cluster assignment column per leiden resolution
CLUSTER_GROUP_PREFIX = "cluster_group_"

# obs columns the page already carries as a filter, so they stay out of the gene table
OBS_EXCLUDED_COLUMNS = ["cell_cycle_phase"]

# columns the page adds for its own bookkeeping, dropped before the gene table is shown
BOOKKEEPING_COLUMNS = [
    "source",
    "source_h5ad_path",
    "cluster_dir",
    "channel_combo",
    "compartment_combo",
    "cell_class",
    "leiden_resolution",
]

# bootstrap significance layers, most directly usable first
SIGNIFICANCE_LAYERS = ("neg_log10_fdr", "fdr")

# Common hover data columns
HOVER_COLUMNS = ["gene_symbol_0", "cluster", "cell_count", "source"]

# Indices for accessing customdata array
GENE_SYMBOL_INDEX = 0
CLUSTER_INDEX = 1
CELL_COUNT_INDEX = 2
SOURCE_INDEX = 3

# =====================
# FUNCTIONS


def find_mozzarellm_dirs(channel_combo: str) -> list:
    """Find every mozzarellm/clusters directory under a channel combo.

    Globs rather than walking fixed levels because the cluster tree gains a
    compartment_combo level when the run defines compartments.
    """
    return sorted(
        d
        for d in glob.glob(
            os.path.join(CLUSTER_ROOT, channel_combo, "**", "mozzarellm", "clusters"),
            recursive=True,
        )
        if os.path.isdir(d)
    )


def parse_mozzarellm_dir(mozzarellm_dir: str) -> tuple:
    """Return the (cell_class, leiden_resolution) a mozzarellm/clusters directory belongs to."""
    leiden_dir = os.path.dirname(os.path.dirname(mozzarellm_dir))
    return os.path.basename(os.path.dirname(leiden_dir)), os.path.basename(leiden_dir)


def has_mozzarellm_analysis(channel_combo: str) -> bool:
    """Check if a channel combo has mozzarellm analysis for any cell_class/leiden_resolution."""
    return len(find_mozzarellm_dirs(channel_combo)) > 0


def has_mozzarellm_for_cell_class(channel_combo: str, cell_class: str) -> bool:
    """Check if mozzarellm exists for channel_combo + cell_class + any leiden_resolution."""
    return any(
        parse_mozzarellm_dir(d)[0] == cell_class
        for d in find_mozzarellm_dirs(channel_combo)
    )


def has_mozzarellm_for_leiden(channel_combo: str, cell_class: str, leiden_res) -> bool:
    """Check if mozzarellm exists for the exact channel_combo + cell_class + leiden_resolution."""
    # Convert to int then string to handle float values like 15.0 -> "15"
    leiden_str = str(int(float(leiden_res)))
    return (cell_class, leiden_str) in [
        parse_mozzarellm_dir(d) for d in find_mozzarellm_dirs(channel_combo)
    ]


# -- Data Load Methods --
@st.cache_data
def find_cluster_h5ads() -> list:
    """List every cluster h5ad written by ``rule format_cluster_anndata``."""
    return sorted(
        glob.glob(os.path.join(CLUSTER_ROOT, "**", "h5ad", "*.h5ad"), recursive=True)
    )


@st.cache_resource
def load_cluster_h5ad(h5ad_path: str):
    """Read one cluster h5ad, the single source for clustering, features and gene metadata."""
    return ad.read_h5ad(h5ad_path)


def cluster_base_dir(h5ad_path: str) -> str:
    """Return the cell class directory that holds a cluster h5ad's ``h5ad/`` subdirectory."""
    return os.path.dirname(os.path.dirname(h5ad_path))


def parse_cluster_levels(h5ad_path: str) -> dict:
    """Return the channel_combo / compartment_combo / cell_class a cluster h5ad belongs to.

    The compartment_combo level is only present when the run splits by compartment.
    """
    parts = os.path.relpath(cluster_base_dir(h5ad_path), CLUSTER_ROOT).split(os.sep)
    level_names = CLUSTER_DIR_LEVELS.get(
        len(parts), [f"dir_level_{i}" for i in range(len(parts))]
    )
    return dict(zip(level_names, parts))


def leiden_resolutions(adata) -> list:
    """List the leiden resolutions an h5ad carries cluster assignments for."""
    return [
        col[len(CLUSTER_GROUP_PREFIX) :]
        for col in adata.obs.columns
        if col.startswith(CLUSTER_GROUP_PREFIX)
    ]


@st.cache_data
def load_cluster_data():
    """Build the gene table the page filters on, one row per gene per leiden resolution."""
    frames = []
    for h5ad_path in find_cluster_h5ads():
        adata = load_cluster_h5ad(h5ad_path)
        dropped = [
            col
            for col in adata.obs.columns
            if col.startswith(CLUSTER_GROUP_PREFIX) or col in OBS_EXCLUDED_COLUMNS
        ]
        genes = adata.obs.drop(columns=dropped).reset_index(drop=True)
        genes.insert(0, "gene_symbol_0", adata.obs_names.to_numpy())
        if "X_phate" in adata.obsm:
            genes["PHATE_0"] = adata.obsm["X_phate"][:, 0]
            genes["PHATE_1"] = adata.obsm["X_phate"][:, 1]

        levels = parse_cluster_levels(h5ad_path)
        for resolution in leiden_resolutions(adata):
            frame = genes.copy()
            frame["cluster"] = adata.obs[
                f"{CLUSTER_GROUP_PREFIX}{resolution}"
            ].to_numpy()
            frame["leiden_resolution"] = resolution
            frame["source"] = os.path.basename(h5ad_path)
            frame["source_h5ad_path"] = h5ad_path
            frame["cluster_dir"] = os.path.join(cluster_base_dir(h5ad_path), resolution)
            for name, part in levels.items():
                frame[name] = part
            frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_active_h5ad(cluster_data):
    """Return the AnnData behind the filtered cluster data, or None when nothing matches."""
    if cluster_data.empty:
        return None
    return load_cluster_h5ad(cluster_data["source_h5ad_path"].iloc[0])


def get_cluster_dir(cluster_data):
    """Return the resolution directory holding the outputs that are not in the h5ad."""
    if cluster_data.empty:
        return None
    return cluster_data["cluster_dir"].iloc[0]


@st.cache_data
def load_montage_data(root_dir, gene_name):
    # Find all montage files
    files = FileSystem.find_files(
        root_dir + "/" + gene_name, include_all=["montages"], extensions=["png"]
    )

    # Extract features from the file paths
    filtered_df = FileSystem.extract_features(root_dir, files)

    # Add additional columns based on the file path structure
    filtered_df["gene"] = filtered_df["file_path"].apply(lambda x: x.split("/")[-3])
    filtered_df["guide"] = filtered_df["file_path"].apply(lambda x: x.split("/")[-2])
    filtered_df["channel"] = filtered_df["file_path"].apply(
        lambda x: x.split("/")[-1].split("__")[0]
    )

    return filtered_df


# -- Cluster scatter methods --
# Extract item value from selected point
def get_item_value_from_point(selected_point, groupby_column):
    # Get value from customdata which contains the hover_data values
    if "customdata" in selected_point and len(selected_point["customdata"]) > 0:
        if groupby_column in HOVER_COLUMNS:
            col_index = HOVER_COLUMNS.index(groupby_column)
            if col_index < len(selected_point["customdata"]):
                return str(selected_point["customdata"][col_index])

    # Fallback to legendgroup as a last resort (for compatibility)
    if "legendgroup" in selected_point:
        return selected_point["legendgroup"]

    return None


# Helper function to create a scatter trace
def make_scatter_trace(x, y, marker, text, customdata, name, showlegend, color=None):
    hovertemplate = (
        "PHATE_0=%{x}<br>"
        "PHATE_1=%{y}<br>"
        f"gene_symbol_0=%{{customdata[{GENE_SYMBOL_INDEX}]}}<br>"
        f"cluster=%{{customdata[{CLUSTER_INDEX}]}}<br>"
        f"cell_count=%{{customdata[{CELL_COUNT_INDEX}]}}<br>"
        f"source=%{{customdata[{SOURCE_INDEX}]}}<br>"
        "<extra></extra>"
    )
    # Optionally override color in marker
    if color is not None:
        marker = dict(marker, color=color)
    return go.Scattergl(
        x=x,
        y=y,
        mode="markers",
        marker=marker,
        text=text,
        customdata=customdata,
        name=name,
        hovertemplate=hovertemplate,
        showlegend=False,
    )


# -- Display helpers --
def get_montage_root(cell_class):
    """Return the directory holding this cell class's montages for the active image format.

    TIFF mode writes a PNG per channel under ``{cell_class}__montages``; zarr mode writes
    one OME-Zarr store per cell crop under ``{cell_class}__examples.zarr``.
    """
    montages_dir = os.path.join(BRIEFLOW_OUTPUT_PATH, "aggregate", "montages")
    if IMAGE_FORMAT == "zarr":
        return os.path.join(montages_dir, f"{cell_class}__examples.zarr")
    return os.path.join(montages_dir, f"{cell_class}__montages")


def display_gene_montages(gene_montages_root, gene):
    gene_dir = os.path.join(gene_montages_root, gene)
    if not os.path.exists(gene_dir):
        st.warning(f"No montage directory found for gene {gene}")
    elif IMAGE_FORMAT == "zarr":
        display_gene_montages_zarr(gene_montages_root, gene)
    else:
        montage_data = load_montage_data(gene_montages_root, gene)
        if montage_data.empty:
            st.write(f"No montage data found for gene {gene}")
        else:
            # Add filters for guide and channel
            available_guides = sorted(montage_data["guide"].unique())
            selected_guide = select_montage_guide(gene, available_guides)

            # Filter the data based on selections
            filtered_montage_data = montage_data[
                (montage_data["guide"] == selected_guide)
            ]

            if len(filtered_montage_data) > 0:
                # Display each image in the filtered data
                for _, row in filtered_montage_data.iterrows():
                    # Construct the full path including the montages directory
                    image_path = os.path.join(gene_montages_root, row["file_path"])
                    channel_name = row["channel"]
                    channel_name = channel_name.replace("CH-", "")

                    try:
                        if os.path.exists(image_path):
                            st.image(image_path, caption=f"Channel: {channel_name}")
                        else:
                            st.error(f"Image file not found: {image_path}")
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")

                # Add download button for overlay TIFF
                overlay_tiff_path = os.path.join(
                    gene_montages_root, gene, selected_guide, "overlay_montage.tiff"
                )

                if os.path.exists(overlay_tiff_path):
                    if STATIC_ASSET_URL_ROOT and STATIC_ASSET_PATH:
                        # Use nginx-served static files when configured
                        relative_path = overlay_tiff_path.replace(STATIC_ASSET_PATH, "")
                        static_url = f"{STATIC_ASSET_URL_ROOT}{relative_path}"
                        st.markdown(f"[Download Overlay TIFF]({static_url})")
                    else:
                        # Fall back to direct download when running locally
                        with open(overlay_tiff_path, "rb") as f:
                            st.download_button(
                                label="Download Overlay TIFF",
                                data=f,
                                file_name=f"{gene}_{selected_guide}_{row['channel']}_overlay.tiff",
                                key=f"download_{gene}_{selected_guide}_{row['channel']}_{uuid.uuid4()}",
                            )
                else:
                    st.warning(f"No overlay tiff found: {overlay_tiff_path}")
            else:
                st.warning(f"No image found for {gene} - {selected_guide}")


def display_gene_montages_zarr(gene_montages_root, gene):
    """Display the per-cell OME-Zarr crops the zarr pipeline writes instead of montage PNGs."""
    available_guides = sorted(list_montage_guides(gene_montages_root, gene))
    if not available_guides:
        st.write(f"No montage data found for gene {gene}")
        return

    selected_guide = select_montage_guide(gene, available_guides)
    crop_paths = load_montage_crops(gene_montages_root, gene, selected_guide)
    if not crop_paths:
        st.warning(f"No image found for {gene} - {selected_guide}")
        return

    channel_names = read_zarr_channel_names(crop_paths[0])
    for crop_path in crop_paths:
        crop = read_image(crop_path)
        if crop.ndim == 2:
            crop = crop[np.newaxis, ...]
        cols = st.columns(len(crop))
        for index, channel_image in enumerate(crop):
            label = (
                channel_names[index]
                if channel_names and index < len(channel_names)
                else f"Channel {index}"
            )
            cols[index].image(
                scale_to_uint8(channel_image), caption=f"Channel: {label}"
            )


@st.cache_data
def list_montage_guides(gene_montages_root, gene):
    """List the guide (sgRNA) directories available for a gene."""
    gene_dir = os.path.join(gene_montages_root, gene)
    if not os.path.isdir(gene_dir):
        return []
    return [
        entry
        for entry in os.listdir(gene_dir)
        if os.path.isdir(os.path.join(gene_dir, entry))
    ]


@st.cache_data
def load_montage_crops(gene_montages_root, gene, guide):
    """List the per-cell OME-Zarr crop stores written for a gene/guide pair."""
    return sorted(
        FileSystem.find_files(
            os.path.join(gene_montages_root, gene, guide), extensions=["zarr"]
        )
    )


def select_montage_guide(gene, available_guides):
    """Render the guide dropdown for a gene and return the selected guide."""
    if f"selected_guide_{gene}" not in st.session_state:
        st.session_state[f"selected_guide_{gene}"] = None

    # Define a callback for when the guide dropdown changes
    def on_guide_select():
        st.session_state[f"selected_guide_{gene}"] = st.session_state[
            f"guide_dropdown_{gene}"
        ]

    # Determine the index of the selected guide in the dropdown
    selected_index = 0
    selected_guide = st.session_state.get(f"selected_guide_{gene}", None)

    if selected_guide in available_guides:
        selected_index = available_guides.index(selected_guide)
    elif available_guides:
        # If no guide is selected yet or the previously selected guide is not available, select the first one
        st.session_state[f"selected_guide_{gene}"] = available_guides[0]

    return st.selectbox(
        "Select Guide",
        available_guides,
        index=selected_index,
        key=f"guide_dropdown_{gene}",  # Use a stable key based on the selected gene
        on_change=on_guide_select,
    )


def scale_to_uint8(channel_image):
    """Rescale a single-channel crop to uint8 so streamlit can display it."""
    arr = channel_image.astype(np.float32)
    low, high = float(arr.min()), float(arr.max())
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (((arr - low) / (high - low)) * 255).astype(np.uint8)


def display_cluster(cluster_data, cell_class=None, channel_combo=None):
    r"""
    :param cluster_data: a dataframe from load_cluster_data
    :param cell_class: the selected cell class filter value
    :param channel_combo: the selected channel combo filter value
    :param container: an st.container or equivalent that UI elements will be added to
    """
    global st
    # Display the data
    if not cluster_data.empty:
        # Always treat grouping column as categorical for discrete color maps
        if st.session_state.groupby_column in cluster_data.columns:
            cluster_data[st.session_state.groupby_column] = cluster_data[
                st.session_state.groupby_column
            ].astype(str)

        # Build a color map using the group names and the color palette
        group_names = cluster_data[st.session_state.groupby_column].unique()

        # Create a color palette optimized for visibility on a black background
        def get_optimized_color_palette(num_colors):
            # Use a perceptually uniform colormap that works well on dark backgrounds
            # Options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
            colormap_name = "turbo"  # Good visibility on dark backgrounds

            # Get evenly spaced colors from the colormap
            cmap = plt.get_cmap(colormap_name)
            colors = [
                mcolors.rgb2hex(cmap(i / (num_colors - 1 if num_colors > 1 else 1)))
                for i in range(num_colors)
            ]

            return colors

        # Get enough colors for all groups
        optimized_palette = get_optimized_color_palette(len(group_names))
        color_map = {group: optimized_palette[i] for i, group in enumerate(group_names)}

        # Always compute selected_data and other_data
        selected_item = st.session_state.get("selected_item", None)
        groupby_column = st.session_state.groupby_column
        selected_data = cluster_data[
            cluster_data[groupby_column].astype(str) == str(selected_item)
        ]
        other_data = cluster_data[
            cluster_data[groupby_column].astype(str) != str(selected_item)
        ]

        # Use plotly.graph_objects for full control
        fig = go.Figure()

        # Plot each group as its own trace so all appear in the legend
        # First phase: Add unselected points (all in gray)
        if selected_item is not None:
            for group in group_names:
                if group != selected_item:
                    group_df = cluster_data[cluster_data[groupby_column] == group]
                    marker = dict(
                        color="gray",  # All unselected points are gray
                        size=8,
                        opacity=0.3,
                    )
                    fig.add_trace(
                        make_scatter_trace(
                            x=group_df["PHATE_0"],
                            y=group_df["PHATE_1"],
                            marker=marker,
                            text=group_df["gene_symbol_0"],
                            customdata=group_df[HOVER_COLUMNS],
                            name=str(group),
                            showlegend=False,
                        )
                    )

            # Second phase: Add selected points on top
            for group in group_names:
                if group == selected_item:
                    group_df = cluster_data[cluster_data[groupby_column] == group]

                    # Get the selected gene if any
                    selected_gene = st.session_state.get("selected_gene", None)

                    # Split the dataframe into selected gene and other genes
                    selected_gene_df = (
                        group_df[group_df["gene_symbol_0"] == selected_gene]
                        if selected_gene
                        else pd.DataFrame()
                    )
                    other_genes_df = (
                        group_df[group_df["gene_symbol_0"] != selected_gene]
                        if selected_gene
                        else group_df
                    )

                    # Add other genes in the selected group
                    if not other_genes_df.empty:
                        marker = dict(
                            color=color_map[group],
                            size=10,
                            opacity=1.0,
                            line=dict(width=2, color="black"),
                        )
                        fig.add_trace(
                            make_scatter_trace(
                                x=other_genes_df["PHATE_0"],
                                y=other_genes_df["PHATE_1"],
                                marker=marker,
                                text=other_genes_df["gene_symbol_0"],
                                customdata=other_genes_df[HOVER_COLUMNS],
                                name=str(group),
                                showlegend=False,
                            )
                        )

                    # Add the selected gene with special highlighting
                    if not selected_gene_df.empty:
                        marker = dict(
                            color=color_map[
                                group
                            ],  # Use the cluster's color instead of red
                            size=15,  # Larger size
                            opacity=1.0,
                            symbol="circle",  # Filled circle
                            line=dict(
                                width=3, color="white"
                            ),  # White border for contrast
                        )
                        fig.add_trace(
                            make_scatter_trace(
                                x=selected_gene_df["PHATE_0"],
                                y=selected_gene_df["PHATE_1"],
                                marker=marker,
                                text=selected_gene_df["gene_symbol_0"],
                                customdata=selected_gene_df[HOVER_COLUMNS],
                                name=f"{selected_gene} (Selected)",
                                showlegend=False,
                            )
                        )
        else:
            # No selection: add all points with their original colors
            for group in group_names:
                group_df = cluster_data[cluster_data[groupby_column] == group]
                marker = dict(
                    color=color_map[group],
                    size=8,
                    opacity=1.0,
                )
                fig.add_trace(
                    make_scatter_trace(
                        x=group_df["PHATE_0"],
                        y=group_df["PHATE_1"],
                        marker=marker,
                        text=group_df["gene_symbol_0"],
                        customdata=group_df[HOVER_COLUMNS],
                        name=str(group),
                        showlegend=False,
                    )
                )

        # Update layout
        fig.update_layout(
            hovermode="closest",
            showlegend=False,
            title="",
            width=1000,
            height=800,
        )

        # Apply saved zoom coordinates if they exist
        if (
            st.session_state.zoom_xrange is not None
            and st.session_state.zoom_yrange is not None
        ):
            fig.update_layout(
                xaxis=dict(range=st.session_state.zoom_xrange),
                yaxis=dict(range=st.session_state.zoom_yrange),
            )

        # Display the plot with click event handling
        event = st.plotly_chart(
            fig, use_container_width=True, key="cluster_plot", on_select="rerun"
        )

        # Handle click events
        if event.selection and event.selection.points:
            selected_point = event.selection.points[0]

            # Get the item value from the selected point
            item_value = get_item_value_from_point(
                selected_point, st.session_state.groupby_column
            )

            # Get the gene value from the selected point
            gene_value = None
            if "customdata" in selected_point and len(selected_point["customdata"]) > 0:
                if GENE_SYMBOL_INDEX < len(selected_point["customdata"]):
                    gene_value = str(selected_point["customdata"][GENE_SYMBOL_INDEX])

            # Update session state if the item has changed
            if item_value and (
                st.session_state.selected_item != item_value
                or st.session_state.selected_gene != gene_value
            ):
                st.session_state.selected_item = item_value
                st.session_state.selected_gene = gene_value

                # Store current zoom state before rerunning
                if hasattr(event, "relayoutData") and event.relayoutData:
                    if (
                        "xaxis.range[0]" in event.relayoutData
                        and "xaxis.range[1]" in event.relayoutData
                    ):
                        st.session_state.zoom_xrange = [
                            event.relayoutData["xaxis.range[0]"],
                            event.relayoutData["xaxis.range[1]"],
                        ]
                    if (
                        "yaxis.range[0]" in event.relayoutData
                        and "yaxis.range[1]" in event.relayoutData
                    ):
                        st.session_state.zoom_yrange = [
                            event.relayoutData["yaxis.range[0]"],
                            event.relayoutData["yaxis.range[1]"],
                        ]

                st.rerun()

        # Save zoom coordinates from the event if available
        if hasattr(event, "relayoutData") and event.relayoutData:
            if (
                "xaxis.range[0]" in event.relayoutData
                and "xaxis.range[1]" in event.relayoutData
            ):
                st.session_state.zoom_xrange = [
                    event.relayoutData["xaxis.range[0]"],
                    event.relayoutData["xaxis.range[1]"],
                ]
            if (
                "yaxis.range[0]" in event.relayoutData
                and "yaxis.range[1]" in event.relayoutData
            ):
                st.session_state.zoom_yrange = [
                    event.relayoutData["yaxis.range[0]"],
                    event.relayoutData["yaxis.range[1]"],
                ]

    else:
        st.write("No cluster data files found.")


def cluster_table(cluster_data):
    # Display data overview
    st.markdown("## Cluster Data Overview")
    if cluster_data.empty:
        st.warning("No cluster data found for the selected filters.")
        return

    table_data = cluster_data.drop(
        columns=[c for c in BOOKKEEPING_COLUMNS if c in cluster_data.columns]
    )
    # If an item is selected, filter the dataframe
    if st.session_state.selected_item:
        table_data = table_data[
            table_data["cluster"].astype(str) == str(st.session_state.selected_item)
        ]

    if table_data.empty:
        st.warning("⚠️ WARNING: No genes found for the selected cluster.")
        return
    st.dataframe(table_data.set_index("gene_symbol_0"))


def feature_table(adata):
    # Feature Data Overview
    st.markdown("## Feature Data Overview")
    st.markdown(
        "Median feature values per gene after center scaling all single cell data on control cells by well."
    )
    if adata is None:
        st.warning("⚠️ WARNING: No cluster h5ad found for the selected filters.")
        return

    feature_df = pd.DataFrame(
        adata.X, index=adata.obs_names, columns=adata.var_names.to_list()
    )
    if "cell_count" in adata.obs.columns:
        feature_df.insert(0, "cell_count", adata.obs["cell_count"].to_numpy())
    feature_df.index.name = "gene_symbol_0"

    # Create a container with a fixed height and scrolling
    with st.container():
        # Display the dataframe with all columns and sorting enabled
        st.dataframe(
            feature_df,
            use_container_width=True,
            height=400,  # Fixed height for scrolling
            column_config={
                # Configure all columns to be sortable
                col: st.column_config.NumberColumn(width="medium")
                for col in feature_df.columns
            },
        )


def gene_significance_chart(adata, gene):
    """Plot each feature's value against its bootstrap significance for one gene.

    Renders nothing unless the h5ad carries bootstrap layers, which only runs with
    gene-level bootstrap results have.
    """
    if adata is None or not gene or gene not in adata.obs_names:
        return
    layer = next((name for name in SIGNIFICANCE_LAYERS if name in adata.layers), None)
    if layer is None:
        return

    row = adata.obs_names.get_loc(gene)
    effect = np.asarray(adata.X[row, :]).ravel()
    significance = np.asarray(adata.layers[layer][row, :]).ravel()
    if layer == "fdr":
        significance = -np.log10(np.clip(significance, 1e-300, None))

    st.markdown("#### Feature Significance")
    fig = go.Figure(
        go.Scattergl(
            x=effect,
            y=significance,
            mode="markers",
            text=adata.var_names.to_list(),
            marker=dict(size=6, opacity=0.7),
            hovertemplate="%{text}<br>value=%{x}<br>-log10 FDR=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Feature value",
        yaxis_title="-log10 FDR",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"significance_{gene}")


def cluster_size_charts(cluster_data):
    if cluster_data.empty:
        return

    # Create two equal-sized columns
    col1, col2 = st.columns([1, 1])

    cluster_dir = get_cluster_dir(cluster_data)

    with col1:
        st.markdown("### Cluster Sizes")
        # Cluster membership lives in obs, so count the genes per cluster instead of
        # reading the PNG the pipeline renders.
        sizes = (
            cluster_data["cluster"]
            .astype(str)
            .value_counts()
            .rename_axis("cluster")
            .reset_index(name="genes")
            .sort_values("cluster", key=lambda values: values.astype(int))
        )
        st.bar_chart(sizes, x="cluster", y="genes")

    with col2:
        st.markdown("### Cluster Enrichment")
        # Construct the path to the enrichment pie chart
        enrichment_pie_path = os.path.join(cluster_dir, "CB-Real__pie_chart.png")

        # Display the plot if it exists
        if os.path.exists(enrichment_pie_path):
            st.image(enrichment_pie_path, use_container_width=True)
        else:
            st.warning(
                f"Cluster enrichment pie chart not found at: {enrichment_pie_path}"
            )


def get_available_llm_combinations(channel_combo: str) -> list:
    """Find all cell_class/resolution combinations that have LLM data."""
    return [
        parse_mozzarellm_dir(d)
        for d in find_mozzarellm_dirs(channel_combo)
        if os.listdir(d)
    ]


def display_cluster_json(cluster_data, container=st.container()):
    if (
        "selected_item" in st.session_state
        and st.session_state.selected_item is not None
    ):
        cluster_dir = get_cluster_dir(cluster_data)
        cluster_id = str(st.session_state.selected_item)

        # Build the path to the individual cluster JSON file in mozzarellm/clusters/
        mozzarellm_clusters_dir = os.path.join(cluster_dir, "mozzarellm", "clusters")
        cluster_json_path = os.path.join(
            mozzarellm_clusters_dir, f"cluster_{cluster_id}.json"
        )

        # Always show the section header
        st.markdown("### LLM Cluster Analysis")

        if os.path.exists(cluster_json_path):
            with open(cluster_json_path, "r") as f:
                c = json.load(f)
            # Card layout using markdown and Streamlit elements
            st.markdown(
                f"""
                <div style='background-color:#1e1e1e; border-radius:10px; padding:20px; margin-bottom:20px; box-shadow:0 2px 8px #00000040;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div>
                            <span style='font-size:1.3em; font-weight:bold; color:#e0e0e0;'>Dominant Process:</span>
                            <span style='font-size:1.3em; color:#60a5fa; font-weight:bold;'>{
                    c.get("dominant_process", "")
                }</span>
                        </div>
                        <div>
                            <span style='background:#1e3a8a; color:#93c5fd; border-radius:6px; padding:4px 12px; font-weight:600;'>Confidence: {
                    c.get("pathway_confidence", "")
                }</span>
                        </div>
                    </div>
                    <div style='margin-top:10px; margin-bottom:10px; font-size:1.1em; color:#d1d5db;'>
                        {c.get("summary", "")}
                    </div>
                    <div style='margin-top:18px;'>
                        <span style='font-weight:600; color:#60a5fa;'>Established Genes:</span>
                        <span style='margin-left:8px;'>{
                    " ".join(
                        [
                            f"<span style='background:#064e3b; color:#6ee7b7; border-radius:4px; padding:2px 8px; margin-right:4px;'>{gene}</span>"
                            for gene in c.get("established_genes", [])
                        ]
                    )
                }</span>
                    </div>
                    <div style='margin-top:10px;'>
                        <span style='font-weight:600; color:#fbbf24;'>Novel Role Genes:</span>
                        <ul style='margin:0; padding-left:20px;'>
                        {
                    "".join(
                        [
                            f"<li><span style='background:#78350f; color:#fcd34d; border-radius:4px; padding:2px 8px; margin-right:4px;'>{gene['gene']}</span> <span style='color:#9ca3af;'>{gene['rationale']}</span></li>"
                            for gene in c.get("novel_role_genes", [])
                        ]
                    )
                }</ul>
                    </div>
                    <div style='margin-top:10px;'>
                        <span style='font-weight:600; color:#c084fc;'>Uncharacterized Genes:</span>
                        <ul style='margin:0; padding-left:20px;'>
                        {
                    "".join(
                        [
                            f"<li><span style='background:#5b21b6; color:#d8b4fe; border-radius:4px; padding:2px 8px; margin-right:4px;'>{gene['gene']}</span> <span style='color:#9ca3af;'>{gene['rationale']}</span></li>"
                            for gene in c.get("uncharacterized_genes", [])
                        ]
                    )
                }</ul>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            # Show placeholder card when LLM data is not available
            current_cell_class = st.session_state.get("cell_class", "unknown")
            current_resolution = st.session_state.get("leiden_resolution", "unknown")
            channel_combo = st.session_state.get("channel_combo", "")

            # Find which combinations have LLM data
            available = get_available_llm_combinations(channel_combo)

            # Smart context: tailor message based on what's wrong
            if not available:
                available_text = "No LLM analysis available for this dataset."
            else:
                # Check if current cell class has any LLM data
                cell_classes_with_llm = set(cc for cc, res in available)
                resolutions_for_current_class = [
                    res for cc, res in available if cc == current_cell_class
                ]

                if current_cell_class in cell_classes_with_llm:
                    # Right cell class, wrong resolution
                    res_list = ", ".join(sorted(resolutions_for_current_class, key=int))
                    available_text = (
                        f"Available for {current_cell_class} at resolution: {res_list}"
                    )
                else:
                    # Wrong cell class
                    available_text = (
                        f"Available for: {', '.join(sorted(cell_classes_with_llm))}"
                    )

            st.markdown(
                f"""
                <div style='background-color:#1e1e1e; border-radius:10px; padding:20px; margin-bottom:20px; box-shadow:0 2px 8px #00000040; border: 1px solid #374151;'>
                    <div style='color:#9ca3af; font-size:1.1em;'>
                        <span style='font-size:1.2em;'>ℹ️</span>
                        LLM analysis is not available for <strong>{current_cell_class}</strong> cells
                        at resolution <strong>{current_resolution}</strong>.
                    </div>
                    <div style='margin-top:12px; color:#6b7280; font-size:0.95em;'>
                        {available_text}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def display_uniprot_info():
    """Show the uniprot entry and function text the h5ad carries in obs for the selected gene."""
    if not st.session_state.selected_gene:
        return
    gene_rows = cluster_data[
        cluster_data["gene_symbol_0"] == st.session_state.selected_gene
    ]
    if gene_rows.empty or "uniprot_entry" not in gene_rows.columns:
        return

    st.write(
        f"Uniprot Entry: [{gene_rows['uniprot_entry'].values[0]}]({gene_rows['uniprot_link'].values[0]})"
    )
    function_text = gene_rows["uniprot_function"].values[0]
    if isinstance(function_text, str) and function_text.strip():
        st.markdown(f"Uniprot Function:\n>{function_text}")
    else:
        st.write("Uniprot Function: Not available")


# -- Search/Filter state management --


def initialize_session_state() -> None:
    """Initialize all session state variables used in the cluster analysis.

    This function sets up all the necessary session state variables with their default values.
    It should be called at the start of the script to ensure all required state variables
    are properly initialized.
    """
    # Initialize basic selection states
    if "selected_item" not in st.session_state:
        st.session_state.selected_item = None
    if "groupby_column" not in st.session_state:
        st.session_state.groupby_column = "cluster"
    if "selected_gene" not in st.session_state:
        st.session_state.selected_gene = None
    if "selected_guide" not in st.session_state:
        st.session_state.selected_guide = None

    # Initialize zoom coordinates
    if "zoom_xrange" not in st.session_state:
        st.session_state.zoom_xrange = None
    if "zoom_yrange" not in st.session_state:
        st.session_state.zoom_yrange = None

    # Initialize search state
    if "last_gene_search" not in st.session_state:
        st.session_state.last_gene_search = ""
    if "last_cluster_search" not in st.session_state:
        st.session_state.last_cluster_search = ""

    # Initialize gene selection dropdowns
    if "selected_gene_global" not in st.session_state:
        st.session_state.selected_gene_global = None
    if "selected_gene_cluster" not in st.session_state:
        st.session_state.selected_gene_cluster = None

    # Initialize cell class
    if "cell_class" not in st.session_state:
        st.session_state.cell_class = "all"

    # Initialize cluster dropdown
    if "cluster_dropdown" not in st.session_state:
        st.session_state.cluster_dropdown = None

    # Initialize filter counter for unique keys
    if "filter_counter" not in st.session_state:
        st.session_state.filter_counter = 0


def on_global_gene_select() -> None:
    """Callback function for global gene selection.

    Updates the selected gene and its associated cluster in the session state.
    When a gene is selected globally, it also updates the cluster selection to match
    the cluster containing the selected gene.
    """
    gene = st.session_state.selected_gene_global
    st.session_state.selected_gene = gene
    # Set cluster to the gene's cluster
    gene_row = cluster_data[cluster_data["gene_symbol_0"] == gene]
    if not gene_row.empty:
        cluster_num = str(gene_row["cluster"].iloc[0])
        st.session_state.selected_item = cluster_num
        st.session_state.cluster_dropdown = cluster_num
        st.session_state.selected_gene_cluster = gene
    else:
        st.session_state.selected_item = None
        st.session_state.cluster_dropdown = None
        st.session_state.selected_gene_cluster = None


def on_cluster_select() -> None:
    """Callback function for cluster selection.

    Updates the selected cluster and its associated gene in the session state.
    When a cluster is selected, it automatically selects the first gene in that cluster.
    If 'Select a cluster to view' is chosen, it clears all selections.
    """
    cluster = st.session_state.cluster_dropdown
    if cluster == "Select a cluster...":
        st.session_state.selected_item = None
        st.session_state.selected_gene = None
        st.session_state.selected_gene_global = None
        st.session_state.selected_gene_cluster = None
    else:
        st.session_state.selected_item = cluster
        # Find the first gene in this cluster
        cluster_genes = get_cluster_genes(cluster_data, cluster)
        if cluster_genes:
            first_gene = cluster_genes[0]
            st.session_state.selected_gene = first_gene
            st.session_state.selected_gene_global = first_gene
            st.session_state.selected_gene_cluster = first_gene
        else:
            st.session_state.selected_gene = None
            st.session_state.selected_gene_global = None
            st.session_state.selected_gene_cluster = None


def on_cluster_gene_select() -> None:
    """Callback function for gene selection within a cluster.

    Updates the selected gene in both global and cluster contexts.
    This ensures that gene selection is synchronized between the global
    and cluster-specific views.
    """
    gene = st.session_state.selected_gene_cluster
    st.session_state.selected_gene = gene
    st.session_state.selected_gene_global = gene


def on_channel_combo_change():
    """Callback function for channel combo selection."""
    st.session_state.channel_combo = st.session_state.channel_combo_radio_main
    # Reset gene selections when filter changes
    st.session_state.selected_gene = None
    st.session_state.selected_gene_global = None
    st.session_state.selected_gene_cluster = None


def on_cell_class_change():
    """Callback function for cell class selection."""
    st.session_state.cell_class = st.session_state.cell_class_radio_main
    # Reset gene selections when filter changes
    st.session_state.selected_gene = None
    st.session_state.selected_gene_global = None
    st.session_state.selected_gene_cluster = None


def on_leiden_resolution_change():
    """Callback function for leiden resolution selection."""
    st.session_state.leiden_resolution = st.session_state.leiden_resolution_radio_main
    # Reset gene selections when filter changes
    st.session_state.selected_gene = None
    st.session_state.selected_gene_global = None
    st.session_state.selected_gene_cluster = None


# Apply filters
def apply_all_filters(data):
    """Apply all filters to the cluster data in the correct order."""
    # Channel Combo filter - handle directly
    channel_combo_options = sorted(data["channel_combo"].unique().tolist())
    # Initialize channel combo in session state if needed
    if "channel_combo" not in st.session_state:
        st.session_state.channel_combo = (
            channel_combo_options[0] if channel_combo_options else None
        )

    # Format channel combo display label
    def format_channel_combo(combo: str) -> str:
        return combo

    # Create the radio button with a stable key
    selected_channel_combo = st.sidebar.radio(
        "**Channel Combo** - *Used to subset features during aggregation*",
        channel_combo_options,
        index=channel_combo_options.index(st.session_state.channel_combo)
        if st.session_state.channel_combo in channel_combo_options
        else 0,
        key="channel_combo_radio_main",
        on_change=on_channel_combo_change,
        format_func=format_channel_combo,
    )
    data = apply_filter(data, "channel_combo", selected_channel_combo)

    # Cell Class filter - handle directly
    # Options come from the data; "All" is the sentinel apply_filter treats as no filter.
    cell_class_options = ["All"] + sorted(data["cell_class"].dropna().unique().tolist())
    # Initialize cell class in session state if needed
    if st.session_state.get("cell_class") not in cell_class_options:
        st.session_state.cell_class = (
            cell_class_options[1] if len(cell_class_options) > 1 else "All"
        )

    # Format cell class display label
    def format_cell_class(cc: str) -> str:
        return cc

    # Create the radio button with a stable key
    selected_cell_class = st.sidebar.radio(
        "**Cell Class** - *Used to subset single cell data with classifier provided during aggregation*",
        cell_class_options,
        index=cell_class_options.index(st.session_state.cell_class),
        key="cell_class_radio_main",
        on_change=on_cell_class_change,
        format_func=format_cell_class,
    )
    data = apply_filter(data, "cell_class", selected_cell_class)

    # Leiden Resolution filter - handle directly
    leiden_options = sorted(
        data["leiden_resolution"].unique().tolist(), key=lambda x: float(x)
    )
    # Initialize leiden resolution in session state if needed
    if "leiden_resolution" not in st.session_state:
        st.session_state.leiden_resolution = (
            leiden_options[0] if leiden_options else None
        )

    # Format leiden resolution display label
    def format_leiden(lr: str) -> str:
        return str(lr)

    # Create the radio button with a stable key
    selected_lr = st.sidebar.radio(
        """**Leiden Resolution** - *Used in the Leiden clustering algorithm to determine gene clusters*""",
        leiden_options,
        index=leiden_options.index(st.session_state.leiden_resolution)
        if st.session_state.leiden_resolution in leiden_options
        else 0,
        key="leiden_resolution_radio_main",
        on_change=on_leiden_resolution_change,
        format_func=format_leiden,
    )
    data = apply_filter(data, "leiden_resolution", selected_lr)

    return data


# Calculate cluster_genes after all filters are applied
def get_cluster_genes(data, cluster_id):
    """Get sorted list of genes for a given cluster."""
    if not cluster_id:
        return []
    try:
        cluster_val = int(cluster_id)
    except Exception:
        cluster_val = cluster_id
    return sorted(data[data["cluster"] == cluster_val]["gene_symbol_0"].unique())


# ===

# Call initialize_session_state at the start of the script
initialize_session_state()

# Apply config defaults on first load
if not st.session_state.get("config_defaults_applied", False):
    try:
        _config = load_config()
        _mozzarellm = _config.get("mozzarellm", {})
        if _mozzarellm:
            if "cell_class" in _mozzarellm:
                st.session_state.cell_class = _mozzarellm["cell_class"]
            if "channel_combo" in _mozzarellm:
                st.session_state.channel_combo = _mozzarellm["channel_combo"]
            if "leiden_resolution" in _mozzarellm:
                st.session_state.leiden_resolution = str(
                    int(_mozzarellm["leiden_resolution"])
                )
    except Exception:
        pass  # If config loading fails, fall back to existing defaults
    st.session_state.config_defaults_applied = True

# Load and filter cluster data
cluster_data = load_cluster_data()

# Sort clusters numerically instead of alphabetically
all_genes = sorted(cluster_data["gene_symbol_0"].unique())
all_clusters = sorted(
    [str(c) for c in cluster_data["cluster"].unique()], key=lambda x: int(x)
)

st.sidebar.title("Filters")
cluster_data = apply_all_filters(cluster_data)
cluster_genes = get_cluster_genes(cluster_data, st.session_state.selected_item)

# --- UI Layout ---
st.title("Cluster Analysis")
st.markdown(
    "*Click a cluster to see details, panning and zooming is easily done through the top right of the cluster panel*"
)

# Add filters section in sidebar FIRST
# Remove duplicate call to apply_all_filters since it's already called above
# cluster_data = apply_all_filters(cluster_data)  # This line is removed

# --- Widget Rendering ---
col1, col2 = st.columns(2)
with col1:
    # Global gene dropdown with placeholder
    gene_placeholder = "Select a gene..."
    gene_options = [gene_placeholder] + all_genes
    gene_val = (
        st.session_state.selected_gene
        if st.session_state.selected_gene in all_genes
        else gene_placeholder
    )
    st.session_state.selected_gene_global = gene_val
    selected_gene = st.selectbox(
        "Gene Search",
        options=gene_options,
        index=gene_options.index(gene_val) if gene_val in gene_options else 0,
        key="selected_gene_global",
        on_change=on_global_gene_select,
    )
    # Only update if a real gene is selected
    if selected_gene != gene_placeholder:
        st.session_state.selected_gene = selected_gene

with col2:
    # Cluster dropdown
    cluster_options = ["Select a cluster..."] + all_clusters
    cluster_val = (
        st.session_state.selected_item
        if st.session_state.selected_item in all_clusters
        else "Select a cluster..."
    )
    st.session_state.cluster_dropdown = cluster_val
    st.selectbox(
        "Cluster Search",
        options=cluster_options,
        index=cluster_options.index(cluster_val)
        if cluster_val in cluster_options
        else 0,
        key="cluster_dropdown",
        on_change=on_cluster_select,
    )

cell_class = st.session_state.cell_class
channel_combo = st.session_state.channel_combo
leiden_resolution = st.session_state.leiden_resolution

cluster_adata = get_active_h5ad(cluster_data)

if not st.session_state.selected_item:
    # No cluster selected: Just show the full width cluster plot
    display_cluster(
        cluster_data,
        cell_class=st.session_state.cell_class,
        channel_combo=st.session_state.channel_combo,
    )
    cluster_table(cluster_data)
    feature_table(cluster_adata)
    cluster_size_charts(cluster_data)

else:
    # Cluster selected: Two columns: plot | detail.
    col1, col2 = st.columns([1, 1])
    with col1:
        display_cluster(
            cluster_data, cell_class=cell_class, channel_combo=channel_combo
        )
        cluster_table(cluster_data)
        feature_table(cluster_adata)
        cluster_size_charts(cluster_data)

    with col2:
        # Selected Gene info
        cell_class = st.session_state.get("cell_class", "all")

        selected_gene_info_df = cluster_data[
            cluster_data["cluster"] == st.session_state.selected_item
        ]
        genes = sorted(selected_gene_info_df["gene_symbol_0"].tolist())
        gene_montages_root = get_montage_root(cell_class)

        ## Cluster Info
        # Create two columns for the title and clear button
        title_col, button_col = st.columns([2, 1])
        with title_col:
            st.write(f"## Cluster {st.session_state.selected_item}: {len(genes)} genes")
        with button_col:
            # Add float styling and red color specifically for the Close Cluster button
            st.markdown(
                """
                <style>
                div[data-testid="stButton"] button {
                    float: right;
                    background-color: #dc2626 !important;
                    border-color: #dc2626 !important;
                    color: white !important;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )
            # Show selected item and clear button if an item is selected
            if st.button("Close Cluster"):
                st.session_state.selected_item = None
                st.session_state.selected_gene = None
                st.rerun()

        display_cluster_json(cluster_data)

        ## Montages
        # Check if gene_montages_root directory exists
        if os.path.exists(gene_montages_root):
            st.markdown("#### Gene Montages")

            # Cluster gene dropdown
            if cluster_genes:
                gene_val = (
                    st.session_state.selected_gene
                    if st.session_state.selected_gene in cluster_genes
                    else cluster_genes[0]
                )
                st.session_state.selected_gene_cluster = gene_val
                st.selectbox(
                    "Select a gene to view (within this cluster)",
                    options=cluster_genes,
                    index=cluster_genes.index(gene_val),
                    key="selected_gene_cluster",
                    on_change=on_cluster_gene_select,
                )
            else:
                st.write("No genes found in this cluster.")

            display_uniprot_info()
            gene_significance_chart(cluster_adata, st.session_state.selected_gene)

            # Display montages only for the selected gene
            if st.session_state.selected_gene:
                display_gene_montages(
                    gene_montages_root, st.session_state.selected_gene
                )
            else:
                # If no gene is selected yet, select the first one
                if genes:
                    st.session_state.selected_gene = genes[0]
                    st.rerun()
                else:
                    st.write("No genes found in this cluster.")
        else:
            st.warning(
                f"⚠️ WARNING: Gene montages root directory does not exist: {gene_montages_root}"
            )
