import streamlit as st
import uuid

from src.theme import (
    ACCENT,
    CATEGORICAL,
    empty_state,
    page_setup,
    sidebar_filters_header,
    style_figure,
)

page_setup(
    "Cluster Analysis",
    "🧫",
    "Gene clusters in the PHATE embedding, with per-cluster annotation and example cell montages.",
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

# Montage crops per row, so a gene's examples read as a grid rather than a column
MONTAGE_COLUMNS = 4

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
        empty_state(f"No montage directory for gene {gene}.")
    elif IMAGE_FORMAT == "zarr":
        display_gene_montages_zarr(gene_montages_root, gene)
    else:
        montage_data = load_montage_data(gene_montages_root, gene)
        if montage_data.empty:
            empty_state(f"No montage data found for gene {gene}.")
        else:
            # Add filters for guide and channel
            available_guides = sorted(montage_data["guide"].unique())
            selected_guide = select_montage_guide(gene, available_guides)

            # Filter the data based on selections
            filtered_montage_data = montage_data[
                (montage_data["guide"] == selected_guide)
            ]

            if len(filtered_montage_data) > 0:
                # Display each image in the filtered data, capped columns per row
                rows = list(filtered_montage_data.iterrows())
                cols = st.columns(min(MONTAGE_COLUMNS, len(rows)))
                for index, (_, row) in enumerate(rows):
                    # Construct the full path including the montages directory
                    image_path = os.path.join(gene_montages_root, row["file_path"])
                    channel_name = row["channel"]
                    channel_name = channel_name.replace("CH-", "")

                    with cols[index % len(cols)]:
                        try:
                            if os.path.exists(image_path):
                                st.image(
                                    image_path,
                                    caption=channel_name,
                                    use_container_width=True,
                                )
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
                    st.caption("No overlay TIFF written for this guide.")
            else:
                empty_state(f"No montage image for {gene} - {selected_guide}.")


def display_gene_montages_zarr(gene_montages_root, gene):
    """Display the per-cell OME-Zarr crops the zarr pipeline writes instead of montage PNGs."""
    available_guides = sorted(list_montage_guides(gene_montages_root, gene))
    if not available_guides:
        empty_state(f"No montage data found for gene {gene}.")
        return

    selected_guide = select_montage_guide(gene, available_guides)
    crop_paths = load_montage_crops(gene_montages_root, gene, selected_guide)
    if not crop_paths:
        empty_state(f"No montage image for {gene} - {selected_guide}.")
        return

    channel_names = read_zarr_channel_names(crop_paths[0])
    for crop_index, crop_path in enumerate(crop_paths):
        crop = read_image(crop_path)
        if crop.ndim == 2:
            crop = crop[np.newaxis, ...]
        st.caption(f"Cell {crop_index + 1} of {len(crop_paths)}")
        cols = st.columns(min(MONTAGE_COLUMNS, len(crop)))
        for index, channel_image in enumerate(crop):
            label = (
                channel_names[index]
                if channel_names and index < len(channel_names)
                else f"Channel {index}"
            )
            cols[index % len(cols)].image(
                scale_to_uint8(channel_image),
                caption=label,
                use_container_width=True,
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
        "Guide",
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

        # Use the theme palette while it has enough distinct colors, then fall back
        # to a perceptually uniform colormap for runs with many clusters
        def get_optimized_color_palette(num_colors):
            if num_colors <= len(CATEGORICAL):
                return CATEGORICAL[:num_colors]

            cmap = plt.get_cmap("turbo")
            return [
                mcolors.rgb2hex(cmap(i / (num_colors - 1 if num_colors > 1 else 1)))
                for i in range(num_colors)
            ]

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
                        color="#C7D0D7",  # All unselected points are muted gray
                        size=7,
                        opacity=0.55,
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
                            size=9,
                            opacity=1.0,
                            line=dict(width=1, color="#FFFFFF"),
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
                            color=color_map[group],  # Use the cluster's color
                            size=16,
                            opacity=1.0,
                            symbol="circle",
                            line=dict(width=2, color="#1F2A37"),
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
                    size=7,
                    opacity=0.9,
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
        style_figure(
            fig,
            hovermode="closest",
            showlegend=False,
            title="",
            height=760,
            xaxis_title="PHATE 0",
            yaxis_title="PHATE 1",
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
        empty_state(
            "No cluster data found for the selected filters.",
            "The cluster step writes one h5ad per channel combo under `cluster/`.",
        )


def cluster_table(cluster_data):
    # Display data overview
    st.subheader("Genes")
    st.caption("One row per gene, with its cluster assignment and gene metadata.")
    if cluster_data.empty:
        empty_state(
            "No cluster data found for the selected filters.",
            "The cluster step writes one h5ad per channel combo under `cluster/`.",
        )
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
        empty_state("No genes found for the selected cluster.")
        return
    st.dataframe(
        table_data.set_index("gene_symbol_0"),
        use_container_width=True,
        height=360,
        column_config={
            "cluster": st.column_config.TextColumn("Cluster", width="small"),
            "cell_count": st.column_config.NumberColumn("Cells", format="%d"),
            "uniprot_link": st.column_config.LinkColumn(
                "UniProt", display_text="entry"
            ),
        },
    )


def feature_table(adata):
    # Feature Data Overview
    st.subheader("Features")
    st.caption(
        "Median feature values per gene after center scaling all single cell data on control cells by well."
    )
    if adata is None:
        empty_state(
            "No cluster h5ad found for the selected filters.",
            "`rule format_cluster_anndata` writes `cluster/**/h5ad/*.h5ad`.",
        )
        return

    feature_df = pd.DataFrame(
        adata.X, index=adata.obs_names, columns=adata.var_names.to_list()
    )
    if "cell_count" in adata.obs.columns:
        feature_df.insert(0, "cell_count", adata.obs["cell_count"].to_numpy())
    feature_df.index.name = "gene_symbol_0"

    # Wide and long, so it opens on demand inside a scrolling expander
    with st.expander(
        f"{feature_df.shape[0]:,} genes x {feature_df.shape[1]:,} columns",
        expanded=False,
    ):
        st.dataframe(
            feature_df,
            use_container_width=True,
            height=400,  # Fixed height for scrolling
            column_config={
                col: st.column_config.NumberColumn(width="medium", format="%.3f")
                for col in feature_df.columns
                if col != "cell_count"
            }
            | {"cell_count": st.column_config.NumberColumn("Cells", format="%d")},
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

    st.markdown("##### Feature significance")
    fig = go.Figure(
        go.Scattergl(
            x=effect,
            y=significance,
            mode="markers",
            text=adata.var_names.to_list(),
            marker=dict(size=6, opacity=0.7, color=ACCENT),
            hovertemplate="%{text}<br>value=%{x}<br>-log10 FDR=%{y}<extra></extra>",
        )
    )
    style_figure(
        fig,
        xaxis_title="Feature value",
        yaxis_title="-log10 FDR",
        height=360,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"significance_{gene}")


def cluster_size_charts(cluster_data):
    if cluster_data.empty:
        return

    st.subheader("Cluster composition")

    # Create two equal-sized columns
    col1, col2 = st.columns([1, 1])

    cluster_dir = get_cluster_dir(cluster_data)

    with col1:
        st.markdown("##### Cluster sizes")
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
        sizes_fig = go.Figure(
            go.Bar(x=sizes["cluster"], y=sizes["genes"], marker_color=ACCENT)
        )
        style_figure(
            sizes_fig,
            height=320,
            yaxis_title="Genes",
            xaxis=dict(
                title="Cluster",
                type="category",
                categoryorder="array",
                categoryarray=sizes["cluster"].tolist(),
            ),
        )
        st.plotly_chart(sizes_fig, use_container_width=True, key="cluster_sizes")

    with col2:
        st.markdown("##### Cluster enrichment")
        # Construct the path to the enrichment pie chart
        enrichment_pie_path = os.path.join(cluster_dir, "CB-Real__pie_chart.png")

        # Display the plot if it exists
        if os.path.exists(enrichment_pie_path):
            st.image(enrichment_pie_path, use_container_width=True)
        else:
            empty_state(
                "No cluster enrichment chart for this resolution.",
                "`rule benchmark_clusters` writes `CB-Real__pie_chart.png` next to the "
                "clustering outputs.",
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
        st.subheader("LLM cluster analysis")

        if os.path.exists(cluster_json_path):
            with open(cluster_json_path, "r") as f:
                c = json.load(f)

            process_col, confidence_col = st.columns([3, 1])
            process_col.metric("Dominant process", c.get("dominant_process", "—"))
            confidence_col.metric(
                "Pathway confidence", str(c.get("pathway_confidence", "—")).title()
            )

            summary = c.get("summary", "")
            if summary:
                st.markdown(summary)

            established = c.get("established_genes", [])
            if established:
                st.markdown("**Established genes**")
                st.markdown(" ".join(f":green-badge[{gene}]" for gene in established))

            novel = c.get("novel_role_genes", [])
            if novel:
                st.markdown("**Novel role genes**")
                for gene in novel:
                    st.markdown(
                        f":orange-badge[{gene['gene']}] {gene.get('rationale', '')}"
                    )

            uncharacterized = c.get("uncharacterized_genes", [])
            if uncharacterized:
                st.markdown("**Uncharacterized genes**")
                for gene in uncharacterized:
                    st.markdown(
                        f":violet-badge[{gene['gene']}] {gene.get('rationale', '')}"
                    )
        else:
            # Show a hint about where the analysis does exist when it is missing here
            current_cell_class = st.session_state.get("cell_class", "unknown")
            current_resolution = st.session_state.get("leiden_resolution", "unknown")
            channel_combo = st.session_state.get("channel_combo", "")

            # Find which combinations have LLM data
            available = get_available_llm_combinations(channel_combo)

            # Smart context: tailor message based on what's wrong
            if not available:
                available_text = (
                    "The `mozzarellm` step writes `mozzarellm/clusters/cluster_*.json` "
                    "next to the clustering outputs."
                )
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

            empty_state(
                f"No LLM analysis for **{current_cell_class}** cells at resolution "
                f"**{current_resolution}**.",
                available_text,
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

    st.markdown(
        f"**UniProt entry:** [{gene_rows['uniprot_entry'].values[0]}]({gene_rows['uniprot_link'].values[0]})"
    )
    function_text = gene_rows["uniprot_function"].values[0]
    if isinstance(function_text, str) and function_text.strip():
        st.markdown(f"> {function_text}")
    else:
        st.caption("No UniProt function annotation for this gene.")


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
        "Channel combo",
        channel_combo_options,
        help="Which channels' features the aggregation step kept.",
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
        "Cell class",
        cell_class_options,
        help="Single-cell subset chosen by the classifier supplied during aggregation.",
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
        "Leiden resolution",
        leiden_options,
        help="Resolution the Leiden algorithm used to form the gene clusters.",
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

sidebar_filters_header("Pick the clustering run to explore.")
cluster_data = apply_all_filters(cluster_data)
cluster_genes = get_cluster_genes(cluster_data, st.session_state.selected_item)

# --- UI Layout ---
# Add filters section in sidebar FIRST
# Remove duplicate call to apply_all_filters since it's already called above
# cluster_data = apply_all_filters(cluster_data)  # This line is removed

# --- Widget Rendering ---
genes_col, clusters_col, res_col, class_col = st.columns(4)
genes_col.metric("Genes", f"{cluster_data['gene_symbol_0'].nunique():,}")
clusters_col.metric("Clusters", f"{cluster_data['cluster'].nunique():,}")
res_col.metric("Leiden resolution", str(st.session_state.leiden_resolution))
class_col.metric("Cell class", str(st.session_state.cell_class))

st.divider()

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
        "Find a gene",
        options=gene_options,
        help="Jumps to the gene and selects the cluster it belongs to.",
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
        "Open a cluster",
        options=cluster_options,
        help="Opens the cluster detail panel; also set by clicking a point in the embedding.",
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
    st.subheader("PHATE embedding")
    st.caption(
        "Click a point to open its cluster; pan and zoom from the toolbar above the plot."
    )
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
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.subheader("PHATE embedding")
        st.caption(
            "Click a point to open its cluster; pan and zoom from the toolbar above the plot."
        )
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
        title_col, button_col = st.columns([3, 2], vertical_alignment="bottom")
        with title_col:
            st.subheader(f"Cluster {st.session_state.selected_item}")
            st.caption(f"{len(genes)} genes")
        with button_col:
            # Show selected item and clear button if an item is selected
            if st.button("Close cluster"):
                st.session_state.selected_item = None
                st.session_state.selected_gene = None
                st.rerun()

        display_cluster_json(cluster_data)

        ## Montages
        # Check if gene_montages_root directory exists
        if os.path.exists(gene_montages_root):
            st.divider()
            st.subheader("Gene montages")
            st.caption("Example cells for one gene and guide in this cluster.")

            # Cluster gene dropdown
            if cluster_genes:
                gene_val = (
                    st.session_state.selected_gene
                    if st.session_state.selected_gene in cluster_genes
                    else cluster_genes[0]
                )
                st.session_state.selected_gene_cluster = gene_val
                st.selectbox(
                    "Gene",
                    options=cluster_genes,
                    help="Genes assigned to this cluster.",
                    index=cluster_genes.index(gene_val),
                    key="selected_gene_cluster",
                    on_change=on_cluster_gene_select,
                )
            else:
                empty_state("No genes found in this cluster.")

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
                    empty_state("No genes found in this cluster.")
        else:
            empty_state(
                "No montages available for this cell class.",
                "The aggregate step writes example cells to `aggregate/montages/`.",
            )
