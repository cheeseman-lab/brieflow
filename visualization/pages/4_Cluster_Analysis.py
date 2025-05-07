import streamlit as st
st.set_page_config(
    page_title="Cluster Analysis - Brieflow Analysis",
    layout="wide",
)

import pandas as pd
import glob
import os
import uuid
import json

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.filesystem import FileSystem
from src.rendering import VisualizationRenderer
from src.filtering import create_filter_radio, apply_filter
from src.config import get_analysis_root_dir

# =====================
# CONSTANTS
ANALYSIS_ROOT = get_analysis_root_dir()
CLUSTER_ROOT = os.path.join(ANALYSIS_ROOT, 'cluster')

# Common hover data columns
HOVER_COLUMNS = ['gene_symbol_0', 'cluster', 'cell_count', 'source']

# Indices for accessing customdata array
GENE_SYMBOL_INDEX = 0
CLUSTER_INDEX = 1
CELL_COUNT_INDEX = 2
SOURCE_INDEX = 3

# =====================
# FUNCTIONS

# Load and merge cluster TSV files
@st.cache_data
def load_cluster_data():
    # Find all relevant TSV files
    tsv_files = glob.glob(f"{CLUSTER_ROOT}/**/phate_leiden_clustering.tsv", recursive=True)

    # Read each file and add source attribute
    dfs = []
    for file_path in tsv_files:
        rel_path = os.path.relpath(file_path, CLUSTER_ROOT)
        dirname = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path, sep='\t')
        df['source_full_path'] = file_path
        df['source'] = base_name
        parts = dirname.split(os.sep)
        for i, part in enumerate(parts):
            df[f'dir_level_{i}'] = part
        
        df.rename(columns={
            'dir_level_0': 'channel_combo',
            'dir_level_1': 'cell_class',
            'dir_level_2': 'leiden_resolution'
        }, inplace=True)

        dfs.append(df)

    # Concatenate all dataframes
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Create a scatter plot with consistent settings
def create_scatter_plot(data, color_column, color_discrete_sequence, color_discrete_map=None):
    fig = px.scatter(
        data,
        x='PHATE_0',
        y='PHATE_1',
        color=color_column,
        hover_data=HOVER_COLUMNS,
        title='PHATE Visualization',
        width=1000,
        height=800,
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map or {}
    )

    # Apply hover template to all traces
    for trace in fig.data:
        trace.hovertemplate = (
            "PHATE_0=%{x}<br>"
            "PHATE_1=%{y}<br>"
            f"gene_symbol_0=%{{customdata[{GENE_SYMBOL_INDEX}]}}<br>"
            f"cluster=%{{customdata[{CLUSTER_INDEX}]}}<br>"
            f"cell_count=%{{customdata[{CELL_COUNT_INDEX}]}}<br>"
            f"source=%{{customdata[{SOURCE_INDEX}]}}<br>"
            "<extra></extra>"
        )

    return fig

# Extract item value from selected point
def get_item_value_from_point(selected_point, groupby_column):
    # Get value from customdata which contains the hover_data values
    if 'customdata' in selected_point and len(selected_point['customdata']) > 0:
        if groupby_column in HOVER_COLUMNS:
            col_index = HOVER_COLUMNS.index(groupby_column)
            if col_index < len(selected_point['customdata']):
                return str(selected_point['customdata'][col_index])

    # Fallback to legendgroup as a last resort (for compatibility)
    if 'legendgroup' in selected_point:
        return selected_point['legendgroup']

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
        mode='markers',
        marker=marker,
        text=text,
        customdata=customdata,
        name=name,
        hovertemplate=hovertemplate,
        showlegend=False,
    )

@st.cache_data
def load_montage_data(root_dir, gene_name):
    # Find all montage files
    files = FileSystem.find_files(
        root_dir + "/" + gene_name,
        include_all=['montages'],
        extensions=['png']
    )

    # Extract features from the file paths
    filtered_df = FileSystem.extract_features(root_dir, files)

    # Add additional columns based on the file path structure
    filtered_df['gene'] = filtered_df['file_path'].apply(lambda x: x.split('/')[-3])
    filtered_df['guide'] = filtered_df['file_path'].apply(lambda x: x.split('/')[-2])
    filtered_df['channel'] = filtered_df['file_path'].apply(lambda x: x.split('/')[-1].split('__')[0])

    return filtered_df

def display_gene_montages(gene_montages_root, gene):
    gene_dir = os.path.join(gene_montages_root, gene)
    if not os.path.exists(gene_dir):
        st.warning(f"No montage directory found for gene {gene}")
    else:
        montage_data = load_montage_data(gene_montages_root, gene)
        if montage_data.empty:
            st.write(f"No montage data found for gene {gene}")
        else:
            # Add filters for guide and channel
            available_guides = sorted(montage_data['guide'].unique())
            
            # Initialize session state for selected guide if it doesn't exist
            if f'selected_guide_{gene}' not in st.session_state:
                st.session_state[f'selected_guide_{gene}'] = None
            
            # Define a callback for when the guide dropdown changes
            def on_guide_select():
                st.session_state[f'selected_guide_{gene}'] = st.session_state[f"guide_dropdown_{gene}"]
            
            # Determine the index of the selected guide in the dropdown
            selected_index = 0
            selected_guide = st.session_state.get(f"selected_guide_{gene}", None)
            
            if selected_guide in available_guides:
                selected_index = available_guides.index(selected_guide)
            elif available_guides:
                # If no guide is selected yet or the previously selected guide is not available, select the first one
                selected_guide = available_guides[0]
                st.session_state[f'selected_guide_{gene}'] = selected_guide
            
            # Create a dropdown to select a guide
            selected_guide = st.selectbox(
                "Select Guide",
                available_guides,
                index=selected_index,
                key=f"guide_dropdown_{gene}",  # Use a stable key based on the selected gene
                on_change=on_guide_select
            )

            # Filter the data based on selections
            filtered_montage_data = montage_data[
                (montage_data['guide'] == selected_guide)
                ]

            if len(filtered_montage_data) > 0:
                # Display each image in the filtered data
                for _, row in filtered_montage_data.iterrows():
                    # Construct the full path including the montages directory
                    image_path = os.path.join(gene_montages_root, row['file_path'])
                    channel_name = row['channel']
                    channel_name = channel_name.replace('CH-', '')

                    try:
                        if os.path.exists(image_path):
                            st.image(image_path, caption=f"Channel: {channel_name}")
                        else:
                            st.error(f"Image file not found: {image_path}")
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")

                # Add download button for overlay TIFF
                overlay_tiff_path = os.path.join(
                    gene_montages_root,
                    gene,
                    selected_guide,
                    f"overlay_montage.tiff"
                )

                if os.path.exists(overlay_tiff_path):
                    with open(overlay_tiff_path, 'rb') as f:
                        st.download_button(
                            label="Download Overlay TIFF",
                            data=f,
                            file_name=f"{gene}_{selected_guide}_{row['channel']}_overlay.tiff",
                            key=f"download_{gene}_{selected_guide}_{row['channel']}_{uuid.uuid4()}"
                        )
                else:
                    st.warning(f"No overlay tiff found: {overlay_tiff_path}")
            else:
                st.warning(f"No image found for {gene} - {selected_guide}")

# =====================
# MAIN CODE

# Initialize session state for selected item and grouping column if they don't exist
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None

if 'groupby_column' not in st.session_state:
    st.session_state.groupby_column = 'cluster'

# Initialize session state for selected gene if it doesn't exist
if 'selected_gene' not in st.session_state:
    st.session_state.selected_gene = None

# Initialize session state for selected guide if it doesn't exist
if 'selected_guide' not in st.session_state:
    st.session_state.selected_guide = None

# Initialize session state for zoom coordinates if they don't exist
if 'zoom_xrange' not in st.session_state:
    st.session_state.zoom_xrange = None
if 'zoom_yrange' not in st.session_state:
    st.session_state.zoom_yrange = None

def display_cluster(cluster_data, container=st.container()):
    r'''
    :param cluster_data: a dataframe from load_cluster_data
    :param container: an st.container or equivalent that UI elements will be added to
    '''
    global st
    # Display the data
    if not cluster_data.empty:
        # Always treat grouping column as categorical for discrete color maps
        if st.session_state.groupby_column in cluster_data.columns:
            cluster_data[st.session_state.groupby_column] = cluster_data[st.session_state.groupby_column].astype(str)

        # Build a color map using the group names and the color palette
        group_names = cluster_data[st.session_state.groupby_column].unique()

        # Create a color palette optimized for visibility on a black background
        def get_optimized_color_palette(num_colors):

            # Use a perceptually uniform colormap that works well on dark backgrounds
            # Options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
            colormap_name = 'turbo'  # Good visibility on dark backgrounds

            # Get evenly spaced colors from the colormap
            cmap = plt.get_cmap(colormap_name)
            colors = [mcolors.rgb2hex(cmap(i / (num_colors - 1 if num_colors > 1 else 1)))
                      for i in range(num_colors)]

            return colors

        # Get enough colors for all groups
        optimized_palette = get_optimized_color_palette(len(group_names))
        color_map = {group: optimized_palette[i] for i, group in enumerate(group_names)}

        # Always compute selected_data and other_data
        selected_item = st.session_state.get("selected_item", None)
        groupby_column = st.session_state.groupby_column
        selected_data = cluster_data[cluster_data[groupby_column].astype(str) == str(selected_item)]
        other_data = cluster_data[cluster_data[groupby_column].astype(str) != str(selected_item)]

        # Use plotly.graph_objects for full control
        fig = go.Figure()

        # Plot each group as its own trace so all appear in the legend
        # First phase: Add unselected points (all in gray)
        if selected_item is not None:
            for group in group_names:
                if group != selected_item:
                    group_df = cluster_data[cluster_data[groupby_column] == group]
                    marker = dict(
                        color='gray',  # All unselected points are gray
                        size=8,
                        opacity=0.3,
                    )
                    fig.add_trace(make_scatter_trace(
                        x=group_df['PHATE_0'],
                        y=group_df['PHATE_1'],
                        marker=marker,
                        text=group_df['gene_symbol_0'],
                        customdata=group_df[HOVER_COLUMNS],
                        name=str(group),
                        showlegend=False,
                    ))

            # Second phase: Add selected points on top
            for group in group_names:
                if group == selected_item:
                    group_df = cluster_data[cluster_data[groupby_column] == group]
                    
                    # Get the selected gene if any
                    selected_gene = st.session_state.get("selected_gene", None)
                    
                    # Split the dataframe into selected gene and other genes
                    selected_gene_df = group_df[group_df['gene_symbol_0'] == selected_gene] if selected_gene else pd.DataFrame()
                    other_genes_df = group_df[group_df['gene_symbol_0'] != selected_gene] if selected_gene else group_df
                    
                    # Add other genes in the selected group
                    if not other_genes_df.empty:
                        marker = dict(
                            color=color_map[group],
                            size=10,
                            opacity=1.0,
                            line=dict(width=2, color='black')
                        )
                        fig.add_trace(make_scatter_trace(
                            x=other_genes_df['PHATE_0'],
                            y=other_genes_df['PHATE_1'],
                            marker=marker,
                            text=other_genes_df['gene_symbol_0'],
                            customdata=other_genes_df[HOVER_COLUMNS],
                            name=str(group),
                            showlegend=False,
                        ))
                    
                    # Add the selected gene with special highlighting
                    if not selected_gene_df.empty:
                        marker = dict(
                            color=color_map[group],  # Use the cluster's color instead of red
                            size=15,      # Larger size
                            opacity=1.0,
                            symbol='circle',  # Filled circle
                            line=dict(width=3, color='white')  # White border for contrast
                        )
                        fig.add_trace(make_scatter_trace(
                            x=selected_gene_df['PHATE_0'],
                            y=selected_gene_df['PHATE_1'],
                            marker=marker,
                            text=selected_gene_df['gene_symbol_0'],
                            customdata=selected_gene_df[HOVER_COLUMNS],
                            name=f"{selected_gene} (Selected)",
                            showlegend=False,
                        ))
        else:
            # No selection: add all points with their original colors
            for group in group_names:
                group_df = cluster_data[cluster_data[groupby_column] == group]
                marker = dict(
                    color=color_map[group],
                    size=8,
                    opacity=1.0,
                )
                fig.add_trace(make_scatter_trace(
                    x=group_df['PHATE_0'],
                    y=group_df['PHATE_1'],
                    marker=marker,
                    text=group_df['gene_symbol_0'],
                    customdata=group_df[HOVER_COLUMNS],
                    name=str(group),
                    showlegend=False,
                ))

        # Update layout
        fig.update_layout(
            hovermode='closest',
            showlegend=False,
            title='',
            width=1000,
            height=800,
        )
        
        # Apply saved zoom coordinates if they exist
        if st.session_state.zoom_xrange is not None and st.session_state.zoom_yrange is not None:
            fig.update_layout(
                xaxis=dict(range=st.session_state.zoom_xrange),
                yaxis=dict(range=st.session_state.zoom_yrange)
            )

        # Display the plot with click event handling
        event = st.plotly_chart(fig, use_container_width=True, key="cluster_plot", on_select="rerun")

        # Handle click events
        if event.selection and event.selection.points:
            selected_point = event.selection.points[0]

            # Get the item value from the selected point
            item_value = get_item_value_from_point(selected_point, st.session_state.groupby_column)
            
            # Get the gene value from the selected point
            gene_value = None
            if 'customdata' in selected_point and len(selected_point['customdata']) > 0:
                if GENE_SYMBOL_INDEX < len(selected_point['customdata']):
                    gene_value = str(selected_point['customdata'][GENE_SYMBOL_INDEX])

            # Update session state if the item has changed
            if item_value and (st.session_state.selected_item != item_value or st.session_state.selected_gene != gene_value):
                st.session_state.selected_item = item_value
                st.session_state.selected_gene = gene_value
                
                # Store current zoom state before rerunning
                if hasattr(event, 'relayoutData') and event.relayoutData:
                    if 'xaxis.range[0]' in event.relayoutData and 'xaxis.range[1]' in event.relayoutData:
                        st.session_state.zoom_xrange = [
                            event.relayoutData['xaxis.range[0]'],
                            event.relayoutData['xaxis.range[1]']
                        ]
                    if 'yaxis.range[0]' in event.relayoutData and 'yaxis.range[1]' in event.relayoutData:
                        st.session_state.zoom_yrange = [
                            event.relayoutData['yaxis.range[0]'],
                            event.relayoutData['yaxis.range[1]']
                        ]
                
                st.rerun()
        
        # Save zoom coordinates from the event if available
        if hasattr(event, 'relayoutData') and event.relayoutData:
            if 'xaxis.range[0]' in event.relayoutData and 'xaxis.range[1]' in event.relayoutData:
                st.session_state.zoom_xrange = [
                    event.relayoutData['xaxis.range[0]'],
                    event.relayoutData['xaxis.range[1]']
                ]
            if 'yaxis.range[0]' in event.relayoutData and 'yaxis.range[1]' in event.relayoutData:
                st.session_state.zoom_yrange = [
                    event.relayoutData['yaxis.range[0]'],
                    event.relayoutData['yaxis.range[1]']
                ]

        # Display data overview
        st.markdown("## Cluster Data Overview")

        # If an item is selected, filter the dataframe
        source_tsv = cluster_data['source_full_path'].unique()[0]
        if os.path.exists(source_tsv):
            table_data = pd.read_csv(source_tsv, sep='\t')
            if st.session_state.selected_item:
                # Convert selected_item to integer since cluster column is int64
                try:
                    selected_item_int = int(st.session_state.selected_item)
                    table_data = table_data[table_data['cluster'] == selected_item_int]
                except ValueError:
                    st.error(f"Invalid cluster value: {st.session_state.selected_item}")
                    return
                
                if len(table_data.index) == 0:
                    st.warning(f"⚠️ WARNING: No data found in the TSV file: {source_tsv}")
                else:
                    table_data.set_index('gene_symbol_0', inplace=True)
                    st.dataframe(table_data)
            else:
                if len(table_data.index) == 0:
                    st.warning(f"⚠️ WARNING: No data found in the TSV file: {source_tsv}")
                else:
                    table_data.set_index('gene_symbol_0', inplace=True)
                    st.dataframe(table_data)
        else:
            st.warning(f"⚠️ WARNING: Source TSV file not found at: {source_tsv}")

        # Feature Data Overview
        st.markdown("## Feature Data Overview")
        st.markdown("Median feature values per gene after center scaling all single cell data on control cells by well.")

        # Construct the feature table path
        feature_table_path = os.path.join(
            ANALYSIS_ROOT,
            "aggregate",
            "tsvs",
            f"CeCl-{selected_cell_class}_ChCo-{selected_channel_combo}__feature_table.tsv"
        )

        # Load and display the feature table if it exists
        if os.path.exists(feature_table_path):
            feature_df = pd.read_csv(feature_table_path, sep='\t')
            feature_df.set_index('gene_symbol_0', inplace=True)

            # Create a container with a fixed height and scrolling
            with st.container():
                # Display the dataframe with all columns and sorting enabled
                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    height=400,  # Fixed height for scrolling
                    column_config={
                        # Configure all columns to be sortable
                        col: st.column_config.NumberColumn(
                            width="medium"
                        ) for col in feature_df.columns
                    }
                )
        else:
            st.warning(f"⚠️ WARNING: Feature table not found at: {feature_table_path}")


    else:
        st.write("No cluster data files found.")

def display_cluster_json(cluster_data, container=st.container()):
    if 'selected_item' in st.session_state and st.session_state.selected_item is not None:

        # Because the interphase folder has mixed case
        cluster_dir = os.path.dirname(cluster_data['source_full_path'].unique()[0])

        # Build the path to the gpt-4o_clusters.json file
        cluster_json_path = os.path.join(
            cluster_dir,
            'mozzarellm',
            'gpt-4o_clusters.json'
        )
        if os.path.exists(cluster_json_path):
            st.markdown("### LLM Cluster Analysis")
            with open(cluster_json_path, 'r') as f:
                cluster_json = json.load(f)
            cluster_id = str(st.session_state.selected_item)
            clusters = cluster_json.get('clusters', {})
            
            if cluster_id in clusters:
                c = clusters[cluster_id]
                # Card layout using markdown and Streamlit elements
                st.markdown(f"""
                    <div style='background-color:#1e1e1e; border-radius:10px; padding:20px; margin-bottom:20px; box-shadow:0 2px 8px #00000040;'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <span style='font-size:1.3em; font-weight:bold; color:#e0e0e0;'>Dominant Process:</span>
                                <span style='font-size:1.3em; color:#60a5fa; font-weight:bold;'>{c.get('dominant_process','')}</span>
                            </div>
                            <div>
                                <span style='background:#1e3a8a; color:#93c5fd; border-radius:6px; padding:4px 12px; font-weight:600;'>Confidence: {c.get('pathway_confidence','')}</span>
                            </div>
                        </div>
                        <div style='margin-top:10px; margin-bottom:10px; font-size:1.1em; color:#d1d5db;'>
                            {c.get('summary','')}
                        </div>
                        <div style='margin-top:18px;'>
                            <span style='font-weight:600; color:#60a5fa;'>Established Genes:</span>
                            <span style='margin-left:8px;'>{" ".join([f"<span style='background:#064e3b; color:#6ee7b7; border-radius:4px; padding:2px 8px; margin-right:4px;'>{gene}</span>" for gene in c.get('established_genes',[])])}</span>
                        </div>
                        <div style='margin-top:10px;'>
                            <span style='font-weight:600; color:#fbbf24;'>Novel Role Genes:</span>
                            <ul style='margin:0; padding-left:20px;'>
                            {"".join([
                                f"<li><span style='background:#78350f; color:#fcd34d; border-radius:4px; padding:2px 8px; margin-right:4px;'>{gene['gene']}</span> <span style='color:#9ca3af;'>{gene['rationale']}</span></li>" for gene in c.get('novel_role_genes',[])
                            ])}</ul>
                        </div>
                        <div style='margin-top:10px;'>
                            <span style='font-weight:600; color:#c084fc;'>Uncharacterized Genes:</span>
                            <ul style='margin:0; padding-left:20px;'>
                            {"".join([
                                f"<li><span style='background:#5b21b6; color:#d8b4fe; border-radius:4px; padding:2px 8px; margin-right:4px;'>{gene['gene']}</span> <span style='color:#9ca3af;'>{gene['rationale']}</span></li>" for gene in c.get('uncharacterized_genes',[])
                            ])}</ul>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Cluster {cluster_id} not found in the analysis. Available clusters are: {', '.join(clusters.keys())}")
                st.info("This might indicate that the cluster analysis was run with different parameters or the JSON file is from a different analysis run.")
                return


# Load cluster data
cluster_data = load_cluster_data()

# Initialize cell_class in session state if it doesn't exist
if 'cell_class' not in st.session_state:
    st.session_state.cell_class = 'all'

# Set up the page layout with two columns
st.title("Cluster Analysis")

# Add cell class filter in the sidebar
st.sidebar.title("Filters")

# Channel Combo
selected_channel_combo = create_filter_radio(cluster_data, 'channel_combo', st.sidebar, "Channel Combo", include_all=False)
cluster_data = apply_filter(cluster_data, 'channel_combo', selected_channel_combo)

# Cell Class
cell_class_options = ['all', 'Mitotic', 'Interphase']  # Add default options
selected_cell_class = st.sidebar.radio(
    "Cell Class",
    cell_class_options,
    index=cell_class_options.index(st.session_state.cell_class) if st.session_state.cell_class in cell_class_options else 0
)
cluster_data = apply_filter(cluster_data, 'cell_class', selected_cell_class)
st.session_state.cell_class = selected_cell_class

# Leiden Resolution
selected_lr = create_filter_radio(cluster_data, 'leiden_resolution', st.sidebar, "Leiden Resolution", include_all=False)
cluster_data = apply_filter(cluster_data, 'leiden_resolution', selected_lr)

if 'selected_item' in st.session_state and st.session_state.selected_item is not None:
    # Create two columns for the main content
    col1, col2 = st.columns([1,1])
else:
    col1 = st.container()
    col2 = st.container()

with col1:
    # Display cluster data in the left column
    display_cluster(cluster_data)

with col2:

    # Selected Gene info
    cell_class = st.session_state.get("cell_class", 'all')

    if 'selected_item' in st.session_state:
        selected_item = st.session_state.get("selected_item", None)
        selected_gene = st.session_state.get("selected_gene", None)
        groupby_column = 'cluster'

        # Cluster section dropdown
        cluster_options = cluster_data[groupby_column].unique()
        selected_index = 0
        if selected_item in cluster_options:
            selected_index = list(cluster_options).index(selected_item)

        def on_cluster_select():
            st.session_state.selected_item = st.session_state.cluster_dropdown
            st.session_state.selected_gene = None  # Reset selected gene when cluster changes

        st.selectbox(
            "Select a cluster to view",
            options=cluster_options,
            index=selected_index,
            key="cluster_dropdown",
            on_change=on_cluster_select
        )
        
        if selected_item:
            selected_gene_info_df = cluster_data[cluster_data[groupby_column] == selected_item]
            genes = selected_gene_info_df['gene_symbol_0'].tolist()
            gene_montages_root = os.path.join(ANALYSIS_ROOT, "aggregate", 'montages', f"{cell_class}__montages")

            # Show selected item and clear button if an item is selected
            if st.session_state.selected_item:
                # Create two columns for the title and button
                title_col, button_col = st.columns([7, 1])
                with title_col:
                    st.write(f"## Cluster {selected_item}: {len(genes)} genes")
                with button_col:
                    if st.button("X"):
                        st.session_state.selected_item = None
                        st.session_state.selected_gene = None
                        st.rerun()
            
            display_cluster_json(cluster_data)
            
            
            # Check if gene_montages_root directory exists
            if os.path.exists(gene_montages_root):
                st.markdown("#### Gene Montages")
                
                # Define a callback for when the dropdown changes
                def on_gene_select():
                    st.session_state.selected_gene = st.session_state[f"gene_dropdown_{selected_item}"]
                
                # Determine the index of the selected gene in the dropdown
                selected_index = 0
                if selected_gene in genes:
                    selected_index = genes.index(selected_gene)
                
                # Create a dropdown to select a gene
                st.selectbox(
                    "Select a gene to view",
                    options=genes,
                    index=selected_index,
                    key=f"gene_dropdown_{selected_item}",  # Use a stable key based on the selected cluster
                    on_change=on_gene_select
                )


                if st.session_state.selected_gene:
                    source_tsv = cluster_data['source_full_path'].unique()[0]
                    if os.path.exists(source_tsv):
                        table_data = pd.read_csv(source_tsv, sep='\t')
                        table_data = table_data[table_data['gene_symbol_0'] == st.session_state.selected_gene]
                        if len(table_data.index) != 0:
                            st.write(f"Uniprot Entry: [{table_data['uniprot_entry'].values[0]}]({table_data['uniprot_link'].values[0]})")
                            function_text = table_data['uniprot_function'].values[0]
                            if isinstance(function_text, str) and function_text.strip():
                                st.markdown(f"Uniprot Function:\n>{function_text}")
                            else:
                                st.write("Uniprot Function: Not available")
                
                # Display montages only for the selected gene
                if selected_gene:
                    display_gene_montages(gene_montages_root, selected_gene)
                else:
                    # If no gene is selected yet, select the first one
                    if genes:
                        st.session_state.selected_gene = genes[0]
                        st.rerun()
                    else:
                        st.write("No genes found in this cluster.")
            else:
                st.warning(f"⚠️ WARNING: Gene montages root directory does not exist: {gene_montages_root}")