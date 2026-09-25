import os

import pandas as pd
import streamlit as st
import yaml

from src.config import BRIEFLOW_OUTPUT_PATH, CONFIG_PATH, SCREEN_PATH, load_config
from src.filesystem import read_table
from src.theme import empty_state, page_setup

page_setup(
    "Screen Overview",
    "🧬",
    "What was screened: the screen definition, the perturbation library and the features measured.",
)

# Number of rows previewed for the library tables
PREVIEW_ROWS = 50

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_raw_yaml(file_path):
    with open(file_path, "r") as file:
        return file.read()


def resolve_path(path):
    """Return the path if it exists, trying fallbacks relative to config and data locations."""
    if not path:
        return None
    if os.path.isfile(path):
        return path
    if not os.path.isabs(path):
        # Try relative to the config file directory (deployment layout)
        alt = os.path.join(os.path.dirname(CONFIG_PATH), path)
        if os.path.isfile(alt):
            return alt
        # Try relative to the original analysis data location
        data_loc = yaml.safe_load(open(SCREEN_PATH)).get("data", {}).get("location", "")
        alt = os.path.join(data_loc, "analysis", path)
        if os.path.isfile(alt):
            return alt
    return None


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
config = load_config()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_screen, tab_library, tab_features = st.tabs(
    ["Screen Info", "Perturbation Library", "Features"]
)

# ========================== Screen Info (raw YAML) =========================
with tab_screen:
    st.caption(os.path.abspath(SCREEN_PATH))
    st.code(load_raw_yaml(SCREEN_PATH), language="yaml")

# ========================== Perturbation Library ===========================
with tab_library:
    # --- Processed barcode library (from config) ---
    barcode_path = resolve_path(config.get("sbs", {}).get("df_barcode_library_fp", ""))

    if barcode_path:
        st.subheader("Barcode Library")
        st.caption(os.path.abspath(barcode_path))
        df_lib = read_table(barcode_path)

        n_guides = len(df_lib)
        gene_col = "gene_symbol" if "gene_symbol" in df_lib.columns else None
        n_genes = df_lib[gene_col].nunique() if gene_col else "N/A"

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Unique Genes", f"{n_genes:,}" if isinstance(n_genes, int) else n_genes
        )
        col2.metric("Total Guides", f"{n_guides:,}")
        col3.metric(
            "Guides per Gene",
            f"{n_guides / n_genes:.1f}"
            if isinstance(n_genes, int) and n_genes
            else "—",
        )

        st.download_button(
            label="Download barcode library",
            data=df_lib.to_csv(sep="\t", index=False).encode("utf-8"),
            file_name=os.path.basename(barcode_path),
            mime="text/tab-separated-values",
            key="download_barcode_lib",
        )

        with st.expander(f"Preview first {PREVIEW_ROWS} rows", expanded=True):
            st.dataframe(
                df_lib.head(PREVIEW_ROWS), use_container_width=True, hide_index=True
            )
    else:
        empty_state(
            "No barcode library found.",
            "Set `sbs.df_barcode_library_fp` in the config to the library the screen used.",
        )

    # --- Raw perturbation library design file (optional) ---
    raw_lib_path = resolve_path(os.environ.get("PERTURBATION_LIBRARY_PATH", ""))

    if raw_lib_path:
        st.divider()
        st.subheader("Raw Perturbation Library Design")
        st.caption(os.path.abspath(raw_lib_path))
        df_raw = read_table(raw_lib_path)

        col_rows, col_cols = st.columns(2)
        col_rows.metric("Rows", f"{len(df_raw):,}")
        col_cols.metric("Columns", f"{len(df_raw.columns):,}")

        st.download_button(
            label="Download raw library design",
            data=df_raw.to_csv(sep="\t", index=False).encode("utf-8"),
            file_name=os.path.basename(raw_lib_path),
            mime="text/tab-separated-values",
            key="download_raw_lib",
        )

        with st.expander(f"Preview first {PREVIEW_ROWS} rows", expanded=False):
            st.dataframe(
                df_raw.head(PREVIEW_ROWS), use_container_width=True, hide_index=True
            )

# ========================== Features =======================================
with tab_features:
    # -- Summary table of all feature sets --
    agg_tsvs_dir = os.path.join(BRIEFLOW_OUTPUT_PATH, "aggregate", "tsvs")
    feature_files = []
    if os.path.isdir(agg_tsvs_dir):
        feature_files = sorted(
            [f for f in os.listdir(agg_tsvs_dir) if f.endswith("__features_genes.tsv")]
        )

    if feature_files:
        METADATA_PREFIXES = (
            "gene_symbol",
            "cell_count",
            "cell_stage",
            "cell_stage_confidence",
            "cell_barcode",
        )

        summary_rows = []
        for fname in feature_files:
            cols = list(
                pd.read_csv(
                    os.path.join(agg_tsvs_dir, fname), sep="\t", nrows=0
                ).columns
            )
            feat_cols = [c for c in cols if not c.startswith(METADATA_PREFIXES)]
            summary_rows.append(
                {
                    "Feature Set": fname.replace("__features_genes.tsv", ""),
                    "Total Columns": len(cols),
                    "Feature Columns": len(feat_cols),
                }
            )

        st.subheader("Feature Sets")
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature Set": st.column_config.TextColumn(width="large"),
                "Total Columns": st.column_config.NumberColumn(format="%d"),
                "Feature Columns": st.column_config.NumberColumn(format="%d"),
            },
        )

        # Download button for selected feature set
        selected_file = st.selectbox(
            "Feature set",
            feature_files,
            help="Pick a feature set to download its column list.",
        )
        fp = os.path.join(agg_tsvs_dir, selected_file)
        cols = list(pd.read_csv(fp, sep="\t", nrows=0).columns)
        feat_cols = [c for c in cols if not c.startswith(METADATA_PREFIXES)]

        md_lines = [f"# Features: {selected_file}\n"]
        md_lines.append(f"Total features: {len(feat_cols)}\n")
        md_lines.append("## Feature Columns\n")
        for c in feat_cols:
            md_lines.append(f"- `{c}`")
        md_content = "\n".join(md_lines)

        st.download_button(
            label="Download feature list (Markdown)",
            data=md_content.encode("utf-8"),
            file_name=f"{selected_file.replace('.tsv', '')}_feature_list.md",
            mime="text/markdown",
        )
    else:
        empty_state(
            "No feature files found.",
            "The aggregate step writes `aggregate/tsvs/*__features_genes.tsv`.",
        )

    # -- Feature description reference --
    st.divider()
    st.subheader("Feature Descriptions")

    # Allow override via env var, else auto-discover from brieflow repo
    feature_doc_path = os.environ.get("FEATURE_DOC_PATH", "")
    if not feature_doc_path:
        _vis_dir = os.path.dirname(os.path.dirname(__file__))
        _brieflow_dir = os.path.dirname(_vis_dir)
        feature_doc_path = os.path.join(
            _brieflow_dir,
            "workflow",
            "lib",
            "external",
            "CP_EMULATOR_FEATURES.md",
        )

    if os.path.isfile(feature_doc_path):
        with open(feature_doc_path, "r") as f:
            feature_doc = f.read()

        with st.expander(
            "View CellProfiler Emulator Feature Documentation", expanded=False
        ):
            st.markdown(feature_doc)

        st.download_button(
            label="Download Feature Documentation (Markdown)",
            data=feature_doc.encode("utf-8"),
            file_name="CP_EMULATOR_FEATURES.md",
            mime="text/markdown",
            key="download_feature_doc",
        )
    else:
        empty_state(
            "Feature documentation not found.",
            "Point `FEATURE_DOC_PATH` at `workflow/lib/external/CP_EMULATOR_FEATURES.md`.",
        )
