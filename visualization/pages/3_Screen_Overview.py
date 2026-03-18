import os

import pandas as pd
import streamlit as st
import yaml

from src.config import CONFIG_PATH, SCREEN_PATH, load_config

st.set_page_config(page_title="Screen Overview - Brieflow Analysis", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_raw_yaml(file_path):
    with open(file_path, "r") as file:
        return file.read()


def read_tabular(path):
    """Read a TSV or CSV based on file extension."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


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

st.title("Screen Overview")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_screen, tab_library, tab_features = st.tabs(
    ["Screen Info", "Perturbation Library", "Features"]
)

# ========================== Screen Info (raw YAML) =========================
with tab_screen:
    st.header("Screen Info")
    st.code(load_raw_yaml(SCREEN_PATH), language="yaml")

# ========================== Perturbation Library ===========================
with tab_library:
    st.header("Perturbation Library")

    # --- Processed barcode library (from config) ---
    barcode_path = resolve_path(config.get("sbs", {}).get("df_barcode_library_fp", ""))

    if barcode_path:
        st.subheader("Barcode Library")
        df_lib = read_tabular(barcode_path)

        n_guides = len(df_lib)
        gene_col = "gene_symbol" if "gene_symbol" in df_lib.columns else None
        n_genes = df_lib[gene_col].nunique() if gene_col else "N/A"

        col1, col2 = st.columns(2)
        col1.metric(
            "Unique Genes", f"{n_genes:,}" if isinstance(n_genes, int) else n_genes
        )
        col2.metric("Total Guides", f"{n_guides:,}")

        st.download_button(
            label="Download barcode library",
            data=df_lib.to_csv(sep="\t", index=False).encode("utf-8"),
            file_name=os.path.basename(barcode_path),
            mime="text/tab-separated-values",
            key="download_barcode_lib",
        )

        st.dataframe(df_lib.head(50), use_container_width=True)
    else:
        st.info("Barcode library not found in config.")

    # --- Raw perturbation library design file (optional) ---
    raw_lib_path = resolve_path(os.environ.get("PERTURBATION_LIBRARY_PATH", ""))

    if raw_lib_path:
        st.divider()
        st.subheader("Raw Perturbation Library Design")
        df_raw = read_tabular(raw_lib_path)

        st.markdown(
            f"**Source:** `{raw_lib_path}` ({len(df_raw):,} rows, {len(df_raw.columns)} columns)"
        )

        st.download_button(
            label="Download raw library design",
            data=df_raw.to_csv(sep="\t", index=False).encode("utf-8"),
            file_name=os.path.basename(raw_lib_path),
            mime="text/tab-separated-values",
            key="download_raw_lib",
        )

        st.dataframe(df_raw.head(50), use_container_width=True)

# ========================== Features =======================================
with tab_features:
    st.header("Features")

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

        st.dataframe(
            pd.DataFrame(summary_rows), use_container_width=True, hide_index=True
        )

        # Download button for selected feature set
        selected_file = st.selectbox("Select feature set to download", feature_files)
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
        st.info("No feature files found. Run the aggregate step first.")

    # -- Feature description reference --
    st.divider()
    st.header("Feature Descriptions")

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
        st.info("Feature documentation not found.")
