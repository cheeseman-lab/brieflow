import re
import os
import glob

import streamlit as st

from src.config import BRIEFLOW_OUTPUT_PATH
from src.theme import empty_state, page_setup

page_setup(
    "Pipeline Statistics",
    "📈",
    "Per-module counts and summaries collected from the run's stats report.",
)

# Find stats file
stats_files = glob.glob(os.path.join(BRIEFLOW_OUTPUT_PATH, "*_stats.txt"))

if not stats_files:
    empty_state(
        "No pipeline stats file found.",
        "The `generate_stats` step writes `*_stats.txt` into the output root.",
    )
    st.stop()

stats_path = stats_files[0]

with open(stats_path, "r") as f:
    stats_content = f.read()


# ---------------------------------------------------------------------------
# Parse the stats file into sections
# ---------------------------------------------------------------------------
def parse_sections(text):
    """Split stats text into (header, body) sections."""
    sections = []
    lines = text.splitlines()
    current_header = None
    current_lines = []

    for line in lines:
        # Match section headers like " PREPROCESSING STATISTICS:"
        m = re.match(r"^\s*([A-Z][A-Z /]+STATISTICS):", line)
        if m:
            if current_header:
                sections.append((current_header, "\n".join(current_lines)))
            current_header = m.group(1).strip()
            current_lines = []
            continue
        # Skip progress lines like "[1/6] Gathering..."  and separator lines
        if (
            re.match(r"^\[[\d/]+\]", line.strip())
            or "====" in line
            or "REPORT COMPLETE" in line
        ):
            continue
        if current_header is not None:
            current_lines.append(line)

    if current_header:
        sections.append((current_header, "\n".join(current_lines)))

    return sections


def body_to_markdown(body):
    """Convert the indented bullet-point body into clean markdown."""
    md_lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            # Blank line = possible sub-section break
            if md_lines and md_lines[-1] != "":
                md_lines.append("")
            continue
        if stripped.startswith("- "):
            md_lines.append(f"- {stripped[2:]}")
        elif stripped.endswith(":"):
            # Sub-header like "Interphase_DAPI_TUBULIN_GH2AX_PHALLOIDIN:"
            md_lines.append(f"\n**{stripped.rstrip(':')}**")
        else:
            md_lines.append(stripped)
    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# Render each section as a block
# ---------------------------------------------------------------------------
sections = parse_sections(stats_content)

# Pretty names for section headers
HEADER_MAP = {
    "PREPROCESSING STATISTICS": "Preprocessing",
    "SBS STATISTICS": "Sequencing by Synthesis (SBS)",
    "PHENOTYPE STATISTICS": "Phenotype",
    "MERGE STATISTICS": "Merge",
    "AGGREGATION STATISTICS": "Aggregation",
    "CLUSTERING STATISTICS": "Clustering",
}

st.sidebar.subheader("Report")
st.sidebar.caption(os.path.basename(stats_path))
st.sidebar.download_button(
    label="Download stats file",
    data=stats_content,
    file_name=os.path.basename(stats_path),
    mime="text/plain",
    use_container_width=True,
)

for index, (header, body) in enumerate(sections):
    display_name = HEADER_MAP.get(header, header.title())
    st.subheader(display_name)
    st.markdown(body_to_markdown(body))
    if index < len(sections) - 1:
        st.divider()
