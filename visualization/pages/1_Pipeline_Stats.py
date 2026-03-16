import re
import os
import glob

import streamlit as st

from src.config import BRIEFLOW_OUTPUT_PATH

st.set_page_config(
    page_title="Pipeline Stats - Brieflow Analysis",
    layout="wide",
)

# Find stats file
stats_files = glob.glob(os.path.join(BRIEFLOW_OUTPUT_PATH, "*_stats.txt"))

if not stats_files:
    st.info(
        "No pipeline stats file found. Run the stats collection step to generate one."
    )
    st.stop()

stats_path = stats_files[0]

with open(stats_path, "r") as f:
    stats_content = f.read()

st.title("Pipeline Statistics")


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

for header, body in sections:
    display_name = HEADER_MAP.get(header, header.title())
    st.header(display_name)
    st.markdown(body_to_markdown(body))
    st.divider()

# Download
st.download_button(
    label="Download Stats File",
    data=stats_content,
    file_name=os.path.basename(stats_path),
    mime="text/plain",
)
