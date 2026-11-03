import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Accent and neutrals, kept in step with .streamlit/config.toml
ACCENT = "#2B6E8F"
MUTED = "#7A8B99"
GRID = "#E8ECEF"

# Categorical sequence shared by the plotly figures and the theme's chart colors
CATEGORICAL = [
    "#2B6E8F",
    "#C97B3C",
    "#5B8C5A",
    "#8E6C9B",
    "#B55A5A",
    "#4F7CA3",
    "#7A8B99",
    "#A38A4B",
]

PLOTLY_TEMPLATE = "brieflow"

# Registered once at import so every page's figures share one look
pio.templates[PLOTLY_TEMPLATE] = go.layout.Template(
    pio.templates["plotly_white"],
    layout=dict(
        colorway=CATEGORICAL,
        font=dict(family="sans-serif", size=13, color="#1F2A37"),
        title=dict(font=dict(size=16)),
        margin=dict(l=48, r=24, t=32, b=48),
        hoverlabel=dict(font_size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, ticks="outside"),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, ticks="outside"),
    ),
)

# Spacing only: tighten the default gap above headings and around the block container
_PAGE_CSS = """
<style>
    div.block-container {padding-top: 2.4rem; padding-bottom: 3rem;}
    h2, h3 {margin-top: 0.6rem;}
    div[data-testid="stMetric"] {padding: 0.25rem 0;}
</style>
"""


def page_setup(title, icon, caption):
    """Set the page config and render the shared title + one-line caption."""
    st.set_page_config(
        page_title=f"{title} - Brieflow Analysis",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)
    st.title(title)
    st.caption(caption)


def sidebar_filters_header(subtitle=None):
    """Render the shared sidebar filter heading."""
    st.sidebar.subheader("Filters")
    if subtitle:
        st.sidebar.caption(subtitle)


def empty_state(message, hint=None):
    """Show a missing-data message, naming the pipeline step that produces the file."""
    st.info(f"{message}\n\n:gray[{hint}]" if hint else message)


def style_figure(fig, **layout):
    """Apply the shared plotly template to a figure and return it."""
    fig.update_layout(template=PLOTLY_TEMPLATE, **layout)
    return fig
