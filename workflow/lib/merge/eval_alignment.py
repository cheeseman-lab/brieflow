"""Helper functions for evaluating the alignment from the merge process steps.

This includes:
- Plotting alignment quality from intial site alignment.
"""

import matplotlib.pyplot as plt


def plot_alignment_quality(
    df_align, det_range, score, xlim=(0, 0.1), ylim=(0, 1), figsize=(10, 6)
):
    """Creates a scatter plot visualizing alignment quality based on determinant and score values.

    Args:
        df_align (pandas.DataFrame): DataFrame containing alignment results. Must have columns:
            'determinant', 'score', 'tile', and 'site'.
        det_range (tuple): (min, max) range for acceptable determinant values.
        score (float): Minimum acceptable score value.
        xlim (tuple, optional): (min, max) range for x-axis (determinant). Defaults to (0, 0.1).
        ylim (tuple, optional): (min, max) range for y-axis (score). Defaults to (0, 1).
        figsize (tuple, optional): Figure size in inches. Defaults to (10, 6).

    Returns:
        matplotlib.figure.Figure: The created figure object.
        matplotlib.axes.Axes: The created axes object.
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Construct filtering condition
    gate = "{0} <= determinant <= {1} & score > {2}".format(*det_range, score)

    # Add labels for each point
    for idx, row in df_align.iterrows():
        ax.annotate(
            f"PH:{row['tile']}\nSBS:{row['site']}",
            (row["determinant"], row["score"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Add threshold lines
    ax.axhline(y=score, color="r", linestyle="--", label=f"Score threshold = {score}")
    ax.axvline(
        x=det_range[0], color="g", linestyle="--", label=f"Det min = {det_range[0]}"
    )
    ax.axvline(
        x=det_range[1], color="g", linestyle="--", label=f"Det max = {det_range[1]}"
    )

    # Shade valid region
    ax.axvspan(
        det_range[0],
        det_range[1],
        ymin=score / ylim[1],
        alpha=0.1,
        color="green",
        label="Valid region",
    )

    # Set axis labels and title
    ax.set_xlabel("Determinant")
    ax.set_ylabel("Score")
    ax.set_title("Alignment Quality Check\nScore vs Determinant")

    # Set axis ranges
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Add legend in top left corner
    ax.legend(loc="upper left")

    # Show grid
    ax.grid(True, alpha=0.3)

    # Calculate and add statistics
    passing = df_align.query(gate).shape[0]
    total = df_align.shape[0]
    stats_text = (
        f"Passing alignments: {passing}/{total}\n({passing / total * 100:.1f}%)"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    return fig, ax
