"""Quality Control and Evaluation for Stitched Well Outputs.

This module provides comprehensive quality control tools for evaluating stitched well images
and segmentation masks. It includes visualization capabilities, statistical analysis, and
interactive tools for examining stitching quality.

Run this in a Jupyter notebook for interactive QC capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage import io, exposure
from matplotlib.patches import Rectangle
import warnings
from matplotlib.widgets import Slider

warnings.filterwarnings("ignore")

# Set up plotting
plt.style.use("default")
sns.set_palette("husl")


def analyze_stitch_config(config, metadata_df, well, data_type):
    """Analyze stitching configuration to verify geometry preservation.

    This function validates that the computed tile shifts preserve the original
    well geometry and provides visual comparisons between stage coordinates
    and computed stitch positions.

    Parameters
    ----------
    config : dict
        Stitching configuration containing total_translation shifts
    metadata_df : pd.DataFrame
        Metadata containing tile positions and stage coordinates
    well : str
        Well identifier to analyze
    data_type : str
        Data type being analyzed ('phenotype' or 'sbs')

    Returns:
    -------
    pd.DataFrame
        Combined dataframe with original coordinates and computed shifts
    """
    print(f"\n=== Analyzing {data_type.upper()} Stitch Config ===")

    shifts = config["total_translation"]
    confidence = config.get("confidence", {}).get(well, {})

    print(f"Total shifts computed: {len(shifts)}")
    print(f"Confidence scores: {len(confidence)}")

    # Extract shift data
    shift_data = []
    for key, shift in shifts.items():
        well_name, tile_id = key.split("/")
        shift_data.append(
            {"tile": int(tile_id), "y_shift": shift[0], "x_shift": shift[1]}
        )

    shift_df = pd.DataFrame(shift_data)

    # Get original stage coordinates
    well_metadata = metadata_df[metadata_df["well"] == well].copy()
    original_coords = well_metadata[["tile", "x_pos", "y_pos"]]

    # Merge with shifts
    combined = original_coords.merge(shift_df, on="tile", how="inner")

    print(f"Matched {len(combined)} tiles between metadata and shifts")

    # Analysis
    print(f"\nShift Statistics:")
    print(
        f"  Y shifts: {shift_df['y_shift'].min()} to {shift_df['y_shift'].max()} "
        f"(range: {shift_df['y_shift'].max() - shift_df['y_shift'].min()})"
    )
    print(
        f"  X shifts: {shift_df['x_shift'].min()} to {shift_df['x_shift'].max()} "
        f"(range: {shift_df['x_shift'].max() - shift_df['x_shift'].min()})"
    )
    print(
        f"  Mean Y: {shift_df['y_shift'].mean():.1f}, Std: {shift_df['y_shift'].std():.1f}"
    )
    print(
        f"  Mean X: {shift_df['x_shift'].mean():.1f}, Std: {shift_df['x_shift'].std():.1f}"
    )

    # Check if stitched positions preserve circular geometry
    stitched_x = combined["x_shift"]
    stitched_y = combined["y_shift"]

    # Center the stitched coordinates
    center_x = stitched_x.mean()
    center_y = stitched_y.mean()

    distances_stitched = np.sqrt(
        (stitched_x - center_x) ** 2 + (stitched_y - center_y) ** 2
    )
    cv_stitched = distances_stitched.std() / distances_stitched.mean()

    # Center the original coordinates for comparison
    orig_center_x = combined["x_pos"].mean()
    orig_center_y = combined["y_pos"].mean()
    distances_orig = np.sqrt(
        (combined["x_pos"] - orig_center_x) ** 2
        + (combined["y_pos"] - orig_center_y) ** 2
    )
    cv_orig = distances_orig.std() / distances_orig.mean()

    print(f"\nGeometry Preservation Check:")
    print(f"  Original well CV: {cv_orig:.3f}")
    print(f"  Stitched well CV: {cv_stitched:.3f}")

    if abs(cv_stitched - cv_orig) < 0.1:
        print(f"  ✅ Circular geometry well preserved!")
    elif cv_stitched < 0.3:
        print(f"  ✅ Good circular geometry in stitched result")
    else:
        print(f"  ⚠️  Geometry may be distorted")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original stage coordinates
    axes[0].scatter(
        combined["x_pos"], combined["y_pos"], c=combined["tile"], cmap="viridis", s=30
    )
    axes[0].set_title(f"{data_type.title()} - Original Stage Coordinates")
    axes[0].set_xlabel("X Position (μm)")
    axes[0].set_ylabel("Y Position (μm)")
    axes[0].axis("equal")

    # Computed shifts (stitched positions)
    axes[1].scatter(
        combined["x_shift"],
        combined["y_shift"],
        c=combined["tile"],
        cmap="viridis",
        s=30,
    )
    axes[1].set_title(f"{data_type.title()} - Computed Stitch Positions")
    axes[1].set_xlabel("X Shift (pixels)")
    axes[1].set_ylabel("Y Shift (pixels)")
    axes[1].axis("equal")

    # Overlay comparison (normalized)
    orig_x_norm = (combined["x_pos"] - combined["x_pos"].min()) / (
        combined["x_pos"].max() - combined["x_pos"].min()
    )
    orig_y_norm = (combined["y_pos"] - combined["y_pos"].min()) / (
        combined["y_pos"].max() - combined["y_pos"].min()
    )

    stitch_x_norm = (combined["x_shift"] - combined["x_shift"].min()) / (
        combined["x_shift"].max() - combined["x_shift"].min()
    )
    stitch_y_norm = (combined["y_shift"] - combined["y_shift"].min()) / (
        combined["y_shift"].max() - combined["y_shift"].min()
    )

    axes[2].scatter(
        orig_x_norm, orig_y_norm, c="blue", alpha=0.6, s=20, label="Original"
    )
    axes[2].scatter(
        stitch_x_norm, stitch_y_norm, c="red", alpha=0.6, s=20, label="Stitched"
    )
    axes[2].set_title(f"{data_type.title()} - Geometry Comparison (Normalized)")
    axes[2].set_xlabel("Normalized X")
    axes[2].set_ylabel("Normalized Y")
    axes[2].legend()
    axes[2].axis("equal")

    plt.tight_layout()
    plt.show()

    return combined


def check_backend_and_setup():
    """Check and setup the correct matplotlib backend for interactivity.

    Interactive widgets require specific matplotlib backends. This function
    checks the current backend and provides guidance for enabling interactivity.

    Returns:
    -------
    bool
        True if backend supports interactive widgets, False otherwise
    """
    backend = plt.get_backend()
    print(f"Current matplotlib backend: {backend}")

    if "inline" in backend.lower():
        print("\n⚠️  Current backend doesn't support interactive widgets!")
        print("To enable interactive sliders, run one of these commands:")
        print("  %matplotlib widget    # Recommended for Jupyter")
        print("  %matplotlib qt        # Alternative option")
        print("  %matplotlib tk        # Another alternative")
        print("\nThen restart your kernel and try again.")
        return False
    elif any(b in backend.lower() for b in ["widget", "qt", "tk", "macosx"]):
        print("✅ Backend supports interactive widgets!")
        return True
    else:
        print(f"⚠️  Backend '{backend}' may not support interactive widgets.")
        print("If sliders don't work, try: %matplotlib widget")
        return True


class StitchQC:
    """Quality control and visualization tools for stitched well outputs.

    This class provides comprehensive QC capabilities including file validation,
    overlay visualization, cell position analysis, and interactive image inspection
    with brightness/contrast controls.
    """

    def __init__(self, base_path, plate, well):
        """Initialize QC for a specific plate/well.

        Parameters
        ----------
        base_path : str or Path
            Base path to analysis outputs (should point to the 'merge' directory)
        plate : str or int
            Plate identifier
        well : str
            Well identifier (e.g., 'A3')
        """
        self.base_path = Path(base_path)
        self.plate = str(plate)
        self.well = well
        prefix = f"P-{plate}_W-{well}__"

        # Define expected file paths
        self.phenotype_image = (
            self.base_path / "stitched_images" / f"{prefix}phenotype_stitched_image.npy"
        )
        self.phenotype_mask = (
            self.base_path / "stitched_masks" / f"{prefix}phenotype_stitched_mask.npy"
        )
        self.phenotype_positions = (
            self.base_path
            / "cell_positions"
            / f"{prefix}phenotype_cell_positions.parquet"
        )
        self.phenotype_overlay = (
            self.base_path / "overlays" / f"{prefix}phenotype_overlay.png"
        )

        self.sbs_image = (
            self.base_path / "stitched_images" / f"{prefix}sbs_stitched_image.npy"
        )
        self.sbs_mask = (
            self.base_path / "stitched_masks" / f"{prefix}sbs_stitched_mask.npy"
        )
        self.sbs_positions = (
            self.base_path / "cell_positions" / f"{prefix}sbs_cell_positions.parquet"
        )
        self.sbs_overlay = self.base_path / "overlays" / f"{prefix}sbs_overlay.png"

        print(f"Initialized QC for Plate {plate}, Well {well}")
        print(f"Base path: {self.base_path}")
        print(f"Looking for files with prefix: {prefix}")
        self.check_files()

    def check_files(self):
        """Check which output files exist and report their status."""
        files = {
            "Phenotype Image": self.phenotype_image,
            "Phenotype Mask": self.phenotype_mask,
            "Phenotype Positions": self.phenotype_positions,
            "Phenotype Overlay": self.phenotype_overlay,
            "SBS Image": self.sbs_image,
            "SBS Mask": self.sbs_mask,
            "SBS Positions": self.sbs_positions,
            "SBS Overlay": self.sbs_overlay,
        }

        print("\n=== File Status ===")
        for name, path in files.items():
            status = "✅ EXISTS" if path.exists() else "❌ MISSING"
            size = f"({path.stat().st_size / 1e6:.1f} MB)" if path.exists() else ""
            print(f"{name:20} {status} {size}")

    def view_overlays(self, figsize=(15, 6)):
        """Display phenotype and SBS overlays side by side.

        Parameters
        ----------
        figsize : tuple, default (15, 6)
            Figure size as (width, height) in inches
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Phenotype overlay
        if self.phenotype_overlay.exists():
            ph_overlay = io.imread(self.phenotype_overlay)
            axes[0].imshow(ph_overlay)
            axes[0].set_title(
                f"Phenotype Overlay\nPlate {self.plate}, Well {self.well}"
            )
            axes[0].axis("off")
        else:
            axes[0].text(
                0.5,
                0.5,
                "Phenotype\nOverlay\nMissing",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
                fontsize=16,
            )
            axes[0].set_title("Phenotype Overlay - MISSING")

        # SBS overlay
        if self.sbs_overlay.exists():
            sbs_overlay = io.imread(self.sbs_overlay)
            axes[1].imshow(sbs_overlay)
            axes[1].set_title(f"SBS Overlay\nPlate {self.plate}, Well {self.well}")
            axes[1].axis("off")
        else:
            axes[1].text(
                0.5,
                0.5,
                "SBS\nOverlay\nMissing",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
                fontsize=16,
            )
            axes[1].set_title("SBS Overlay - MISSING")

        plt.tight_layout()
        plt.show()

    def analyze_cell_positions(self):
        """Analyze cell position data and create comprehensive summary plots.

        This method loads cell position data from both phenotype and SBS modalities
        and creates detailed visualizations including cell counts, area distributions,
        spatial distributions, and overlay comparisons.

        Returns:
        -------
        tuple
            (phenotype_positions, sbs_positions) DataFrames if available, None otherwise
        """
        # Load position data
        ph_pos = None
        sbs_pos = None

        if self.phenotype_positions.exists():
            ph_pos = pd.read_parquet(self.phenotype_positions)
            print(f"Phenotype: {len(ph_pos)} cells")

        if self.sbs_positions.exists():
            sbs_pos = pd.read_parquet(self.sbs_positions)
            print(f"SBS: {len(sbs_pos)} cells")

        if ph_pos is None and sbs_pos is None:
            print("No position data available")
            return None, None

        # Create analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Cell Position Analysis - Plate {self.plate}, Well {self.well}",
            fontsize=16,
        )

        # Plot 1: Cell counts by modality
        counts = []
        labels = []
        if ph_pos is not None:
            counts.append(len(ph_pos))
            labels.append("Phenotype")
        if sbs_pos is not None:
            counts.append(len(sbs_pos))
            labels.append("SBS")

        axes[0, 0].bar(labels, counts, color=["skyblue", "lightcoral"])
        axes[0, 0].set_title("Cell Counts by Modality")
        axes[0, 0].set_ylabel("Number of Cells")
        for i, count in enumerate(counts):
            axes[0, 0].text(i, count + max(counts) * 0.01, str(count), ha="center")

        # Plot 2: Cell area distributions
        if ph_pos is not None:
            axes[0, 1].hist(
                ph_pos["area"], bins=50, alpha=0.7, label="Phenotype", color="skyblue"
            )
        if sbs_pos is not None:
            axes[0, 1].hist(
                sbs_pos["area"], bins=50, alpha=0.7, label="SBS", color="lightcoral"
            )
        axes[0, 1].set_title("Cell Area Distributions")
        axes[0, 1].set_xlabel("Cell Area (pixels)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()

        # Plot 3: Tile distribution (if available)
        if ph_pos is not None and "tile" in ph_pos.columns:
            tile_counts = ph_pos["tile"].value_counts().sort_index()
            axes[0, 2].bar(
                tile_counts.index, tile_counts.values, color="skyblue", alpha=0.7
            )
            axes[0, 2].set_title("Phenotype Cells per Tile")
            axes[0, 2].set_xlabel("Tile ID")
            axes[0, 2].set_ylabel("Cell Count")
        else:
            axes[0, 2].text(
                0.5,
                0.5,
                "No Tile\nData",
                ha="center",
                va="center",
                transform=axes[0, 2].transAxes,
                fontsize=14,
            )
            axes[0, 2].set_title("Phenotype Tiles - No Data")

        # Plot 4: Spatial distribution - Phenotype
        if ph_pos is not None:
            scatter = axes[1, 0].scatter(
                ph_pos["j"],
                ph_pos["i"],
                c=ph_pos.get("tile", 0),
                s=1,
                alpha=0.6,
                cmap="tab10",
            )
            axes[1, 0].set_title("Phenotype Cell Positions")
            axes[1, 0].set_xlabel("J (Column)")
            axes[1, 0].set_ylabel("I (Row)")
            axes[1, 0].invert_yaxis()
            if "tile" in ph_pos.columns:
                plt.colorbar(scatter, ax=axes[1, 0], label="Tile ID")
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No Phenotype\nPosition Data",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
                fontsize=14,
            )

        # Plot 5: Spatial distribution - SBS
        if sbs_pos is not None:
            scatter = axes[1, 1].scatter(
                sbs_pos["j"],
                sbs_pos["i"],
                c=sbs_pos.get("tile", 0),
                s=1,
                alpha=0.6,
                cmap="tab10",
            )
            axes[1, 1].set_title("SBS Cell Positions")
            axes[1, 1].set_xlabel("J (Column)")
            axes[1, 1].set_ylabel("I (Row)")
            axes[1, 1].invert_yaxis()
            if "tile" in sbs_pos.columns:
                plt.colorbar(scatter, ax=axes[1, 1], label="Tile ID")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No SBS\nPosition Data",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=14,
            )

        # Plot 6: Position overlay comparison
        if ph_pos is not None and sbs_pos is not None:
            axes[1, 2].scatter(
                ph_pos["j"],
                ph_pos["i"],
                s=1,
                alpha=0.5,
                label="Phenotype",
                color="blue",
            )
            axes[1, 2].scatter(
                sbs_pos["j"], sbs_pos["i"], s=1, alpha=0.5, label="SBS", color="red"
            )
            axes[1, 2].set_title("Position Overlay Comparison")
            axes[1, 2].set_xlabel("J (Column)")
            axes[1, 2].set_ylabel("I (Row)")
            axes[1, 2].invert_yaxis()
            axes[1, 2].legend()
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "Missing Data\nfor Comparison",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
                fontsize=14,
            )

        plt.tight_layout()
        plt.show()

        return ph_pos, sbs_pos

    def check_stitching_quality_efficient(
        self,
        sample_region=None,
        preview_downsample=20,
        brightness_range=(0.1, 2.0),
        contrast_range=(0.5, 3.0),
    ):
        """Memory-efficient stitching quality check with interactive controls.

        This method provides interactive brightness and contrast adjustment for
        examining stitched images. It uses memory mapping and downsampling to
        handle large images efficiently.

        Parameters
        ----------
        sample_region : tuple, optional
            (start_i, end_i, start_j, end_j) region to examine at full resolution
        preview_downsample : int, default 20
            Downsampling factor for full well preview (higher = lower memory usage)
        brightness_range : tuple, default (0.1, 2.0)
            Min and max values for brightness adjustment slider
        contrast_range : tuple, default (0.5, 3.0)
            Min and max values for contrast adjustment slider

        Returns:
        -------
        tuple
            (figure, brightness_slider, contrast_slider) for interactive control
        """
        # Check matplotlib backend
        backend = plt.get_backend()
        print(f"Matplotlib backend: {backend}")
        if "inline" in backend.lower():
            print("Warning: Interactive widgets may not work with 'inline' backend.")
            print("Try running: %matplotlib widget")
            print("Or: %matplotlib qt")

        # Create figure with space for sliders
        fig = plt.figure(figsize=(16, 14))

        # Create main subplot area (leave space at bottom for sliders)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.2], hspace=0.4, bottom=0.1)
        axes = [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]

        # Slider area
        slider_ax1 = plt.axes([0.15, 0.05, 0.25, 0.03])
        slider_ax2 = plt.axes([0.55, 0.05, 0.25, 0.03])

        fig.suptitle(
            f"Stitching Quality Check - Plate {self.plate}, Well {self.well}",
            fontsize=16,
        )

        # Store image data and display objects for slider updates
        image_data = {}
        display_objects = {}

        def adjust_image_display(img_array, brightness=1.0, contrast=1.0):
            """Apply brightness and contrast adjustments to image array."""
            # Normalize to 0-1 range first
            img_norm = (img_array - img_array.min()) / (
                img_array.max() - img_array.min() + 1e-8
            )
            # Apply contrast (multiply) then brightness (add)
            adjusted = np.clip(contrast * img_norm + (brightness - 1.0), 0, 1)
            return adjusted

        def update_display(val=None):
            """Update all image displays with current slider values."""
            brightness = brightness_slider.val
            contrast = contrast_slider.val

            for key, img_data in image_data.items():
                adjusted = adjust_image_display(img_data, brightness, contrast)
                display_objects[key].set_data(adjusted)

            fig.canvas.draw_idle()

        # Process phenotype image
        if self.phenotype_image.exists():
            # Memory map the array for efficiency
            ph_img = np.load(self.phenotype_image, mmap_mode="r")
            print(f"Phenotype image shape: {ph_img.shape}")
            print(f"Estimated size: {ph_img.nbytes / 1e9:.1f} GB")

            # Create downsampled preview
            ph_preview = ph_img[::preview_downsample, ::preview_downsample]
            image_data["ph_preview"] = ph_preview

            # Initial display with default brightness/contrast
            ph_preview_adj = adjust_image_display(ph_preview)
            display_objects["ph_preview"] = axes[0][0].imshow(
                ph_preview_adj, cmap="gray"
            )
            axes[0][0].set_title(
                f"Phenotype Full Well\n(downsampled {preview_downsample}x)"
            )
            axes[0][0].axis("off")

            # Handle sample region
            if sample_region:
                start_i, end_i, start_j, end_j = sample_region

                # Validate region bounds
                start_i = max(0, min(start_i, ph_img.shape[0]))
                end_i = max(start_i, min(end_i, ph_img.shape[0]))
                start_j = max(0, min(start_j, ph_img.shape[1]))
                end_j = max(start_j, min(end_j, ph_img.shape[1]))

                # Extract only the requested region
                ph_sample = np.array(ph_img[start_i:end_i, start_j:end_j])
                image_data["ph_sample"] = ph_sample

                ph_sample_adj = adjust_image_display(ph_sample)
                display_objects["ph_sample"] = axes[0][1].imshow(
                    ph_sample_adj, cmap="gray"
                )
                axes[0][1].set_title(
                    f"Phenotype Sample Region\n[{start_i}:{end_i}, {start_j}:{end_j}]\n"
                    f"Size: {ph_sample.shape}"
                )

                # Add rectangle to preview showing sample region
                rect = Rectangle(
                    (start_j // preview_downsample, start_i // preview_downsample),
                    (end_j - start_j) // preview_downsample,
                    (end_i - start_i) // preview_downsample,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                axes[0][0].add_patch(rect)
            else:
                # Show center region
                h, w = ph_img.shape
                center_h, center_w = h // 2, w // 2
                size = min(1000, min(h, w) // 8)

                start_i = center_h - size
                end_i = center_h + size
                start_j = center_w - size
                end_j = center_w + size

                ph_center = np.array(ph_img[start_i:end_i, start_j:end_j])
                image_data["ph_sample"] = ph_center

                ph_center_adj = adjust_image_display(ph_center)
                display_objects["ph_sample"] = axes[0][1].imshow(
                    ph_center_adj, cmap="gray"
                )
                axes[0][1].set_title(f"Phenotype Center Region\n{ph_center.shape}")

            axes[0][1].axis("off")

        # Process SBS image
        if self.sbs_image.exists():
            # Memory map the SBS array
            sbs_img = np.load(self.sbs_image, mmap_mode="r")
            print(f"SBS image shape: {sbs_img.shape}")
            print(f"Estimated size: {sbs_img.nbytes / 1e9:.1f} GB")

            # Create downsampled preview
            sbs_downsample = max(1, max(sbs_img.shape) // 1000)
            sbs_preview = sbs_img[::sbs_downsample, ::sbs_downsample]
            image_data["sbs_preview"] = sbs_preview

            sbs_preview_adj = adjust_image_display(sbs_preview)
            display_objects["sbs_preview"] = axes[1][0].imshow(
                sbs_preview_adj, cmap="gray"
            )
            axes[1][0].set_title(f"SBS Full Well\n(downsampled {sbs_downsample}x)")
            axes[1][0].axis("off")

            # Handle sample region for SBS
            if sample_region:
                start_i, end_i, start_j, end_j = sample_region

                # Calculate scale factor between phenotype and SBS
                if self.phenotype_image.exists():
                    scale_h = sbs_img.shape[0] / ph_img.shape[0]
                    scale_w = sbs_img.shape[1] / ph_img.shape[1]
                else:
                    scale_h = scale_w = 0.25  # Default assumption for 10x vs 40x

                # Scale region coordinates for SBS
                sbs_start_i = max(0, int(start_i * scale_h))
                sbs_end_i = min(sbs_img.shape[0], int(end_i * scale_h))
                sbs_start_j = max(0, int(start_j * scale_w))
                sbs_end_j = min(sbs_img.shape[1], int(end_j * scale_w))

                sbs_sample = np.array(
                    sbs_img[sbs_start_i:sbs_end_i, sbs_start_j:sbs_end_j]
                )
                image_data["sbs_sample"] = sbs_sample

                sbs_sample_adj = adjust_image_display(sbs_sample)
                display_objects["sbs_sample"] = axes[1][1].imshow(
                    sbs_sample_adj, cmap="gray"
                )
                axes[1][1].set_title(
                    f"SBS Sample Region\n[{sbs_start_i}:{sbs_end_i}, {sbs_start_j}:{sbs_end_j}]\n"
                    f"Size: {sbs_sample.shape}"
                )
            else:
                # Show center region
                h, w = sbs_img.shape
                center_h, center_w = h // 2, w // 2
                size = min(500, min(h, w) // 4)

                sbs_center = np.array(
                    sbs_img[
                        center_h - size : center_h + size,
                        center_w - size : center_w + size,
                    ]
                )
                image_data["sbs_sample"] = sbs_center

                sbs_center_adj = adjust_image_display(sbs_center)
                display_objects["sbs_sample"] = axes[1][1].imshow(
                    sbs_center_adj, cmap="gray"
                )
                axes[1][1].set_title(f"SBS Center Region\n{sbs_center.shape}")

            axes[1][1].axis("off")

        # Create brightness and contrast sliders
        brightness_slider = Slider(
            slider_ax1,
            "Brightness",
            brightness_range[0],
            brightness_range[1],
            valinit=1.0,
            valstep=0.05,
            valfmt="%.2f",
            facecolor="lightblue",
            edgecolor="black",
        )
        contrast_slider = Slider(
            slider_ax2,
            "Contrast",
            contrast_range[0],
            contrast_range[1],
            valinit=1.0,
            valstep=0.05,
            valfmt="%.2f",
            facecolor="lightgreen",
            edgecolor="black",
        )

        # Connect sliders to update function
        brightness_slider.on_changed(update_display)
        contrast_slider.on_changed(update_display)

        # Add text instructions
        fig.text(
            0.5,
            0.01,
            "Drag sliders to adjust brightness and contrast",
            ha="center",
            fontsize=12,
            style="italic",
        )

        plt.tight_layout()
        plt.show()

        # Keep references to prevent garbage collection
        fig._brightness_slider = brightness_slider
        fig._contrast_slider = contrast_slider

        return fig, brightness_slider, contrast_slider

    def check_stitching_quality_static(
        self,
        sample_region=None,
        preview_downsample=20,
        brightness_levels=[0.3, 0.7, 1.0, 1.5, 2.0],
    ):
        """Non-interactive stitching quality check with multiple brightness levels.

        This method provides a static visualization showing multiple brightness
        levels side by side. Use this if interactive sliders don't work in your
        environment.

        Parameters
        ----------
        sample_region : tuple, optional
            (start_i, end_i, start_j, end_j) region to examine at full resolution
        preview_downsample : int, default 20
            Downsampling factor for full well preview
        brightness_levels : list, default [0.3, 0.7, 1.0, 1.5, 2.0]
            Different brightness levels to display
        """
        # Handle single brightness level case
        if len(brightness_levels) == 1:
            fig, axes = plt.subplots(2, 1, figsize=(8, 10))
            axes = axes.reshape(2, 1)
        else:
            fig, axes = plt.subplots(
                2, len(brightness_levels), figsize=(4 * len(brightness_levels), 8)
            )
            if len(brightness_levels) == 1:
                axes = axes.reshape(2, 1)

        fig.suptitle(
            f"Brightness Comparison - Plate {self.plate}, Well {self.well}",
            fontsize=16,
        )

        def adjust_image_display(img_array, brightness=1.0):
            """Apply brightness adjustment to image array."""
            img_norm = (img_array - img_array.min()) / (
                img_array.max() - img_array.min() + 1e-8
            )
            adjusted = np.clip(img_norm * brightness, 0, 1)
            return adjusted

        # Process phenotype image
        if self.phenotype_image.exists():
            ph_img = np.load(self.phenotype_image, mmap_mode="r")

            if sample_region:
                start_i, end_i, start_j, end_j = sample_region
                start_i = max(0, min(start_i, ph_img.shape[0]))
                end_i = max(start_i, min(end_i, ph_img.shape[0]))
                start_j = max(0, min(start_j, ph_img.shape[1]))
                end_j = max(start_j, min(end_j, ph_img.shape[1]))
                ph_sample = np.array(ph_img[start_i:end_i, start_j:end_j])
                print(
                    f"Phenotype region: [{start_i}:{end_i}, {start_j}:{end_j}], "
                    f"shape: {ph_sample.shape}"
                )
            else:
                h, w = ph_img.shape
                center_h, center_w = h // 2, w // 2
                size = min(1000, min(h, w) // 8)
                ph_sample = np.array(
                    ph_img[
                        center_h - size : center_h + size,
                        center_w - size : center_w + size,
                    ]
                )

            # Show different brightness levels
            for i, brightness in enumerate(brightness_levels):
                adjusted = adjust_image_display(ph_sample, brightness)
                axes[0, i].imshow(adjusted, cmap="gray")
                axes[0, i].set_title(f"Phenotype\nBrightness: {brightness}")
                axes[0, i].axis("off")

        # Process SBS image if available
        if self.sbs_image.exists():
            sbs_img = np.load(self.sbs_image, mmap_mode="r")

            if sample_region:
                # Scale coordinates for SBS
                if self.phenotype_image.exists():
                    scale_h = sbs_img.shape[0] / ph_img.shape[0]
                    scale_w = sbs_img.shape[1] / ph_img.shape[1]
                else:
                    scale_h = scale_w = 0.25

                sbs_start_i = max(0, int(start_i * scale_h))
                sbs_end_i = min(sbs_img.shape[0], int(end_i * scale_h))
                sbs_start_j = max(0, int(start_j * scale_w))
                sbs_end_j = min(sbs_img.shape[1], int(end_j * scale_w))
                sbs_sample = np.array(
                    sbs_img[sbs_start_i:sbs_end_i, sbs_start_j:sbs_end_j]
                )
                print(
                    f"SBS region: [{sbs_start_i}:{sbs_end_i}, {sbs_start_j}:{sbs_end_j}], "
                    f"shape: {sbs_sample.shape}"
                )
            else:
                h, w = sbs_img.shape
                center_h, center_w = h // 2, w // 2
                size = min(500, min(h, w) // 4)
                sbs_sample = np.array(
                    sbs_img[
                        center_h - size : center_h + size,
                        center_w - size : center_w + size,
                    ]
                )

            # Show different brightness levels
            for i, brightness in enumerate(brightness_levels):
                adjusted = adjust_image_display(sbs_sample, brightness)
                axes[1, i].imshow(adjusted, cmap="gray")
                axes[1, i].set_title(f"SBS\nBrightness: {brightness}")
                axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()

    def view_region(self, center_row, center_col, size=1000, brightness=2.0):
        """View a square region centered at specified coordinates.

        This is a convenience method for quickly examining specific regions
        of stitched images with adjustable brightness.

        Parameters
        ----------
        center_row : int
            Center row coordinate of region to view
        center_col : int
            Center column coordinate of region to view
        size : int, default 1000
            Size of square region (will be size x size pixels)
        brightness : float, default 2.0
            Brightness multiplier for display
        """
        half_size = size // 2
        start_i = center_row - half_size
        end_i = center_row + half_size
        start_j = center_col - half_size
        end_j = center_col + half_size

        print(
            f"Viewing {size}x{size} region centered at ({center_row}, {center_col}) "
            f"with brightness {brightness}"
        )

        # Use static version with single brightness level
        self.check_stitching_quality_static(
            sample_region=(start_i, end_i, start_j, end_j),
            brightness_levels=[brightness],
        )


# Enable interactive mode
plt.ion()


def quick_qc(base_path, plate, well):
    """Perform quick quality control check for a single well.

    This function provides a streamlined QC workflow that checks file status,
    displays overlays, and analyzes cell positions for a single well.

    Parameters
    ----------
    base_path : str
        Path to analysis outputs
    plate : str or int
        Plate identifier
    well : str
        Well identifier

    Returns:
    -------
    tuple
        (phenotype_positions, sbs_positions) DataFrames
    """
    qc = StitchQC(base_path, plate, well)
    qc.view_overlays()
    return qc.analyze_cell_positions()


def batch_qc_report(base_path, plate_wells):
    """Generate batch QC reports for multiple wells.

    This function processes multiple wells and generates a summary report
    showing file availability, cell counts, and processing status for each well.

    Parameters
    ----------
    base_path : str
        Path to analysis outputs
    plate_wells : list of tuples
        List of (plate, well) tuples to process, e.g., [(1, 'A01'), (1, 'A02')]

    Returns:
    -------
    pd.DataFrame
        Summary dataframe with QC statistics for each well
    """
    print("BATCH QC REPORT")
    print("=" * 60)

    summary = []

    for plate, well in plate_wells:
        qc = StitchQC(base_path, plate, well)

        # Quick file check
        ph_exists = qc.phenotype_positions.exists()
        sbs_exists = qc.sbs_positions.exists()

        ph_count = 0
        sbs_count = 0

        if ph_exists:
            ph_count = len(pd.read_parquet(qc.phenotype_positions))
        if sbs_exists:
            sbs_count = len(pd.read_parquet(qc.sbs_positions))

        summary.append(
            {
                "plate": plate,
                "well": well,
                "ph_exists": ph_exists,
                "sbs_exists": sbs_exists,
                "ph_cells": ph_count,
                "sbs_cells": sbs_count,
                "status": "OK"
                if ph_exists and sbs_exists and ph_count > 0 and sbs_count > 0
                else "ISSUE",
            }
        )

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    return summary_df

def create_tile_arrangement_qc_plot(
    cell_positions_df: pd.DataFrame, output_path: str, data_type: str = "phenotype"
):
    """Create QC plot showing tile arrangement and cell distribution."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{data_type.title()} Tile Arrangement QC", fontsize=16)

    # Use the correct column for preserved tile mapping
    tile_column = (
        "original_tile_id"
        if "original_tile_id" in cell_positions_df.columns
        else "tile"
    )

    # Plot 1: Cell positions colored by tile
    ax1 = axes[0, 0]
    if tile_column in cell_positions_df.columns:
        scatter = ax1.scatter(
            cell_positions_df["j"],
            cell_positions_df["i"],
            c=cell_positions_df[tile_column],
            cmap="tab20",
            s=0.1,
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax1, label="Original Tile ID")
    ax1.set_title("Cell Positions by Original Tile")
    ax1.set_xlabel("X Position (pixels)")
    ax1.set_ylabel("Y Position (pixels)")
    ax1.invert_yaxis()

    # Plot 2: Stage coordinates with tile numbers
    ax2 = axes[0, 1]
    if (
        "stage_x" in cell_positions_df.columns
        and "stage_y" in cell_positions_df.columns
    ):
        tile_info = (
            cell_positions_df.groupby(tile_column)
            .agg({"stage_x": "first", "stage_y": "first"})
            .reset_index()
        )

        ax2.scatter(tile_info["stage_x"], tile_info["stage_y"], s=50)
        for _, row in tile_info.iterrows():
            if not pd.isna(row["stage_x"]) and not np.isnan(row["stage_x"]):
                ax2.annotate(
                    f"{int(row[tile_column])}",
                    (row["stage_x"], row["stage_y"]),
                    fontsize=8,
                    ha="center",
                )
    ax2.set_title("Tile Arrangement (Stage Coordinates)")
    ax2.set_xlabel("Stage X (μm)")
    ax2.set_ylabel("Stage Y (μm)")

    # Plot 3: Cells per tile histogram
    ax3 = axes[1, 0]
    if tile_column in cell_positions_df.columns:
        tile_counts = cell_positions_df[tile_column].value_counts()
        ax3.hist(tile_counts.values, bins=50, alpha=0.7)
        ax3.axvline(
            tile_counts.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {tile_counts.mean():.1f}",
        )
        ax3.legend()
    ax3.set_title("Distribution of Cells per Original Tile")
    ax3.set_xlabel("Cells per Tile")
    ax3.set_ylabel("Number of Tiles")

    # Plot 4: Tile boundaries overlay
    ax4 = axes[1, 1]
    if "tile_i" in cell_positions_df.columns and "tile_j" in cell_positions_df.columns:
        ax4.scatter(
            cell_positions_df["tile_j"],
            cell_positions_df["tile_i"],
            c=cell_positions_df[tile_column],
            cmap="tab20",
            s=0.1,
            alpha=0.7,
        )
    ax4.set_title("Relative Positions within Original Tiles")
    ax4.set_xlabel("Tile-relative X")
    ax4.set_ylabel("Tile-relative Y")
    ax4.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"QC plot saved to: {output_path}")


def verify_cell_id_preservation(
    enhanced_positions: pd.DataFrame,
    original_data_path: str,
    well: str,
    data_type: str = "phenotype",
) -> None:
    """
    Verify the quality of cell ID preservation by comparing with original data.
    Call this after running the enhanced pipeline to check quality.
    """
    print(f"\n=== VERIFYING CELL ID PRESERVATION ===")
    print(f"Well: {well}, Data type: {data_type}")

    try:
        # Load original data
        if Path(original_data_path).exists():
            original_data = pd.read_parquet(original_data_path)
            print(f"✅ Loaded original data: {len(original_data)} cells")
        else:
            print(f"❌ Original data not found: {original_data_path}")
            return

        # Overall statistics
        total_stitched = len(enhanced_positions)
        direct_mapped = len(
            enhanced_positions[enhanced_positions["mapping_method"] == "direct_mapping"]
        )
        position_estimated = len(
            enhanced_positions[
                enhanced_positions["mapping_method"] == "position_estimate"
            ]
        )

        print(f"\n📊 Overall Statistics:")
        print(f"  Original cells: {len(original_data)}")
        print(f"  Stitched cells: {total_stitched}")
        print(
            f"  Direct mappings: {direct_mapped} ({direct_mapped / total_stitched * 100:.1f}%)"
        )
        print(
            f"  Position estimates: {position_estimated} ({position_estimated / total_stitched * 100:.1f}%)"
        )

        # Per-tile analysis
        print(f"\n🔍 Per-Tile Analysis:")
        tile_comparison = []

        for tile_id in enhanced_positions["original_tile_id"].unique():
            if pd.isna(tile_id) or tile_id == -1:
                continue

            tile_id = int(tile_id)

            # Count in stitched data
            stitched_count = len(
                enhanced_positions[enhanced_positions["original_tile_id"] == tile_id]
            )

            # Count in original data
            original_count = len(original_data[original_data["tile"] == tile_id])

            recovery_rate = stitched_count / original_count if original_count > 0 else 0

            tile_comparison.append(
                {
                    "tile": tile_id,
                    "original": original_count,
                    "stitched": stitched_count,
                    "recovery": recovery_rate,
                }
            )

        if tile_comparison:
            tile_df = pd.DataFrame(tile_comparison)
            mean_recovery = tile_df["recovery"].mean()
            good_tiles = len(tile_df[tile_df["recovery"] > 0.8])

            print(f"  Mean recovery rate: {mean_recovery:.3f}")
            print(f"  Tiles with >80% recovery: {good_tiles}/{len(tile_df)}")
            print(
                f"  Total original cells accounted: {tile_df['stitched'].sum()}/{tile_df['original'].sum()}"
            )

            # Show worst performing tiles
            worst_tiles = tile_df.nsmallest(3, "recovery")
            if len(worst_tiles) > 0:
                print(f"\n⚠️  Tiles needing attention:")
                for _, row in worst_tiles.iterrows():
                    print(
                        f"    Tile {row['tile']}: {row['stitched']}/{row['original']} ({row['recovery']:.3f})"
                    )

        print(f"\n✅ Verification complete!")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()


def estimate_stitch_sbs_coordinate_based(
    metadata_df: pd.DataFrame,
    well: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
) -> Dict[str, Dict]:
    """Coordinate-based stitching for SBS data with correct scaling."""

    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        print(f"No SBS tiles found for well {well}")
        return {"total_translation": {}, "confidence": {well: {}}}

    coords = well_metadata[["x_pos", "y_pos"]].values
    tile_ids = well_metadata["tile"].values
    tile_size = (1200, 1200)  # SBS tile size in pixels

    print(f"Creating coordinate-based SBS stitch config for {len(tile_ids)} tiles")

    # Use proven spacing detection
    from scipy.spatial.distance import pdist

    distances = pdist(coords)
    actual_spacing = np.percentile(distances[distances > 0], 10)  # 10th percentile

    print(f"SBS detected spacing: {actual_spacing:.1f} μm")
    print(f"SBS tile size: {tile_size} pixels")

    # FIXED: Use pixel size from metadata
    if (
        "pixel_size_x" in well_metadata.columns
        and "pixel_size_y" in well_metadata.columns
    ):
        # Use pixel size from metadata (in μm per pixel)
        pixel_size_um = well_metadata["pixel_size_x"].iloc[0]  # μm per pixel
        pixels_per_micron = 1.0 / pixel_size_um  # pixels per μm
        print(f"SBS pixel size from metadata: {pixel_size_um:.6f} μm/pixel")
        print(f"SBS pixels per micron: {pixels_per_micron:.4f}")

        # Verify pixel_size_y matches pixel_size_x
        pixel_size_y = well_metadata["pixel_size_y"].iloc[0]
        if abs(pixel_size_um - pixel_size_y) > 1e-6:
            print(
                f"⚠️  Warning: pixel_size_x ({pixel_size_um:.6f}) != pixel_size_y ({pixel_size_y:.6f})"
            )
    else:
        print(
            "⚠️  pixel_size_x/y not found in metadata, falling back to calculated values"
        )
        # Fallback: SBS specs: 1560 μm field of view, 1200 pixels -> 0.7692 pixels/μm
        sbs_field_of_view_um = 1560.0  # μm
        pixels_per_micron = tile_size[0] / sbs_field_of_view_um
        print(f"SBS fallback - field of view: {sbs_field_of_view_um} μm")
        print(f"SBS fallback - pixels per micron: {pixels_per_micron:.4f}")

    x_min, y_min = coords.min(axis=0)

    total_translation = {}
    confidence = {}

    for i, tile_id in enumerate(tile_ids):
        x_pos, y_pos = coords[i]

        # Convert directly to pixel coordinates using correct scale
        pixel_x = int((x_pos - x_min) * pixels_per_micron)
        pixel_y = int((y_pos - y_min) * pixels_per_micron)

        total_translation[f"{well}/{tile_id}"] = [pixel_y, pixel_x]

        # High confidence since using direct coordinates
        confidence[f"coord_{i}"] = [[pixel_y, pixel_x], [pixel_y, pixel_x], 0.9]

    print(f"Generated {len(total_translation)} SBS coordinate-based positions")

    # Verify output size and spacing
    y_shifts = [shift[0] for shift in total_translation.values()]
    x_shifts = [shift[1] for shift in total_translation.values()]

    if len(y_shifts) > 1:
        pixel_spacings = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                stage_dist = np.sqrt(
                    (coords[i][0] - coords[j][0]) ** 2
                    + (coords[i][1] - coords[j][1]) ** 2
                )
                pixel_dist = np.sqrt(
                    (y_shifts[i] - y_shifts[j]) ** 2 + (x_shifts[i] - x_shifts[j]) ** 2
                )
                if stage_dist > 0:
                    pixel_spacings.append(pixel_dist / stage_dist)

        if pixel_spacings:
            avg_pixel_spacing = np.mean(pixel_spacings)
            print(
                f"Verification - Average pixel spacing ratio: {avg_pixel_spacing:.4f} pixels/μm"
            )

            # Check for expected tile overlap
            expected_tile_spacing_pixels = actual_spacing * pixels_per_micron
            actual_avg_spacing = np.mean(
                [
                    np.sqrt(
                        (y_shifts[i] - y_shifts[j]) ** 2
                        + (x_shifts[i] - x_shifts[j]) ** 2
                    )
                    for i in range(len(y_shifts))
                    for j in range(i + 1, len(y_shifts))
                ]
            )

            if actual_avg_spacing > 0:
                overlap_percent = (
                    (tile_size[0] - actual_avg_spacing) / tile_size[0] * 100
                )
                print(f"SBS tile overlap: {overlap_percent:.1f}%")
                if overlap_percent < 0:
                    print("⚠️  Warning: Negative overlap detected - tiles may have gaps")
                elif overlap_percent > 50:
                    print(
                        "⚠️  Warning: Very high overlap detected - may indicate scaling issues"
                    )

    final_size = (max(y_shifts) + tile_size[0], max(x_shifts) + tile_size[1])
    memory_gb = final_size[0] * final_size[1] * 2 / 1e9

    print(f"SBS final image size: {final_size}")
    print(f"SBS memory estimate: {memory_gb:.1f} GB")

    return {"total_translation": total_translation, "confidence": {well: confidence}}
