#!/usr/bin/env python3
"""
Load BrieFlow OME-Zarr outputs in Napari
Handles both images and labels correctly.
"""

import napari
import zarr
import numpy as np
from pathlib import Path


def load_omezarr_to_napari(zarr_path: str):
    """
    Load OME-Zarr (image + labels) into Napari viewer.
    
    Args:
        zarr_path: Path to .zarr directory
        
    Raises:
        FileNotFoundError: If zarr path doesn't exist
        ValueError: If zarr structure is invalid
    """
    viewer = napari.Viewer()
    zarr_path = Path(zarr_path)
    
    # Validate path exists
    if not zarr_path.exists():
        raise FileNotFoundError(
            f"OME-Zarr path does not exist: {zarr_path}\n"
            f"Please check the path and try again."
        )
    
    if not zarr_path.is_dir():
        raise ValueError(
            f"Path is not a directory: {zarr_path}\n"
            f"OME-Zarr files should be directories with .zarr extension."
        )
    
    # Open the zarr group
    try:
        root = zarr.open_group(str(zarr_path), mode='r')
    except Exception as e:
        raise ValueError(
            f"Failed to open zarr group: {zarr_path}\n"
            f"Error: {e}\n"
            f"This may not be a valid Zarr file."
        )
    
    # Get multiscales metadata
    multiscales = root.attrs.get('multiscales', [{}])[0]
    if not multiscales:
        raise ValueError(
            f"No multiscales metadata found in {zarr_path}\n"
            f"This may not be a valid OME-Zarr file."
        )
    
    axes_info = multiscales.get('axes', [])
    axes_names = [ax['name'] for ax in axes_info]
    
    if not axes_names:
        raise ValueError(
            f"No axes information found in {zarr_path}\n"
            f"OME-Zarr files must specify axes (e.g., 'c', 'y', 'x')."
        )
    
    # Load the highest resolution image (scale 0)
    try:
        image_data = root['0'][:]
    except KeyError:
        raise ValueError(
            f"Scale '0' not found in {zarr_path}\n"
            f"OME-Zarr files must contain at least one resolution level named '0'."
        )
    except Exception as e:
        raise ValueError(
            f"Failed to load image data from {zarr_path}\n"
            f"Error: {e}"
        )
    
    # Get pixel size if available
    pixel_size = root.attrs.get('omero', {}).get('pixel_size', {})
    
    # Get channel names if available
    channel_axis = next((i for i, ax in enumerate(axes_names) if ax == 'c'), None)
    
    # Build scale - exclude channel axis if present
    scale = []
    if pixel_size:
        # Build scale based on axes (excluding channel axis for napari)
        for i, ax in enumerate(axes_names):
            if i == channel_axis:
                continue  # Skip channel axis
            if ax in ['x', 'y']:
                scale.append(pixel_size.get(ax, 1.0))
            else:
                scale.append(1.0)
    else:
        # Default scale (excluding channel if present)
        scale = [1.0] * (len(axes_names) - (1 if channel_axis is not None else 0))
    
    if channel_axis is not None:
        # This is a multichannel image
        n_channels = image_data.shape[channel_axis]
        
        # Try to get channel names from omero metadata
        omero_channels = root.attrs.get('omero', {}).get('channels', [])
        if omero_channels:
            channel_names = [ch.get('label', f'Channel {i}') for i, ch in enumerate(omero_channels)]
        else:
            channel_names = [f'Channel {i}' for i in range(n_channels)]
        
        print(f"Loading {n_channels} channels: {channel_names}")
        print(f"Image shape: {image_data.shape}")
        print(f"Axes: {axes_names}")
        print(f"Scale (excluding channel): {scale}")
        
        # Generate colormaps for all channels
        # Use a cycling list of colormaps to handle any number of channels
        base_colormaps = ['gray', 'green', 'red', 'cyan', 'magenta', 'yellow', 'blue']
        colormaps = [base_colormaps[i % len(base_colormaps)] for i in range(n_channels)]
        
        # Calculate contrast limits per channel for better visualization
        contrast_limits = []
        for i in range(n_channels):
            # Get data for this channel
            channel_data = np.take(image_data, i, axis=channel_axis)
            # Use percentile-based contrast for better visualization
            vmin = np.percentile(channel_data, 1)
            vmax = np.percentile(channel_data, 99.5)
            contrast_limits.append((vmin, vmax))
        
        # Add as multichannel image
        viewer.add_image(
            image_data,
            name=zarr_path.stem,
            channel_axis=channel_axis,
            scale=scale,  # Scale excludes channel dimension
            colormap=colormaps,
            blending='additive',
            contrast_limits=contrast_limits,
            interpolation2d='nearest'  # Use nearest neighbor for sharpest display
        )
        
        print(f"Contrast limits applied per channel for optimal display")
    else:
        # Single channel image
        print(f"Loading single channel image")
        print(f"Image shape: {image_data.shape}")
        
        # Calculate contrast limits for better visualization
        vmin = np.percentile(image_data, 1)
        vmax = np.percentile(image_data, 99.5)
        
        viewer.add_image(
            image_data,
            name=zarr_path.stem,
            scale=scale,
            colormap='gray',
            contrast_limits=(vmin, vmax),
            interpolation2d='nearest'  # Use nearest neighbor for sharpest display
        )
    
    # Load labels if they exist
    labels_path = zarr_path / 'labels'
    if labels_path.exists():
        labels_group = zarr.open_group(str(labels_path), mode='r')
        label_names = labels_group.attrs.get('labels', [])
        
        print(f"\nFound {len(label_names)} label groups: {label_names}")
        
        for label_name in label_names:
            try:
                label_group = labels_group[label_name]
                # Get highest resolution labels (scale 0)
                label_data = label_group['0'][:]
                
                # Get label-specific metadata
                label_multiscales = label_group.attrs.get('multiscales', [{}])[0]
                label_axes = label_multiscales.get('axes', [])
                label_axes_names = [ax['name'] for ax in label_axes]
                
                # Build scale for labels (usually YX, no channel axis)
                label_scale = []
                for ax in label_axes_names:
                    if ax in ['x', 'y']:
                        label_scale.append(pixel_size.get(ax, 1.0))
                    else:
                        label_scale.append(1.0)
                
                print(f"  Loading '{label_name}': shape {label_data.shape}, {len(np.unique(label_data))-1} objects")
                
                viewer.add_labels(
                    label_data,
                    name=label_name,
                    scale=label_scale
                )
            except Exception as e:
                print(f"  Warning: Could not load label '{label_name}': {e}")
    
    print(f"\n Loaded {zarr_path.name} successfully!")
    return viewer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        zarr_path = sys.argv[1]
    else:
        print(" Error: No zarr path provided")
        print("\nUsage: python load_omezarr_in_napari.py <path_to_zarr_file>")
        print("\nExample:")
        print("  python load_omezarr_in_napari.py ./output/phenotype/omezarr/sample.zarr")
        sys.exit(1)
    
    try:
        print(f"Loading OME-Zarr: {zarr_path}\n")
        viewer = load_omezarr_to_napari(zarr_path)
        print("\n Successfully loaded! Close the Napari window when done.")
        napari.run()
    except FileNotFoundError as e:
        print(f"\nFile Error:\n{e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nValidation Error:\n{e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\nImport Error:\n{e}")
        print("\nMake sure you have installed napari and zarr:")
        print("  pip install napari zarr")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected Error:\n{e}")
        print("\nIf this persists, please report the issue with:")
        print(f"  - Zarr path: {zarr_path}")
        print(f"  - Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

