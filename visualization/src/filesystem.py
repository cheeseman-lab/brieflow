import glob
import os
import re
import pandas as pd
import streamlit as st


class FileSystem:
    """
    Utility class for file system operations related to evaluation files.
    """

    @staticmethod
    @st.cache_data
    def find_files(root_dir, include_any=None, include_all=None, extensions=None):
        """
        Find all files with specified extensions in the directory tree with optional path filtering.

        Args:
            root_dir: The root directory to search in
            includes_any: List of strings where if the path includes any of the values, it is included
            include_all: List of strings where the path must include each and every element
            extensions: List of file extensions to search for (default: ['png', 'tsv'])

        Returns:
            A list of file paths that match the filtering criteria
        """
        # Use default extensions if none provided
        if extensions is None:
            extensions = ["png", "tsv", "zarr"]

        # Find all files with specified extensions
        all_files = []
        for ext in extensions:
            files = glob.glob(f"{root_dir}/**/*.{ext}", recursive=True)
            all_files.extend(files)

        # Apply filtering if specified
        filtered_files = all_files

        # Filter for paths that include any of the specified strings
        if include_any and len(include_any) > 0:
            filtered_files = [
                f
                for f in filtered_files
                if any(
                    item in os.path.normpath(f).split(os.sep) for item in include_any
                )
            ]

        # Filter for paths that include all of the specified strings
        if include_all and len(include_all) > 0:
            filtered_files = [
                f
                for f in filtered_files
                if all(
                    item in os.path.normpath(f).split(os.sep) for item in include_all
                )
            ]

        # Return the array of matching file paths
        return filtered_files

    @staticmethod
    def extract_well_id(file_path):
        r"""
        Extract well identifier (format: W-[A-Z]\d+) from a file path.

        Args:
            file_path: Path to the file

        Returns:
            Well ID if found, None otherwise
        """
        match = re.search(r"W-([A-Z]\d+)", file_path)
        return match.group(1) if match else None

    @staticmethod
    def extract_plate_id(file_path):
        r"""
        Extract plate identifier (format: P-\d+) from a file path.

        Args:
            file_path: Path to the file

        Returns:
            Plate ID if found, None otherwise
        """
        match = re.search(r"P-(\d+)", file_path)
        return match.group(1) if match else None

    @staticmethod
    def extract_metric_name(file_path):
        """
        Extract metric name from the basename (the part after the last "__").

        Args:
            file_path: Path to the file

        Returns:
            Metric name if found, None otherwise
        """
        base = os.path.splitext(os.path.basename(file_path))[0]
        if "__" in base:
            return base.split("__")[-1]
        return None

    @staticmethod
    def extract_leiden_resolution(file_path):
        r"""
        Extract Leiden resolution (format: LR-\d+) from a file path.

        Args:
            file_path: Path to the file

        Returns:
            Leiden resolution if found, None otherwise
        """
        match = re.search(r"(LR-\d+)", file_path)
        return match.group(1) if match else None

    @staticmethod
    def extract_features(root_dir, files):
        """
        Extract features from PNG and TSV files including path information.

        Args:
            root_dir: Root directory to use as base for relative paths
            files: List of file paths to process
            omit_folders: Set of folder names to omit from directory levels (default: {'eval'})

        Returns:
            DataFrame containing extracted features and path information
        """
        features = []

        for file in files:
            # Convert to relative path
            rel_path = os.path.relpath(file, root_dir)
            dirname = os.path.dirname(rel_path)
            basename = os.path.basename(file)
            name, ext = os.path.splitext(basename)

            # Basic feature dictionary with new fields
            feature = {
                "file_path": rel_path,
                "dir": dirname,
                "basename": name,
                "ext": ext.lstrip("."),  # Remove the leading dot from extension
            }

            # Extract identifiers and metadata from file path
            feature["well_id"] = FileSystem.extract_well_id(rel_path)
            feature["plate_id"] = FileSystem.extract_plate_id(rel_path)
            feature["metric_name"] = FileSystem.extract_metric_name(rel_path)
            # feature['leiden_resolution'] = FileSystem.extract_leiden_resolution(rel_path)

            # Add directory levels, skipping omitted folders
            parts = dirname.split(os.sep)
            for i, part in enumerate(parts):
                feature[f"dir_level_{i}"] = part
            features.append(feature)

        return pd.DataFrame(features)
