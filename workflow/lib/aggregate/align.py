"""This module provides functions for data preparation, PCA analysis, and normalization.

It includes utilities for aligning data, performing PCA transformations, and applying
typical variation normalization (TVN) on embeddings based on control perturbation units.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import linalg


def prepare_alignment_data(
    metadata, features, batch_cols, pert_col, control_key, pert_id_col
):
    """Prepare batch values and split metadata and feature DataFrames.

    Args:
        metadata (pd.DataFrame): Input DataFrame containing metadata.
        features (pd.DataFrame): DataFrame containing feature data.
        batch_cols (list): List of column names used to generate batch values.
        pert_col (str): Column name representing perturbation labels.
        control_key (str): Key for identifying control samples in the metadata.
        pert_id_col (str): Column name for perturbation IDs.

    Returns:
        tuple: metadata (pd.DataFrame), features (numpy.ndarray)
    """
    metadata = metadata.copy()

    # Create batch values
    batch_values = metadata[batch_cols[0]].astype(str)
    for col in batch_cols[1:]:
        batch_values = batch_values + "_" + metadata[col].astype(str)

    # Add batch values to metadata
    metadata["batch_values"] = batch_values

    # Add unique number suffix to perturbation names based on pert_id_col
    if control_key is not None and pert_col is not None and pert_id_col is not None:
        control_mask = metadata[pert_col] == control_key
        metadata.loc[control_mask, pert_col] = (
            control_key + "_" + metadata.loc[control_mask, pert_id_col].astype(str)
        )

    # Extract feature data
    features = features.to_numpy()

    return metadata, features


def pca_variance_plot(features, variance_threshold=0.95, random_state=42):
    """Perform PCA analysis and create an explained variance plot.

    Args:
        features (np.ndarray): Array containing feature data to be analyzed.
        variance_threshold (float): Cumulative variance threshold. Defaults to 0.95.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - n_components (int): Number of components needed to reach the variance threshold.
            - fig (matplotlib.figure.Figure): Figure object for the explained variance plot.
    """
    # Copy and scale data
    features = features.copy()
    features = centerscale_by_batch(features)

    # Initialize and fit PCA
    pca = PCA(random_state=random_state)
    pca_transformed = pca.fit_transform(features)

    # Create DataFrame with PCA results
    n_components_total = pca_transformed.shape[1]
    pca_df = pd.DataFrame(
        pca_transformed,
        columns=[f"pca_{n}" for n in range(n_components_total)],
    )

    # Find number of components needed for threshold
    cumsum = pca.explained_variance_ratio_.cumsum()
    n_components = np.argwhere(cumsum >= variance_threshold)[0][0] + 1

    # Create variance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumsum, "-")
    ax.axhline(
        variance_threshold,
        linestyle="--",
        color="red",
        label=f"{variance_threshold * 100}% Threshold",
    )
    ax.axvline(n_components, linestyle="--", color="blue", label=f"n={n_components}")
    ax.set_ylabel("Cumulative fraction of variance explained")
    ax.set_xlabel("Number of principal components included")
    ax.set_title("PCA Explained Variance Ratio")
    ax.grid(True)
    ax.legend()

    return n_components, fig


def embed_by_pca(
    features: np.ndarray,
    metadata: pd.DataFrame = None,
    variance_or_ncomp=128,
    batch_col: str | None = None,
) -> np.ndarray:
    """Embed the whole input data using principal component analysis (PCA).

    Note that we explicitly center & scale the data (by batch) before an embedding operation with `PCA`.
    Centering and scaling is done by batch if `batch_col` is not None, and on the whole data otherwise.
    Also note that `PCA` transformer also does mean-centering on the whole data prior to the PCA operation.

    Args:
        features (np.ndarray): Features to transform
        metadata (pd.DataFrame): Metadata. Defaults to None.
        variance_or_ncomp (float, optional): Variance or number of components to keep after PCA.
            Defaults to 128 (n_components). If between 0 and 1, select the number of components such that
            the amount of variance that needs to be explained is greater than the percentage specified.
            If 1, a single component is kept, and if None, all components are kept.
        batch_col (str, optional): Column name for batch information. Defaults to None.

    Returns:
        np.ndarray: Transformed data using PCA.
    """
    features = centerscale_by_batch(features, metadata, batch_col)
    if features.shape[0] > 50000:
        sample = features[:50000]
        pca_estimate = PCA(n_components=variance_or_ncomp).fit(sample)
        variance_or_ncomp = pca_estimate.n_components_
    features = PCA(n_components=variance_or_ncomp).fit_transform(features)
    return features


def tvn_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: str,
    batch_col: str | None = None,
) -> np.ndarray:
    """Apply TVN (Typical Variation Normalization) to the data based on the control perturbation units.

    Note that the data is first centered and scaled based on the control units.

    Args:
        embeddings (np.ndarray): The embeddings to be normalized.
        metadata (pd.DataFrame): The metadata containing information about the samples.
        pert_col (str): The column name in the metadata DataFrame that represents the perturbation labels.
        control_key (str): The control perturbation label.
        batch_col (str, optional): Column name in the metadata DataFrame representing the batch labels
            to be used for CORAL normalization. Defaults to None.

    Returns:
        np.ndarray: The normalized embeddings.
    """
    embeddings = centerscale_on_controls(embeddings, metadata, pert_col, control_key)
    ctrl_ind = metadata[pert_col].str.startswith(control_key).to_list()
    embeddings = PCA().fit(embeddings[ctrl_ind]).transform(embeddings)
    embeddings = centerscale_on_controls(
        embeddings, metadata, pert_col, control_key, batch_col
    )
    target_cov = np.cov(embeddings[ctrl_ind], rowvar=False, ddof=1) + 0.5 * np.eye(
        embeddings.shape[1]
    )
    if batch_col is not None:
        batches = metadata[batch_col].unique()
        for batch in batches:
            batch_ind = metadata[batch_col] == batch
            batch_control_ind = (
                batch_ind & (metadata[pert_col].str.startswith(control_key)).to_list()
            )
            source_cov = np.cov(
                embeddings[batch_control_ind], rowvar=False, ddof=1
            ) + 0.5 * np.eye(embeddings.shape[1])
            embeddings[batch_ind] = np.matmul(
                embeddings[batch_ind], linalg.fractional_matrix_power(source_cov, -0.5)
            )
            embeddings[batch_ind] = np.matmul(
                embeddings[batch_ind], linalg.fractional_matrix_power(target_cov, 0.5)
            )
    return embeddings


def centerscale_by_batch(
    features: np.ndarray, metadata: pd.DataFrame = None, batch_col: str | None = None
) -> np.ndarray:
    """Center and scale the input features by each batch. Not using any controls at all.

    We are using this prior to embedding high-dimensional data with PCA.

    Args:
        features (np.ndarray): Input features to be centered and scaled.
        metadata (pd.DataFrame): Metadata information for the input features.
        batch_col (str): Name of the column in metadata that contains batch information.

    Returns:
        np.ndarray: Centered and scaled features.
    """
    if batch_col is None:
        features = StandardScaler(copy=False).fit_transform(features)
    else:
        if metadata is None:
            raise ValueError("metadata must be provided if batch_col is not None")
        batches = metadata[batch_col].unique()
        for batch in batches:
            ind = metadata[batch_col] == batch
            features[ind, :] = StandardScaler(copy=False).fit_transform(
                features[ind, :]
            )
    return features


def centerscale_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: str,
    batch_col: str | None = None,
) -> np.ndarray:
    """Center and scale the embeddings on the control perturbation units in the metadata.

    If batch information is provided, the embeddings are centered and scaled by batch.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aligned.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        pert_col (str, optional): The column in the metadata containing perturbation information.
        control_key (str, optional): The key for non-targeting controls in the metadata.
        batch_col (str, optional): Column name in the metadata representing the batch labels.
            Defaults to None.

    Returns:
        numpy.ndarray: The aligned embeddings.
    """
    if batch_col is not None:
        batches = metadata[batch_col].unique()
        for batch in batches:
            batch_ind = metadata[batch_col] == batch
            batch_control_ind = (
                batch_ind & (metadata[pert_col].str.startswith(control_key)).to_list()
            )
            embeddings[batch_ind] = (
                StandardScaler(copy=False)
                .fit(embeddings[batch_control_ind])
                .transform(embeddings[batch_ind])
            )
        return embeddings

    control_ind = metadata[pert_col].str.startswith(control_key).to_list()
    return StandardScaler(copy=False).fit(embeddings[control_ind]).transform(embeddings)
