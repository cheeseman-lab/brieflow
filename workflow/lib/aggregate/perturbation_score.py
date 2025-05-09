import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def get_top_differential_features(
    cell_data,
    feature_cols,
    gene,
    pert_col,
    control_key,
    n_features: int = 200,
):
    """Get the top differentially expressed features between control cells and `gene`.
    """
    control_mask = cell_data[pert_col].str.startswith(control_key)
    gene_mask = cell_data[pert_col] == gene

    control_vals = cell_data.loc[control_mask, feature_cols].to_numpy(dtype=np.float32)
    gene_vals = cell_data.loc[gene_mask, feature_cols].to_numpy(dtype=np.float32)

    _, pvals = stats.ttest_ind(control_vals, gene_vals, axis=0, equal_var=False)

    top_idx = np.argsort(pvals)[:n_features]
    return [feature_cols[i] for i in top_idx]


def get_perturbation_scores(cell_data, gene, diff_exp_features, pert_col, control_key):
    """Compute perturbation scores for a given gene using a projection-based method."""

    feature_data = cell_data[diff_exp_features]
    gene_mask = cell_data[pert_col] == gene
    control_mask = cell_data[pert_col].str.startswith(control_key)

    gene_features = feature_data[gene_mask]
    control_features = feature_data[control_mask]

    # Compute signature vector (beta)
    beta = (gene_features.mean() - control_features.mean()).to_numpy()
    norm_squared = beta @ beta

    # Project all cells onto the signature
    projection = (feature_data @ beta) / norm_squared

    # Fit linear model to binary label
    binary_labels = gene_mask.astype(int)
    lin_model = LinearRegression().fit(projection.values.reshape(-1, 1), binary_labels)
    raw_scores = lin_model.predict(projection.values.reshape(-1, 1))

    # Scale scores to [0, 1]
    scaled_scores = (raw_scores - raw_scores.min()) / (
        raw_scores.max() - raw_scores.min()
    )
    return scaled_scores
