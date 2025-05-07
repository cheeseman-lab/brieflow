import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def get_top_differential_features(cell_data, feature_cols, gene, pert_col, control_key, n_features=100):
    """Get top differentially expressed features between control and gene using t-test"""
    results = []

    control_cells = cell_data[cell_data[pert_col].str.startswith(control_key)]
    gene_cells = cell_data[cell_data[pert_col] == gene]

    for feature in feature_cols:
        if feature not in control_cells.columns or feature not in gene_cells.columns:
            continue

        control_values = control_cells[feature].dropna()
        gene_values = gene_cells[feature].dropna()

        if len(control_values) < 2 or len(gene_values) < 2:
            continue

        t_stat, p_value = stats.ttest_ind(control_values, gene_values, equal_var=False)

        results.append({
            'feature': feature,
            'pvalue': p_value,
            'control_mean': control_values.mean(),
            'gene_mean': gene_values.mean()
        })

    results_df = pd.DataFrame(results).sort_values('pvalue')
    return results_df.head(n_features)['feature'].tolist()

def get_perturbation_score(cell_data, gene, diff_exp_features, pert_col, control_key):
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
    scaled_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    return scaled_scores