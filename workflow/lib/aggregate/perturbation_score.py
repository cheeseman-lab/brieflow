import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls
from lib.aggregate.cell_data_utils import split_cell_data


def calculate_perturbation_scores(
    cell_data,
    gene,
    feature_cols,
    perturbation_col="gene_symbol_0",
    n_differential_features=200,
):
    """Per-cell perturbation scores via 5-fold out-of-fold logistic regression with top-k feature selection.

    AUROC guide:
      - < 0.6  → basically noise; don’t filter (return NaN scores and keep all cells)
      - 0.6–0.75 → weak/moderate separation; filter cautiously
      - > 0.75 → decent separation; filtering makes sense
      - > 0.85–0.9 → strong separation; filtering always safe and effective
    """
    y = (cell_data[perturbation_col] == gene).astype(int).to_numpy()
    X_all = cell_data[feature_cols].to_numpy()

    # select top-k differential features (ANOVA F-test)
    k = min(n_differential_features, X_all.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k).fit(X_all, y)
    X = selector.transform(X_all)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, scores)

    return pd.Series(scores, index=cell_data.index), auc


def perturbation_score(
    cell_data, metadata_cols, perturbation_name_col, control_key, auc_threshold=0.6
):
    # start with all perturbation scores = 0
    cell_data["perturbation_score"] = np.nan
    metadata_cols.append("perturbation_score")

    perturbation_col = cell_data[perturbation_name_col]
    perturbed_genes = [
        gene
        for gene in perturbation_col.unique().tolist()
        if not gene.startswith(control_key)
    ]
    nt_idx = perturbation_col.index[
        perturbation_col.str.startswith(control_key)
    ].to_numpy()

    for gene in perturbed_genes:
        print(f"Processing {gene}...")
        gene_idx = perturbation_col.index[perturbation_col == gene].to_numpy()
        nt_keep = np.random.choice(
            nt_idx, size=min(len(gene_idx), len(nt_idx)), replace=False
        )
        keep_idx = np.union1d(gene_idx, nt_keep)
        gene_subset_df = cell_data.iloc[keep_idx].copy()
        original_idx = gene_subset_df.index.copy()
        gene_subset_df = gene_subset_df.reset_index(drop=True)

        # SCALE PERTURBATION GENE AND CONTROL FEATURES

        feature_cols = gene_subset_df.columns.difference(metadata_cols, sort=False)
        metadata, features = split_cell_data(gene_subset_df, metadata_cols)
        metadata, features = prepare_alignment_data(
            metadata,
            features,
            ["plate", "well"],
            "gene_symbol_0",
            "nontargeting",
            "sgRNA_0",
        )
        features = features.astype(np.float32)
        features = centerscale_on_controls(
            features,
            metadata,
            "gene_symbol_0",
            "nontargeting",
            "batch_values",
        )
        features = pd.DataFrame(features, columns=feature_cols)
        gene_subset_df = pd.concat([metadata, features], axis=1)

        # REMOVE LOW PERTURBATION SCORE CELLS

        perturbation_scores, auc = calculate_perturbation_scores(
            gene_subset_df,
            gene,
            feature_cols,
            perturbation_col="gene_symbol_0",
        )

        # if auc is greater than the cutoff, this perturbation is significant and we give perturbed cells their score
        # otherwise, this perturbation is not significant and the score is nan
        if auc > auc_threshold:
            print(f"!! {gene} qualified for perturbation scoring with AUC of {auc:.3f}")

            # set the gene rows in original cell dataset to perturbation scores
            perturbation_scores.index = original_idx
            cell_data.loc[gene_idx, "perturbation_score"] = perturbation_scores[
                gene_idx
            ]
