import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score


def get_perturbation_score(
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
