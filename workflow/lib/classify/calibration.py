"""This module provides functions for model confidence calibration using manually labeled data.

It includes utilities for post-hoc confidence calibration using isotonic or sigmoid
methods, joining classified data with manual labels, and replacing confidence values
while preserving predicted class labels.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from lib.aggregate.cell_classification import CellClassifier


def resolve_join_keys(classify_by: str) -> Tuple[List[str], str]:
    """Public helper to get join keys and id column for a classify_by mode.

    Returns (join_keys, id_col) where:
    - For 'cell'/'cells'/'cp': (['label','plate','well','tile'], 'label')
    - For 'vacuole'/'vacuoles'/'vac': (['vacuole_id','plate','well','tile'], 'vacuole_id')
    """
    return _resolve_join_keys(classify_by)


def calibrate_confidence(
    *,
    # data
    master_phenotype_df: pd.DataFrame,
    classified_metadata: pd.DataFrame,
    manual_labeled_data: pd.DataFrame,
    # semantics
    classify_by: str,
    class_title: str,
    classifier_path: Union[str, Path],
    # calibration controls
    confidence_correction: Optional[
        str
    ] = None,  # None or 'post-hoc' (aliases accepted)
    calibration_method: str = "isotonic",  # 'isotonic' or 'sigmoid'
    test_plate: Optional[Iterable] = None,
    test_well: Optional[Iterable] = None,
    min_samples_isotonic: int = 50,
    # misc
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Union[str, int, bool]]]:
    """Post-hoc confidence calibration & in-place replacement of <class_title>_confidence.

    - Keeps predicted class labels unchanged.
    - Replaces only the confidence column where calibrated probabilities are available.
    - Joins rows via normalized keys for robust alignment.

    Args:
        master_phenotype_df: DataFrame with all objects and features.
        classified_metadata: DataFrame with predicted classes and confidences.
        manual_labeled_data: DataFrame with manually labeled objects for calibration.
        classify_by: 'cell' or 'vacuole' (determines join keys).
        class_title: Column name for predicted class labels (e.g., 'phenotype').
        classifier_path: Path to the trained CellClassifier file.
        confidence_correction: None or 'post-hoc' (currently only 'post-hoc' is supported).
        calibration_method: 'isotonic' or 'sigmoid' (fallback to 'sigmoid' if isotonic is unsafe).
        test_plate: Optional list of plate identifiers to filter by (None = no filter).
        test_well: Optional list of well identifiers to filter by (None = no filter).
        min_samples_isotonic: Minimum samples per class to use isotonic safely.
        verbose: Whether to print progress messages.

    Returns:
        (out_df, info)
        out_df: Classified metadata with calibrated <class_title>_confidence where applicable.
        info: dict with metadata:
              {
                'correction': str,
                'calibration_method': str,
                'fallback_from_isotonic': bool,
                'rows_with_features': int,
                'rows_replaced': int,
                'rows_unmodified': int,
                'used_label_encoder': bool,
                'filtered_by_plate': bool,
                'filtered_by_well': bool,
              }

    Notes:
        Designed to be easily extended to additional correction types (registry pattern ready).
    """
    cc = (
        None
        if confidence_correction is None
        else str(confidence_correction).strip().lower()
    )
    if not cc or cc in {"", "none"}:
        if verbose:
            print(
                "confidence_correction=None → pass-through original classified_metadata."
            )
        return classified_metadata.copy(deep=True), {
            "correction": "none",
            "calibration_method": "",
            "fallback_from_isotonic": False,
            "rows_with_features": 0,
            "rows_replaced": 0,
            "rows_unmodified": len(classified_metadata),
            "used_label_encoder": False,
            "filtered_by_plate": test_plate is not None,
            "filtered_by_well": test_well is not None,
        }

    # normalize supported correction type(s)
    if cc in {"post-hoc", "posthoc", "post_hoc"}:
        correction_name = "post-hoc"
    else:
        raise ValueError(
            f"Unsupported confidence_correction={confidence_correction!r}. Currently only 'post-hoc' is supported."
        )

    join_keys, id_col = _resolve_join_keys(classify_by)

    # Required columns sanity checks (minimal; detailed checks happen during merges)
    required_cls_cols = set(join_keys + [class_title, f"{class_title}_confidence"])
    missing_cls = required_cls_cols - set(classified_metadata.columns)
    if missing_cls:
        raise KeyError(
            f"classified_metadata missing required columns: {sorted(missing_cls)}"
        )

    master_norm = _normalize_keys_inplace(
        master_phenotype_df.copy(deep=True), id_col=id_col
    )
    cls_norm = _normalize_keys_inplace(
        classified_metadata.copy(deep=True), id_col=id_col
    )
    manual_norm = _normalize_keys_inplace(
        manual_labeled_data.copy(deep=True), id_col=id_col
    )

    master_norm = _apply_test_filters(master_norm, test_plate, test_well)
    cls_norm = _apply_test_filters(cls_norm, test_plate, test_well)
    manual_norm = _apply_test_filters(manual_norm, test_plate, test_well)

    clf = CellClassifier.load(classifier_path)
    est_prefit = _get_prefit_estimator(clf)
    feature_cols = list(getattr(clf, "features", []))
    if not feature_cols:
        raise ValueError(
            "No feature list found on the loaded classifier (clf.features is empty)."
        )

    # Build master features table (one row per join key)
    feat_master = (
        master_norm[join_keys + feature_cols].drop_duplicates(subset=join_keys).copy()
    )

    manual_join = (
        manual_norm[join_keys + [class_title]]
        .dropna(subset=join_keys + [class_title])
        .copy()
    )
    cal_df = manual_join.merge(feat_master, on=join_keys, how="inner")
    if cal_df.empty:
        raise ValueError(
            "No overlap between manual_labeled_data and feature table on the specified keys."
        )

    # Ensure all requested features exist
    missing_feats = [c for c in feature_cols if c not in cal_df.columns]
    if missing_feats:
        raise ValueError(f"Missing required features for calibration: {missing_feats}")

    X_manual = cal_df[feature_cols].to_numpy()
    y_manual_raw = cal_df[class_title].to_numpy()
    y_manual_enc, used_le = _encode_labels_if_needed(clf, y_manual_raw)

    # Choose calibration method safely
    chosen_method, did_fallback = _choose_calibration_method(
        requested=calibration_method,
        y_manual=y_manual_enc,
        min_samples_isotonic=min_samples_isotonic,
    )

    # Fit calibrator on the prefit estimator
    cal_wrapper = CalibratedClassifierCV(
        estimator=est_prefit, cv="prefit", method=chosen_method
    )
    cal_wrapper.fit(X_manual, y_manual_enc)

    target_df = cls_norm.merge(
        feat_master, on=join_keys, how="left", suffixes=("", "_feat")
    )
    feat_missing_mask = target_df[feature_cols].isna().any(axis=1)
    total_candidates = int((~feat_missing_mask).sum())
    if verbose and feat_missing_mask.any():
        print(
            f"Note: {int(feat_missing_mask.sum())} rows lack features; keeping their original confidence."
        )

    if total_candidates == 0:
        # Nothing to replace; return original as-is
        if verbose:
            print(
                "No rows with complete features in target set; returning original classified_metadata."
            )
        return classified_metadata.copy(deep=True), {
            "correction": correction_name,
            "calibration_method": chosen_method,
            "fallback_from_isotonic": did_fallback,
            "rows_with_features": 0,
            "rows_replaced": 0,
            "rows_unmodified": len(classified_metadata),
            "used_label_encoder": used_le,
            "filtered_by_plate": test_plate is not None,
            "filtered_by_well": test_well is not None,
        }

    X_target = target_df.loc[~feat_missing_mask, feature_cols].to_numpy()

    probs_cal = cal_wrapper.predict_proba(X_target)
    cal_classes = getattr(cal_wrapper, "classes_", None)
    if cal_classes is None:
        raise RuntimeError("Calibrated classifier does not expose classes_.")
    class_index = {cls_val: idx for idx, cls_val in enumerate(cal_classes)}

    pred_col = class_title
    conf_col = f"{class_title}_confidence"

    # Build encoded predictions to index probs (respect label encoder if present)
    pred_vals = target_df.loc[~feat_missing_mask, pred_col].to_numpy()

    if used_le:
        # Transform original class ids to encoded space; invalid transform → None
        def to_enc_safe(v):
            try:
                return clf.label_encoder.transform([v])[0]
            except Exception:
                return None
    else:

        def to_enc_safe(v):
            return v

    enc_vals = np.array([to_enc_safe(v) for v in pred_vals], dtype=object)
    idxs = np.array([class_index.get(v, None) for v in enc_vals], dtype=object)

    # Prepare original confidences to preserve when class mapping is unknown
    orig_conf = target_df.loc[~feat_missing_mask, conf_col].to_numpy(dtype=float)

    cal_conf = np.empty(len(idxs), dtype=float)
    for i, j in enumerate(idxs):
        if j is None:
            cal_conf[i] = orig_conf[i]  # keep original if we can't map the class
        else:
            cal_conf[i] = float(probs_cal[i, j])

    out_df = classified_metadata.copy(deep=True)

    # Create a normalized copy to find row positions robustly
    left_norm = _normalize_keys_inplace(out_df.copy(), id_col=id_col)
    left_norm["__tmp_idx__"] = np.arange(len(left_norm), dtype=np.int64)

    corrected_subset = target_df.loc[~feat_missing_mask, join_keys].copy()
    corrected_subset["__row_ix__"] = np.arange(len(corrected_subset), dtype=np.int64)

    merged_pos = corrected_subset.merge(
        left_norm[join_keys + ["__tmp_idx__"]],
        on=join_keys,
        how="inner",
        validate="m:1",
    )

    pos_in_left = merged_pos["__tmp_idx__"].to_numpy()
    rows_replaced = int(len(pos_in_left))

    # Set the calibrated values into the original DF
    if conf_col not in out_df.columns:
        out_df[conf_col] = np.nan
    out_df.iloc[pos_in_left, out_df.columns.get_loc(conf_col)] = cal_conf[
        :rows_replaced
    ]

    if verbose:
        print(
            f"Replaced {rows_replaced} confidences via normalized-key merge "
            f"(method={chosen_method}, cv='prefit')."
        )

    info = {
        "correction": correction_name,
        "calibration_method": chosen_method,
        "fallback_from_isotonic": did_fallback,
        "rows_with_features": total_candidates,
        "rows_replaced": rows_replaced,
        "rows_unmodified": int(total_candidates - rows_replaced),
        "used_label_encoder": used_le,
        "filtered_by_plate": test_plate is not None,
        "filtered_by_well": test_well is not None,
    }
    return out_df, info


def _canon_plate_value(v) -> Optional[str]:
    """Normalize plate values to 'string int' (e.g., 1 -> '1'), preserving None."""
    if pd.isna(v):
        return None
    try:
        f = float(v)
        if np.isfinite(f):
            return str(int(round(f)))
    except Exception:
        pass
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _canon_well_value(v) -> Optional[str]:
    """Normalize well values to uppercase strings, preserving None."""
    return None if pd.isna(v) else str(v).strip().upper()


def _normalize_keys_inplace(df: pd.DataFrame, *, id_col: str) -> pd.DataFrame:
    """In-place normalization of plate, well, tile, and id_col in the given DataFrame."""
    if "plate" in df.columns:
        df["plate"] = df["plate"].apply(_canon_plate_value)
    if "well" in df.columns:
        df["well"] = df["well"].apply(_canon_well_value)
    if "tile" in df.columns:
        df["tile"] = pd.to_numeric(df["tile"], errors="coerce").astype("Int64")
    if id_col in df.columns:
        df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    return df


def _apply_test_filters(
    df: pd.DataFrame,
    test_plate: Optional[Iterable] = None,
    test_well: Optional[Iterable] = None,
) -> pd.DataFrame:
    """Filter by TEST_PLATE / TEST_WELL semantics after key normalization."""
    out = df
    if test_plate is not None:
        plate_set = set(_canon_plate_value(p) for p in test_plate)
        out = out[out["plate"].isin(plate_set)]
    if test_well is not None:
        well_set = set(_canon_well_value(w) for w in test_well)
        out = out[out["well"].isin(well_set)]
    return out


def _resolve_join_keys(classify_by: str) -> Tuple[List[str], str]:
    """Return (join_keys, id_col) given classify_by."""
    ctype = str(classify_by).lower()
    if ctype in {"cell", "cells", "cp"}:
        return ["label", "plate", "well", "tile"], "label"
    if ctype in {"vacuole", "vacuoles", "vac"}:
        return ["vacuole_id", "plate", "well", "tile"], "vacuole_id"
    raise ValueError(
        f"Unsupported classify_by value: {classify_by!r}. Use 'cell' or 'vacuole'."
    )


def _get_prefit_estimator(trained_classifier) -> object:
    """Extract the prefit estimator from your CellClassifier or sklearn Pipeline.

    Returns the estimator to be wrapped by CalibratedClassifierCV(cv='prefit').

    Args:
        trained_classifier: A CellClassifier instance or a sklearn estimator.

    Returns:
        The underlying sklearn estimator (not a Pipeline).
    """
    est = getattr(trained_classifier, "model", trained_classifier)
    if hasattr(est, "named_steps"):  # sklearn Pipeline
        est = list(est.named_steps.values())[-1]
    return est


def _choose_calibration_method(
    requested: str,
    y_manual: Sequence,
    min_samples_isotonic: int = 50,
) -> Tuple[str, bool]:
    """Ensure a safe calibration method. If 'isotonic' but any class has < threshold samples, fallback to 'sigmoid'.

    Args:
        requested: 'isotonic' or 'sigmoid' (case-insensitive).
        y_manual: Array-like of manual class labels (encoded if needed).
        min_samples_isotonic: Minimum samples per class to use isotonic safely.

    Returns:
        (chosen_method, did_fallback)
    """
    method = str(requested).lower()
    did_fallback = False
    if method == "isotonic":
        cls_counts = pd.Series(y_manual).value_counts()
        if (cls_counts < min_samples_isotonic).any():
            method = "sigmoid"
            did_fallback = True
    elif method != "sigmoid":
        # Unknown → default to sigmoid for safety
        method = "sigmoid"
        did_fallback = requested.lower() != "sigmoid"
    return method, did_fallback


def _encode_labels_if_needed(clf, y: Sequence) -> Tuple[np.ndarray, bool]:
    """Use clf.label_encoder if present; otherwise return y as-is."""
    if getattr(clf, "label_encoder", None) is not None:
        return clf.label_encoder.transform(y), True
    return np.asarray(y), False
