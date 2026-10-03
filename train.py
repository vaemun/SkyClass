"""Train SkyClass models and write metrics, diagnostics, and artifacts."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from data_pipeline import BANDS, CLASSES, fetch_sdss, prepare_photometry

SEED = 42
COLOR_NAMES = ("u-g", "g-r", "r-i", "i-z", "u-r", "g-i", "r-z", "u-z")
FEATURE_SETS = {
    "A_colours_only_psf": [f"psf_color_{name}" for name in COLOR_NAMES],
    "B_colours_plus_r": [f"psf_color_{name}" for name in COLOR_NAMES] + ["psf_r"],
    "C_colours_plus_concentration": [f"psf_color_{name}" for name in COLOR_NAMES]
    + [f"concentration_{band}" for band in BANDS],
    "D_colours_concentration_errors": [f"psf_color_{name}" for name in COLOR_NAMES]
    + [f"concentration_{band}" for band in BANDS]
    + [f"psf_color_err_{name}" for name in COLOR_NAMES],
}


def score_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict:
    report = classification_report(
        y_true,
        predictions,
        labels=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=list(CLASSES), average="macro", zero_division=0)),
        "per_class": {
            label: {
                key: float(report[label][key]) if key != "support" else int(report[label][key])
                for key in ("precision", "recall", "f1-score", "support")
            }
            for label in CLASSES
        },
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(CLASSES)).tolist(),
    }


def spatial_groups(frame: pd.DataFrame) -> np.ndarray:
    ra_block = np.floor(frame["ra"].astype(float) / 15).astype(int)
    dec_block = np.floor((frame["dec"].astype(float) + 90) / 10).astype(int)
    return (ra_block.astype(str) + ":" + dec_block.astype(str)).to_numpy()


def make_xgb(params: dict, seed: int = SEED) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        **params,
    )


def make_comparison_models(seed: int, xgb_params: dict) -> dict:
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed)),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "XGBoost": make_xgb(xgb_params, seed=seed),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, max_iter=140, early_stopping=True, n_iter_no_change=12, random_state=seed)),
    }


def weighted_fit(model, features: pd.DataFrame, labels: pd.Series):
    weights = compute_sample_weight(class_weight="balanced", y=labels)
    is_xgboost = isinstance(model, XGBClassifier)
    fit_labels = pd.Categorical(labels, categories=CLASSES).codes if is_xgboost else labels
    if is_xgboost and (fit_labels < 0).any():
        raise ValueError("Training labels must be one of STAR, GALAXY, or QSO")
    if hasattr(model, "steps"):
        model.fit(features, fit_labels, **{f"{model.steps[-1][0]}__sample_weight": weights})
    else:
        model.fit(features, fit_labels, sample_weight=weights)
    return model


def snr_sample_weights(frame: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    """Multiply balanced class weights by clipped, normalized mean five-band S/N."""
    errors = frame[[f"psfMagErr_{band}" for band in BANDS]].to_numpy(dtype=float)
    snr = 1.0 / (np.log(10.0) / 2.5 * np.maximum(errors, 1e-6))
    quality = np.sqrt(np.mean(snr, axis=1))
    quality /= np.median(quality)
    quality = np.clip(quality, 0.25, 4.0)
    return compute_sample_weight(class_weight="balanced", y=labels) * quality


def fit_with_weights(model, features: pd.DataFrame, labels: pd.Series, weights: np.ndarray):
    fit_labels = pd.Categorical(labels, categories=CLASSES).codes if isinstance(model, XGBClassifier) else labels
    if isinstance(model, XGBClassifier) and (fit_labels < 0).any():
        raise ValueError("Training labels must be one of STAR, GALAXY, or QSO")
    model.fit(features, fit_labels, sample_weight=weights)
    return model


def predict_labels(model, features: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(features)
    if isinstance(model, XGBClassifier):
        return np.asarray(CLASSES)[predictions.astype(int)]
    return predictions


def reliability_metrics(labels: pd.Series, probabilities: np.ndarray, bins: int = 10) -> dict:
    labels_array = labels.to_numpy()
    class_index = {name: index for index, name in enumerate(CLASSES)}
    targets = np.eye(len(CLASSES))[np.array([class_index[label] for label in labels_array])]
    brier = float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    correct = predicted == labels_array
    edges = np.linspace(0, 1, bins + 1)
    ece, calibration = 0.0, []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += float(mask.mean()) * abs(accuracy - mean_confidence)
            calibration.append({"lower": float(lower), "upper": float(upper), "rows": int(mask.sum()), "mean_confidence": mean_confidence, "accuracy": accuracy})
    one_vs_rest = {}
    for index, label in enumerate(CLASSES):
        curve = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (probabilities[:, index] >= lower) & (probabilities[:, index] < upper if upper < 1 else probabilities[:, index] <= upper)
            if mask.any():
                curve.append({"lower": float(lower), "upper": float(upper), "rows": int(mask.sum()), "mean_probability": float(probabilities[mask, index].mean()), "observed_fraction": float(targets[mask, index].mean())})
        one_vs_rest[label] = curve
    return {"multiclass_brier": brier, "top_label_ece": float(ece), "top_label_reliability": calibration, "one_vs_rest_reliability": one_vs_rest}


def abstention_metrics(labels: pd.Series, probabilities: np.ndarray) -> list[dict]:
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    rows = []
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        retained = confidence >= threshold
        if retained.any():
            score = score_predictions(labels.iloc[np.flatnonzero(retained)], predicted[retained])
            rows.append({"threshold": threshold, "coverage": float(retained.mean()), "rows": int(retained.sum()), "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]})
        else:
            rows.append({"threshold": threshold, "coverage": 0.0, "rows": 0, "accuracy": None, "macro_f1": None})
    return rows


def bootstrap_intervals(labels: pd.Series, predictions: np.ndarray, draws: int = 500) -> dict:
    label_values = labels.to_numpy()
    rng = np.random.default_rng(SEED + 31)
    samples = {"accuracy": [], "macro_f1": [], "qso_recall": []}
    for _ in range(draws):
        indices = rng.integers(0, len(label_values), size=len(label_values))
        truth, predicted = label_values[indices], predictions[indices]
        samples["accuracy"].append(float(np.mean(truth == predicted)))
        samples["macro_f1"].append(float(f1_score(truth, predicted, labels=list(CLASSES), average="macro", zero_division=0)))
        qso = truth == "QSO"
        samples["qso_recall"].append(float(np.mean(predicted[qso] == "QSO")) if qso.any() else float("nan"))
    return {name: {"lower_95": float(np.nanquantile(values, 0.025)), "upper_95": float(np.nanquantile(values, 0.975))} for name, values in samples.items()} | {"draws": draws, "method": "Percentile bootstrap of spatial holdout rows; resampling is not block-level."}


def noisy_training_copy(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Perturb magnitudes by independent Gaussian errors, capped at 0.5 mag."""
    augmented = frame.copy()
    rng = np.random.default_rng(seed)
    for band in BANDS:
        psf_sigma = np.clip(frame[f"psfMagErr_{band}"].to_numpy(), 0, 0.5)
        model_sigma = np.clip(frame[f"modelMagErr_{band}"].to_numpy(), 0, 0.5)
        augmented[f"psfMag_{band}"] = frame[f"psfMag_{band}"].to_numpy() + rng.normal(0, psf_sigma)
        augmented[f"modelMag_{band}"] = frame[f"modelMag_{band}"].to_numpy() + rng.normal(0, model_sigma)
    prepared, _ = prepare_photometry(augmented)
    return prepared


def tune_xgb(frame: pd.DataFrame, train_indices: np.ndarray, groups: np.ndarray, features: list[str]) -> tuple[dict, list[dict]]:
    local_groups = groups[train_indices]
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 1)
    subtrain, validation = next(split.split(train_indices, frame.iloc[train_indices]["class"], local_groups))
    fit_indices = train_indices[subtrain]
    validation_indices = train_indices[validation]
    candidates = [
        {"n_estimators": 220, "max_depth": 4, "learning_rate": 0.06, "subsample": 0.85, "colsample_bytree": 0.9, "reg_lambda": 2.0},
        {"n_estimators": 320, "max_depth": 6, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.9, "reg_lambda": 2.0},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.06, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 4.0},
    ]
    results = []
    for params in candidates:
        model = make_xgb(params)
        weighted_fit(model, frame.iloc[fit_indices][features], frame.iloc[fit_indices]["class"])
        predicted = predict_labels(model, frame.iloc[validation_indices][features])
        results.append({"params": params, "validation_macro_f1": score_predictions(frame.iloc[validation_indices]["class"], predicted)["macro_f1"]})
    best = max(results, key=lambda item: item["validation_macro_f1"])
    return best["params"], results


def binned_scores(frame: pd.DataFrame, predictions: np.ndarray, column: str, edges: list[float], labels: list[str]) -> list[dict]:
    bins = pd.cut(frame[column], bins=edges, labels=labels, include_lowest=True)
    result = []
    for label in labels:
        mask = bins == label
        if int(mask.sum()) == 0:
            continue
        score = score_predictions(frame.loc[mask, "class"], predictions[mask.to_numpy()])
        result.append({"bin": label, "rows": int(mask.sum()), "accuracy": score["accuracy"], "macro_f1": score["macro_f1"], "per_class": score["per_class"]})
    return result


def uncertainty_quintiles(frame: pd.DataFrame, predictions: np.ndarray, exclude_error_outliers: bool) -> dict:
    errors = frame[[f"psfMagErr_{band}" for band in BANDS]]
    outliers = errors.ge(1.0).any(axis=1).to_numpy()
    included = ~outliers if exclude_error_outliers else np.ones(len(frame), dtype=bool)
    selected = frame.loc[included].reset_index(drop=True)
    selected_predictions = predictions[included]
    mean_error = errors.mean(axis=1).to_numpy()[included]
    bins = pd.qcut(pd.Series(mean_error).rank(method="first"), q=5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    rows = []
    for label in bins.cat.categories:
        mask = (bins == label).to_numpy()
        if mask.any():
            score = score_predictions(selected.loc[mask, "class"], selected_predictions[mask])
            rows.append({"bin": str(label), "rows": int(mask.sum()), "mean_error_mag": float(mean_error[mask].mean()), "accuracy": score["accuracy"], "macro_f1": score["macro_f1"], "per_class": score["per_class"]})
    return {"excluded_error_outlier_rows": int(outliers.sum()) if exclude_error_outliers else 0, "included_rows": int(included.sum()), "quintiles": rows}


def high_error_distribution(frame: pd.DataFrame) -> dict:
    error_columns = [f"psfMagErr_{band}" for band in BANDS]
    high_error = frame[error_columns].ge(1.0).any(axis=1)
    bins = pd.cut(frame["psf_r"], [-np.inf, 17, 18, 19, 20, 21, np.inf], labels=["<17", "17-18", "18-19", "19-20", "20-21", ">=21"], include_lowest=True)
    class_counts = frame.loc[high_error, "class"].value_counts().reindex(CLASSES, fill_value=0)
    magnitude_counts = {}
    for magnitude_bin in bins.cat.categories:
        mask = high_error & bins.eq(magnitude_bin)
        magnitude_counts[str(magnitude_bin)] = {
            "total": int(mask.sum()),
            "by_class": frame.loc[mask, "class"].value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict(),
        }
    return {"threshold_mag": 1.0, "rows": int(high_error.sum()), "by_class": class_counts.astype(int).to_dict(), "by_dereddened_psf_r_bin": magnitude_counts}


def bootstrap_classification(labels: pd.Series, predictions: np.ndarray, draws: int, seed: int) -> dict:
    label_values = labels.to_numpy()
    rng = np.random.default_rng(seed)
    class_indices = {label: np.flatnonzero(label_values == label) for label in CLASSES}
    samples = {"accuracy": [], "macro_f1": []}
    for label in CLASSES:
        samples.update({f"{label}_precision": [], f"{label}_recall": [], f"{label}_f1": []})
    for _ in range(draws):
        indices = np.concatenate([rng.choice(class_indices[label], size=len(class_indices[label]), replace=True) for label in CLASSES if len(class_indices[label])])
        truth, predicted = label_values[indices], predictions[indices]
        samples["accuracy"].append(float(np.mean(truth == predicted)))
        samples["macro_f1"].append(float(f1_score(truth, predicted, labels=list(CLASSES), average="macro", zero_division=0)))
        for label in CLASSES:
            mask = truth == label
            predicted_mask = predicted == label
            tp = int(np.sum(mask & predicted_mask))
            precision = tp / int(predicted_mask.sum()) if predicted_mask.any() else 0.0
            recall = float(np.mean(predicted[mask] == label)) if mask.any() else float("nan")
            samples[f"{label}_precision"].append(precision)
            samples[f"{label}_recall"].append(recall)
            samples[f"{label}_f1"].append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    intervals = {}
    for name, values in samples.items():
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        intervals[name] = (
            {"lower_95": float(np.quantile(finite_values, 0.025)), "upper_95": float(np.quantile(finite_values, 0.975))}
            if len(finite_values)
            else {"lower_95": None, "upper_95": None}
        )
    return {"draws": draws, "method": "Stratified percentile bootstrap, resampling independently within each class", "intervals": intervals}


def calibrated_probabilities(probabilities: np.ndarray, calibrators: list[IsotonicRegression]) -> np.ndarray:
    calibrated = np.column_stack([calibrator.predict(probabilities[:, index]) for index, calibrator in enumerate(calibrators)])
    totals = calibrated.sum(axis=1, keepdims=True)
    return np.divide(calibrated, totals, out=np.full_like(calibrated, 1.0 / len(CLASSES)), where=totals > 0)


def fit_isotonic_calibration(probabilities: np.ndarray, labels: pd.Series) -> list[IsotonicRegression]:
    calibrators = []
    for index, label in enumerate(CLASSES):
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        calibrator.fit(probabilities[:, index], (labels.to_numpy() == label).astype(int))
        calibrators.append(calibrator)
    return calibrators


def bootstrap_probability_metrics(labels: pd.Series, probabilities: np.ndarray, draws: int, seed: int) -> dict:
    labels_array = labels.to_numpy()
    classes = {label: np.flatnonzero(labels_array == label) for label in CLASSES}
    rng = np.random.default_rng(seed)
    values = {"multiclass_brier": [], "top_label_ece": []}
    for _ in range(draws):
        indices = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in classes.values() if len(indices)])
        result = reliability_metrics(pd.Series(labels_array[indices]), probabilities[indices], bins=10)
        values["multiclass_brier"].append(result["multiclass_brier"])
        values["top_label_ece"].append(result["top_label_ece"])
    return {name: {"lower_95": float(np.quantile(samples, 0.025)), "upper_95": float(np.quantile(samples, 0.975))} for name, samples in values.items()}


def bootstrap_qso_rates(predictions: np.ndarray, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = {"qso_recall": [], "qso_to_star_rate": []}
    for _ in range(draws):
        resample = rng.integers(0, len(predictions), size=len(predictions))
        values = predictions[resample]
        samples["qso_recall"].append(float(np.mean(values == "QSO")))
        samples["qso_to_star_rate"].append(float(np.mean(values == "STAR")))
    return {key: {"lower_95": float(np.quantile(value, 0.025)), "upper_95": float(np.quantile(value, 0.975))} for key, value in samples.items()}


def bootstrap_qso_rate_difference(high_predictions: np.ndarray, adjacent_predictions: np.ndarray, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        high = high_predictions[rng.integers(0, len(high_predictions), len(high_predictions))]
        adjacent = adjacent_predictions[rng.integers(0, len(adjacent_predictions), len(adjacent_predictions))]
        differences.append(float(np.mean(high == "STAR") - np.mean(adjacent == "STAR")))
    return {"point_difference": float(np.mean(high_predictions == "STAR") - np.mean(adjacent_predictions == "STAR")), "lower_95": float(np.quantile(differences, 0.025)), "upper_95": float(np.quantile(differences, 0.975)), "definition": "QSO predicted STAR rate in z=2.5-3 minus pooled z=2-2.5 and z=3-5 rate"}


def bootstrap_paired_macro_f1_difference(labels: pd.Series, first: np.ndarray, second: np.ndarray, draws: int, seed: int) -> dict:
    values = labels.to_numpy()
    class_rows = [np.flatnonzero(values == label) for label in CLASSES if np.any(values == label)]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        indices = np.concatenate([rng.choice(rows, size=len(rows), replace=True) for rows in class_rows])
        deltas.append(float(f1_score(values[indices], first[indices], labels=list(CLASSES), average="macro", zero_division=0) - f1_score(values[indices], second[indices], labels=list(CLASSES), average="macro", zero_division=0)))
    return {"difference_first_minus_second": float(f1_score(values, first, labels=list(CLASSES), average="macro", zero_division=0) - f1_score(values, second, labels=list(CLASSES), average="macro", zero_division=0)), "lower_95": float(np.quantile(deltas, 0.025)), "upper_95": float(np.quantile(deltas, 0.975)), "method": "paired stratified bootstrap on identical evaluation rows"}


def add_abstention_intervals(rows: list[dict], labels: pd.Series, predictions: np.ndarray, scores: np.ndarray, draws: int, seed: int) -> list[dict]:
    for index, row in enumerate(rows):
        retained = scores >= row["threshold"]
        if retained.any():
            row["accuracy_bootstrap_95"] = bootstrap_classification(labels.iloc[np.flatnonzero(retained)].reset_index(drop=True), predictions[retained], draws, seed + index)["intervals"]["accuracy"]
    return rows


def colour_snr(frame: pd.DataFrame) -> np.ndarray:
    signal = np.column_stack([frame[f"psf_color_{name}"].to_numpy(dtype=float) for name in COLOR_NAMES])
    uncertainty = np.column_stack([frame[f"psf_color_err_{name}"].to_numpy(dtype=float) for name in COLOR_NAMES])
    per_color = np.divide(np.abs(signal), uncertainty, out=np.zeros_like(signal), where=uncertainty > 0)
    return np.sqrt(np.mean(np.square(per_color), axis=1))


def colour_snr_abstention(labels: pd.Series, predictions: np.ndarray, snr: np.ndarray) -> list[dict]:
    rows = []
    for threshold in (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0):
        retained = snr >= threshold
        if retained.any():
            score = score_predictions(labels.iloc[np.flatnonzero(retained)], predictions[retained])
            rows.append({"threshold": threshold, "coverage": float(retained.mean()), "rows": int(retained.sum()), "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]})
        else:
            rows.append({"threshold": threshold, "coverage": 0.0, "rows": 0, "accuracy": None, "macro_f1": None})
    return rows


def r_bin_calibration(frame: pd.DataFrame, labels: pd.Series, raw: np.ndarray, calibrated: np.ndarray) -> list[dict]:
    bins = pd.cut(frame["psf_r"], [-np.inf, 18, 19, 20, 21, np.inf], labels=["<18", "18-19", "19-20", "20-21", ">=21"], include_lowest=True)
    rows = []
    for magnitude_bin in bins.cat.categories:
        mask = (bins == magnitude_bin).to_numpy()
        if not mask.any():
            continue
        raw_metrics = reliability_metrics(labels.iloc[np.flatnonzero(mask)].reset_index(drop=True), raw[mask], bins=10)
        calibrated_metrics = reliability_metrics(labels.iloc[np.flatnonzero(mask)].reset_index(drop=True), calibrated[mask], bins=10)
        rows.append({"r_bin": str(magnitude_bin), "rows": int(mask.sum()), "raw": {"multiclass_brier": raw_metrics["multiclass_brier"], "top_label_ece": raw_metrics["top_label_ece"]}, "isotonic": {"multiclass_brier": calibrated_metrics["multiclass_brier"], "top_label_ece": calibrated_metrics["top_label_ece"]}})
    return rows


def faint_model_comparison(frame: pd.DataFrame, prediction_map: dict[str, np.ndarray], draws: int) -> dict:
    bins = pd.cut(frame["psf_r"], [19, 20, 21, np.inf], labels=["19-20", "20-21", ">=21"], include_lowest=False)
    output = {}
    for magnitude_bin in bins.cat.categories:
        mask = (bins == magnitude_bin).to_numpy()
        if not mask.any():
            continue
        bin_labels = frame.loc[mask, "class"].reset_index(drop=True)
        output[str(magnitude_bin)] = {"rows": int(mask.sum()), "models": {}}
        bin_predictions = {}
        for model_name, all_predictions in prediction_map.items():
            predictions = all_predictions[mask]
            bin_predictions[model_name] = predictions
            output[str(magnitude_bin)]["models"][model_name] = {
                "score": score_predictions(bin_labels, predictions),
                "bootstrap_95": bootstrap_classification(bin_labels, predictions, draws, SEED + 220 + len(output[str(magnitude_bin)]["models"])),
            }
        for model_name in ("monte_carlo_augmentation", "snr_weighted"):
            output[str(magnitude_bin)]["models"][model_name]["paired_macro_f1_difference_vs_baseline_95"] = bootstrap_paired_macro_f1_difference(
                bin_labels, bin_predictions[model_name], bin_predictions["baseline"], draws, SEED + 250 + len(output) + (0 if model_name == "monte_carlo_augmentation" else 30)
            )
    return output


def importance_for_model(model, frame: pd.DataFrame, indices: np.ndarray, feature_names: list[str], rows: int, seed: int) -> dict:
    sample_indices = indices[np.linspace(0, len(indices) - 1, min(rows, len(indices)), dtype=int)]
    features = frame.iloc[sample_indices][feature_names]
    labels = frame.iloc[sample_indices]["class"]
    encoded_labels = pd.Categorical(labels, categories=CLASSES).codes
    permutation = permutation_importance(model, features, encoded_labels, scoring="f1_macro", n_repeats=5, random_state=seed, n_jobs=-1)
    permutation_rows = sorted(
        [{"feature": name, "mean_macro_f1_drop": float(mean), "std": float(std)} for name, mean, std in zip(feature_names, permutation.importances_mean, permutation.importances_std)],
        key=lambda row: row["mean_macro_f1_drop"], reverse=True,
    )
    contributions = model.get_booster().predict(xgboost.DMatrix(features), pred_contribs=True)
    shap_rows = []
    for class_index, class_name in enumerate(CLASSES):
        means = np.abs(contributions[:, class_index, :-1]).mean(axis=0)
        shap_rows.extend({"class": class_name, "feature": name, "mean_abs_shap": float(value)} for name, value in zip(feature_names, means))
    return {"evaluation_rows": len(sample_indices), "permutation": permutation_rows, "tree_shap": shap_rows}


def save_error_figures(metrics: dict, output_dir: Path) -> dict:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    confusion = np.asarray(metrics["headline_score"]["confusion_matrix"], dtype=float)
    normalized = confusion / confusion.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(CLASSES)), CLASSES)
    axis.set_yticks(range(len(CLASSES)), CLASSES)
    axis.set(xlabel="Predicted class", ylabel="True class", title="Row-normalized confusion · PSF colours only")
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            axis.text(column, row, f"{normalized[row, column]:.3f}\n(n={int(confusion[row, column])})", ha="center", va="center", color="white" if normalized[row, column] > 0.55 else "black")
    figure.colorbar(image, ax=axis, label="Fraction of true class")
    paths["confusion"] = "figures/confusion_row_normalized.png"
    figure.savefig(output_dir / paths["confusion"], dpi=170)
    plt.close(figure)

    redshift = metrics["error_analysis"]["qso_redshift_bins_colours_only_xgboost"]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(redshift))
    recall = np.array([row["qso_recall"] for row in redshift])
    lower = np.array([row["qso_recall_bootstrap_95"]["lower_95"] for row in redshift])
    upper = np.array([row["qso_recall_bootstrap_95"]["upper_95"] for row in redshift])
    axis.errorbar(x, recall, yerr=[recall - lower, upper - recall], marker="o", capsize=4, color="#087e78")
    axis.set_xticks(x, [row["bin"] for row in redshift])
    axis.set(ylim=(0, 1), xlabel="Spectroscopic redshift bin", ylabel="QSO recall", title="QSO recovery versus spectroscopic redshift")
    axis.grid(alpha=0.25)
    paths["redshift"] = "figures/qso_recall_by_redshift.png"
    figure.savefig(output_dir / paths["redshift"], dpi=170)
    plt.close(figure)

    faint = metrics["noise_aware_faint_bins"]
    model_names = list(next(iter(faint.values()))["models"])
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    styles = {"baseline": "#087e78", "monte_carlo_augmentation": "#c3543f", "snr_weighted": "#285a9f"}
    for name in model_names:
        values, lows, highs = [], [], []
        for magnitude_bin in faint:
            result = faint[magnitude_bin]["models"][name]
            value = result["score"]["macro_f1"]
            interval = result["bootstrap_95"]["intervals"]["macro_f1"]
            values.append(value)
            lows.append(value - interval["lower_95"])
            highs.append(interval["upper_95"] - value)
        axis.errorbar(list(faint), values, yerr=[lows, highs], marker="o", capsize=4, label=name.replace("_", " "), color=styles.get(name))
    axis.set(ylim=(0, 1), xlabel="Dereddened PSF r bin", ylabel="Macro-F1", title="Noise-aware training on faint spatial-holdout objects")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    paths["noise_faint"] = "figures/noise_aware_faint_bins.png"
    figure.savefig(output_dir / paths["noise_faint"], dpi=170)
    plt.close(figure)

    calibration = metrics["calibration"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    class_colors = {"STAR": "#087e78", "GALAXY": "#c3543f", "QSO": "#285a9f"}
    for probability_key, probability_name, line_style in (("raw", "raw", "--"), ("isotonic", "isotonic", "-")):
        for class_name in CLASSES:
            reliability = calibration[probability_key]["one_vs_rest_reliability"][class_name]
            axes[0].plot([row["mean_probability"] for row in reliability], [row["observed_fraction"] for row in reliability], marker="o", markersize=3, linestyle=line_style, label=f"{class_name} {probability_name}", color=class_colors[class_name], alpha=0.85)
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="#58676b")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted class probability", ylabel="Observed class fraction", title="One-vs-rest reliability · test fold")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    per_r = calibration["by_r_bin"]
    xpos = np.arange(len(per_r))
    for metric_key, axis, title in (("multiclass_brier", axes[1], "Multiclass Brier by r"), ("top_label_ece", axes[2], "Top-label ECE by r")):
        axis.plot(xpos, [entry["raw"][metric_key] for entry in per_r], marker="o", label="Uncalibrated", color="#c3543f")
        axis.plot(xpos, [entry["isotonic"][metric_key] for entry in per_r], marker="o", label="Isotonic", color="#087e78")
        axis.set_xticks(xpos, [entry["r_bin"] for entry in per_r])
        axis.set(xlabel="Dereddened PSF r bin", ylabel=metric_key, title=title)
        axis.legend(frameon=False)
    paths["calibration"] = "figures/calibration_reliability_and_r_bins.png"
    figure.savefig(output_dir / paths["calibration"], dpi=170)
    plt.close(figure)

    abstention = metrics["abstention"]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for key, label, color in (("max_calibrated_probability", "Max calibrated probability", "#087e78"), ("colour_snr_rms", "Colour S/N RMS", "#c3543f")):
        rows = abstention[key]
        axis.plot([row["coverage"] for row in rows], [row["accuracy"] for row in rows], marker="o", label=label, color=color)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Coverage", ylabel="Accuracy among retained rows", title="Abstention trade-off on spatial test fold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    paths["abstention"] = "figures/abstention_accuracy_coverage.png"
    figure.savefig(output_dir / paths["abstention"], dpi=170)
    plt.close(figure)

    importance = metrics["feature_importance"]["sets"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for row_index, set_name in enumerate(("A_colours_only_psf", "D_colours_concentration_errors")):
        entry = importance[set_name]
        perm = entry["permutation"][:10]
        axes[row_index, 0].barh([item["feature"] for item in perm][::-1], [item["mean_macro_f1_drop"] for item in perm][::-1], color="#087e78")
        axes[row_index, 0].set(title=f"{set_name}: permutation", xlabel="Macro-F1 drop")
        shap = sorted(entry["tree_shap"], key=lambda item: item["mean_abs_shap"], reverse=True)[:10]
        axes[row_index, 1].barh([item["feature"] for item in shap][::-1], [item["mean_abs_shap"] for item in shap][::-1], color="#c3543f")
        axes[row_index, 1].set(title=f"{set_name}: mean |TreeSHAP|", xlabel="Mean absolute contribution")
    paths["importance"] = "figures/feature_importance_A_D.png"
    figure.savefig(output_dir / paths["importance"], dpi=170)
    plt.close(figure)
    return paths


def build_interpretations(metrics: dict) -> dict[str, str]:
    score = metrics["headline_score"]
    confusion = score["confusion_matrix"]
    star_support = sum(confusion[CLASSES.index("STAR")])
    qso_support = sum(confusion[CLASSES.index("QSO")])
    star_to_qso = confusion[CLASSES.index("STAR")][CLASSES.index("QSO")] / star_support
    qso_to_star = confusion[CLASSES.index("QSO")][CLASSES.index("STAR")] / qso_support
    confusion_text = (
        f"The row-normalized matrix shows {star_to_qso:.1%} of true STARs assigned QSO and {qso_to_star:.1%} of true QSOs assigned STAR. "
        f"On this holdout that is {metrics['spatial_holdout']['star_to_qso_confusion']} STAR→QSO and {metrics['spatial_holdout']['qso_to_star_confusion']} QSO→STAR cases; both directions are kept visible because they have different class costs."
    )

    redshift_test = metrics["redshift_hypothesis_test"]
    if redshift_test["difference"]["lower_95"] > 0:
        verdict = "The bootstrap interval is above zero, supporting elevated assignment to STAR in this bin relative to its adjacent-z comparison."
    elif redshift_test["difference"]["upper_95"] < 0:
        verdict = "The bootstrap interval is below zero, contradicting elevated assignment to STAR in this bin relative to adjacent redshifts."
    else:
        verdict = "The bootstrap interval includes zero, so these data do not resolve whether z=2.5–3 QSOs are more often assigned STAR than adjacent-redshift QSOs."
    redshift_text = verdict + " A predicted STAR label is only a classifier-level proxy for stellar-locus overlap; these measurements do not establish the physical colour-locus mechanism."

    faintest = metrics["noise_aware_faint_bins"][">=21"]["models"]
    improvement_statements = []
    for model_name in ("monte_carlo_augmentation", "snr_weighted"):
        interval = faintest[model_name]["paired_macro_f1_difference_vs_baseline_95"]
        result = "positive over baseline" if interval["lower_95"] > 0 else "negative versus baseline" if interval["upper_95"] < 0 else "not distinguishable from baseline within this paired interval"
        improvement_statements.append(f"{model_name.replace('_', ' ')} is {result}")
    noise_text = (
        f"In the faintest r bin, baseline macro-F1 is {faintest['baseline']['score']['macro_f1']:.3f}, Monte Carlo augmentation is {faintest['monte_carlo_augmentation']['score']['macro_f1']:.3f}, and S/N weighting is {faintest['snr_weighted']['score']['macro_f1']:.3f}. "
        f"The paired 95% intervals imply that {improvement_statements[0]} and {improvement_statements[1]}, using identical faint test rows."
    )

    raw_cal = metrics["calibration"]["raw"]
    iso_cal = metrics["calibration"]["isotonic"]
    brier_delta = iso_cal["multiclass_brier"] - raw_cal["multiclass_brier"]
    ece_delta = iso_cal["top_label_ece"] - raw_cal["top_label_ece"]
    calibration_text = (
        f"Isotonic calibration was fit only on the grouped calibration fold ({metrics['calibration']['calibration_rows']:,} rows), then evaluated on the untouched spatial test fold. "
        f"Test Brier changed from {raw_cal['multiclass_brier']:.4f} to {iso_cal['multiclass_brier']:.4f} ({brier_delta:+.4f}) and top-label ECE from {raw_cal['top_label_ece']:.4f} to {iso_cal['top_label_ece']:.4f} ({ece_delta:+.4f}); per-r-bin results are reported separately because calibration may vary with brightness."
    )

    max_prob = next(row for row in metrics["abstention"]["max_calibrated_probability"] if row["threshold"] == 0.9)
    snr_threshold = next(row for row in metrics["abstention"]["colour_snr_rms"] if row["threshold"] == 5.0)
    abstention_text = (
        f"At max calibrated probability >=0.9, accuracy is {max_prob['accuracy']:.3f} at {max_prob['coverage']:.1%} coverage. "
        f"Requiring colour-S/N RMS >=5 yields {snr_threshold['accuracy']:.3f} accuracy at {snr_threshold['coverage']:.1%} coverage; confidence and measurement significance therefore reject different objects and should not be treated as interchangeable abstention rules."
    )

    set_results = metrics["feature_importance"]["sets"]
    a_perm = set_results["A_colours_only_psf"]["permutation"]
    a_perm_rank = {row["feature"]: rank for rank, row in enumerate(a_perm, start=1)}
    d_perm = set_results["D_colours_concentration_errors"]["permutation"]
    d_perm_rank = {row["feature"]: rank for rank, row in enumerate(d_perm, start=1)}
    dominant = all(a_perm_rank.get(feature, len(a_perm) + 1) <= 2 for feature in ("psf_color_u-g", "psf_color_g-r"))
    verdict_text = "Both u-g and g-r rank in the top two, supporting joint dominance." if dominant else f"They do not jointly occupy the top two in set A (u-g rank {a_perm_rank.get('psf_color_u-g')}, g-r rank {a_perm_rank.get('psf_color_g-r')}); the claim that both dominate is not supported by permutation importance."
    d_concentration = sum(1 for row in d_perm[:5] if row["feature"].startswith("concentration_"))
    importance_text = f"For colour-only set A, {verdict_text} In set D, {d_concentration} of the top five permutation features are concentration features; morphology/error inputs change what the model relies on, so A and D importances are interpreted separately."

    bootstrap = metrics["headline_bootstrap_95"]["XGBoost"]
    bootstrap_text = f"The headline colour-only XGBoost accuracy is {score['accuracy']:.4f} (95% CI {bootstrap['intervals']['accuracy']['lower_95']:.4f}–{bootstrap['intervals']['accuracy']['upper_95']:.4f}) and macro-F1 is {score['macro_f1']:.4f} (95% CI {bootstrap['intervals']['macro_f1']['lower_95']:.4f}–{bootstrap['intervals']['macro_f1']['upper_95']:.4f}). Intervals are class-stratified test-row bootstrap intervals, not spatial-block uncertainty intervals."

    return {
        "per_class_and_confusion": confusion_text,
        "qso_redshift_hypothesis": redshift_text,
        "noise_aware_faint_bins": noise_text,
        "calibration": calibration_text,
        "abstention": abstention_text,
        "explainability_A_D": importance_text,
        "headline_bootstrap": bootstrap_text,
    }


def save_sky_coverage_plot(frame: pd.DataFrame, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"STAR": "#087e78", "GALAXY": "#c3543f", "QSO": "#285a9f"}
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for label in CLASSES:
        subset = frame.loc[frame["class"].eq(label)]
        axis.scatter(subset["ra"], subset["dec"], s=1.0, alpha=0.42, label=label, c=colors[label], rasterized=True)
    axis.set(xlim=(360, 0), ylim=(-5, 75), xlabel="Right ascension (deg; reversed)", ylabel="Declination (deg)", title="SkyClass deterministic-hash sample coverage")
    axis.grid(alpha=0.18)
    axis.legend(markerscale=5, frameon=False, ncol=3, loc="lower center")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "plot": str(output_path),
        "rows": int(len(frame)),
        "ra_min_deg": float(frame["ra"].min()),
        "ra_max_deg": float(frame["ra"].max()),
        "dec_min_deg": float(frame["dec"].min()),
        "dec_max_deg": float(frame["dec"].max()),
        "ra_dec_block_count": int(pd.Series(spatial_groups(frame)).nunique()),
    }


def compare_cross_validation(frame: pd.DataFrame, groups: np.ndarray, features: list[str]) -> dict:
    labels = frame["class"].reset_index(drop=True)
    fixed_xgb = {"n_estimators": 220, "max_depth": 4, "learning_rate": 0.06, "subsample": 0.85, "colsample_bytree": 0.9, "reg_lambda": 2.0}
    schemes = {
        "grouped_spatial_5fold": list(GroupKFold(n_splits=5).split(frame, labels, groups)),
        "stratified_5fold": list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(frame, labels)),
    }
    results = {}
    for model_name in ("LogisticRegression", "RandomForest", "XGBoost", "MLP"):
        results[model_name] = {}
        for scheme, folds in schemes.items():
            fold_scores = []
            for fold_index, (train_index, validation_index) in enumerate(folds):
                model = make_comparison_models(SEED + fold_index, fixed_xgb)[model_name]
                weighted_fit(model, frame.iloc[train_index][features], labels.iloc[train_index])
                prediction = predict_labels(model, frame.iloc[validation_index][features])
                fold_scores.append(score_predictions(labels.iloc[validation_index], prediction)["macro_f1"])
            results[model_name][scheme] = {"fold_macro_f1": fold_scores, "mean": float(np.mean(fold_scores)), "sd": float(np.std(fold_scores, ddof=1))}
        grouped_mean = results[model_name]["grouped_spatial_5fold"]["mean"]
        stratified_mean = results[model_name]["stratified_5fold"]["mean"]
        delta = grouped_mean - stratified_mean
        results[model_name]["grouped_minus_stratified"] = float(delta)
        results[model_name]["interpretation"] = "no spatial leakage effect detected" if abs(delta) < 0.01 else "CV schemes differ by at least 0.01 macro-F1; this comparison alone does not establish the cause"
    return {"folds": 5, "features": features, "xgboost_params_fixed_before_cv": fixed_xgb, "models": results}


def write_report(metrics: dict, output_dir: Path) -> None:
    spatial = metrics["spatial_holdout"]
    ablations = metrics["feature_ablations_xgboost"]
    score_columns = ["Accuracy", "Macro-F1"] + [f"{label} {metric}" for label in CLASSES for metric in ("P", "R", "F1")]
    table_header = "| Model / feature set | " + " | ".join(score_columns) + " |"
    table_divider = "|---|" + "|".join(["---:"] * len(score_columns)) + "|"

    def score_values(score: dict) -> list[str]:
        values = [f"{score['accuracy']:.4f}", f"{score['macro_f1']:.4f}"]
        for label in CLASSES:
            values.extend(f"{score['per_class'][label][metric]:.4f}" for metric in ("precision", "recall", "f1-score"))
        return values

    def interval_text(interval: dict, metric: str) -> str:
        values = interval["intervals"][metric]
        return f"{values['lower_95']:.4f}–{values['upper_95']:.4f}"

    lines = [
        "# SkyClass DR17 evaluation",
        "",
        f"Data rows: {metrics['data']['retained_rows']:,} retained from {metrics['data']['input_rows']:,}.",
        f"Sampling: capped per class at {metrics['data']['limit_per_class']:,}; retained counts {metrics['data']['class_counts']}. Natural DR17 class proportions are not represented.",
        f"Sky coverage: RA {metrics['sky_coverage']['ra_min_deg']:.2f}–{metrics['sky_coverage']['ra_max_deg']:.2f} deg, Dec {metrics['sky_coverage']['dec_min_deg']:.2f}–{metrics['sky_coverage']['dec_max_deg']:.2f} deg, {metrics['sky_coverage']['ra_dec_block_count']} occupied RA/Dec blocks. Map: `sky_coverage.png`.",
        "Sampling query per class: `SELECT TOP N ... FROM SpecObj s JOIN PhotoObj p ON s.bestobjid=p.objid WHERE s.class='<class>' AND s.zWarning=0 ORDER BY CHECKSUM(s.specobjid), s.specobjid`. Exact expanded SQL is cached in the query sidecar. The prior `data/raw_sdss_dr17.csv` is preserved and superseded by the deterministic hash sample.",
        "",
        "## Spatial holdout: PSF colours only",
        "",
        table_header,
        table_divider,
    ]
    for name, score in metrics["model_comparison_colours_only"].items():
        lines.append("| " + name + " | " + " | ".join(score_values(score)) + " |")
    lines += ["", "## Feature ablation: XGBoost", "", table_header.replace("Model / feature set", "Feature set"), table_divider]
    for name, score in ablations.items():
        lines.append("| " + name + " | " + " | ".join(score_values(score["score"])) + " |")
    lines += [
        "",
        "## Generalization and selection shift",
        "",
        "Five-fold comparison using the same PSF-colour features and fixed model settings:",
        "",
        "| Model | Grouped macro-F1 mean ± SD | Stratified macro-F1 mean ± SD | Grouped - stratified | Interpretation |",
        "|---|---:|---:|---:|---|",
        *[f"| {name} | {result['grouped_spatial_5fold']['mean']:.4f} ± {result['grouped_spatial_5fold']['sd']:.4f} | {result['stratified_5fold']['mean']:.4f} ± {result['stratified_5fold']['sd']:.4f} | {result['grouped_minus_stratified']:.4f} | {result['interpretation']} |" for name, result in metrics["cross_validation_comparison"]["models"].items()],
        "",
        f"Bright-to-faint: train on r < 18 ({metrics['bright_to_faint']['train_rows']:,}); evaluate on r > 19 ({metrics['bright_to_faint']['test_rows']:,}). Full-faint macro-F1 {metrics['bright_to_faint']['score']['macro_f1']:.4f}, class-stratified bootstrap 95% CI {interval_text(metrics['bright_to_faint']['bootstrap_95'], 'macro_f1')}.",
        f"Matched same-range control: bright holdout macro-F1 {metrics['bright_to_faint']['matched_control']['bright_same_range_score']['macro_f1']:.4f} (95% CI {interval_text(metrics['bright_to_faint']['matched_control']['bright_same_range_bootstrap_95'], 'macro_f1')}); matched faint macro-F1 {metrics['bright_to_faint']['matched_control']['faint_matched_score']['macro_f1']:.4f} (95% CI {interval_text(metrics['bright_to_faint']['matched_control']['faint_matched_bootstrap_95'], 'macro_f1')}); both have {metrics['bright_to_faint']['matched_control']['bright_rows']:,} rows and identical class counts.",
        "",
        "| Bright/faint evaluation | Macro-F1 | 95% CI | STAR recall | GALAXY recall | QSO recall |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {title} | {score['macro_f1']:.4f} | {interval_text(intervals, 'macro_f1')} | "
            + " | ".join(f"{score['per_class'][label]['recall']:.4f} (95% CI {interval_text(intervals, f'{label}_recall')})" for label in CLASSES)
            + " |"
            for title, score, intervals in (
                ("Full faint", metrics["bright_to_faint"]["score"], metrics["bright_to_faint"]["bootstrap_95"]),
                ("Bright same-range control", metrics["bright_to_faint"]["matched_control"]["bright_same_range_score"], metrics["bright_to_faint"]["matched_control"]["bright_same_range_bootstrap_95"]),
                ("Faint size/class-matched", metrics["bright_to_faint"]["matched_control"]["faint_matched_score"], metrics["bright_to_faint"]["matched_control"]["faint_matched_bootstrap_95"]),
            )
        ],
        "",
        "The labelled sample is spectroscopically targeted, not a random census of photometric objects. In particular, the bright-to-faint and magnitude-bin results describe transfer within this labelled sample; they do not remove targeting/selection bias. The spatial holdout reduces local-neighbour leakage but does not make the labelled set representative.",
        "The matched control holds test size and class counts constant, so its bright/faint comparison reduces sample-size and class-mix effects. It remains observational: it cannot isolate a causal effect of brightness from correlated measurement quality or spectroscopic targeting.",
        "",
        f"Rows with any PSF-band error >= 1 mag: {metrics['error_analysis']['psf_error_ge_1mag']['rows']:,}; per class {metrics['error_analysis']['psf_error_ge_1mag']['by_class']}.",
        "| Dereddened r bin | Total high-error rows | STAR | GALAXY | QSO |",
        "|---|---:|---:|---:|---:|",
        *[f"| {magnitude_bin} | {counts['total']} | {counts['by_class']['STAR']} | {counts['by_class']['GALAXY']} | {counts['by_class']['QSO']} |" for magnitude_bin, counts in metrics["error_analysis"]["psf_error_ge_1mag"]["by_dereddened_psf_r_bin"].items()],
        "",
        "Uncertainty quintiles are recalculated on the same spatial-holdout predictions, once including and once excluding rows with any PSF error >= 1 mag:",
        "",
        f"Including outliers: highest-quintile macro-F1 {metrics['error_analysis']['uncertainty_quintiles_including_error_outliers']['quintiles'][-1]['macro_f1']:.4f} ({metrics['error_analysis']['uncertainty_quintiles_including_error_outliers']['quintiles'][-1]['rows']:,} rows).",
        f"Excluding {metrics['error_analysis']['uncertainty_quintiles_excluding_error_outliers']['excluded_error_outlier_rows']:,} outliers: highest-quintile macro-F1 {metrics['error_analysis']['uncertainty_quintiles_excluding_error_outliers']['quintiles'][-1]['macro_f1']:.4f} ({metrics['error_analysis']['uncertainty_quintiles_excluding_error_outliers']['quintiles'][-1]['rows']:,} rows).",
        "",
        "",
        "## Calibration, abstention, and robustness",
        "",
        f"XGBoost colour-only multiclass Brier score: {metrics['calibration']['multiclass_brier']:.4f}; top-label ECE: {metrics['calibration']['top_label_ece']:.4f}.",
        f"Monte Carlo training augmentation (two independent Gaussian perturbations per training object): accuracy {metrics['monte_carlo_noise_augmentation']['score']['accuracy']:.4f}, macro-F1 {metrics['monte_carlo_noise_augmentation']['score']['macro_f1']:.4f}.",
        f"S/N-weighted colour-only XGBoost: accuracy {metrics['snr_weighted_model']['score']['accuracy']:.4f}, macro-F1 {metrics['snr_weighted_model']['score']['macro_f1']:.4f}.",
        f"Majority-class baseline ({metrics['majority_class_baseline']['predicted_class']}): accuracy {metrics['majority_class_baseline']['score']['accuracy']:.4f}, macro-F1 {metrics['majority_class_baseline']['score']['macro_f1']:.4f}.",
        f"STAR→QSO confusion: {metrics['spatial_holdout']['star_to_qso_confusion']}; QSO→STAR confusion: {metrics['spatial_holdout']['qso_to_star_confusion']}.",
        f"Macro-F1 row-bootstrap 95% interval: {metrics['bootstrap_95_percent_confidence_intervals']['macro_f1']['lower_95']:.4f}–{metrics['bootstrap_95_percent_confidence_intervals']['macro_f1']['upper_95']:.4f}.",
        "",
        "Top permutation-importance features: " + ", ".join(f"{row['feature']} ({row['mean_macro_f1_drop']:.4f})" for row in metrics["feature_importance"]["permutation"]["results"][:3]) + ".",
        "Top mean absolute TreeSHAP contributions: " + ", ".join(f"{row['feature']} ({row['mean_abs_shap']:.4f})" for row in sorted(metrics["feature_importance"]["xgboost_tree_shap"]["mean_absolute_contributions"], key=lambda row: row["mean_abs_shap"], reverse=True)[:3]) + ".",
        "",
        "### QSO redshift bins (colour-only spatial holdout)",
        "",
        "| Redshift | Rows | Mean z | QSO recall |",
        "|---|---:|---:|---:|",
        *[f"| {row['bin']} | {row['rows']} | {row['mean_redshift']:.3f} | {row['qso_recall']:.4f} |" for row in metrics["error_analysis"]["qso_redshift_bins_colours_only_xgboost"]],
        "",
        "Monte Carlo augmentation uses independent Gaussian errors per band and caps each reported sigma at 0.5 mag to limit pathological uncertainty values.",
        "",
        "Reliability bins, accuracy-versus-coverage, bootstrap intervals, SHAP and permutation importance, full split proportions, and STAR↔QSO confusion counts are recorded in `metrics.json`.",
    ]
    headline_lines = [
        "",
        "## Phase 3 Error Analysis",
        "",
        "### Per-class performance and STAR↔QSO confusion",
        "",
        f"![Row-normalized confusion matrix](figures/{metrics['figures']['confusion']})",
        metrics["interpretations"]["per_class_and_confusion"],
        "",
        "| Headline model/feature set | Accuracy | Macro-F1 | STAR P | STAR R | STAR F1 | GALAXY P | GALAXY R | GALAXY F1 | QSO P | QSO R | QSO F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for display_name, bootstrap_key, score in [
        *((name, name, value) for name, value in metrics["model_comparison_colours_only"].items()),
        *((name, name, value["score"]) for name, value in metrics["feature_ablations_xgboost"].items()),
    ]:
        intervals = metrics["headline_bootstrap_95"][bootstrap_key]["intervals"]
        cells = []
        for metric_name, interval_key, class_name in (("accuracy", "accuracy", None), ("macro_f1", "macro_f1", None)):
            interval = intervals[interval_key]
            cells.append(f"{score[metric_name]:.4f} ({interval['lower_95']:.4f}–{interval['upper_95']:.4f})")
        for class_name in CLASSES:
            for short_name, score_key in (("P", "precision"), ("R", "recall"), ("F1", "f1-score")):
                interval_key = f"{class_name}_{'f1' if score_key == 'f1-score' else score_key}"
                interval = intervals[interval_key]
                value = score["per_class"][class_name][score_key]
                cells.append(f"{value:.4f} ({interval['lower_95']:.4f}–{interval['upper_95']:.4f})")
        headline_lines.append("| " + display_name + " | " + " | ".join(cells) + " |")
    headline_lines += [
        "",
        "### QSO redshift and the z≈2.5–3 hypothesis",
        "",
        f"![QSO recall by redshift with bootstrap intervals](figures/{metrics['figures']['redshift']})",
        metrics["interpretations"]["qso_redshift_hypothesis"],
        "",
        f"Test: {metrics['redshift_hypothesis_test']['hypothesis']}. The z=2.5–3 bin has {metrics['redshift_hypothesis_test']['z_2_5_to_3_rows']} QSOs; the adjacent comparison has {metrics['redshift_hypothesis_test']['adjacent_rows']}.",
        "",
        "| z bin | Rows | QSO recall (95% CI) | QSO assigned STAR (95% CI) |",
        "|---|---:|---:|---:|",
        *[f"| {row['bin']} | {row['rows']} | {row['qso_recall']:.4f} ({row['qso_recall_bootstrap_95']['lower_95']:.4f}–{row['qso_recall_bootstrap_95']['upper_95']:.4f}) | {row['qso_to_star_rate']:.4f} ({row['qso_to_star_bootstrap_95']['lower_95']:.4f}–{row['qso_to_star_bootstrap_95']['upper_95']:.4f}) |" for row in metrics["error_analysis"]["qso_redshift_bins_colours_only_xgboost"]],
        "",
        "### Noise-aware models on faint objects",
        "",
        f"![Faint-bin noise-aware comparison with bootstrap intervals](figures/{metrics['figures']['noise_faint']})",
        metrics["interpretations"]["noise_aware_faint_bins"],
        "",
        "| r bin | Model | Rows | Macro-F1 (95% CI) | Δ macro-F1 vs baseline (paired 95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for magnitude_bin, bin_data in metrics["noise_aware_faint_bins"].items():
        for model_name, result in bin_data["models"].items():
            score_ci = result["bootstrap_95"]["intervals"]["macro_f1"]
            difference = result.get("paired_macro_f1_difference_vs_baseline_95")
            difference_text = "baseline" if difference is None else f"{difference['difference_first_minus_second']:.4f} ({difference['lower_95']:.4f}–{difference['upper_95']:.4f})"
            headline_lines.append(f"| {magnitude_bin} | {model_name} | {bin_data['rows']} | {result['score']['macro_f1']:.4f} ({score_ci['lower_95']:.4f}–{score_ci['upper_95']:.4f}) | {difference_text} |")
    headline_lines += [
        "",
        "### Calibration",
        "",
        f"![Reliability diagrams and per-r calibration scores](figures/{metrics['figures']['calibration']})",
        metrics["interpretations"]["calibration"],
        "",
        "| r bin | Rows | Raw Brier (95% CI) | Isotonic Brier (95% CI) | Raw ECE (95% CI) | Isotonic ECE (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for entry in metrics["calibration"]["by_r_bin"]:
        raw_ci = entry["raw"]["bootstrap_95"]
        iso_ci = entry["isotonic"]["bootstrap_95"]
        headline_lines.append(f"| {entry['r_bin']} | {entry['rows']} | {entry['raw']['multiclass_brier']:.4f} ({raw_ci['multiclass_brier']['lower_95']:.4f}–{raw_ci['multiclass_brier']['upper_95']:.4f}) | {entry['isotonic']['multiclass_brier']:.4f} ({iso_ci['multiclass_brier']['lower_95']:.4f}–{iso_ci['multiclass_brier']['upper_95']:.4f}) | {entry['raw']['top_label_ece']:.4f} ({raw_ci['top_label_ece']['lower_95']:.4f}–{raw_ci['top_label_ece']['upper_95']:.4f}) | {entry['isotonic']['top_label_ece']:.4f} ({iso_ci['top_label_ece']['lower_95']:.4f}–{iso_ci['top_label_ece']['upper_95']:.4f}) |")
    headline_lines += [
        "",
        "### Abstention",
        "",
        f"![Accuracy versus coverage under two abstention rules](figures/{metrics['figures']['abstention']})",
        metrics["interpretations"]["abstention"],
        "",
        "| Rule | Threshold | Rows | Coverage | Accuracy (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for rule, rows in (("Max calibrated probability", metrics["abstention"]["max_calibrated_probability"]), ("Colour S/N RMS", metrics["abstention"]["colour_snr_rms"])):
        for row in rows:
            accuracy_ci = row.get("accuracy_bootstrap_95")
            interval = "n/a" if accuracy_ci is None else f"{accuracy_ci['lower_95']:.4f}–{accuracy_ci['upper_95']:.4f}"
            headline_lines.append(f"| {rule} | {row['threshold']:.2f} | {row['rows']} | {row['coverage']:.4f} | {row['accuracy']:.4f} ({interval}) |" if row["accuracy"] is not None else f"| {rule} | {row['threshold']:.2f} | 0 | 0.0000 | n/a |")
    headline_lines += [
        "",
        "### Explainability: feature sets A and D",
        "",
        f"![Permutation and TreeSHAP importance for A and D](figures/{metrics['figures']['importance']})",
        metrics["interpretations"]["explainability_A_D"],
        "",
        "| Feature set | Method | Rank | Feature | Importance |",
        "|---|---|---:|---|---:|",
    ]
    for set_name, importance in metrics["feature_importance"]["sets"].items():
        for rank, row in enumerate(importance["permutation"][:8], start=1):
            headline_lines.append(f"| {set_name} | Permutation macro-F1 drop | {rank} | {row['feature']} | {row['mean_macro_f1_drop']:.5f} |")
        shap_aggregate = {}
        for row in importance["tree_shap"]:
            shap_aggregate.setdefault(row["feature"], []).append(row["mean_abs_shap"])
        top_shap = sorted(((name, float(np.mean(values))) for name, values in shap_aggregate.items()), key=lambda item: item[1], reverse=True)[:8]
        for rank, (feature, value) in enumerate(top_shap, start=1):
            headline_lines.append(f"| {set_name} | Mean absolute TreeSHAP (class-mean) | {rank} | {feature} | {value:.5f} |")
    headline_lines += ["", metrics["interpretations"]["headline_bootstrap"]]
    lines.extend(headline_lines)
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    raw = fetch_sdss(args.cache, args.limit_per_class, refresh=args.refresh)
    raw_class_counts = raw["class"].value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict()
    frame, cleaning = prepare_photometry(raw)
    if frame.empty:
        raise RuntimeError("No rows remain after photometry validation")
    duplicate_objid_rows = int(frame["objid"].duplicated(keep=False).sum())
    duplicate_objid_count = int(frame["objid"].duplicated().sum())
    conflicting_ids = frame.groupby("objid")["class"].nunique()
    conflicting_ids = set(conflicting_ids[conflicting_ids > 1].index)
    conflicting_rows = int(frame["objid"].isin(conflicting_ids).sum())
    duplicate_rows_removed = int(frame.loc[~frame["objid"].isin(conflicting_ids), "objid"].duplicated().sum())
    frame = frame.loc[~frame["objid"].isin(conflicting_ids)].drop_duplicates("objid", keep="first").copy()
    frame = frame.reset_index(drop=True)
    sky_coverage = save_sky_coverage_plot(frame, output_dir / "sky_coverage.png")
    groups = spatial_groups(frame)

    group_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(group_split.split(frame, frame["class"], groups))
    y_train, y_test = frame.iloc[train_idx]["class"], frame.iloc[test_idx]["class"]
    xgb_params, tuning = tune_xgb(frame, train_idx, groups, FEATURE_SETS["A_colours_only_psf"])
    cv_comparison = compare_cross_validation(frame, groups, FEATURE_SETS["A_colours_only_psf"])

    color_features = FEATURE_SETS["A_colours_only_psf"]
    baseline_models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED)),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "XGBoost": make_xgb(xgb_params),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, max_iter=140, early_stopping=True, n_iter_no_change=12, random_state=SEED)),
    }
    comparison = {}
    color_test_predictions = {}
    color_test_probabilities = {}
    for name, model in baseline_models.items():
        weighted_fit(model, frame.iloc[train_idx][color_features], y_train)
        prediction = predict_labels(model, frame.iloc[test_idx][color_features])
        comparison[name] = score_predictions(y_test, prediction)
        color_test_predictions[name] = prediction
        color_test_probabilities[name] = model.predict_proba(frame.iloc[test_idx][color_features])
        joblib.dump(model, model_dir / f"{name.lower()}_colours_only.joblib")

    calibration_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 17)
    calibration_fit_local, calibration_local = next(
        calibration_split.split(train_idx, y_train, groups[train_idx])
    )
    calibration_fit_idx = train_idx[calibration_fit_local]
    calibration_idx = train_idx[calibration_local]
    if set(groups[calibration_fit_idx]).intersection(groups[calibration_idx]) or set(groups[calibration_fit_idx]).intersection(groups[test_idx]) or set(groups[calibration_idx]).intersection(groups[test_idx]):
        raise AssertionError("Spatial group leakage across calibration fit, calibration, and test partitions")
    calibration_model = make_xgb(xgb_params, seed=SEED + 17)
    weighted_fit(calibration_model, frame.iloc[calibration_fit_idx][color_features], frame.iloc[calibration_fit_idx]["class"])
    calibration_fold_probabilities = calibration_model.predict_proba(frame.iloc[calibration_idx][color_features])
    isotonic_models = fit_isotonic_calibration(calibration_fold_probabilities, frame.iloc[calibration_idx]["class"])
    test_raw_calibration_probabilities = calibration_model.predict_proba(frame.iloc[test_idx][color_features])
    test_isotonic_probabilities = calibrated_probabilities(test_raw_calibration_probabilities, isotonic_models)
    test_raw_calibration_labels = np.asarray(CLASSES)[test_raw_calibration_probabilities.argmax(axis=1)]
    test_isotonic_labels = np.asarray(CLASSES)[test_isotonic_probabilities.argmax(axis=1)]
    calibration_raw_metrics = reliability_metrics(y_test.reset_index(drop=True), test_raw_calibration_probabilities, bins=10)
    calibration_isotonic_metrics = reliability_metrics(y_test.reset_index(drop=True), test_isotonic_probabilities, bins=10)
    calibration_raw_metrics["bootstrap_95"] = bootstrap_probability_metrics(y_test.reset_index(drop=True), test_raw_calibration_probabilities, args.bootstrap_draws, SEED + 701)
    calibration_isotonic_metrics["bootstrap_95"] = bootstrap_probability_metrics(y_test.reset_index(drop=True), test_isotonic_probabilities, args.bootstrap_draws, SEED + 702)
    calibration_frame = test_frame = frame.iloc[test_idx].reset_index(drop=True)
    calibration_by_r = r_bin_calibration(calibration_frame, y_test.reset_index(drop=True), test_raw_calibration_probabilities, test_isotonic_probabilities)
    for index, entry in enumerate(calibration_by_r):
        bins_for_test = pd.cut(calibration_frame["psf_r"], [-np.inf, 18, 19, 20, 21, np.inf], labels=["<18", "18-19", "19-20", "20-21", ">=21"], include_lowest=True)
        bin_mask = (bins_for_test.astype(str) == entry["r_bin"]).to_numpy()
        bin_labels = y_test.reset_index(drop=True).iloc[np.flatnonzero(bin_mask)].reset_index(drop=True)
        for key, probabilities in (("raw", test_raw_calibration_probabilities), ("isotonic", test_isotonic_probabilities)):
            entry[key]["bootstrap_95"] = bootstrap_probability_metrics(bin_labels, probabilities[bin_mask], args.bootstrap_draws, SEED + 710 + index + (0 if key == "raw" else 20))
    calibration_results = {
        "method": "One-vs-rest isotonic regression with renormalized class probabilities",
        "fit_partition": "GroupShuffleSplit calibration fold drawn only from outer spatial training rows; calibration model fitted on remaining spatial training rows",
        "calibration_rows": int(len(calibration_idx)),
        "model_fit_rows": int(len(calibration_fit_idx)),
        "test_rows": int(len(test_idx)),
        "calibration_fit_test_group_overlap": 0,
        "calibration_test_group_overlap": 0,
        "calibration_fit_calibration_group_overlap": 0,
        "raw": calibration_raw_metrics,
        "isotonic": calibration_isotonic_metrics,
        "by_r_bin": calibration_by_r,
    }
    joblib.dump({"estimator": calibration_model, "calibrators": isotonic_models}, model_dir / "xgboost_colours_only_isotonic.joblib")

    ablations = {}
    ablation_predictions = {}
    ablation_models = {}
    for name, features in FEATURE_SETS.items():
        model = make_xgb(xgb_params)
        weighted_fit(model, frame.iloc[train_idx][features], y_train)
        prediction = predict_labels(model, frame.iloc[test_idx][features])
        ablations[name] = {"score": score_predictions(y_test, prediction), "features": features}
        ablation_predictions[name] = prediction
        ablation_models[name] = model
        joblib.dump(model, model_dir / f"xgboost_{name}.joblib")

    headline_bootstrap = {}
    for name, score in comparison.items():
        headline_bootstrap[name] = bootstrap_classification(y_test, color_test_predictions[name], args.bootstrap_draws, SEED + 801 + len(headline_bootstrap))
    for name, prediction in ablation_predictions.items():
        headline_bootstrap[name] = bootstrap_classification(y_test, prediction, args.bootstrap_draws, SEED + 821 + len(headline_bootstrap))

    stratified_scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fold_train, fold_test) in enumerate(cv.split(frame.iloc[train_idx], y_train), start=1):
        fold_train_idx, fold_test_idx = train_idx[fold_train], train_idx[fold_test]
        model = make_xgb(xgb_params, seed=SEED + fold)
        weighted_fit(model, frame.iloc[fold_train_idx][color_features], frame.iloc[fold_train_idx]["class"])
        prediction = predict_labels(model, frame.iloc[fold_test_idx][color_features])
        stratified_scores.append(score_predictions(frame.iloc[fold_test_idx]["class"], prediction)["macro_f1"])

    primary_prediction = ablation_predictions["A_colours_only_psf"]
    primary_model = baseline_models["XGBoost"]
    primary_probabilities = color_test_probabilities["XGBoost"]
    test_frame = frame.iloc[test_idx].reset_index(drop=True)
    brightness_edges = [-np.inf, 17, 18, 19, 20, 21, np.inf]
    brightness_labels = ["<17", "17-18", "18-19", "19-20", "20-21", ">=21"]
    brightness = binned_scores(test_frame.assign(prediction=primary_prediction), primary_prediction, "psf_r", brightness_edges, brightness_labels)
    noise_quintiles_all = uncertainty_quintiles(test_frame, primary_prediction, exclude_error_outliers=False)
    noise_quintiles_clean = uncertainty_quintiles(test_frame, primary_prediction, exclude_error_outliers=True)
    high_error_rows = high_error_distribution(frame)

    redshift_rows = []
    qso_mask = test_frame["class"].eq("QSO") & np.isfinite(pd.to_numeric(test_frame["redshift"], errors="coerce"))
    z_frame = test_frame.loc[qso_mask].copy()
    z_prediction = primary_prediction[qso_mask.to_numpy()]
    z_bins = pd.cut(z_frame["redshift"], bins=[-np.inf, 0, 0.5, 1, 2, 2.5, 3, 5, np.inf], labels=["<=0", "0-0.5", "0.5-1", "1-2", "2-2.5", "2.5-3", "3-5", ">5"], include_lowest=True)
    for label in z_bins.cat.categories:
        mask = z_bins == label
        if mask.any():
            indices = np.flatnonzero(mask.to_numpy())
            bin_prediction = z_prediction[indices]
            redshift_rows.append({"bin": str(label), "rows": int(mask.sum()), "qso_recall": float(np.mean(bin_prediction == "QSO")), "qso_to_star_rate": float(np.mean(bin_prediction == "STAR")), "qso_recall_bootstrap_95": bootstrap_qso_rates(bin_prediction, args.bootstrap_draws, SEED + 501 + len(redshift_rows))["qso_recall"], "qso_to_star_bootstrap_95": bootstrap_qso_rates(bin_prediction, args.bootstrap_draws, SEED + 601 + len(redshift_rows))["qso_to_star_rate"], "mean_redshift": float(z_frame.loc[mask, "redshift"].mean())})

    snr_model = make_xgb(xgb_params)
    snr_weights = snr_sample_weights(frame.iloc[train_idx], y_train)
    fit_with_weights(snr_model, frame.iloc[train_idx][color_features], y_train, snr_weights)
    snr_prediction = predict_labels(snr_model, frame.iloc[test_idx][color_features])
    snr_score = score_predictions(y_test, snr_prediction)
    joblib.dump(snr_model, model_dir / "xgboost_colours_only_snr_weighted.joblib")

    noise_copies = [noisy_training_copy(frame.iloc[train_idx], SEED + draw + 1) for draw in range(2)]
    augmented_train = pd.concat([frame.iloc[train_idx], *noise_copies], ignore_index=True)
    noise_model = make_xgb(xgb_params)
    weighted_fit(noise_model, augmented_train[color_features], augmented_train["class"])
    noise_prediction = predict_labels(noise_model, frame.iloc[test_idx][color_features])
    noise_score = score_predictions(y_test, noise_prediction)
    joblib.dump(noise_model, model_dir / "xgboost_colours_only_mc_augmented.joblib")

    majority_label = y_train.value_counts().idxmax()
    majority_prediction = np.full(len(test_idx), majority_label, dtype=object)
    majority_score = score_predictions(y_test, majority_prediction)

    calibration = reliability_metrics(y_test, primary_probabilities, bins=10)
    abstention = abstention_metrics(y_test.reset_index(drop=True), primary_probabilities)
    bootstrap = bootstrap_intervals(y_test, primary_prediction, draws=args.bootstrap_draws)

    snr_prediction_map = {
        "baseline": primary_prediction,
        "monte_carlo_augmentation": noise_prediction,
        "snr_weighted": snr_prediction,
    }
    noise_faint_bins = faint_model_comparison(test_frame, snr_prediction_map, args.bootstrap_draws)
    max_probability_abstention = add_abstention_intervals(
        abstention_metrics(y_test.reset_index(drop=True), test_isotonic_probabilities),
        y_test.reset_index(drop=True), test_isotonic_labels, test_isotonic_probabilities.max(axis=1), args.bootstrap_draws, SEED + 840,
    )
    test_colour_snr = colour_snr(test_frame)
    colour_snr_abstention_rows = add_abstention_intervals(
        colour_snr_abstention(y_test.reset_index(drop=True), test_isotonic_labels, test_colour_snr),
        y_test.reset_index(drop=True), test_isotonic_labels, test_colour_snr, args.bootstrap_draws, SEED + 860,
    )

    importance_sets = {
        "A_colours_only_psf": importance_for_model(primary_model, frame, test_idx, FEATURE_SETS["A_colours_only_psf"], args.importance_rows, SEED),
        "D_colours_concentration_errors": importance_for_model(ablation_models["D_colours_concentration_errors"], frame, test_idx, FEATURE_SETS["D_colours_concentration_errors"], args.importance_rows, SEED + 1),
    }

    importance_sample_size = min(args.importance_rows, len(test_idx))
    importance_indices = np.linspace(0, len(test_idx) - 1, importance_sample_size, dtype=int)
    importance_features = frame.iloc[test_idx[importance_indices]][color_features]
    importance_labels = y_test.iloc[importance_indices]
    permutation = permutation_importance(
        primary_model,
        importance_features,
        pd.Categorical(importance_labels, categories=CLASSES).codes,
        scoring="f1_macro",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )
    permutation_rows = sorted(
        [{"feature": name, "mean_macro_f1_drop": float(mean), "std": float(std)} for name, mean, std in zip(color_features, permutation.importances_mean, permutation.importances_std)],
        key=lambda row: row["mean_macro_f1_drop"],
        reverse=True,
    )
    shap_rows = min(args.importance_rows, len(test_idx))
    shap_indices = np.linspace(0, len(test_idx) - 1, shap_rows, dtype=int)
    shap_values = primary_model.get_booster().predict(
        xgboost.DMatrix(frame.iloc[test_idx[shap_indices]][color_features]),
        pred_contribs=True,
    )
    shap_importance = []
    for class_index, class_name in enumerate(CLASSES):
        per_feature = np.abs(shap_values[:, class_index, :-1]).mean(axis=0)
        shap_importance.extend({"class": class_name, "feature": name, "mean_abs_shap": float(value)} for name, value in zip(color_features, per_feature))

    train_ids, test_ids = set(frame.iloc[train_idx]["objid"]), set(frame.iloc[test_idx]["objid"])
    if train_ids.intersection(test_ids):
        raise AssertionError("objid leakage across the spatial train/test split")
    split_proportions = {
        split_name: {
            "rows": int(len(indices)),
            "class_counts": frame.iloc[indices]["class"].value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict(),
            "class_proportions": frame.iloc[indices]["class"].value_counts(normalize=True).reindex(CLASSES, fill_value=0).to_dict(),
        }
        for split_name, indices in (("train", train_idx), ("test", test_idx))
    }
    split_proportions["id_overlap_count"] = len(train_ids.intersection(test_ids))

    bright_idx = np.flatnonzero(frame["psf_r"].to_numpy() < 18)
    faint_idx = np.flatnonzero(frame["psf_r"].to_numpy() > 19)
    selection = {"train_rows": int(len(bright_idx)), "test_rows": int(len(faint_idx)), "train_class_counts": frame.iloc[bright_idx]["class"].value_counts().reindex(CLASSES, fill_value=0).to_dict(), "test_class_counts": frame.iloc[faint_idx]["class"].value_counts().reindex(CLASSES, fill_value=0).to_dict()}
    if len(bright_idx) and len(faint_idx) and frame.iloc[bright_idx]["class"].nunique() > 1:
        bright_fit_idx, bright_validation_idx = train_test_split(
            bright_idx,
            test_size=0.2,
            random_state=SEED,
            stratify=frame.iloc[bright_idx]["class"],
        )
        selection_model = make_xgb(xgb_params)
        selection_features = FEATURE_SETS["D_colours_concentration_errors"]
        weighted_fit(selection_model, frame.iloc[bright_fit_idx][selection_features], frame.iloc[bright_fit_idx]["class"])
        bright_validation_prediction = predict_labels(selection_model, frame.iloc[bright_validation_idx][selection_features])
        selection["bright_validation_rows"] = int(len(bright_validation_idx))
        selection["bright_validation_score"] = score_predictions(frame.iloc[bright_validation_idx]["class"], bright_validation_prediction)
        selection["bright_control_bootstrap_95"] = bootstrap_classification(
            frame.iloc[bright_validation_idx]["class"], bright_validation_prediction, args.bootstrap_draws, SEED + 101
        )
        faint_prediction = predict_labels(selection_model, frame.iloc[faint_idx][selection_features])
        selection["score"] = score_predictions(frame.iloc[faint_idx]["class"], faint_prediction)
        selection["bootstrap_95"] = bootstrap_classification(frame.iloc[faint_idx]["class"], faint_prediction, args.bootstrap_draws, SEED + 102)
        selection["macro_f1_degradation"] = selection["bright_validation_score"]["macro_f1"] - selection["score"]["macro_f1"]

        bright_counts = frame.iloc[bright_validation_idx]["class"].value_counts().reindex(CLASSES, fill_value=0).astype(int)
        rng = np.random.default_rng(SEED + 103)
        matched_faint_idx = np.concatenate([
            rng.choice(faint_idx[frame.iloc[faint_idx]["class"].to_numpy() == label], size=int(bright_counts[label]), replace=False)
            for label in CLASSES
        ])
        matched_faint_prediction = predict_labels(selection_model, frame.iloc[matched_faint_idx][selection_features])
        selection["matched_control"] = {
            "criterion": "faint test sampled to match bright-control test row count and per-class support exactly",
            "bright_rows": int(len(bright_validation_idx)),
            "faint_rows": int(len(matched_faint_idx)),
            "class_counts": bright_counts.to_dict(),
            "bright_same_range_score": selection["bright_validation_score"],
            "bright_same_range_bootstrap_95": selection["bright_control_bootstrap_95"],
            "faint_matched_score": score_predictions(frame.iloc[matched_faint_idx]["class"], matched_faint_prediction),
            "faint_matched_bootstrap_95": bootstrap_classification(frame.iloc[matched_faint_idx]["class"], matched_faint_prediction, args.bootstrap_draws, SEED + 104),
        }
        joblib.dump(selection_model, model_dir / "xgboost_bright_to_faint_D.joblib")
    else:
        selection["score"] = None
        selection["bright_validation_score"] = None
        selection["macro_f1_degradation"] = None
        selection["bootstrap_95"] = None
        selection["matched_control"] = None
        selection["note"] = "Insufficient bright/faint data or fewer than two classes in the bright training subset."

    qso_to_star = int(comparison["XGBoost"]["confusion_matrix"][CLASSES.index("QSO")][CLASSES.index("STAR")])
    star_to_qso = int(comparison["XGBoost"]["confusion_matrix"][CLASSES.index("STAR")][CLASSES.index("QSO")])
    z2p5_3 = z_prediction[(z_frame["redshift"].to_numpy() >= 2.5) & (z_frame["redshift"].to_numpy() < 3.0)]
    z_adjacent = z_prediction[((z_frame["redshift"].to_numpy() >= 2.0) & (z_frame["redshift"].to_numpy() < 2.5)) | ((z_frame["redshift"].to_numpy() >= 3.0) & (z_frame["redshift"].to_numpy() < 5.0))]
    qso_locus_test = {
        "hypothesis": "QSO objects at 2.5 <= z < 3 are more often classified as STAR than adjacent-z QSOs",
        "proxy": "classifier assignment to STAR; not direct measurement of colour-locus proximity",
        "z_2_5_to_3_rows": int(len(z2p5_3)),
        "adjacent_rows": int(len(z_adjacent)),
        "difference_qso_to_star_rate": bootstrap_qso_rate_difference(z2p5_3, z_adjacent, args.bootstrap_draws, SEED + 901) if len(z2p5_3) and len(z_adjacent) else None,
    }
    calibration_payload = {
        **calibration_results,
        "multiclass_brier": calibration_isotonic_metrics["multiclass_brier"],
        "top_label_ece": calibration_isotonic_metrics["top_label_ece"],
        "top_label_reliability": calibration_isotonic_metrics["top_label_reliability"],
        "one_vs_rest_reliability": calibration_isotonic_metrics["one_vs_rest_reliability"],
    }
    metrics = {
        "data": {**cleaning, "duplicate_objid_rows": duplicate_objid_rows, "duplicate_objid_repeated_rows": duplicate_objid_count, "conflicting_objid_count": len(conflicting_ids), "conflicting_objid_rows_removed": conflicting_rows, "duplicate_rows_removed": duplicate_rows_removed, "limit_per_class": args.limit_per_class, "raw_class_counts": raw_class_counts, "class_counts": frame["class"].value_counts().reindex(CLASSES, fill_value=0).to_dict(), "sampling_strategy": "Capped per class, up to limit_per_class, to retain evaluation support for all three classes; not natural class proportions.", "sampling": "Separate class-filtered DR17 TOP requests ordered by CHECKSUM(s.specobjid), s.specobjid; deterministic hash-ranked sample.", "query_metadata": str(Path(args.cache).with_suffix(".query.json")), "supersedes": "data/raw_sdss_dr17.csv (preserved unchanged)", "source": "SDSS DR17 SkyServer SQL REST"},
        "sky_coverage": sky_coverage,
        "spatial_holdout": {"train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)), "heldout_blocks": int(pd.Series(groups[test_idx]).nunique()), "block_definition": "floor(RA/15 degrees), floor((Dec+90)/10 degrees)", "test_class_counts": y_test.value_counts().reindex(CLASSES, fill_value=0).to_dict(), "splits": split_proportions, "duplicate_objid_overlap_count": len(train_ids.intersection(test_ids)), "duplicate_objid_overlap_checked": True, "star_to_qso_confusion": star_to_qso, "qso_to_star_confusion": qso_to_star},
        "hyperparameter_tuning": {"validation": "20% grouped spatial split within spatial training partition; test data not used", "candidates": tuning, "selected": xgb_params},
        "model_comparison_colours_only": comparison,
        "feature_ablations_xgboost": ablations,
        "stratified_5fold_colours_only": {"fold_macro_f1": stratified_scores, "macro_f1_mean": float(np.mean(stratified_scores)), "macro_f1_std": float(np.std(stratified_scores))},
        "cross_validation_comparison": cv_comparison,
        "error_analysis": {"psf_r_brightness_bins_colours_only_xgboost": brightness, "psf_error_ge_1mag": high_error_rows, "uncertainty_quintiles_including_error_outliers": noise_quintiles_all, "uncertainty_quintiles_excluding_error_outliers": noise_quintiles_clean, "qso_redshift_bins_colours_only_xgboost": redshift_rows},
        "redshift_hypothesis_test": qso_locus_test,
        "headline_score": comparison["XGBoost"],
        "headline_bootstrap_95": headline_bootstrap,
        "noise_aware_faint_bins": noise_faint_bins,
        "bright_to_faint": selection,
        "majority_class_baseline": {"predicted_class": majority_label, "score": majority_score},
        "calibration": calibration_payload,
        "abstention": {"confidence_definition": "maximum isotonic-calibrated class probability; colour S/N is RMS significance over eight PSF colours", "accuracy_vs_coverage": max_probability_abstention, "max_calibrated_probability": max_probability_abstention, "colour_snr_rms": colour_snr_abstention_rows},
        "bootstrap_95_percent_confidence_intervals": bootstrap,
        "feature_importance": {"permutation": {"evaluation_rows": importance_sample_size, "repeats": 5, "scoring": "macro-F1 drop after feature permutation", "results": permutation_rows}, "xgboost_tree_shap": {"evaluation_rows": shap_rows, "method": "XGBoost pred_contribs (TreeSHAP)", "mean_absolute_contributions": shap_importance}, "sets": importance_sets},
        "snr_weighted_model": {"score": snr_score, "weight_rule": "balanced class weight multiplied by sqrt(mean ugriz S/N), normalized by median and clipped to [0.25, 4.0]"},
        "monte_carlo_noise_augmentation": {"score": noise_score, "training_rows_original": int(len(train_idx)), "independent_augmented_copies_per_training_row": 2, "noise_model": "independent Gaussian PSF/model magnitude perturbations using reported per-band magnitude errors", "seed": SEED + 1},
        "software": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "seed": SEED,
    }
    metrics["figures"] = save_error_figures(metrics, output_dir)
    metrics["interpretations"] = build_interpretations(metrics)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    config = {"seed": SEED, "classes": list(CLASSES), "feature_sets": FEATURE_SETS, "selected_xgboost_params": xgb_params, "redshift_is_feature": False, "coordinates_are_features": False, "primary_model": "XGBoost", "primary_feature_set": "A_colours_only_psf", "ablation_model": "XGBoost"}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(metrics, output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/raw_sdss_dr17_hash.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--limit-per-class", type=int, default=33_333)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--importance-rows", type=int, default=3000)
    args = parser.parse_args()
    metrics = run(args)
    print(json.dumps({"retained_rows": metrics["data"]["retained_rows"], "spatial_macro_f1": {name: score["macro_f1"] for name, score in metrics["model_comparison_colours_only"].items()}, "artifacts": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()