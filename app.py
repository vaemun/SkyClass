from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from data_pipeline import BANDS
from train import CLASSES, COLOR_NAMES

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="SkyClass | SDSS photometric classifier", page_icon="✦", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink: #171d20; --muted: #58676b; --paper: #f3f5f2; --line: #ccd5d1; --teal: #087e78; --coral: #c3543f; }
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { max-width: 1320px; padding-top: 1.6rem; }
    h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
    .eyebrow { color: var(--teal); font: 500 0.78rem 'DM Mono', monospace; text-transform: uppercase; }
    .mono { font-family: 'DM Mono', monospace; }
    [data-testid="stMetric"] { border-top: 2px solid var(--line); padding-top: .7rem; }
    [data-testid="stMetricValue"] { color: var(--teal); }
    div[data-testid="stForm"] { border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.2rem; background: #fafbf9; }
    .stButton > button, .stFormSubmitButton > button { border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_metrics():
    path = ARTIFACTS / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def input_features(values: dict[str, dict[str, float]]) -> dict[str, float]:
    psf = {band: values[band]["psf"] - values[band]["extinction"] for band in BANDS}
    model = {band: values[band]["model"] - values[band]["extinction"] for band in BANDS}
    color_pairs = {
        "u-g": ("u", "g"), "g-r": ("g", "r"), "r-i": ("r", "i"), "i-z": ("i", "z"),
        "u-r": ("u", "r"), "g-i": ("g", "i"), "r-z": ("r", "z"), "u-z": ("u", "z"),
    }
    features = {f"psf_color_{name}": psf[first] - psf[second] for name, (first, second) in color_pairs.items()}
    features["psf_r"] = psf["r"]
    for band in BANDS:
        features[f"concentration_{band}"] = psf[band] - model[band]
    for name, (first, second) in color_pairs.items():
        features[f"psf_color_err_{name}"] = float(np.hypot(values[first]["error"], values[second]["error"]))
    return features


def select_model():
    choices = {
        "XGBoost · colours only": ("xgboost_A_colours_only_psf.joblib", "A_colours_only_psf"),
        "XGBoost · colours + r": ("xgboost_B_colours_plus_r.joblib", "B_colours_plus_r"),
        "XGBoost · colours + concentration": ("xgboost_C_colours_plus_concentration.joblib", "C_colours_plus_concentration"),
        "XGBoost · colours + concentration + errors": ("xgboost_D_colours_concentration_errors.joblib", "D_colours_concentration_errors"),
        "Logistic regression · colours only": ("logisticregression_colours_only.joblib", "A_colours_only_psf"),
        "Random forest · colours only": ("randomforest_colours_only.joblib", "A_colours_only_psf"),
        "MLP · colours only": ("mlp_colours_only.joblib", "A_colours_only_psf"),
        "XGBoost · S/N weighted colours": ("xgboost_colours_only_snr_weighted.joblib", "A_colours_only_psf"),
        "XGBoost · Monte Carlo noise augmented": ("xgboost_colours_only_mc_augmented.joblib", "A_colours_only_psf"),
    }
    available = {label: value for label, value in choices.items() if (ARTIFACTS / "models" / value[0]).exists()}
    if not available:
        return None, None
    label = st.selectbox("Model", list(available))
    filename, feature_set = available[label]
    return ARTIFACTS / "models" / filename, feature_set


metrics = load_metrics()
st.markdown("<div class='eyebrow'>SDSS DR17 · five-band photometry</div>", unsafe_allow_html=True)
st.title("SkyClass")
st.caption("Photometric classification with the failure modes left in view.")

predict_tab, analysis_tab, methods_tab = st.tabs(["Classify an object", "Error analysis", "Evaluation"])

with predict_tab:
    left, right = st.columns([1.35, 0.9], gap="large")
    with left:
        st.subheader("Enter ugriz photometry")
        st.caption("Magnitudes are observed values. Extinction is subtracted before colours are formed.")
        model_path, feature_set = select_model()
        with st.form("photometry_form"):
            columns = st.columns(5, gap="small")
            inputs = {}
            defaults = {"u": (19.3, 0.05, 19.2, 0.12), "g": (18.1, 0.02, 18.0, 0.08), "r": (17.6, 0.02, 17.5, 0.06), "i": (17.4, 0.02, 17.3, 0.04), "z": (17.3, 0.04, 17.2, 0.03)}
            for column, band in zip(columns, BANDS):
                with column:
                    st.markdown(f"**{band.upper()}**")
                    psf, error, model, extinction = defaults[band]
                    inputs[band] = {
                        "psf": st.number_input(f"PSF mag {band}", value=psf, step=0.01, format="%.3f", key=f"psf_{band}"),
                        "error": st.number_input(f"PSF error {band}", value=error, min_value=0.0, step=0.01, format="%.4f", key=f"err_{band}"),
                        "model": st.number_input(f"Model mag {band}", value=model, step=0.01, format="%.3f", key=f"model_{band}"),
                        "extinction": st.number_input(f"Extinction {band}", value=extinction, step=0.01, format="%.3f", key=f"ext_{band}"),
                    }
            submitted = st.form_submit_button("Classify", type="primary", use_container_width=True)
        if submitted and model_path:
            classifier = joblib.load(model_path)
            values = input_features(inputs)
            feature_names = json.loads((ARTIFACTS / "config.json").read_text(encoding="utf-8"))["feature_sets"][feature_set]
            row = pd.DataFrame([{name: values[name] for name in feature_names}])
            raw_prediction = classifier.predict(row)[0]
            predicted = CLASSES[int(raw_prediction)] if isinstance(raw_prediction, (int, np.integer)) else str(raw_prediction)
            probabilities = classifier.predict_proba(row)[0]
            classes = [CLASSES[int(label)] if isinstance(label, (int, np.integer)) else str(label) for label in classifier.classes_]
            st.session_state["last_prediction"] = {"class": predicted, "probabilities": dict(zip(classes, map(float, probabilities))), "features": values, "model": Path(model_path).name}
        elif submitted:
            st.error("No trained model found. Run `python train.py` to download data, train models, and create artifacts.")
    with right:
        st.subheader("Prediction")
        result = st.session_state.get("last_prediction")
        if result:
            st.markdown(f"<div class='eyebrow'>Predicted class</div><h1>{result['class']}</h1>", unsafe_allow_html=True)
            st.caption(f"Model artifact: `{result['model']}`")
            probability_frame = pd.DataFrame({"Class": list(result["probabilities"]), "Probability": list(result["probabilities"].values())}).set_index("Class")
            st.bar_chart(probability_frame, x_label="Class", y_label="Predicted probability", color="#087e78")
            with st.expander("Derived features"):
                st.dataframe(pd.DataFrame([result["features"]]).T.rename(columns={0: "Value"}), use_container_width=True)
        else:
            st.info("Fit a model with `python train.py`, then enter an object's five-band measurements.")
        if metrics:
            primary = metrics["model_comparison_colours_only"]["XGBoost"]
            st.metric("Spatial holdout macro-F1 · colours only", f"{primary['macro_f1']:.3f}")

with analysis_tab:
    if not metrics:
        st.info("Evaluation artifacts are not present yet. Run `python train.py` first.")
    else:
        st.subheader("Where performance changes")
        bins = metrics["error_analysis"]
        coverage_path = ARTIFACTS / "sky_coverage.png"
        if coverage_path.exists():
            st.image(str(coverage_path), caption="RA/Dec coverage of the deterministic hash sample")
        brightness = pd.DataFrame(bins["psf_r_brightness_bins_colours_only_xgboost"])
        if not brightness.empty:
            st.markdown("**Spatial holdout by dereddened PSF r magnitude**")
            st.bar_chart(brightness.set_index("bin")[["macro_f1"]], color="#087e78")
            st.dataframe(brightness[["bin", "rows", "accuracy", "macro_f1"]], hide_index=True, use_container_width=True)
        high_error = bins["psf_error_ge_1mag"]
        st.markdown(f"**PSF error >= 1 mag:** {high_error['rows']:,} rows")
        st.dataframe(pd.DataFrame([{"r bin": magnitude_bin, **counts, **counts["by_class"]} for magnitude_bin, counts in high_error["by_dereddened_psf_r_bin"].items()]).drop(columns=["by_class"], errors="ignore"), hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        for column, title, key in (
            (c1, "Including error outliers", "uncertainty_quintiles_including_error_outliers"),
            (c2, "Excluding any PSF error >= 1 mag", "uncertainty_quintiles_excluding_error_outliers"),
        ):
            with column:
                st.markdown(f"**{title}**")
                noise = pd.DataFrame(bins[key]["quintiles"])
                st.dataframe(noise[["bin", "rows", "mean_error_mag", "accuracy", "macro_f1"]], hide_index=True, use_container_width=True)
        redshift = pd.DataFrame(bins["qso_redshift_bins_colours_only_xgboost"])
        if not redshift.empty:
            st.markdown("**QSO recall by spectroscopic redshift (diagnostic only)**")
            redshift_figure = ARTIFACTS / "figures" / "qso_recall_by_redshift.png"
            if redshift_figure.exists():
                st.image(str(redshift_figure))
            st.bar_chart(redshift.set_index("bin")[["qso_recall"]], color="#c3543f")
            st.dataframe(redshift[["bin", "rows", "qso_recall", "qso_to_star_rate", "mean_redshift"]], hide_index=True, use_container_width=True)
            st.info(metrics["interpretations"]["qso_redshift_hypothesis"])
        st.warning("Redshift is never an input feature. These bins are post-hoc diagnostics on the spectroscopically selected sample.")

with methods_tab:
    if not metrics:
        st.info("No evaluation artifacts found. Run `python train.py` to build the report.")
    else:
        st.subheader("Spatial holdout · PSF colours only")
        comparison = metrics["model_comparison_colours_only"]
        score_table = pd.DataFrame({name: {"Accuracy": value["accuracy"], "Macro-F1": value["macro_f1"], **{f"{label} {metric}": value["per_class"][label][source] for label in CLASSES for metric, source in (("P", "precision"), ("R", "recall"), ("F1", "f1-score"))}} for name, value in comparison.items()}).T
        st.dataframe(score_table.style.format("{:.3f}"), use_container_width=True)
        st.subheader("XGBoost feature ablation")
        ablations = metrics["feature_ablations_xgboost"]
        ablation_table = pd.DataFrame({name: {"Accuracy": value["score"]["accuracy"], "Macro-F1": value["score"]["macro_f1"], **{f"{label} {metric}": value["score"]["per_class"][label][source] for label in CLASSES for metric, source in (("P", "precision"), ("R", "recall"), ("F1", "f1-score"))}} for name, value in ablations.items()}).T
        st.dataframe(ablation_table.style.format("{:.3f}"), use_container_width=True)
        cv = metrics["stratified_5fold_colours_only"]
        spatial = metrics["spatial_holdout"]
        selection = metrics["bright_to_faint"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Spatial test rows", f"{spatial['test_rows']:,}", f"{spatial['heldout_blocks']} held-out blocks")
        c2.metric("Stratified 5-fold macro-F1", f"{cv['macro_f1_mean']:.3f}", f"SD {cv['macro_f1_std']:.3f}")
        c3.metric("Bright → faint macro-F1", f"{selection['score']['macro_f1']:.3f}" if selection["score"] else "Unavailable")
        baseline = metrics["majority_class_baseline"]
        snr = metrics["snr_weighted_model"]["score"]
        augmented = metrics["monte_carlo_noise_augmentation"]["score"]
        st.subheader("Robustness and calibration")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Majority baseline macro-F1", f"{baseline['score']['macro_f1']:.3f}", baseline["predicted_class"])
        c2.metric("S/N-weighted macro-F1", f"{snr['macro_f1']:.3f}")
        c3.metric("Noise-augmented macro-F1", f"{augmented['macro_f1']:.3f}")
        c4.metric("Multiclass Brier / top-label ECE", f"{metrics['calibration']['multiclass_brier']:.3f} / {metrics['calibration']['top_label_ece']:.3f}")
        calibration_figure = ARTIFACTS / metrics["figures"]["calibration"]
        if calibration_figure.exists():
            st.image(str(calibration_figure), caption="Raw versus isotonic calibration on the untouched spatial test fold")
        calibration_rows = []
        for name, key in (("Uncalibrated", "raw"), ("Isotonic", "isotonic")):
            result = metrics["calibration"][key]
            calibration_rows.append({"Model": name, "Brier": result["multiclass_brier"], "Brier 95% CI": f"{result['bootstrap_95']['multiclass_brier']['lower_95']:.4f}–{result['bootstrap_95']['multiclass_brier']['upper_95']:.4f}", "ECE": result["top_label_ece"], "ECE 95% CI": f"{result['bootstrap_95']['top_label_ece']['lower_95']:.4f}–{result['bootstrap_95']['top_label_ece']['upper_95']:.4f}"})
        st.dataframe(pd.DataFrame(calibration_rows), hide_index=True, use_container_width=True)
        st.caption(metrics["interpretations"]["calibration"])
        st.markdown("**Calibration by r bin**")
        st.dataframe(pd.DataFrame([{"r bin": row["r_bin"], "rows": row["rows"], "raw Brier": row["raw"]["multiclass_brier"], "isotonic Brier": row["isotonic"]["multiclass_brier"], "raw ECE": row["raw"]["top_label_ece"], "isotonic ECE": row["isotonic"]["top_label_ece"]} for row in metrics["calibration"]["by_r_bin"]]), hide_index=True, use_container_width=True)
        reliability = pd.DataFrame(metrics["calibration"]["top_label_reliability"])
        if not reliability.empty:
            st.markdown("**Top-label reliability**")
            st.dataframe(reliability, hide_index=True, use_container_width=True)
        abstention = pd.DataFrame(metrics["abstention"]["accuracy_vs_coverage"])
        if not abstention.empty:
            st.markdown("**Accuracy as low-confidence predictions are withheld**")
            abstention_chart = alt.Chart(abstention).mark_line(point=True, color="#c3543f").encode(
                x=alt.X("coverage:Q", scale=alt.Scale(domain=[0, 1]), title="Coverage"),
                y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="Accuracy"),
                tooltip=["threshold:Q", "coverage:Q", "accuracy:Q"],
            )
            st.altair_chart(abstention_chart, use_container_width=True)
            st.dataframe(abstention, hide_index=True, use_container_width=True)
        st.markdown("**Colour-S/N RMS abstention**")
        st.dataframe(pd.DataFrame(metrics["abstention"]["colour_snr_rms"]), hide_index=True, use_container_width=True)
        abstention_figure = ARTIFACTS / metrics["figures"]["abstention"]
        if abstention_figure.exists():
            st.image(str(abstention_figure))
        st.caption(metrics["interpretations"]["abstention"])
        st.subheader("Noise-aware faint-bin comparison")
        noise_figure = ARTIFACTS / metrics["figures"]["noise_faint"]
        if noise_figure.exists():
            st.image(str(noise_figure))
        noise_rows = []
        for magnitude_bin, content in metrics["noise_aware_faint_bins"].items():
            for model_name, result in content["models"].items():
                interval = result["bootstrap_95"]["intervals"]["macro_f1"]
                noise_rows.append({"r bin": magnitude_bin, "Model": model_name, "Rows": content["rows"], "Macro-F1": result["score"]["macro_f1"], "Macro-F1 95% CI": f"{interval['lower_95']:.4f}–{interval['upper_95']:.4f}"})
        st.dataframe(pd.DataFrame(noise_rows), hide_index=True, use_container_width=True)
        st.caption(metrics["interpretations"]["noise_aware_faint_bins"])
        importance = metrics["feature_importance"]
        st.subheader("Colour importance")
        importance_figure = ARTIFACTS / metrics["figures"]["importance"]
        if importance_figure.exists():
            st.image(str(importance_figure))
        st.caption(metrics["interpretations"]["explainability_A_D"])
        permutation = pd.DataFrame(importance["permutation"]["results"])
        shap = pd.DataFrame(importance["xgboost_tree_shap"]["mean_absolute_contributions"])
        if not permutation.empty:
            st.dataframe(permutation, hide_index=True, use_container_width=True)
        if not shap.empty:
            st.caption("TreeSHAP mean absolute contribution by class")
            st.dataframe(shap, hide_index=True, use_container_width=True)
        st.subheader("Bootstrap 95% intervals")
        st.json(metrics["bootstrap_95_percent_confidence_intervals"])
        st.caption("Stratified folds share the same spatial footprint and can place neighbouring sky blocks in both train and validation. The primary grouped holdout keeps complete 15° RA × 10° Dec blocks out of training.")
        st.subheader("Grouped versus stratified five-fold CV")
        cv_rows = []
        for name, result in metrics["cross_validation_comparison"]["models"].items():
            cv_rows.append({"Model": name, "Grouped mean": result["grouped_spatial_5fold"]["mean"], "Grouped SD": result["grouped_spatial_5fold"]["sd"], "Stratified mean": result["stratified_5fold"]["mean"], "Stratified SD": result["stratified_5fold"]["sd"], "Grouped - stratified": result["grouped_minus_stratified"], "Interpretation": result["interpretation"]})
        st.dataframe(pd.DataFrame(cv_rows), hide_index=True, use_container_width=True)
        control = selection["matched_control"]
        if control:
            st.subheader("Bright/faint control, size and class matched")
            st.caption(control["criterion"])
            st.dataframe(pd.DataFrame([
                {"Evaluation": "Bright same-range control", "Rows": control["bright_rows"], "Macro-F1": control["bright_same_range_score"]["macro_f1"], **{f"{label} recall": control["bright_same_range_score"]["per_class"][label]["recall"] for label in CLASSES}},
                {"Evaluation": "Faint matched", "Rows": control["faint_rows"], "Macro-F1": control["faint_matched_score"]["macro_f1"], **{f"{label} recall": control["faint_matched_score"]["per_class"][label]["recall"] for label in CLASSES}},
            ]), hide_index=True, use_container_width=True)
        st.caption("The matched control holds evaluation size and class counts constant. Bright-to-faint differences remain observational and cannot separate brightness from correlated noise or spectroscopic targeting.")
        st.warning("Spectroscopic targeting is colour- and magnitude-dependent. Metrics quantify performance on this labelled sample, not a representative census of all SDSS photometric detections.")
        st.download_button("Download metrics.json", (ARTIFACTS / "metrics.json").read_bytes(), file_name="skyclass-metrics.json", mime="application/json")