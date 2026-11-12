import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# Configuration
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "artifacts" / "data"
REPORTS_DIR = ROOT_DIR / "artifacts" / "reports"

MONITORING_REPORT_PATH = REPORTS_DIR / "monitoring_report.json"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.json"
BASELINE_DATA_PATH = DATA_DIR / "baseline_data.csv"
NEW_BATCH_DATA_PATH = DATA_DIR / "new_batch.csv"


st.set_page_config(
    page_title="ModelWatch | Model Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# UI styling
# ============================================================
st.markdown(
    """
    <style>
      .block-container {max-width: 1480px; padding-top: 1.35rem; padding-bottom: 2rem;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.22);}
      [data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
      .mw-brand {font-size:1.45rem; font-weight:800; letter-spacing:-.4px;}
      .mw-subtitle {color:rgba(180,190,205,.78); font-size:.88rem;}
      .mw-hero {
        padding:1.35rem 1.55rem; border-radius:18px;
        background:linear-gradient(115deg,rgba(56,115,190,.20),rgba(30,160,130,.11));
        border:1px solid rgba(128,128,128,.24); margin-bottom:1.15rem;
      }
      .mw-hero h1 {font-size:1.75rem; margin:0 0 .25rem 0;}
      .mw-hero p {margin:0; opacity:.78;}
      .mw-card {
        border:1px solid rgba(128,128,128,.22); border-radius:14px;
        padding:1rem 1.1rem; background:rgba(128,128,128,.035);
      }
      .mw-muted {opacity:.72; font-size:.88rem;}
      .mw-good {padding:.85rem 1rem;border-radius:12px;background:rgba(30,160,100,.12);border:1px solid rgba(30,160,100,.3);}
      .mw-alert {padding:.85rem 1rem;border-radius:12px;background:rgba(220,65,65,.10);border:1px solid rgba(220,65,65,.3);}
      div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.20);padding:.7rem .85rem;border-radius:12px;background:rgba(128,128,128,.025);}
      .stButton button {border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# File loading and helpers
# ============================================================
@st.cache_data(show_spinner=False)
def load_json(path_string: str, modified_time: float):
    with open(path_string, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_csv(path_string: str, modified_time: float):
    return pd.read_csv(path_string)


def read_json(path: Path):
    if not path.exists():
        return None
    return load_json(str(path), path.stat().st_mtime)


def read_csv(path: Path):
    if not path.exists():
        return None
    return load_csv(str(path), path.stat().st_mtime)


def number_or_none(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def format_score(value):
    value = number_or_none(value)
    return "N/A" if value is None else f"{value:.3f}"


def load_dashboard_data():
    report = read_json(MONITORING_REPORT_PATH)
    if report is None:
        return None, {}, {}, {}, {}, {}, {}

    evaluation = report.get("model_evaluation", {})
    drift = report.get("data_drift", {})
    alerts = report.get("alerts", {})

    separate_drift = read_json(DRIFT_REPORT_PATH)
    if separate_drift:
        for key in (
            "feature_results", "total_features", "drifted_features",
            "drift_percentage", "drift_detected", "method",
        ):
            if drift.get(key) is None:
                drift[key] = separate_drift.get(key)

    baseline_metrics = evaluation.get("baseline_metrics", {})
    current_metrics = evaluation.get(
        "new_batch_metrics", evaluation.get("metrics", {})
    )
    comparison = evaluation.get("metric_comparison", {})
    return (
        report, evaluation, drift, alerts,
        baseline_metrics, current_metrics, comparison,
    )


# ============================================================
# Sidebar navigation
# ============================================================
(
    report, evaluation, drift, alerts,
    baseline_metrics, current_metrics, comparison,
) = load_dashboard_data()

with st.sidebar:
    st.markdown('<div class="mw-brand">📊 ModelWatch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mw-subtitle">Model monitoring workspace</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("↻ Refresh latest reports", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Model Performance",
            "Data Drift",
            "Data Explorer",
            "Alerts & Recommendations",
            "Understanding Results",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.caption("ModelWatch · Local monitoring interface")


st.markdown(
    """
    <div class="mw-hero">
      <h1>Model Monitoring Center</h1>
      <p>Understand model quality, detect data changes, and investigate monitoring signals.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if report is None:
    st.error("Monitoring report not found.")
    st.info(
        "Generate the report first, then refresh this dashboard."
    )
    st.stop()


# ============================================================
# Overview
# ============================================================
if page == "Overview":
    st.subheader("Overview")
    st.caption("A concise snapshot of the latest monitoring run.")

    status = str(report.get("status", "UNKNOWN")).upper()
    if status == "HEALTHY":
        st.success("Monitoring status: HEALTHY")
    elif status == "ALERT":
        st.warning("Monitoring status: ALERT — investigate active signals.")
    else:
        st.info(f"Monitoring status: {status}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", evaluation.get("model_name", "N/A"), help="The algorithm selected and saved during model training.")
    c2.metric("New-batch records", evaluation.get("new_batch_rows", "N/A"), help="Number of incoming records evaluated by the monitoring pipeline.")
    c3.metric("Drifted features", drift.get("drifted_features", "N/A"), help="Number of input features whose distributions differ significantly from the baseline.")
    drift_pct = number_or_none(drift.get("drift_percentage"))
    c4.metric("Drift percentage", "N/A" if drift_pct is None else f"{drift_pct:.2f}%", help="Percentage of checked input features flagged as drifted.")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Performance snapshot")
        p1, p2 = st.columns(2)
        p1.metric("Accuracy", format_score(current_metrics.get("accuracy")))
        p2.metric("F1-score", format_score(current_metrics.get("f1_score")))
    with right:
        st.markdown("#### Monitoring signals")
        if alerts.get("data_drift", drift.get("drift_detected", False)):
            st.error("Data drift detected")
        else:
            st.success("No data drift alert")
        if alerts.get(
            "performance_degradation", evaluation.get("performance_alert", False)
        ):
            st.error("Performance degradation detected")
        else:
            st.success("No performance degradation alert")

    st.caption(f"Report generated (UTC): {report.get('generated_at_utc', 'N/A')}")
    if report.get("data_note"):
        st.info(report["data_note"])


# ============================================================
# Model Performance
# ============================================================
elif page == "Model Performance":
    st.subheader("Model Performance")
    st.caption("Compare baseline evaluation with the latest incoming batch. Higher scores generally indicate better classification performance.")

    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    cols = st.columns(5)
    for col, name in zip(cols, metric_names):
        col.metric(name.replace("_", " ").title(), format_score(current_metrics.get(name)), help={"accuracy": "Percentage of all predictions that were correct.", "precision": "Of the records predicted as class 1 (benign), how many were actually class 1.", "recall": "Of all actual class 1 (benign) records, how many the model correctly identified.", "f1_score": "A balance between precision and recall.", "roc_auc": "How well the model ranks class 1 above class 0 across thresholds."}[name])

    rows = []
    for name in metric_names:
        old = number_or_none(baseline_metrics.get(name))
        new = number_or_none(current_metrics.get(name))
        if old is not None or new is not None:
            rows.append({
                "Metric": name.replace("_", " ").title(),
                "Baseline": old,
                "New batch": new,
                "Difference": new - old if old is not None and new is not None else None,
            })

    if rows:
        df = pd.DataFrame(rows)
        long_df = df.melt(
            id_vars="Metric",
            value_vars=["Baseline", "New batch"],
            var_name="Dataset",
            value_name="Score",
        ).dropna(subset=["Score"])
        fig = px.bar(
            long_df, x="Metric", y="Score", color="Dataset",
            barmode="group", range_y=[0, 1], text_auto=".3f",
        )
        fig.update_layout(height=430, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        with st.expander("How should I read these performance metrics?"):
            st.markdown(
                """
                - **Accuracy:** Share of all predictions that were correct.
                - **Precision:** Of the records predicted as the positive class (benign = 1), how many were actually positive.
                - **Recall:** Of all actual positive records, how many the model correctly identified.
                - **F1-score:** A balanced score combining precision and recall. It is useful when both missed positives and incorrect positive predictions matter.
                - **ROC-AUC:** How well the model ranks positive examples above negative examples across different classification thresholds. A value closer to 1 is generally better.
                - **Baseline:** The model's reference evaluation score from Step 7.
                - **New batch:** The score calculated on the latest incoming batch.
                - **Difference:** New-batch score minus baseline score. A negative value means the score decreased.
                """
            )
    else:
        st.info("No performance metrics are available in the monitoring report.")


# ============================================================
# Data Drift
# ============================================================
elif page == "Data Drift":
    st.subheader("Data Drift")
    st.caption("Feature-level statistical comparison between baseline and incoming data. A low p-value is evidence that a feature distribution changed.")

    d1, d2, d3 = st.columns(3)
    d1.metric("Features checked", drift.get("total_features", "N/A"))
    d2.metric("Drifted features", drift.get("drifted_features", "N/A"))
    pct = number_or_none(drift.get("drift_percentage"))
    d3.metric("Drift percentage", "N/A" if pct is None else f"{pct:.2f}%")

    feature_results = drift.get("feature_results", []) or []
    if feature_results:
        drift_df = pd.DataFrame(feature_results)
        aliases = {
            "feature_name": "feature",
            "pvalue": "p_value",
            "statistic": "ks_statistic",
            "is_drift": "drift_detected",
        }
        for old, new in aliases.items():
            if old in drift_df.columns and new not in drift_df.columns:
                drift_df.rename(columns={old: new}, inplace=True)

        st.dataframe(drift_df, use_container_width=True, hide_index=True)
        with st.expander("How should I understand the drift results?"):
            st.markdown(
                """
                - **Feature:** Input column being checked.
                - **KS statistic:** Measures the largest distance between the baseline and new-batch cumulative distributions. Larger values indicate a bigger distribution difference.
                - **p-value:** Indicates how much evidence the test provides against the assumption that both samples come from the same distribution. In this project, p < 0.05 is treated as evidence of drift.
                - **Drift detected:** Whether the feature crossed the configured significance rule.
                - **Drift percentage:** Drifted features divided by all checked features, multiplied by 100.

                A drift alert means the incoming inputs look different from the reference data. It does **not** automatically mean the model's predictions became inaccurate.
                """
            )

        if "p_value" in drift_df.columns and "feature" in drift_df.columns:
            plot_df = drift_df.copy()
            plot_df["p_value"] = pd.to_numeric(plot_df["p_value"], errors="coerce")
            plot_df = plot_df.dropna(subset=["p_value"])
            if not plot_df.empty:
                plot_df["-log10(p-value)"] = -plot_df["p_value"].clip(
                    lower=1e-300
                ).apply(lambda x: __import__("math").log10(x))
                fig = px.bar(
                    plot_df.sort_values("p_value"),
                    x="feature",
                    y="-log10(p-value)",
                    hover_data=["p_value"],
                    title="KS-test evidence by feature",
                )
                fig.add_hline(
                    y=-__import__("math").log10(0.05),
                    line_dash="dash",
                    annotation_text="p-value = 0.05",
                )
                fig.update_layout(height=440, xaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature-level drift results are not present in the report.")


# ============================================================
# Data Explorer
# ============================================================
elif page == "Data Explorer":
    st.subheader("Data Explorer")
    st.caption("Inspect one numerical feature at a time. Compare the shape, center, and spread of the baseline and incoming distributions.")

    baseline_df = read_csv(BASELINE_DATA_PATH)
    new_df = read_csv(NEW_BATCH_DATA_PATH)

    if baseline_df is None or new_df is None:
        st.warning("Baseline or new-batch CSV was not found in artifacts/data/.")
    else:
        features = [
            col for col in baseline_df.columns
            if col in new_df.columns and col != "target"
            and pd.api.types.is_numeric_dtype(baseline_df[col])
            and pd.api.types.is_numeric_dtype(new_df[col])
        ]
        if features:
            feature = st.selectbox("Select feature", features)
            base = baseline_df[feature].dropna()
            batch = new_df[feature].dropna()

            a, b = st.columns(2)
            a.metric("Baseline mean", f"{base.mean():.4f}")
            b.metric("New-batch mean", f"{batch.mean():.4f}")

            combined = pd.concat([
                pd.DataFrame({"Value": base, "Dataset": "Baseline"}),
                pd.DataFrame({"Value": batch, "Dataset": "New batch"}),
            ], ignore_index=True)

            fig = px.histogram(
                combined,
                x="Value",
                color="Dataset",
                barmode="overlay",
                histnorm="probability density",
                opacity=0.58,
                marginal="box",
                title=f"Distribution comparison — {feature}",
            )
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shared numerical features are available.")


# ============================================================
# Alerts & Recommendations
# ============================================================
elif page == "Alerts & Recommendations":
    st.subheader("Alerts & Recommendations")
    st.caption("Review the two monitoring signals independently.")

    drift_alert = bool(
        alerts.get("data_drift", drift.get("drift_detected", False))
    )
    performance_alert = bool(
        alerts.get(
            "performance_degradation",
            evaluation.get("performance_alert", False),
        )
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Data drift")
        if drift_alert:
            st.error("Data drift alert is active.")
            st.write(
                "Inspect the flagged features, p-values, feature units, missing "
                "values, and the incoming data source."
            )
        else:
            st.success("No overall data drift alert.")
    with right:
        st.markdown("#### Performance degradation")
        if performance_alert:
            st.error("Performance alert is active.")
            st.write(
                "Review class-wise errors, confusion matrix, label quality, and "
                "whether the batch is representative before considering retraining."
            )
        else:
            st.success("No performance degradation alert.")

    st.divider()
    if report.get("data_note"):
        st.warning(report["data_note"])


# ============================================================
# Understanding Results
# ============================================================
elif page == "Understanding Results":
    st.subheader("Understanding Your Monitoring Results")
    st.caption(
        "A simple guide to the values, charts, and alerts shown in ModelWatch. "
        "Use this page whenever you want to understand what an output means."
    )

    st.info(
        "This page explains the monitoring results generated by your project. "
        "It does not change, recalculate, or make decisions about the model."
    )

    with st.expander("1. What does the Overview page tell me?", expanded=True):
        st.markdown(
            """
            The Overview page gives you a quick health check of the latest monitoring run.

            - **Model:** The machine-learning algorithm used to make predictions.
            - **New-batch records:** How many incoming rows were evaluated.
            - **Drifted features:** How many input columns showed a statistically noticeable change.
            - **Drift percentage:** The share of checked features that were flagged.
            - **Monitoring status:** A combined status based on the drift and performance alerts.

            **HEALTHY** means neither configured alert is active. **ALERT** means at least one alert needs attention. It does not necessarily mean the model has failed.
            """
        )

    with st.expander("2. What do the model performance metrics mean?"):
        st.markdown(
            """
            These metrics compare the model's predictions with the known labels in the evaluation data.

            | Metric | Simple meaning | How to read it |
            |---|---|---|
            | **Accuracy** | How often the model was correct overall. | 0.95 means 95 out of 100 predictions were correct. |
            | **Precision** | How reliable positive predictions were. | Higher precision means fewer false positive predictions. |
            | **Recall** | How many actual positive cases the model found. | Higher recall means fewer positive cases were missed. |
            | **F1-score** | A combined balance of precision and recall. | Useful when both precision and recall matter. |
            | **ROC-AUC** | How well the model separates the two classes across thresholds. | Closer to 1 generally indicates stronger separation; 0.5 is similar to random ranking. |

            In this project, **class 1 means benign** and **class 0 means malignant**. Therefore, precision and recall refer to the benign class when calculated with the default positive label.
            """
        )

    with st.expander("3. What is the baseline vs new-batch comparison?"):
        st.markdown(
            """
            - **Baseline:** The reference score saved during the original model evaluation.
            - **New batch:** The score calculated when the saved model is evaluated on the latest incoming batch.
            - **Difference:** New-batch score minus baseline score.

            A negative difference means the metric decreased. A positive difference means it increased. Compare the same metric across both columns; do not compare accuracy directly with F1-score because they measure different things.
            """
        )

    with st.expander("4. What is data drift?"):
        st.markdown(
            """
            Data drift happens when the values or patterns in incoming input features change compared with the reference data used earlier.

            Example: if the baseline average value of a feature is around 10, but the incoming batch is mostly around 15, its distribution may have shifted.

            Drift can happen because of changes in data collection, user behaviour, equipment, the environment, or the population being measured. Drift is a reason to investigate the data; it is not proof that the model is wrong.
            """
        )

    with st.expander("5. How do I read the KS statistic and p-value?"):
        st.markdown(
            """
            This project uses the **Kolmogorov–Smirnov (KS) two-sample test** to compare numerical feature distributions.

            - **KS statistic:** The largest distance between the two cumulative distribution curves. A larger value indicates a greater difference between the samples.
            - **p-value:** How surprising a difference this large would be if both samples came from the same distribution.
            - **p-value below 0.05:** The project treats this as evidence that the feature has drifted.
            - **p-value equal to or above 0.05:** The test did not find enough evidence to flag that feature.

            A p-value is not the probability that the feature has drifted. It also does not measure how important the feature is to the model.
            """
        )

    with st.expander("6. How do I understand the charts?"):
        st.markdown(
            """
            **Performance comparison chart**
            - Each group represents one metric.
            - One bar shows the baseline score and the other shows the new-batch score.
            - A shorter new-batch bar means that metric decreased.

            **KS-test evidence chart**
            - Each bar represents a feature.
            - It visualizes the p-value using a negative logarithmic scale.
            - A taller bar means a smaller p-value and stronger statistical evidence of a distribution difference.
            - The dashed line marks the p = 0.05 reference threshold.

            **Feature distribution chart**
            - The two overlaid distributions show baseline and incoming values for the selected feature.
            - Similar shapes suggest similar distributions.
            - A shift in location, spread, or shape may indicate a change.
            - The box plot helps compare the median and spread.
            """
        )

    with st.expander("7. What do the alerts mean?"):
        st.markdown(
            """
            **Data drift alert**
            - Means one or more feature distributions crossed the configured drift rule.
            - Check the flagged features, their p-values, source data, units, and missing values.

            **Performance degradation alert**
            - Means the new-batch F1-score fell below the baseline by more than the configured tolerance.
            - Check the labels, class distribution, false positives, false negatives, and whether the batch is representative.

            **Overall ALERT**
            - Appears when either the data drift alert or performance degradation alert is active.
            - Drift and performance are separate signals: drift can happen without a performance drop.
            """
        )

    with st.expander("8. What does the F1 drop threshold mean?"):
        threshold = evaluation.get("f1_drop_threshold")
        f1_drop = evaluation.get("f1_drop")
        if threshold is not None:
            st.write(
                f"The configured F1-score drop tolerance in the latest report is **{threshold}**."
            )
        if f1_drop is not None:
            st.write(f"The measured baseline-to-new-batch F1-score drop is **{f1_drop}**.")
        st.markdown(
            """
            The project raises a performance alert when the F1-score drop is greater than the configured tolerance. For example, a tolerance of 0.05 means a drop of more than five percentage points triggers the alert.
            """
        )

    with st.expander("9. Is this real production monitoring?"):
        st.markdown(
            """
            No. This assignment uses a **simulated incoming batch** created from the Breast Cancer Wisconsin reference dataset. Its original labels are retained so the project can demonstrate offline evaluation.

            This is useful for demonstrating the monitoring workflow, but the results should not be described as verified live-production performance. In a real production system, labels would usually arrive later from a trusted source and would need validation.
            """
        )

    with st.expander("10. What should I do when I see ALERT?"):
        st.markdown(
            """
            1. Open **Alerts & Recommendations** to see which signal is active.
            2. If data drift is active, inspect the flagged features in **Data Drift**.
            3. Use **Data Explorer** to compare the baseline and incoming feature distributions.
            4. If performance degradation is active, review the metric comparison and validate the incoming labels.
            5. Do not retrain automatically just because drift is detected. First confirm that the data is valid and representative.
            """
        )

    st.divider()
    st.markdown("#### Quick glossary")
    glossary = pd.DataFrame(
        [
            ("Baseline", "Reference data or reference model score used for comparison."),
            ("New batch", "The latest incoming dataset evaluated by the monitoring workflow."),
            ("Feature", "An input column used by the machine-learning model."),
            ("Target / label", "The known answer the model is trying to predict."),
            ("Data drift", "A change in the distribution of incoming input data."),
            ("False positive", "The model predicts the positive class when the actual class is negative."),
            ("False negative", "The model predicts the negative class when the actual class is positive."),
            ("Threshold", "A configured limit used to decide when an alert should be raised."),
        ],
        columns=["Term", "Meaning"],
    )
    st.dataframe(glossary, use_container_width=True, hide_index=True)

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "ModelWatch · Monitoring dashboard. Incoming-batch evaluation in this assignment "
    "uses simulated data and retained labels; it is not verified live-production performance."
)
