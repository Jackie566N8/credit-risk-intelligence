import streamlit as st

from dashboard.charts import make_cv_metric_chart, make_feature_importance_chart
from dashboard.components import numeric_table, show_raw_text


def render_model_detail_page(reports: dict, default_model: str | None = None) -> None:
    st.subheader("Single Model Detail")
    if not reports:
        st.warning("No model result files found. Run `python train_credit_models.py` first.")
        return

    model_names = list(reports.keys())
    index = model_names.index(default_model) if default_model in model_names else 0
    selected_model = st.selectbox(
        "Model",
        options=model_names,
        index=index,
        format_func=lambda name: reports[name].label,
    )
    report = reports[selected_model]

    data_cols = st.columns(4)
    data_cols[0].metric("Modeling Rows", report.data_summary.get("Modeling rows used", "-"))
    data_cols[1].metric("Default Count", report.data_summary.get("Default count", "-"))
    data_cols[2].metric("Default Rate", report.data_summary.get("Default rate", "-"))
    data_cols[3].metric("Result File", report.path.name)

    tab_cv, tab_holdout, tab_features, tab_raw = st.tabs(
        ["Cross Validation", "Holdout Test", "Feature Importance", "Raw Text"]
    )
    with tab_cv:
        numeric_table(report.cv_metrics)
        fig = make_cv_metric_chart(report.cv_metrics, report.label)
        if fig is not None:
            st.pyplot(fig)

    with tab_holdout:
        st.write("Optimized threshold")
        numeric_table(report.holdout_metrics)
        st.write("Default threshold 0.50")
        numeric_table(report.default_threshold_metrics)

    with tab_features:
        if report.feature_importance.empty:
            st.info("This model does not expose feature importance or coefficients.")
        else:
            top_n = st.slider("Top features", min_value=5, max_value=20, value=15)
            st.pyplot(make_feature_importance_chart(report.feature_importance, report.label, top_n))
            numeric_table(report.feature_importance.head(top_n))

    with tab_raw:
        show_raw_text(report.path, report.raw_text)
