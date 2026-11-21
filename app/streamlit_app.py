"""FairPrice Streamlit UI.

    streamlit run app/streamlit_app.py

The UI only collects input and displays results; all ML logic lives in the fairprice package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `fairprice` importable

import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st

from fairprice import config
from fairprice.explain import explain_prediction, make_explainer
from fairprice.predict import (ModelNotTrainedError, fair_price_verdict, load_ann_model, load_background,
                               load_model, predict_value, validate_input)

st.set_page_config(page_title="FairPrice | California Housing Valuation", layout="wide")

LOCATION_PRESETS = {
    "Los Angeles": (34.05, -118.24),
    "San Francisco": (37.77, -122.42),
    "San Diego": (32.72, -117.16),
    "San Jose": (37.34, -121.89),
    "Sacramento": (38.58, -121.49),
    "Fresno": (36.74, -119.79),
    "Custom location": (36.0, -119.5),
}
VERDICT_STYLE = {"Good Deal": ("Good deal", st.success), "Fair": ("Fairly priced", st.info),
                 "Overpriced": ("Overpriced", st.error)}
FRIENDLY_NAMES = {
    "MedInc": "Median income", "HouseAge": "House age", "AveRooms": "Rooms per home",
    "Population": "Population", "AveOccup": "Occupants per home", "Latitude": "Latitude",
    "Longitude": "Longitude", "BedroomRatio": "Bedroom ratio", "RoomsPerPerson": "Rooms per person",
    "DistToCity": "Distance to nearest major city",
}
FIGURE_SECTIONS = {
    "Data exploration": {
        "01_target_distribution": "House values cluster below $300k, with a hard cap at $500k",
        "02_feature_distributions": "Several features are heavily skewed by a few extreme districts",
        "03_outliers_boxplot": "Outliers in rooms and occupancy, handled by percentile clipping",
        "05_geographic_value_map": "Coastal districts near LA and the Bay Area are the most expensive",
        "04_correlation_heatmap_raw": "Correlations before feature engineering",
        "06_correlation_heatmap_engineered": "Correlations after feature engineering",
    },
    "Model performance": {
        "07_model_comparison": "Cross-validation and test error for every model",
        "11_ann_training_curve": "Neural network learning curve with early stopping",
        "08_residual_analysis": "Prediction errors of the selected model",
        "09_feature_importance": "Which features the selected model relies on most",
    },
    "Explainability": {
        "10_shap_summary": "How each feature pushes individual predictions up or down",
    },
}
LEADERBOARD_COLUMNS = {
    "model": "Model", "family": "Type", "cv_rmse_mean": "CV RMSE", "cv_rmse_std": "CV RMSE std",
    "train_rmse": "Train RMSE", "test_rmse": "Test RMSE", "test_mae": "Test MAE", "test_r2": "Test R²",
    "test_mae_dollars": "Typical error ($)", "overfit_gap": "Train-test gap", "fit_seconds": "Training time (s)",
}

st.markdown("""
<style>
    .block-container {padding-top: 2.2rem; max-width: 1200px;}
    h1 {font-weight: 650; letter-spacing: -0.5px; margin-bottom: 0;}
    .subtitle {color: #5f6b7a; font-size: 1.05rem; margin-top: 0.2rem; margin-bottom: 1.4rem;}
    div[data-testid="stMetricValue"] {font-weight: 600;}
    .model-note {color: #5f6b7a; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    pipeline, meta = load_model()
    return pipeline, meta, make_explainer(pipeline, load_background()), load_ann_model()


def money(value: float) -> str:
    return f"${value:,.0f}"


def md(text: str) -> str:
    """Escape $ so Streamlit markdown doesn't render dollar amounts as LaTeX math."""
    return text.replace("$", "\\$")


def contribution_chart(table: pd.DataFrame):
    top = table.head(8).iloc[::-1]
    colors = ["#2f6f4f" if v > 0 else "#a33b3b" for v in top["contribution_dollars"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([FRIENDLY_NAMES.get(f, f) for f in top["feature"]], top["contribution_dollars"], color=colors)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.xaxis.set_major_formatter(lambda x, _: f"{'+' if x > 0 else ''}{x / 1000:,.0f}k")
    ax.set_xlabel("Effect on the estimate ($)")
    ax.set_title("What moved the estimate", loc="left", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ load models
try:
    pipeline, meta, explainer, ann_pipeline = get_models()
except ModelNotTrainedError as err:
    st.error(str(err))
    st.code("python run_pipeline.py --fast", language="bash")
    st.stop()

mae = meta["test_mae_dollars"]
ann_meta = meta.get("ann")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.subheader("Selected model")
    st.metric("Algorithm", meta["model_name"])
    col_a, col_b = st.columns(2)
    col_a.metric("Test R²", f"{meta['test_metrics']['r2']:.3f}")
    col_b.metric("Typical error", money(mae))
    if ann_meta:
        st.subheader("Neural network")
        col_c, col_d = st.columns(2)
        col_c.metric("Test R²", f"{ann_meta['test_metrics']['r2']:.3f}")
        col_d.metric("Typical error", money(ann_meta["test_mae_dollars"]))
    st.caption(f"Last trained {meta['trained_at'].replace('T', ' at ')}")
    st.divider()
    st.caption(md("Based on the 1990 California census. Each record describes a neighbourhood "
                  "(census block group), so estimates are the neighbourhood's median home value "
                  "in 1990 dollars. Values above $500k are beyond what the model has learned."))

# ------------------------------------------------------------------ header
st.title("FairPrice")
st.markdown('<p class="subtitle">California housing valuation. Estimate what homes in a '
            'neighbourhood are worth, see what drives the number, and check an asking price '
            'against it.</p>', unsafe_allow_html=True)

tab_value, tab_performance, tab_about = st.tabs(["Valuation", "Model performance", "About the data"])

# ------------------------------------------------------------------ valuation tab
with tab_value:
    location = st.selectbox("Location", list(LOCATION_PRESETS), index=0)
    preset_lat, preset_lon = LOCATION_PRESETS[location]

    with st.form("inputs"):
        st.markdown("**Neighbourhood details**")
        c1, c2, c3 = st.columns(3)
        income = c1.number_input("Median household income ($ per year)", min_value=0, value=45_000, step=1_000)
        house_age = c1.number_input("Median house age (years)", min_value=0, value=25, step=1)
        rooms = c2.number_input("Average rooms per home", min_value=0.0, value=5.0, step=0.5)
        bedrooms = c2.number_input("Average bedrooms per home", min_value=0.0, value=1.0, step=0.1)
        population = c3.number_input("Population", min_value=0, value=1_200, step=100)
        occupancy = c3.number_input("Average occupants per home", min_value=0.0, value=3.0, step=0.1)
        c4, c5, c6 = st.columns(3)
        lat = c4.number_input("Latitude", value=preset_lat, format="%.4f", key=f"lat_{location}")
        lon = c5.number_input("Longitude", value=preset_lon, format="%.4f", key=f"lon_{location}")
        asking = c6.number_input("Asking price ($, optional)", min_value=0, value=0, step=5_000,
                                 help="Enter a listing price to see whether it looks fair. Leave at 0 to skip.")
        submitted = st.form_submit_button("Estimate value", type="primary")

    if submitted:
        inputs = {
            "MedInc": income / 10_000,  # dataset unit is $10,000s
            "HouseAge": house_age, "AveRooms": rooms, "AveBedrms": bedrooms,
            "Population": population, "AveOccup": occupancy, "Latitude": lat, "Longitude": lon,
        }
        errors, warnings = validate_input(inputs, meta["feature_ranges"])
        for message in errors:
            st.error(message)
        if errors:
            st.stop()
        for message in warnings:
            st.warning(message)

        predicted = predict_value(pipeline, inputs)
        base, table = explain_prediction(pipeline, explainer, pd.DataFrame([inputs]))

        st.divider()
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.metric("Estimated median home value", money(predicted))
            st.caption(md(f"Likely range {money(max(predicted - mae, 0))} to {money(predicted + mae)}, "
                          f"based on the model's typical error of {money(mae)}."))

            if asking > 0:
                result = fair_price_verdict(predicted, asking, tolerance=mae)
                label, show = VERDICT_STYLE[result["verdict"]]
                direction = "above" if result["difference_pct"] > 0 else "below"
                show(md(f"**{label}.** The asking price of {money(asking)} is "
                        f"{abs(result['difference_pct']):.1f}% {direction} our estimate. Anything between "
                        f"{money(result['fair_low'])} and {money(result['fair_high'])} would be "
                        f"within the model's normal margin of error."))

            reasons = [f"{FRIENDLY_NAMES.get(r.feature, r.feature).lower()} "
                       f"({'+' if r.contribution_dollars > 0 else '-'}{money(abs(r.contribution_dollars))})"
                       for r in table.head(3).itertuples()]
            st.markdown(md(f"**Main factors:** {', '.join(reasons)}."))
            st.caption(md(f"The model starts from an average of {money(base)} and adjusts it for "
                          f"this neighbourhood's characteristics."))

            if ann_pipeline is not None:
                ann_value = predict_value(ann_pipeline, inputs)
                gap = ann_value - predicted
                agreement = "agree closely" if abs(gap) <= mae / 2 else "differ noticeably"
                st.markdown(md(f"**Second opinion**  \n"
                               f"{meta['model_name']}: {money(predicted)}  \n"
                               f"Neural network: {money(ann_value)}"))
                st.caption(md(f"The two models {agreement} ({money(abs(gap))} apart). When independent "
                              f"models agree, the estimate is more reliable."))
        with right:
            st.pyplot(contribution_chart(table))
            st.caption("Green bars raise the estimate, red bars lower it.")
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=6, height=260)

        with st.expander("Detailed breakdown"):
            shown = pd.DataFrame({
                "Factor": table["feature"].map(lambda f: FRIENDLY_NAMES.get(f, f)),
                "Value": table["value"].round(3),
                "Effect on estimate ($)": table["contribution_dollars"].round(0),
            })
            st.dataframe(shown, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ performance tab
with tab_performance:
    if config.LEADERBOARD_PATH.exists():
        st.subheader("Model comparison")
        st.markdown("Four conventional models and one neural network were trained on the same "
                    "features, tuned with randomized search and compared with 5-fold cross-validation. "
                    "The model with the lowest cross-validation error is used for valuations; the test "
                    "set is only used to report the final score.")
        board = pd.read_csv(config.LEADERBOARD_PATH).rename(columns=LEADERBOARD_COLUMNS)
        st.dataframe(board, hide_index=True, use_container_width=True)

    for section, figures in FIGURE_SECTIONS.items():
        available = [(name, caption) for name, caption in figures.items()
                     if (config.FIGURES_DIR / f"{name}.png").exists()]
        if not available:
            continue
        st.subheader(section)
        for start in range(0, len(available), 2):
            cols = st.columns(2)
            for col, (name, caption) in zip(cols, available[start:start + 2]):
                col.image(str(config.FIGURES_DIR / f"{name}.png"), caption=md(caption),
                          use_container_width=True)

# ------------------------------------------------------------------ about tab
with tab_about:
    st.subheader("About the data")
    st.markdown(md(
        "The model is trained on the California Housing dataset from the 1990 US Census: 20,640 "
        "neighbourhoods described by median income, house age, average rooms and bedrooms, population, "
        "average occupancy and location.\n\n"
        "**How the data was prepared**\n"
        "- 992 neighbourhoods with values capped at $500k were removed, because their true value is unknown.\n"
        "- Extreme values (for example, an average of 1,243 occupants per home) were clipped to the "
        "1st and 99th percentiles of the training data.\n"
        "- Three features were added: bedroom ratio, rooms per person and distance to the nearest major city.\n\n"
        "**Limitations**\n"
        "- Estimates are neighbourhood medians, not prices for individual houses.\n"
        "- Values are in 1990 dollars and are not adjusted for inflation.\n"
        "- Accuracy drops for the most expensive neighbourhoods, where there were fewer training examples."
    ))
