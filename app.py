import streamlit as st
import pandas as pd
import joblib
import os
from src.data_preprocessing import preprocess_data
from src.evaluation import evaluate_model
from src.prediction import make_prediction
from src.visualization import (
    plot_price_distribution,
    plot_sqft_vs_price,
    plot_neighborhood_boxplot,
    plot_quality_bar,
    plot_heatmap,
    plot_predictions_vs_actual,
    plot_residuals,
)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Stacking Regressor",
    page_icon="",
    layout="wide",
)

st.title("House Price Stacking Regressor Dashboard")
st.markdown("---")

# ── PATH RESOLUTION ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "houses.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "stacking_model.pkl")


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
    st.error("Setup Incomplete! Missing dataset or trained model.")
    st.info(
        "1. Make sure `data/houses.csv` exists.\n"
        "2. Run `python -m src.model_training` to train and save the model."
    )
    st.stop()

df    = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("Pipeline Navigation")
page = st.sidebar.radio(
    "Go To:",
    [
        "Dataset Summary",
        "Exploratory Data Analysis",
        "Model Performance Evaluation",
        "Predict House Price",
    ],
)

# ══════════════════════════════════════════════════════════════════════════════
# 1 · DATASET SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dataset Summary":
    st.header("Dataset Overview")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Dimensions")
        st.info(f"Records: **{df.shape[0]}** | Features: **{df.shape[1]}**")
    with col2:
        st.subheader("Missing Value Check")
        st.write(df.isnull().sum())

    st.subheader("Statistical Profiles")
    st.dataframe(df.describe(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2 · EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Exploratory Data Analysis":
    st.header("Interactive Regression EDA")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_price_distribution(df),    use_container_width=True)
        st.plotly_chart(plot_neighborhood_boxplot(df),  use_container_width=True)
    with col2:
        st.plotly_chart(plot_sqft_vs_price(df),         use_container_width=True)
        st.plotly_chart(plot_quality_bar(df),           use_container_width=True)

    st.markdown("---")
    st.plotly_chart(plot_heatmap(df), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3 · MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance Evaluation":
    st.header("⚡ Stacking Regressor Performance Metrics")

    _, X_test, _, y_test, _ = preprocess_data(df)
    metrics, y_test_orig, y_pred_orig = evaluate_model(model, X_test, y_test)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score",  f"{metrics['r2']:.4f}")
        st.caption("Variance explained by model (1.0 = perfect).")
    with col2:
        st.metric("MAE",  f"${metrics['mae']:,.0f}")
        st.caption("Average absolute prediction error.")
    with col3:
        st.metric("RMSE", f"${metrics['rmse']:,.0f}")
        st.caption("Penalises large errors more heavily.")
    with col4:
        st.metric("MAPE", f"{metrics['mape']:.2f}%")
        st.caption("Mean absolute percentage error.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_predictions_vs_actual(y_test_orig, y_pred_orig), use_container_width=True)
    with col2:
        st.plotly_chart(plot_residuals(y_test_orig, y_pred_orig), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4 · PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict House Price":
    st.header("House Price Predictor Engine")
    st.write("Fill in the property details below to get an estimated sale price.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Size & Structure")
        total_sqft    = st.slider("Total Square Footage",       800,  4500, 1800)
        lot_area      = st.slider("Lot Area (sq ft)",          3000, 20000, 8000)
        basement_sqft = st.slider("Basement Square Footage",      0,  2000,  400)
        bedrooms      = st.selectbox("Bedrooms",   [1, 2, 3, 4, 5, 6], index=2)
        bathrooms     = st.selectbox("Bathrooms",  [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], index=2)

    with col2:
        st.subheader("Property Details")
        year_built    = st.number_input("Year Built",         1950, 2022, 1990)
        remodel_year  = st.number_input("Last Remodel Year",  1950, 2023, 2005)
        overall_qual  = st.slider("Overall Quality (1–10)",       1,   10,    7)
        fireplaces    = st.selectbox("Fireplaces",            [0, 1, 2, 3], index=1)
        has_pool      = st.selectbox("Swimming Pool",         ["No", "Yes"])

    with col3:
        st.subheader("Location & Amenities")
        neighborhood  = st.selectbox("Neighborhood",   ["Downtown", "Historic", "Rural", "Suburbs", "Waterfront"])
        condition     = st.selectbox("House Condition",["Excellent", "Good", "Fair", "Poor"])
        garage_type   = st.selectbox("Garage Type",    ["Attached", "Detached", "None"])
        garage_cars   = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3], index=1)
        heating       = st.selectbox("Heating Type",   ["Electric", "Gas", "Oil", "Solar"])

    st.markdown("---")

    if st.button("Estimate Sale Price", use_container_width=True):
        predicted_price = make_prediction(
            model,
            year_built=int(year_built),
            remodel_year=int(remodel_year),
            lot_area=int(lot_area),
            total_sqft=int(total_sqft),
            basement_sqft=int(basement_sqft),
            bedrooms=int(bedrooms),
            bathrooms=float(bathrooms),
            garage_cars=int(garage_cars),
            fireplaces=int(fireplaces),
            has_pool=1 if has_pool == "Yes" else 0,
            overall_qual=int(overall_qual),
            neighborhood=neighborhood,
            condition=condition,
            garage_type=garage_type,
            heating=heating,
        )
        st.subheader("Stacking Regressor Estimate")
        st.success(f"Estimated Sale Price: **${predicted_price:,.0f}**")
        low  = predicted_price * 0.93
        high = predicted_price * 1.07
        st.info(f"Confidence Range (±7%): **${low:,.0f}** – **${high:,.0f}**")
