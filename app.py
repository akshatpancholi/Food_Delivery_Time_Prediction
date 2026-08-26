import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Food Delivery Time Predictor", layout="wide")

st.title("🛵 Food Delivery Time Prediction")
st.markdown(
    "Predict how long a food delivery will take using five different regression "
    "models — **Linear**, **Ridge**, **Lasso**, and **Polynomial** Regression — "
    "trained on real delivery data."
)

MODELS_DIR = Path("models")
DATA_PATH = "Food_Delivery_Times.csv"

# -----------------------------
# Category options (must match training-time encoding)
# -----------------------------
WEATHER_OPTIONS = ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
TRAFFIC_OPTIONS = ["High", "Low", "Medium"]
TIME_OPTIONS = ["Afternoon", "Evening", "Morning", "Night"]
VEHICLE_OPTIONS = ["Bike", "Car", "Scooter"]

DUMMY_COLUMN_ORDER = [
    "Distance_km", "Preparation_Time_min", "Courier_Experience_yrs",
    "Weather_Clear", "Weather_Foggy", "Weather_Rainy", "Weather_Snowy", "Weather_Windy",
    "Traffic_Level_High", "Traffic_Level_Low", "Traffic_Level_Medium",
    "Time_of_Day_Afternoon", "Time_of_Day_Evening", "Time_of_Day_Morning", "Time_of_Day_Night",
    "Vehicle_Type_Bike", "Vehicle_Type_Car", "Vehicle_Type_Scooter",
]

# LabelEncoder assigns codes in alphabetical order of the categories it sees
WEATHER_ENC = {v: i for i, v in enumerate(WEATHER_OPTIONS)}
TRAFFIC_ENC = {v: i for i, v in enumerate(TRAFFIC_OPTIONS)}
TIME_ENC = {v: i for i, v in enumerate(TIME_OPTIONS)}
VEHICLE_ENC = {v: i for i, v in enumerate(VEHICLE_OPTIONS)}

MODEL_INFO = {
    "Linear Regression (One-Hot Encoded)": {
        "file": "linear_regression_dummies_model.pkl",
        "type": "dummies",
    },
    "Linear Regression (Label Encoded)": {
        "file": "linear_regression_encoded_model.pkl",
        "type": "encoded",
    },
    "Ridge Regression": {
        "file": "ridge_regression_model.pkl",
        "type": "dummies",
    },
    "Lasso Regression": {
        "file": "lasso_regression_model.pkl",
        "type": "dummies",
    },
    "Polynomial Regression (degree 2)": {
        "file": "polynomial_regression_model.pkl",
        "type": "poly",
    },
}


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource
def load_model(filename):
    with open(MODELS_DIR / filename, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


data = load_data()


# -----------------------------
# Feature builders
# -----------------------------
def build_dummy_row(distance, prep_time, experience, weather, traffic, time_of_day, vehicle):
    row = {
        "Distance_km": distance,
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": experience,
    }
    for w in WEATHER_OPTIONS:
        row[f"Weather_{w}"] = 1 if w == weather else 0
    for t in TRAFFIC_OPTIONS:
        row[f"Traffic_Level_{t}"] = 1 if t == traffic else 0
    for tod in TIME_OPTIONS:
        row[f"Time_of_Day_{tod}"] = 1 if tod == time_of_day else 0
    for v in VEHICLE_OPTIONS:
        row[f"Vehicle_Type_{v}"] = 1 if v == vehicle else 0
    return pd.DataFrame([row])[DUMMY_COLUMN_ORDER]


def build_encoded_row(order_id, distance, prep_time, experience, weather, traffic, time_of_day, vehicle):
    row = {
        "Order_ID": order_id,
        "Distance_km": distance,
        "Weather": WEATHER_ENC[weather],
        "Traffic_Level": TRAFFIC_ENC[traffic],
        "Time_of_Day": TIME_ENC[time_of_day],
        "Vehicle_Type": VEHICLE_ENC[vehicle],
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": experience,
    }
    return pd.DataFrame([row])


def predict_with_model(model_name, inputs):
    info = MODEL_INFO[model_name]
    model = load_model(info["file"])

    if info["type"] == "dummies":
        X = build_dummy_row(
            inputs["distance"], inputs["prep_time"], inputs["experience"],
            inputs["weather"], inputs["traffic"], inputs["time_of_day"], inputs["vehicle"],
        )
        return float(model.predict(X)[0])

    if info["type"] == "encoded":
        X = build_encoded_row(
            inputs["order_id"], inputs["distance"], inputs["prep_time"], inputs["experience"],
            inputs["weather"], inputs["traffic"], inputs["time_of_day"], inputs["vehicle"],
        )
        return float(model.predict(X)[0])

    if info["type"] == "poly":
        X = build_dummy_row(
            inputs["distance"], inputs["prep_time"], inputs["experience"],
            inputs["weather"], inputs["traffic"], inputs["time_of_day"], inputs["vehicle"],
        )
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X.values)
        return float(model.predict(X_poly)[0])

    raise ValueError(f"Unknown model type for {model_name}")


# -----------------------------
# Sidebar: Inputs
# -----------------------------
st.sidebar.header("📦 Order Details")

distance = st.sidebar.slider("Distance (km)", 0.5, 20.0, 7.5, 0.1)
prep_time = st.sidebar.slider("Preparation Time (min)", 5, 30, 15)
experience = st.sidebar.slider("Courier Experience (yrs)", 0.0, 10.0, 3.0, 0.5)

weather = st.sidebar.selectbox("Weather", WEATHER_OPTIONS)
traffic = st.sidebar.selectbox("Traffic Level", TRAFFIC_OPTIONS)
time_of_day = st.sidebar.selectbox("Time of Day", TIME_OPTIONS)
vehicle = st.sidebar.selectbox("Vehicle Type", VEHICLE_OPTIONS)

order_id = st.sidebar.number_input(
    "Order ID (used only by the 'Label Encoded' model)",
    min_value=1, max_value=100000, value=1, step=1,
)

inputs = {
    "distance": distance,
    "prep_time": prep_time,
    "experience": experience,
    "weather": weather,
    "traffic": traffic,
    "time_of_day": time_of_day,
    "vehicle": vehicle,
    "order_id": order_id,
}

st.sidebar.markdown("---")
st.sidebar.header("🤖 Model")
model_choice = st.sidebar.selectbox(
    "Choose a model", list(MODEL_INFO.keys()) + ["Compare All Models"]
)

# -----------------------------
# Main: Prediction
# -----------------------------
if model_choice == "Compare All Models":
    st.subheader("📊 Predictions from All Models")

    results = {}
    for name in MODEL_INFO:
        try:
            results[name] = predict_with_model(name, inputs)
        except Exception as e:
            results[name] = None
            st.warning(f"{name} failed: {e}")

    col1, col2 = st.columns([1, 1])

    with col1:
        for name, pred in results.items():
            if pred is not None:
                st.metric(name, f"{pred:.1f} min")

    with col2:
        valid = {k: v for k, v in results.items() if v is not None}
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(list(valid.keys()), list(valid.values()), color="#1f77b4")
        ax.set_xlabel("Predicted Delivery Time (min)")
        ax.set_title("Model Comparison")
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.subheader(f"🔮 Prediction — {model_choice}")
    prediction = predict_with_model(model_choice, inputs)
    st.success(f"Estimated Delivery Time: **{prediction:.1f} minutes**")

    st.markdown("#### Order Summary")
    summary_df = pd.DataFrame([{
        "Distance (km)": distance,
        "Prep Time (min)": prep_time,
        "Courier Experience (yrs)": experience,
        "Weather": weather,
        "Traffic": traffic,
        "Time of Day": time_of_day,
        "Vehicle": vehicle,
    }])
    st.table(summary_df)

# -----------------------------
# Dataset Exploration
# -----------------------------
st.markdown("---")
with st.expander("📁 Explore the Training Dataset"):
    st.write(f"**Rows:** {len(data)}  |  **Columns:** {len(data.columns)}")
    st.dataframe(data.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.hist(data["Delivery_Time_min"].dropna(), bins=25, color="#1f77b4")
        ax.set_xlabel("Delivery Time (min)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Delivery Times")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.scatter(data["Distance_km"], data["Delivery_Time_min"], alpha=0.4, color="#ff7f0e")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Delivery Time (min)")
        ax.set_title("Distance vs Delivery Time")
        st.pyplot(fig)

st.markdown("---")
st.caption("Built with Streamlit & scikit-learn · Regression models comparison project")
