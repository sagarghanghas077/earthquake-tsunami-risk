import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# Load model & scaler
# ===============================
model = joblib.load("models/tsunami_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURES = [
    "magnitude", "cdi", "mmi", "sig", "nst",
    "dmin", "gap", "depth", "latitude",
    "longitude", "Year", "Month"
]

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Tsunami Risk Predictor 🌊",
    layout="wide"
)

st.title("🌍 Global Earthquake–Tsunami Risk Prediction System")
st.markdown(
    "Predict tsunami risk using seismic parameters from global earthquake data (2001–2022)."
)

# ===============================
# Sidebar – Inputs
# ===============================
st.sidebar.header("🧭 Earthquake Parameters")

inputs = [
    st.sidebar.slider("Magnitude", 6.5, 9.1, 7.0),
    st.sidebar.slider("CDI", 0, 9, 4),
    st.sidebar.slider("MMI", 1, 9, 5),
    st.sidebar.slider("Significance", 600, 3000, 1000),
    st.sidebar.slider("NST", 0, 950, 100),
    st.sidebar.slider("Dmin", 0.0, 18.0, 1.0),
    st.sidebar.slider("Gap", 0.0, 240.0, 80.0),
    st.sidebar.slider("Depth (km)", 0.0, 700.0, 50.0),
    st.sidebar.slider("Latitude", -90.0, 90.0, 0.0),
    st.sidebar.slider("Longitude", -180.0, 180.0, 0.0),
    st.sidebar.slider("Year", 2001, 2022, 2015),
    st.sidebar.slider("Month", 1, 12, 6),
]

# ===============================
# Prediction
# ===============================
if st.button("🚨 Predict Tsunami Risk"):
    X = scaler.transform([inputs])
    prob = model.predict_proba(X)[0][1]

    st.subheader("🔮 Prediction Result")

    if prob >= 0.5:
        st.error(f"🌊 **Tsunami Risk Detected** ({prob:.2%})")
    else:
        st.success(f"✅ **No Tsunami Risk** ({prob:.2%})")

    # ===============================
    # Probability Visualization
    # ===============================
    st.subheader("📊 Prediction Confidence")

    prob_df = pd.DataFrame({
        "Outcome": ["No Tsunami", "Tsunami"],
        "Probability": [1 - prob, prob]
    })

    fig, ax = plt.subplots()
    ax.bar(prob_df["Outcome"], prob_df["Probability"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Tsunami Risk Probability")

    st.pyplot(fig)

    # ===============================
    # Feature Importance
    # ===============================
    st.subheader("🧠 Model Explainability – Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )
        ax.invert_yaxis()
        ax.set_title("Feature Importance")

        st.pyplot(fig)
    else:
        st.info("Feature importance is not available for this model.")

# ===============================
# Dataset Exploration Section
# ===============================
st.divider()
st.subheader("📈 Explore Historical Earthquake Data")

@st.cache_data
def load_data():
    return pd.read_csv("data/raw/earthquake_data_tsunami.csv")

df = load_data()

chart_option = st.selectbox(
    "Choose a visualization",
    [
        "Magnitude Distribution",
        "Depth vs Tsunami Occurrence",
        "Tsunami Class Distribution"
    ]
)

if chart_option == "Magnitude Distribution":
    fig, ax = plt.subplots()
    ax.hist(df["magnitude"], bins=30)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Count")
    ax.set_title("Earthquake Magnitude Distribution")
    st.pyplot(fig)

elif chart_option == "Depth vs Tsunami Occurrence":
    fig, ax = plt.subplots()
    ax.scatter(df["depth"], df["tsunami"], alpha=0.5)
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Tsunami (0 = No, 1 = Yes)")
    ax.set_title("Depth vs Tsunami Occurrence")
    st.pyplot(fig)

elif chart_option == "Tsunami Class Distribution":
    fig, ax = plt.subplots()
    df["tsunami"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Tsunami")
    ax.set_ylabel("Count")
    ax.set_title("Tsunami vs Non-Tsunami Events")
    st.pyplot(fig)

# ===============================
# Footer
# ===============================
st.caption(
    "Built with Python, Scikit-learn & Streamlit | End-to-End ML Deployment Project"
)
