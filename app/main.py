
# streamlit application
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap

# title and description
st.set_page_config(
    page_title = "Shambani - Smart Crop Recomender",
    page_icon = "🌾",
    layout = "centered"
)

st.title("🌾 SHAMBANI")
st.markdown(
    """
    #### Intelligent Crop Recommendation System for Farmers  
    Provide your soil and weather conditions to discover which crops are most suitable for cultivation.  
    The model uses soil nutrients (NPK), pH, rainfall, temperature, and humidity to make predictions.
    """
)

# resource loader
@st.cache_resource
def load_resources():
    # loading encoder
    encoder = joblib.load("app/../resources/encoder.pkl")
    
    # laoding scaler
    scaler = joblib.load("app/../resources/scaler.pkl")

    # loading model
    model = XGBClassifier()
    model.load_model("app/../resources/crop_model.json")

    return encoder, scaler, model

encoder, scaler, model = load_resources()

# user inputs
st.sidebar.header(
    "🧪 Enter Your Soil and Weather Details"
)

N = st.sidebar.number_input(
    "Nitrogen (N)", min_value=0, max_value=200, value=90
)

P = st.sidebar.number_input(
    "Phosphorous (P)", min_value=0, max_value=200, value=42
)

K = st.sidebar.number_input(
    "Potassium (K)", min_value=0, max_value=200, value=43
)

temperature = st.sidebar.number_input(
    "Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1
)

humidity = st.sidebar.number_input(
    "Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1
)

ph = st.sidebar.number_input(
    "Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1

)

rainfall = st.sidebar.number_input(
    "Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, step=0.1
)

# input data
input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

if st.button("🌱 Recommend Crops"):
    X_scaled = scaler.transform(input_data)
    probabilities = model.predict_proba(X_scaled)[0]

    # getting top 3 predictions
    top3_idx = np.argsort(probabilities)[::-1][:3]
    top3_crops = encoder.inverse_transform(top3_idx)
    top3_probs = probabilities[top3_idx] * 100

    # displaying top recomendations
    st.subheader(f"Recomended Crop: **{top3_crops[0]}**")
    st.markdown(f"**Confidence:** {top3_probs[0]:.2f}%")

    # horizontal bar chart for top 3 crops
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.barh(top3_crops[::-1], top3_probs[::-1], color=['#76b041', '#a6ce39', '#dce775'])
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    st.pyplot(fig)

    # displaying crops insights
    st.markdown("---")
    st.markdown("### 🌿 Crop Suitability Insights")

    for i, crop in enumerate(top3_crops):
        st.markdown(f"""
        **{i+1}. {crop}**  
        - Suitability: {top3_probs[i]:.1f}%  
        - Conditions match based on current soil & climate parameters.
        """)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        st.markdown("---")
        st.markdown("### 🔍 Feature Impact Visualization (SHAP)")
        shap_fig = shap.force_plot(
            explainer.expected_value[0],
            shap_values[0],
            pd.DataFrame(input_data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']),
            matplotlib=True,
            show=False
        )
        st.pyplot(bbox_inches='tight', dpi=80)
    except Exception:
        st.info("SHAP visualization not supported in this environment. Skipping feature impact plot.")

else:
    st.info("👈 Input your soil and weather data in the sidebar, then click **Recommend Crops**.")
