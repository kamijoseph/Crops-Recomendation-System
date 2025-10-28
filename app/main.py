
# streamlit application
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

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
