import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .rainfall-yes {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .rainfall-no {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .sidebar .sidebar-content {
        background-color: #e8f4f8;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# Load Model and Artifacts
# ===========================

@st.cache_resource
def load_model_artifacts():
    try:
        models_dir = Path(__file__).parent / 'models'

        model = joblib.load(models_dir / 'best_rainfall_model.pkl')
        scaler = joblib.load(models_dir / 'scaler.pkl')
        label_encoder = joblib.load(models_dir / 'label_encoder.pkl')
        feature_names = joblib.load(models_dir / 'feature_names.pkl')
        metadata = joblib.load(models_dir / 'model_metadata.pkl')
        
        return {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': feature_names,
            'metadata': metadata
        }
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.stop()

artifacts = load_model_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
label_encoder = artifacts['label_encoder']
feature_names = artifacts['feature_names']
metadata = artifacts['metadata']

# ===========================
# App Header
# ===========================

st.title("🌧️ Rainfall Prediction System")

# ===========================
# Sidebar - Input Parameters
# ===========================

st.sidebar.header("🌤️ Input Weather Parameters")
st.sidebar.markdown("Adjust the sliders to input weather conditions:")

input_data = {}


input_data['pressure'] = st.sidebar.slider(
    "Atmospheric Pressure (hPa)",
    min_value=995.0,
    max_value=1040.0,
    value=1015.0,
    step=0.1,
    help="Atmospheric pressure in hectopascals"
)

input_data['maxtemp'] = st.sidebar.slider(
    "Maximum Temperature (°C)",
    min_value=5.0,
    max_value=40.0,
    value=22.0,
    step=0.1,
    help="Maximum temperature for the day"
)

input_data['temparature'] = st.sidebar.slider(
    "Current Temperature (°C)",
    min_value=4.0,
    max_value=35.0,
    value=18.0,
    step=0.1,
    help="Current temperature"
)

input_data['mintemp'] = st.sidebar.slider(
    "Minimum Temperature (°C)",
    min_value=2.0,
    max_value=32.0,
    value=15.0,
    step=0.1,
    help="Minimum temperature for the day"
)

input_data['dewpoint'] = st.sidebar.slider(
    "Dew Point (°C)",
    min_value=-2.0,
    max_value=28.0,
    value=12.0,
    step=0.1,
    help="Dew point temperature"
)

input_data['humidity'] = st.sidebar.slider(
    "Humidity (%)",
    min_value=30,
    max_value=100,
    value=70,
    step=1,
    help="Relative humidity percentage"
)

input_data['cloud'] = st.sidebar.slider(
    "Cloud Coverage (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
    help="Percentage of sky covered by clouds"
)

input_data['sunshine'] = st.sidebar.slider(
    "Sunshine Hours",
    min_value=0.0,
    max_value=13.0,
    value=5.0,
    step=0.1,
    help="Hours of sunshine"
)
input_data['winddirection'] = st.sidebar.slider(
    "Wind Direction (°)",
    min_value=0,
    max_value=360,
    value=180,
    step=10,
    help="Wind direction in degrees (0° = North, 90° = East, 180° = South, 270° = West)"
)

input_data['windspeed'] = st.sidebar.slider(
    "Wind Speed (km/h)",
    min_value=0.0,
    max_value=70.0,
    value=20.0,
    step=0.5,
    help="Wind speed in kilometers per hour"
)

st.sidebar.markdown("---")

predict_button = st.sidebar.button("Predict Rainfall", type="primary")

# ===========================
# Main Content - Prediction
# ===========================

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Input Summary")
    

    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df.T.rename(columns={0: 'Value'}))

with col_right:
    st.subheader("Prediction Result")
    
    if predict_button:

        input_df = pd.DataFrame([input_data])

        input_df = input_df[feature_names]
        
        input_scaled = scaler.transform(input_df)

        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        prediction = model.predict(input_scaled_df)[0]
        prediction_proba = model.predict_proba(input_scaled_df)[0]

        prediction_label = label_encoder.inverse_transform([prediction])[0]
        
        # Display prediction
        if prediction_label == 'yes':
            st.markdown(
                f'<div class="prediction-box rainfall-yes">🌧️ RAINFALL EXPECTED</div>',
                unsafe_allow_html=True
            )
            probability = prediction_proba[1] * 100
        else:
            st.markdown(
                f'<div class="prediction-box rainfall-no">☀️ NO RAINFALL EXPECTED</div>',
                unsafe_allow_html=True
            )
            probability = prediction_proba[0] * 100
        
        st.markdown(f"### Confidence: {probability:.2f}%")
        

# ===========================
# Footer
# ===========================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; padding-top: 20px;'>
        <strong>Rainfall Predictor</strong><br>
        Created by <a href='https://kindo-tk.github.io/tk.github.io/' target='_blank'>Tufan Kundu</a> ·
        <a href='https://github.com/kindo-tk' target='_blank'>GitHub</a> ·
        <a href='https://www.linkedin.com/in/tufan-kundu-577945221/' target='_blank'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)