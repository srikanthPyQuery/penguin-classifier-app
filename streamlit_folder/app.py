import streamlit as st
import pandas as pd
import joblib
import numpy as np


# 1. Load the saved assets
@st.cache_resource
def load_model():
    return joblib.load('penguin_classification_v1.pkl')


assets = load_model()
model = assets['final_model']
scaler = assets['scaler']
pca = assets['pca']
le_sex = assets['label_encoder_sex']
le_island = assets['label_encoder_island']

# 2. App UI
st.title("🐧 Penguin Species Predictor")
st.markdown("Enter the physical characteristics to identify the species.")

col1, col2 = st.columns(2)

with col1:
    island = st.selectbox("Island", ["Biscoe", "Dream", "Torgersen"])
    culmen_length = st.number_input("Culmen Length (mm)", value=40.0)
    culmen_depth = st.number_input("Culmen Depth (mm)", value=15.0)

with col2:
    sex = st.selectbox("Sex", ["MALE", "FEMALE"])
    flipper_length = st.number_input("Flipper Length (mm)", value=200.0)
    body_mass = st.number_input("Body Mass (g)", value=4000.0)

# 3. Prediction Logic
if st.button("Predict Species"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length],
        'culmen_depth_mm': [culmen_depth],
        'flipper_length_mm': [flipper_length],
        'body_mass_g': [body_mass],
        'sex': [sex]
    })

    # Transform inputs using the saved encoders/scalers
    input_data['island'] = le_island.transform(input_data['island'])
    input_data['sex'] = le_sex.transform(input_data['sex'])

    # Scale and PCA
    scaled_data = scaler.transform(input_data)
    pca_data = pca.transform(scaled_data)

    # Predict
    prediction = model.predict(pca_data)[0]
    probability = np.max(model.predict_proba(pca_data)) * 100

    # Display Result
    st.success(f"The predicted species is **{prediction}**")
    st.info(f"Confidence Level: {probability:.2f}%")