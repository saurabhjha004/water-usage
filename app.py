# app.py
import streamlit as st
import pandas as pd
import joblib

# Load saved models and encoder
encoder = joblib.load("models/encoder.joblib")
clf = joblib.load("models/irrigation_classifier.joblib")
reg = joblib.load("models/water_regressor.joblib")

# App title
st.title("🌾 Farm Irrigation & Water Usage Predictor")
st.markdown("Predict optimal irrigation method and water requirements based on crop, soil, and season.")

# Input section
st.header("📝 Input Parameters")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    crop = st.selectbox(
        "Select Crop Type",
        ["Cotton", "Carrot", "Sugarcane", "Tomato", "Soybean", 
         "Rice", "Maize", "Wheat", "Barley", "Potato"]
    )

with col2:
    soil = st.selectbox(
        "Select Soil Type",
        ["Loamy", "Peaty", "Silty", "Clay", "Sandy"]
    )

with col3:
    season = st.selectbox(
        "Select Season",
        ["Kharif", "Zaid", "Rabi"]
    )

# Prediction button
if st.button("Predict Irrigation & Water Needs", type="primary"):
    # Create DataFrame from inputs
    new_data = pd.DataFrame({
        "Crop_Type": [crop],
        "Soil_Type": [soil],
        "Season": [season]
    })
    
    # Encode features
    X_new = encoder.transform(new_data)
    
    # Get predictions
    irrigation = clf.predict(X_new)[0]
    water_usage = reg.predict(X_new)[0]
    
    # Display results
    st.header("📊 Prediction Results")
    
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric(label="Recommended Irrigation Type", value=irrigation)
        
    with result_col2:
        st.metric(
            label="Predicted Water Usage", 
            value=f"{water_usage:,.2f} m³",
            help="Cubic meters required for the growing season"
        )
    
    # Add explanation
    st.info("💡 Predictions are based on machine learning models trained on historical farming data.")

# Instructions
st.sidebar.markdown("""
**How to Use:**
1. Select crop type from dropdown
2. Choose soil type from your farmland
3. Pick the growing season
4. Click the prediction button
                    
**Allowed Values:**  
- Crops: 10 common varieties  
- Soil: 5 standard types  
- Seasons: 3 Indian agricultural seasons
""")