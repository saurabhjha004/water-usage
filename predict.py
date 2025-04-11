import pandas as pd
import joblib

# Load saved models and encoder
encoder = joblib.load("models/encoder.joblib")
clf = joblib.load("models/irrigation_classifier.joblib")
reg = joblib.load("models/water_regressor.joblib")

# User input
# Get user input
crop = input("Enter Crop Type (e.g., Tomato, Cotton): ")
soil = input("Enter Soil Type (e.g., Silty, Loamy): ")
season = input("Enter Season (Kharif, Rabi, Zaid): ")

new_data = pd.DataFrame({
    'Crop_Type': [crop],
    'Soil_Type': [soil],
    'Season': [season]
})

# Preprocess and predict
X_new = encoder.transform(new_data)
irrigation = clf.predict(X_new)[0]
water_usage = reg.predict(X_new)[0]

print(f"\nRecommended Irrigation Type: {irrigation}")
print(f"Predicted Water Usage: {water_usage:.2f} cubic meters")