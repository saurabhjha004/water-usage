import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv("data/agriculture_dataset.csv")

# Encode categorical features
encoder = OneHotEncoder(drop="first", sparse_output=False)
X = encoder.fit_transform(df[['Crop_Type', 'Soil_Type', 'Season']])
y_class = df['Irrigation_Type']
y_reg = df['Water_Usage(cubic meters)']

# Split data
X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# Train models
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_class)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train_reg)

# Evaluate
y_pred_class = clf.predict(X_test)
print("Classification Results:")
print(f"Accuracy: {accuracy_score(y_test_class, y_pred_class):.2f}")
print(classification_report(y_test_class, y_pred_class))

y_pred_reg = reg.predict(X_test)
print("\nRegression Results:")
print(f"RMSE: {mean_squared_error(y_test_reg, y_pred_reg, squared=False):.2f}")
print(f"R²: {r2_score(y_test_reg, y_pred_reg):.2f}")

# Save models and encoder
joblib.dump(encoder, "models/encoder.joblib")
joblib.dump(clf, "models/irrigation_classifier.joblib")
joblib.dump(reg, "models/water_regressor.joblib")
print("\nModels saved to /models directory!")