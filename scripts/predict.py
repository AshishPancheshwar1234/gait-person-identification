import pandas as pd
import numpy as np
import joblib
import os

# Safe base path (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

unknown_path = os.path.join(BASE_DIR, "data", "unknown")
model_path = os.path.join(BASE_DIR, "models", "gait_model.pkl")

print("Loading unknown sensor data...")

# ---------- Cleaning functions ----------
def clean_acc(file_path):
    df = pd.read_csv(file_path, comment="#")
    df = df.iloc[:, :3]  # take first 3 columns
    df.columns = ["acc_x", "acc_y", "acc_z"]
    return df


def clean_gyro(file_path):
    df = pd.read_csv(file_path, comment="#")
    df = df.iloc[:, :3]
    df.columns = ["gyro_x", "gyro_y", "gyro_z"]
    return df


# Load and clean
acc_df = clean_acc(os.path.join(unknown_path, "acc.csv"))
gyro_df = clean_gyro(os.path.join(unknown_path, "gyro.csv"))

# Make row count equal
min_len = min(len(acc_df), len(gyro_df))
acc_df = acc_df.iloc[:min_len]
gyro_df = gyro_df.iloc[:min_len]

# Merge
merged_df = pd.concat([acc_df, gyro_df], axis=1)

# Extract features
feature_dict = {}
for col in merged_df.columns:
    feature_dict[f"{col}_mean"] = merged_df[col].mean()
    feature_dict[f"{col}_std"] = merged_df[col].std()
    feature_dict[f"{col}_max"] = merged_df[col].max()
    feature_dict[f"{col}_min"] = merged_df[col].min()

feature_df = pd.DataFrame([feature_dict])

print("Loading trained model...")
clf = joblib.load(model_path)

# Ensure same feature order as training
feature_df = feature_df[clf.feature_names_in_]

prediction = clf.predict(feature_df)[0]
print(f"Predicted person: {prediction}")
