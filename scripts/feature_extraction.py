import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
merged_data_path = os.path.join(current_dir, "..", "data", "merged")
processed_data_path = os.path.join(current_dir, "..", "data", "processed")

os.makedirs(processed_data_path, exist_ok=True)

features_list = []

# Window parameters
WINDOW_SIZE = 100      # samples per window (≈2 sec if 50Hz)
STEP_SIZE = 50         # overlap (50%)

print("Starting window-based feature extraction...")

for file in os.listdir(merged_data_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(merged_data_path, file))

        label = df["label"].iloc[0]

        # Convert numeric columns
        for col in df.columns:
            if col != "label":
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop label column for windowing
        sensor_df = df.drop("label", axis=1)

        total_rows = len(sensor_df)

        # Sliding window
        for start in range(0, total_rows - WINDOW_SIZE, STEP_SIZE):
            window = sensor_df.iloc[start:start + WINDOW_SIZE]

            feature_dict = {"label": label}

            for col in window.columns:
                feature_dict[f"{col}_mean"] = window[col].mean()
                feature_dict[f"{col}_std"] = window[col].std()
                feature_dict[f"{col}_max"] = window[col].max()
                feature_dict[f"{col}_min"] = window[col].min()

            features_list.append(feature_dict)

features_df = pd.DataFrame(features_list)
features_file = os.path.join(processed_data_path, "features.csv")
features_df.to_csv(features_file, index=False)

print(f"Feature extraction complete.")
print(f"Total feature rows created: {len(features_df)}")
print(f"Features saved -> {features_file}")
