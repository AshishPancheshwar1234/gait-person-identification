import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(BASE_DIR, "data", "processed", "features.csv")

print("Loading features...")
df = pd.read_csv(features_path)

# Separate label
labels = df["label"]
features = df.drop("label", axis=1)

# Ensure numeric type
features = features.apply(pd.to_numeric, errors="coerce")

augmented_rows = []

for _, row in features.iterrows():
    noise = np.random.normal(0, 0.02, size=len(row))
    new_row = row + noise
    augmented_rows.append(new_row)

aug_df = pd.DataFrame(augmented_rows, columns=features.columns)

# Add labels back
aug_df["label"] = labels.values

# Combine original + augmented
final_df = pd.concat([df, aug_df], ignore_index=True)
final_df.to_csv(features_path, index=False)

print("Data augmentation complete.")
print(f"New dataset size: {len(final_df)}")
