import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(current_dir, "..", "data", "raw")
merged_data_path = os.path.join(current_dir, "..", "data", "merged")

os.makedirs(merged_data_path, exist_ok=True)

print("Starting sensor data merge...")

for person_folder in os.listdir(raw_data_path):
    person_path = os.path.join(raw_data_path, person_folder)

    if os.path.isdir(person_path):

        # Find all accelerometer files
        acc_files = [f for f in os.listdir(person_path) if f.startswith("acc")]
        
        for acc_file in acc_files:
            trial_id = acc_file.replace("acc_", "").replace(".csv", "")
            gyro_file = f"gyro_{trial_id}.csv"

            acc_path = os.path.join(person_path, acc_file)
            gyro_path = os.path.join(person_path, gyro_file)

            if os.path.exists(gyro_path):

                acc_df = pd.read_csv(acc_path, skiprows=3)
                gyro_df = pd.read_csv(gyro_path, skiprows=3)

                # Rename columns
                acc_df = acc_df.rename(columns={
                    acc_df.columns[1]: "acc_x",
                    acc_df.columns[2]: "acc_y",
                    acc_df.columns[3]: "acc_z"
                })

                gyro_df = gyro_df.rename(columns={
                    gyro_df.columns[1]: "gyro_x",
                    gyro_df.columns[2]: "gyro_y",
                    gyro_df.columns[3]: "gyro_z"
                })

                acc_df = acc_df[["acc_x", "acc_y", "acc_z"]]
                gyro_df = gyro_df[["gyro_x", "gyro_y", "gyro_z"]]

                merged_df = pd.concat([acc_df, gyro_df], axis=1)
                merged_df["label"] = person_folder

                merged_filename = f"{person_folder}_{trial_id}.csv"
                merged_path = os.path.join(merged_data_path, merged_filename)

                merged_df.to_csv(merged_path, index=False)

                print(f"Merged: {merged_filename}")

print("All trials merged successfully.")
