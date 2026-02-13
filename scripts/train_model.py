import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

# Absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_path = os.path.join(current_dir, "..", "data", "processed", "features.csv")
model_path = os.path.join(current_dir, "..", "models", "gait_model.pkl")

# Ensure models folder exists
os.makedirs(os.path.join(current_dir, "..", "models"), exist_ok=True)

print("Loading features for training...")
features_df = pd.read_csv(processed_data_path)

# Check if file is empty
if features_df.empty:
    raise ValueError(f"features.csv is empty! Check {processed_data_path}")

X = features_df.drop("label", axis=1)
y = features_df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training complete. Test Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(clf, model_path)
print(f"Model saved -> {model_path}")
