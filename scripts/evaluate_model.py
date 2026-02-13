import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

features_path = os.path.join(BASE_DIR, "data", "processed", "features.csv")
model_path = os.path.join(BASE_DIR, "models", "gait_model.pkl")
eval_path = os.path.join(BASE_DIR, "evaluation")

os.makedirs(eval_path, exist_ok=True)

print("Loading features and model...")
df = pd.read_csv(features_path)
X = df.drop("label", axis=1)
y = df["label"]

model = joblib.load(model_path)
y_pred = model.predict(X)

# Classification report
report = classification_report(y, y_pred)

with open(os.path.join(eval_path, "classification_report.txt"), "w") as f:
    f.write(report)

print("Classification report saved.")

# Confusion matrix
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(eval_path, "confusion_matrix.png"))

print("Confusion matrix saved.")
