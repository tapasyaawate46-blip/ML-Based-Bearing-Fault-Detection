
# ML-Based Bearing Fault Detection from FFT Features

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Step 1: Define project paths

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_DIR, "Dataset", "bearing_features.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Project directory:", PROJECT_DIR)
print("Results directory:", RESULTS_DIR)

# Step 2: Load MATLAB-generated features

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("\nTotal samples:", len(df))
print("Available columns:", df.columns.tolist())

# Step 3: Validate and clean the dataset

feature_columns = [
    "RMS",
    "StdDev",
    "PeakAmplitude",
    "DominantFrequency",
    "SpectralEnergy"
]

required_columns = feature_columns + ["Condition"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df = df[required_columns].copy()

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required_columns)

if df.empty:
    raise ValueError("Dataset contains no valid rows.")

df["Condition"] = df["Condition"].astype(int)

if set(df["Condition"].unique()) != {0, 1}:
    raise ValueError("Condition labels must be 0 and 1.")

print("\nValid samples:", len(df))
print("\nClass distribution:")
print(df["Condition"].map({
    0: "Normal",
    1: "Inner-Race Fault"
}).value_counts())

# Step 4: Prepare training and testing data

X = df[feature_columns]
y = df["Condition"]

train_indices = []
test_indices = []

for label in sorted(y.unique()):
    class_indices = np.flatnonzero(y.to_numpy() == label)

    if len(class_indices) < 5:
        raise ValueError(f"Not enough samples for class {label}")

    split_position = int(len(class_indices) * 0.80)

    train_indices.extend(class_indices[:split_position - 1])
    test_indices.extend(class_indices[split_position:])

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 5: Train the Random Forest classifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nRandom Forest training completed.")

# Step 6: Predict and evaluate the model

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
class_names = ["Normal", "Inner-Race Fault"]

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

# Step 7: Generate and save the confusion matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

fig, ax = plt.subplots(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Random Forest - Confusion Matrix")
plt.tight_layout()

fig.savefig(
    os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Confusion matrix saved.")

# Step 8: Calculate and plot feature importance

importance = model.feature_importances_
sorted_indices = np.argsort(importance)

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(
    np.array(feature_columns)[sorted_indices],
    importance[sorted_indices]
)

ax.set_title("Random Forest - Feature Importance")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()

fig.savefig(
    os.path.join(RESULTS_DIR, "feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Feature importance graph saved.")

# Step 9: Save model evaluation results

results_path = os.path.join(RESULTS_DIR, "model_results.txt")

with open(results_path, "w", encoding="utf-8") as file:
    file.write("Bearing Fault Detection - Random Forest\n\n")
    file.write(f"Total samples: {len(df)}\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n")
    file.write(f"Test accuracy: {accuracy * 100:.2f}%\n\n")

    file.write("Features used:\n")
    for feature in feature_columns:
        file.write(f"- {feature}\n")

    file.write("\nClassification Report:\n")
    file.write(report)

    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))

    file.write("\n\nFeature Importance:\n")
    for name, score in sorted(
        zip(feature_columns, importance),
        key=lambda item: item[1],
        reverse=True
    ):
        file.write(f"{name}: {score:.6f}\n")

    file.write(
        "\nNote: Training and test windows come from the same "
        "source recordings. Performance on independent recordings "
        "has not been evaluated.\n"
    )

print("\nEvaluation results saved.")

# Step 10: Display completion message

print("\nAll outputs saved successfully.")
print("Output folder:", RESULTS_DIR)
