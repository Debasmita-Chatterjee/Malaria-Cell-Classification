# evaluation_metrics.py
# --------------------------------------------------------
# PART 4: Model Validation and Evaluation Metrics
# Author: Tomin Jacob | ISI Internship G9
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from malaria_cnn_model import X_test, y_test  # Import test data from the training file

# --------------------------------------------------------
# Load Trained Model
# --------------------------------------------------------
print("\n🚀 Model Validation & Evaluation Started...\n")
model = load_model("malaria_cnn_model.h5")
print("✅ CNN model loaded successfully!")

# --------------------------------------------------------
# Predict on Test Data
# --------------------------------------------------------
print("🔍 Generating predictions on test dataset...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
print("✅ Predictions generated successfully!\n")

# --------------------------------------------------------
# Calculate Metrics
# --------------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# --------------------------------------------------------
# Display Metrics Nicely
# --------------------------------------------------------
print("📊 MODEL EVALUATION METRICS")
print("-" * 45)
print(f"🧠 Accuracy :  {accuracy * 100:.2f}%")
print(f"🎯 Precision:  {precision * 100:.2f}%")
print(f"📈 Recall   :  {recall * 100:.2f}%")
print(f"⚖️  F1 Score :  {f1 * 100:.2f}%")
print("-" * 45)

# --------------------------------------------------------
# Save Metrics to Text File
# --------------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open("metrics_report.txt", "w") as f:
    f.write("MODEL EVALUATION REPORT\n")
    f.write("=" * 40 + "\n")
    f.write(f"Generated on: {timestamp}\n\n")
    f.write(f"Accuracy :  {accuracy * 100:.2f}%\n")
    f.write(f"Precision:  {precision * 100:.2f}%\n")
    f.write(f"Recall   :  {recall * 100:.2f}%\n")
    f.write(f"F1 Score :  {f1 * 100:.2f}%\n\n")
    f.write("CONFUSION MATRIX:\n")
    f.write(np.array2string(cm))
    f.write("\n\nDETAILED CLASSIFICATION REPORT:\n")
    f.write(classification_report(y_true, y_pred, target_names=["Infected", "Uninfected"]))
print("📝 Metrics saved successfully to 'metrics_report.txt'\n")

# --------------------------------------------------------
# Plot & Save Confusion Matrix
# --------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=["Infected", "Uninfected"],
            yticklabels=["Infected", "Uninfected"])
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label", fontsize=11)
plt.title("Confusion Matrix - Malaria Cell Classification", fontsize=13)
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
print("📸 Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# --------------------------------------------------------
# Classification Report
# --------------------------------------------------------
print("\n🧩 DETAILED CLASSIFICATION REPORT")
print(classification_report(y_true, y_pred, target_names=["Infected", "Uninfected"]))
print("✅ Evaluation completed successfully!\n")
