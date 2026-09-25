import numpy as np

from sklearn.ensemble import IsolationForest

from src.evaluation.evaluator import (
    DocumentAIEvaluator
)


# ============================================================
# TRAINING DATA
# ============================================================
#
# 5-dimensional document feature vector:
#
# 1. total_amount
# 2. row_count
# 3. avg_ocr_confidence
# 4. validation_issue_count
# 5. processing_time
#
# These represent NORMAL documents.
# ============================================================

X_train = np.array([
    [500, 5, 0.96, 0, 2.1],
    [520, 6, 0.95, 0, 2.2],
    [480, 5, 0.97, 0, 2.0],
    [510, 6, 0.94, 0, 2.3],
    [495, 5, 0.96, 0, 2.1],
    [530, 6, 0.95, 0, 2.2],
    [505, 5, 0.97, 0, 2.0],
    [515, 6, 0.96, 0, 2.1],
    [490, 5, 0.95, 0, 2.2],
    [525, 6, 0.94, 0, 2.3],
])


# ============================================================
# UNSEEN TEST DATA
# ============================================================

X_test = np.array([
    # Normal
    [505, 5, 0.96, 0, 2.1],

    # Normal
    [515, 6, 0.95, 0, 2.2],

    # Anomaly
    [5000, 30, 0.40, 5, 15.0],

    # Normal
    [495, 5, 0.97, 0, 2.0],

    # Anomaly
    [12000, 50, 0.30, 8, 25.0],

    # Normal
    [520, 6, 0.96, 0, 2.1],
])


# ============================================================
# GROUND TRUTH
# ============================================================
#
# 0 = normal
# 1 = anomaly
# ============================================================

y_true = np.array([
    0,
    0,
    1,
    0,
    1,
    0
])


# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

print("=" * 60)
print("ANOMALY TRAIN / TEST EVALUATION")
print("=" * 60)

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

model.fit(X_train)


# ============================================================
# PREDICT UNSEEN TEST DATA
# ============================================================

print("Predicting unseen test data...")

raw_predictions = model.predict(
    X_test
)


# sklearn:
#
#  1  = normal
# -1  = anomaly
#
# Convert to:
#
#  0 = normal
#  1 = anomaly
# ============================================================

y_pred = np.where(
    raw_predictions == -1,
    1,
    0
)


# ============================================================
# EVALUATION
# ============================================================

evaluator = DocumentAIEvaluator()

metrics = evaluator.evaluate_anomaly(
    y_true,
    y_pred
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Precision : "
    f"{metrics['precision']}"
)

print(
    f"Recall    : "
    f"{metrics['recall']}"
)

print(
    f"F1 Score  : "
    f"{metrics['f1_score']}"
)

print(
    f"Anomaly Rate : "
    f"{metrics['anomaly_rate']}"
)

print(
    f"True Positive  : "
    f"{metrics['true_positive']}"
)

print(
    f"False Positive : "
    f"{metrics['false_positive']}"
)

print(
    f"False Negative : "
    f"{metrics['false_negative']}"
)

print(
    f"True Negative  : "
    f"{metrics['true_negative']}"
)


# ============================================================
# PAGE-LEVEL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("TEST SAMPLE PREDICTIONS")
print("=" * 60)

for index, (
    actual,
    predicted
) in enumerate(
    zip(
        y_true,
        y_pred
    ),
    start=1
):

    actual_label = (
        "ANOMALY"
        if actual == 1
        else "NORMAL"
    )

    predicted_label = (
        "ANOMALY"
        if predicted == 1
        else "NORMAL"
    )

    print(
        f"Sample {index}: "
        f"Actual={actual_label} | "
        f"Predicted={predicted_label}"
    )


print("\n")
print("=" * 60)
print("TRAIN / TEST EVALUATION COMPLETE")
print("=" * 60)
