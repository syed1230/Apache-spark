
"""
Student Assistance: Decision Tree demonstration model.

Training labels are generated from explicit synthetic rules.
Evaluation measures how well the model reproduces those rules,
not how accurately it predicts real student outcomes.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Math",
    "Science",
    "English",
]

CATEGORIES = ["High support", "Moderate support", "On track"]


def _demo_training_data():
    """Generate reproducible synthetic examples with transparent rules."""
    rng = np.random.default_rng(42)
    count = 1200

    data = pd.DataFrame({
        "Attendance": rng.integers(30, 101, count).astype(float),
        "Study_Hours": rng.integers(1, 9, count).astype(float),
        "Math": rng.integers(20, 101, count).astype(float),
        "Science": rng.integers(20, 101, count).astype(float),
        "English": rng.integers(20, 101, count).astype(float),
    })

    average = data[["Math", "Science", "English"]].mean(axis=1)
    attendance = data["Attendance"]
    study_hours = data["Study_Hours"]

    conditions = [
        (average < 40)
        | (attendance < 55)
        | ((study_hours < 2) & (average < 55)),

        (average < 60)
        | (attendance < 75)
        | (study_hours < 3),
    ]

    data["Support_Category"] = np.select(
        conditions,
        ["High support", "Moderate support"],
        default="On track",
    )

    return data


def _build_model():
    """Train once and evaluate on a separate held-out test set."""
    data = _demo_training_data()

    X = data[FEATURES]
    y = data["Support_Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=CATEGORIES
        ).tolist(),
        "class_labels": CATEGORIES,
        "test_samples": len(y_test),
        "training_samples": len(y_train),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=CATEGORIES,
            output_dict=True,
            zero_division=0,
        ),
        "data_note": (
            "Synthetic data: metrics measure reproduction of generated "
            "rules, not validated real-world student outcomes."
        ),
    }

    return model, metrics


# Reuse the trained model instead of training on every prediction.
_MODEL, _METRICS = _build_model()


def get_model_evaluation():
    """Return model evaluation results for display in the dashboard."""
    return _METRICS.copy()


def predict_student_support(student):
    """Predict a support category and return an explanation and suggestions."""
    values = pd.DataFrame([{
        feature: float(student[feature])
        for feature in FEATURES
    }], columns=FEATURES)

    category = str(_MODEL.predict(values)[0])

    average = sum(
        float(student[subject])
        for subject in ("Math", "Science", "English")
    ) / 3

    explanation = (
        f"Entered marks average: {average:.1f}/100; "
        f"attendance: {float(student['Attendance']):.0f}%; "
        f"study time: {float(student['Study_Hours']):.1f} hours/day. "
        "This is a demonstration prediction based on synthetic data."
    )

    if category == "High support":
        suggestions = [
            "Review the topics with the lowest marks with a teacher or tutor.",
            "Build a manageable daily study routine and practise regularly.",
            "Check attendance barriers and seek help if classes are being missed.",
        ]
    elif category == "Moderate support":
        suggestions = [
            "Focus practice on the weakest subject.",
            "Aim for consistent attendance and review notes weekly.",
            "Track quiz results to see whether your study plan is helping.",
        ]
    else:
        suggestions = [
            "Maintain the current study routine and attendance.",
            "Practise challenging questions to deepen understanding.",
            "Review progress regularly rather than relying on one prediction.",
        ]

    return {
        "category": category,
        "explanation": explanation,
        "suggestions": suggestions,
    }
