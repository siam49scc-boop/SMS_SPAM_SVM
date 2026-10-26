"""Train the pipeline and save reproducible metrics, charts and model."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, accuracy_score,
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from sms_classifier import (
    build_pipeline, load_dataset, parameter_grid, setup_nltk, top_features,
)

ROOT = Path(__file__).resolve().parent


def save_evaluation(model, X_test, y_test, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    predicted = model.predict(X_test)
    scores = model.decision_function(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "spam_precision": float(precision_score(y_test, predicted, zero_division=0)),
        "spam_recall": float(recall_score(y_test, predicted, zero_division=0)),
        "spam_f1": float(f1_score(y_test, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "majority_baseline_accuracy": float((y_test == 0).mean()),
        "confusion_matrix_ham_spam": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predicted, labels=[0, 1], target_names=["ham", "spam"],
            output_dict=True, zero_division=0,
        ),
    }
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, labels=[0, 1], display_labels=["Ham", "Spam"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title("SVM: held-out test set")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    PrecisionRecallDisplay.from_predictions(y_test, scores, name="Linear SVM", ax=ax)
    ax.set_title("Precision–recall from SVM decision scores")
    fig.tight_layout()
    fig.savefig(results_dir / "precision_recall_curve.png", dpi=170)
    plt.close(fig)

    errors = pd.DataFrame({
        "message": X_test.to_numpy(),
        "true_label": np.where(y_test.to_numpy() == 1, "spam", "ham"),
        "predicted_label": np.where(predicted == 1, "spam", "ham"),
        "decision_score": scores,
    })
    errors = errors[errors.true_label != errors.predicted_label]
    errors.to_csv(results_dir / "misclassified_messages.csv", index=False)
    spam, ham = top_features(model)
    spam.to_csv(results_dir / "top_spam_features.csv", index=False)
    ham.to_csv(results_dir / "top_ham_features.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "spam.csv")
    parser.add_argument("--quick", action="store_true", help="Two candidates instead of 24.")
    parser.add_argument("--jobs", type=int, default=1, help="1 uses less memory; -1 uses all CPUs.")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    setup_nltk()
    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        df.message, df.target, test_size=0.20, random_state=42, stratify=df.target,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        build_pipeline(), parameter_grid(args.quick), scoring="f1", cv=cv,
        n_jobs=args.jobs, refit=True, error_score="raise", verbose=1,
    )
    search.fit(X_train, y_train)
    results_dir = args.output / "results"
    metrics = save_evaluation(search.best_estimator_, X_test, y_test, results_dir)
    metrics.update({
        "dataset": df.attrs["dataset_summary"],
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "training_rows": len(X_train), "test_rows": len(X_test),
        "random_state": 42, "cv_folds": 5,
        "search_mode": "quick" if args.quick else "full",
        "best_parameters": search.best_params_,
        "best_training_cv_f1": float(search.best_score_),
        "cv_note": "Selection score from training folds; not an unbiased post-selection estimate.",
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "nltk": nltk.__version__},
    })
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(search.cv_results_).to_csv(results_dir / "cv_results.csv", index=False)
    model_dir = args.output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, model_dir / "sms_spam_model.joblib")
    print(classification_report(y_test, search.best_estimator_.predict(X_test),
                                target_names=["ham", "spam"], zero_division=0))
    print("Best parameters:", search.best_params_)
    print(f"Held-out spam F1: {metrics['spam_f1']:.4f}")
    print("Saved model:", model_dir / "sms_spam_model.joblib")


if __name__ == "__main__":
    main()
