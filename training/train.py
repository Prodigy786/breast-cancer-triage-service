import os
import sys
import json
import argparse
import warnings
import yaml
import joblib
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")


def train(data_path: str, output_dir: str, model_version: str, params_path: str = "params.yaml") -> None:
    # 1. Load parameters
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)["train"]

    seed = params.get("seed", 42)
    test_size = params.get("test_size", 0.2)
    cv_folds = params.get("cv_folds", 5)

    lr_params = params.get("logistic_regression", {})
    rf_params = params.get("random_forest", {})

    # 2. Read dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "target"]
    assert len(feature_cols) == 30, f"Expected 30 features, got {len(feature_cols)}"

    X = df[feature_cols]
    # In original load_breast_cancer dataset, 0=malignant, 1=benign.
    # Service contract states malignant is the positive class (1).
    y = (df["target"] == 0).astype(int)

    # 3. Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples.")
    print(f"Class distribution (train): Malignant={sum(y_train)}, Benign={len(y_train)-sum(y_train)}")
    print(f"Class distribution (test): Malignant={sum(y_test)}, Benign={len(y_test)-sum(y_test)}")

    # 4. Candidate pipelines
    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=lr_params.get("C", 1.0),
                max_iter=lr_params.get("max_iter", 1000),
                solver=lr_params.get("solver", "lbfgs"),
                random_state=seed
            ))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=rf_params.get("n_estimators", 100),
                max_depth=rf_params.get("max_depth", 5),
                random_state=seed
            ))
        ])
    }

    # 5. 5-fold Stratified Cross-Validation on training split
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_results = {}

    for name, pipeline in candidates.items():
        recalls, accuracies, rocs = [], [], []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)
            probs = pipeline.predict_proba(X_val)[:, 1]

            recalls.append(recall_score(y_val, preds, pos_label=1))
            accuracies.append(accuracy_score(y_val, preds))
            rocs.append(roc_auc_score(y_val, probs))

        cv_results[name] = {
            "mean_cv_malignant_recall": float(np.mean(recalls)),
            "std_cv_malignant_recall": float(np.std(recalls)),
            "mean_cv_accuracy": float(np.mean(accuracies)),
            "std_cv_accuracy": float(np.std(accuracies)),
            "mean_cv_roc_auc": float(np.mean(rocs)),
            "std_cv_roc_auc": float(np.std(rocs)),
        }
        print(f"Candidate [{name}] CV Malignant Recall: {cv_results[name]['mean_cv_malignant_recall']:.4f}, "
              f"Accuracy: {cv_results[name]['mean_cv_accuracy']:.4f}, ROC AUC: {cv_results[name]['mean_cv_roc_auc']:.4f}")

    # 6. Select model with highest mean CV malignant recall
    selected_name = max(cv_results, key=lambda k: cv_results[k]["mean_cv_malignant_recall"])
    selected_pipeline = candidates[selected_name]
    print(f"\n--> Selected Model: {selected_name} (Mean CV Recall: {cv_results[selected_name]['mean_cv_malignant_recall']:.4f})")

    # Fit selected pipeline on all training data
    selected_pipeline.fit(X_train, y_train)

    # 7. Evaluate on held-out test set
    test_preds = selected_pipeline.predict(X_test)
    test_probs = selected_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "test_malignant_recall": float(recall_score(y_test, test_preds, pos_label=1)),
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
    }

    print("\nHeld-out Test Set Evaluation:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Gate check: test malignant recall >= 0.95
    if test_metrics["test_malignant_recall"] < 0.95:
        print(f"ERROR: Test malignant recall {test_metrics['test_malignant_recall']:.4f} is below threshold 0.95!", file=sys.stderr)
        sys.exit(1)

    print("Quality gate passed: Test malignant recall >= 0.95.")

    # 8. Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(selected_pipeline, model_path)
    print(f"Serialized model pipeline saved to {model_path}")

    model_card = {
        "model_version": model_version,
        "model_family": selected_name,
        "classifier_class": selected_pipeline.named_steps["classifier"].__class__.__name__,
        "scikit_learn_version": sklearn.__version__,
        "random_seed": seed,
        "feature_count": len(feature_cols),
        "ordered_features": feature_cols,
        "cv_evaluation": cv_results,
        "test_metrics": test_metrics,
    }

    model_card_path = os.path.join(output_dir, "model_card.json")
    with open(model_card_path, "w") as f:
        json.dump(model_card, f, indent=2)
    print(f"Model card saved to {model_card_path}")

    # Write metrics.json (for DVC metrics)
    metrics_path = "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "selected_model": selected_name,
            **test_metrics,
            "cv_malignant_recall": cv_results[selected_name]["mean_cv_malignant_recall"],
            "cv_accuracy": cv_results[selected_name]["mean_cv_accuracy"],
            "cv_roc_auc": cv_results[selected_name]["mean_cv_roc_auc"]
        }, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reproducible classification model")
    parser.add_argument("--data-path", type=str, default="data/raw/breast_cancer.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--model-version", type=str, default="1.0.0")
    args = parser.parse_args()

    train(args.data_path, args.output_dir, args.model_version)
