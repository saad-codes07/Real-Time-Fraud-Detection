"""
train.py - Train an XGBoost fraud detector and track everything in MLflow.

Pipeline (in order):
  1. Load the Kaggle credit card CSV.
  2. Light feature engineering.
  3. Split into TRAIN / VALIDATION / TEST with StratifiedShuffleSplit.
  4. Apply SMOTE to the TRAIN set ONLY.
  5. Train XGBoost, with early stopping that watches AUPRC on the validation set.
  6. Pick a decision threshold on the validation set.
  7. Evaluate once on the untouched test set.
  8. Log params, metrics, a PR-curve plot and the model to MLflow, and save
     the model locally for the API to load.

Run from the project root:
    python model_training/train.py --data-path data/creditcard.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw plots without needing a screen (works in Docker)
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

TARGET = "Class"  # 1 = fraud, 0 = legitimate


# --------------------------------------------------------------------------- #
# Step 1-2: load data and build features
# --------------------------------------------------------------------------- #
def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # "Time" is the number of seconds since the FIRST transaction in the dataset.
    # That number would mean nothing for a brand-new live transaction, so we
    # convert it to "hour of day" (0-23), which is meaningful at any time.
    df["Hour"] = (df["Time"] // 3600) % 24
    df = df.drop(columns=["Time"])

    # Note: we do NOT scale features. Tree models like XGBoost split on
    # thresholds, so feature scale does not matter to them.
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


# --------------------------------------------------------------------------- #
# Step 3: stratified splits
# --------------------------------------------------------------------------- #
def make_splits(X, y, seed: int):
    """60% train / 20% validation / 20% test, each keeping the same fraud ratio.

    Frauds are only ~0.17% of rows. A plain random split could leave the test
    set with very few frauds. 'Stratified' guarantees the same ratio everywhere.
    """
    # First cut: set aside 20% as the final TEST set.
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trainval_idx, test_idx = next(outer.split(X, y))
    X_trainval, y_trainval = X.iloc[trainval_idx], y.iloc[trainval_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Second cut: from the remaining 80%, take 25% (= 20% of the total) as VALIDATION.
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, val_idx = next(inner.split(X_trainval, y_trainval))
    X_train, y_train = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
    X_val, y_val = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


# --------------------------------------------------------------------------- #
# Step 6: choose a decision threshold
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true, y_proba) -> float:
    """Return the probability cut-off that maximises F1 on the given data.

    The model outputs a fraud PROBABILITY. We must decide 'above what number
    do we call it fraud?'. The default 0.5 is rarely best for rare events.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one more entry than thresholds, so drop the last.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1)])


def evaluate(y_true, y_proba, threshold: float, prefix: str) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        f"{prefix}_auprc": average_precision_score(y_true, y_proba),  # MAIN METRIC
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_proba),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_tp": int(tp),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tn": int(tn),
    }


def pr_curve_figure(y_true, y_proba, auprc: float):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"XGBoost (AUPRC = {auprc:.3f})")
    ax.axhline(y_true.mean(), ls="--", color="gray", label="Random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test set)")
    ax.legend()
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    X, y = load_and_prepare(args.data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = make_splits(X, y, args.seed)
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    print(f"Fraud rate -> train {y_train.mean():.4%}, val {y_val.mean():.4%}, test {y_test.mean():.4%}")

    # Step 4: SMOTE on TRAIN ONLY.
    # SMOTE invents synthetic fraud rows by interpolating between real ones.
    # If we applied it before splitting, synthetic copies of test frauds would
    # leak into training and our scores would look better than reality.
    # Validation and test keep the real, imbalanced distribution on purpose.
    scale_pos_weight = 1.0
    if args.use_smote:
        smote = SMOTE(
            sampling_strategy=args.smote_ratio,  # fraud becomes this fraction of legit count
            k_neighbors=5,
            random_state=args.seed,
        )
        X_fit, y_fit = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {len(X_fit)} rows, fraud share {y_fit.mean():.2%}")
    else:
        # Alternative to SMOTE: tell XGBoost to weight fraud rows more heavily.
        X_fit, y_fit = X_train, y_train
        scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    params = {
        "n_estimators": 2000,  # upper limit; early stopping decides the real number
        "learning_rate": 0.05,
        "max_depth": args.max_depth,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",  # XGBoost optimises/monitors AUPRC directly
        "early_stopping_rounds": 50,
        "tree_method": "hist",  # fast algorithm
        "random_state": args.seed,
        "n_jobs": -1,
    }

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="xgboost-smote" if args.use_smote else "xgboost-weighted"):
        mlflow.log_params({**params, "use_smote": args.use_smote, "smote_ratio": args.smote_ratio})

        # Step 5: train. Early stopping watches the VALIDATION set (real distribution)
        # and stops when AUPRC has not improved for 50 rounds, preventing overfitting.
        model = XGBClassifier(**params)
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=100)
        mlflow.log_metric("best_iteration", model.best_iteration)

        # Step 6: threshold chosen on validation (never on test!).
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = best_f1_threshold(y_val, val_proba)
        mlflow.log_metric("threshold", threshold)

        # Step 7: one final evaluation on the untouched test set.
        test_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            **evaluate(y_val, val_proba, threshold, "val"),
            **evaluate(y_test, test_proba, threshold, "test"),
        }
        mlflow.log_metrics(metrics)
        print(f"\nTEST AUPRC: {metrics['test_auprc']:.4f}")
        print(f"TEST precision {metrics['test_precision']:.3f} | recall {metrics['test_recall']:.3f} | F1 {metrics['test_f1']:.3f}")

        # Step 8: save artifacts.
        fig = pr_curve_figure(y_test, test_proba, metrics["test_auprc"])
        mlflow.log_figure(fig, "pr_curve.png")
        plt.close(fig)
        mlflow.xgboost.log_model(model.get_booster(), artifact_path="model")

        # Local copy for the FastAPI service (it needs the model + threshold + feature order).
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(out_dir / "model.json")
        meta = {
            "threshold": threshold,
            "features": list(X.columns),
            "test_auprc": metrics["test_auprc"],
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(out_dir / "metadata.json"))
        print(f"Saved model to {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Train fraud detection model")
    p.add_argument("--data-path", default="data/creditcard.csv")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="fraud-detection")
    p.add_argument("--use-smote", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-use-smote trains with class weights instead")
    p.add_argument("--smote-ratio", type=float, default=0.1,
                   help="fraud:legit ratio after SMOTE (0.1 = 1 fraud per 10 legit)")
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
