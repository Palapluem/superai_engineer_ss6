from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - handled at runtime
    XGBClassifier = None


META_COLUMNS = {
    "window_start_ts",
    "window_end_ts",
    "session_id",
    "class_th",
    "label",
    "source",
    "class_en",
    "category",
    "class",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Classical ML models from windows_all.csv feature windows. "
            "Default target is binary fall-risk label."
        )
    )
    parser.add_argument("--data", default="windows_all.csv", help="Input CSV path.")
    parser.add_argument(
        "--target",
        default="label",
        help="Target column, e.g. label, class_en, category.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output folder name under models/ and reports/. Defaults to target name.",
    )
    parser.add_argument(
        "--models",
        default="random_forest,svm_rbf,xgboost",
        help="Comma-separated models: random_forest, svm_rbf, xgboost.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-feature-plot",
        type=int,
        default=30,
        help="Number of top features to show in feature importance plots.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "run"


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def select_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    ignored = set(META_COLUMNS)
    ignored.discard(target_col)
    ignored.add(target_col)

    feature_cols: list[str] = []
    for col in df.columns:
        if col in ignored:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No numeric feature columns were found.")
    return feature_cols


def build_preprocessor(feature_cols: list[str], scale: bool) -> ColumnTransformer:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps)
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(
    requested: list[str],
    feature_cols: list[str],
    num_classes: int,
    random_state: int,
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}

    if "random_forest" in requested:
        models["random_forest"] = Pipeline(
            steps=[
                ("prep", build_preprocessor(feature_cols, scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if "svm_rbf" in requested:
        models["svm_rbf"] = Pipeline(
            steps=[
                ("prep", build_preprocessor(feature_cols, scale=True)),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=10.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if "xgboost" in requested:
        if XGBClassifier is None:
            print("WARN: xgboost is not installed, skipping xgboost.", file=sys.stderr)
        else:
            objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
            eval_metric = "logloss" if num_classes == 2 else "mlogloss"
            xgb_params: dict[str, Any] = {
                "n_estimators": 350,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "objective": objective,
                "eval_metric": eval_metric,
                "random_state": random_state,
                "n_jobs": -1,
                "tree_method": "hist",
            }
            if num_classes > 2:
                xgb_params["num_class"] = num_classes

            models["xgboost"] = Pipeline(
                steps=[
                    ("prep", build_preprocessor(feature_cols, scale=False)),
                    ("model", XGBClassifier(**xgb_params)),
                ]
            )

    unknown = sorted(set(requested) - {"random_forest", "svm_rbf", "xgboost"})
    if unknown:
        raise ValueError(f"Unknown model name(s): {', '.join(unknown)}")
    if not models:
        raise ValueError("No models are available to train.")
    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if len(class_names) == 2:
        positive_index = class_names.index("1") if "1" in class_names else 1
        metrics["positive_class"] = class_names[positive_index]
        metrics["positive_recall"] = recall_score(
            y_true, y_pred, pos_label=positive_index, zero_division=0
        )
        metrics["positive_precision"] = precision_score(
            y_true, y_pred, pos_label=positive_index, zero_division=0
        )
        metrics["positive_f1"] = f1_score(
            y_true, y_pred, pos_label=positive_index, zero_division=0
        )
    return metrics


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    report_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(report_dir / f"confusion_matrix_{model_name}.csv", encoding="utf-8")

    width = max(7, min(15, len(class_names) * 1.25))
    fig, ax = plt.subplots(figsize=(width, width * 0.8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(report_dir / f"confusion_matrix_{model_name}.png", dpi=180)
    plt.close(fig)


def extract_feature_importance(model: Pipeline, feature_cols: list[str]) -> np.ndarray | None:
    estimator = model.named_steps.get("model")
    if estimator is None:
        return None
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        return np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    return None


def save_feature_importance(
    model: Pipeline,
    feature_cols: list[str],
    model_name: str,
    report_dir: Path,
    max_feature_plot: int,
) -> None:
    importances = extract_feature_importance(model, feature_cols)
    if importances is None or len(importances) != len(feature_cols):
        return

    fi = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi.to_csv(report_dir / f"feature_importance_{model_name}.csv", index=False, encoding="utf-8")

    top = fi.head(max_feature_plot).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.27)))
    ax.barh(top["feature"], top["importance"], color="#0f766e")
    ax.set_title(f"Top Feature Importance: {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(report_dir / f"feature_importance_{model_name}.png", dpi=180)
    plt.close(fig)


def safe_cv_folds(y: np.ndarray, requested_folds: int) -> int:
    if requested_folds <= 1:
        return 0
    _, counts = np.unique(y, return_counts=True)
    return int(min(requested_folds, counts.min()))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    run_name = slugify(args.run_name or f"{data_path.stem}_{args.target}")
    model_dir = data_path.parent / "models" / run_name
    report_dir = data_path.parent / "reports" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {data_path.name}.")

    df = df.dropna(subset=[args.target]).copy()
    target_series = df[args.target].astype(str)
    feature_cols = select_feature_columns(df, args.target)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target_series)
    class_names = [str(x) for x in label_encoder.classes_]
    X = df[feature_cols].copy()

    class_counts = target_series.value_counts().sort_index()
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "target_col": args.target,
        "rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "class_names": class_names,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    write_json(report_dir / "dataset_summary.json", metadata)
    write_json(model_dir / "feature_columns.json", feature_cols)
    write_json(model_dir / "class_names.json", class_names)

    stratify = y if np.unique(y).size > 1 and np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    models = build_models(requested_models, feature_cols, len(class_names), args.random_state)

    fold_count = safe_cv_folds(y_train, args.cv_folds)
    cv = (
        StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=args.random_state)
        if fold_count >= 2
        else None
    )

    metrics_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        print(f"\n== Training {model_name} ==")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, class_names)
        metrics.update({"model": model_name, "train_rows": len(X_train), "test_rows": len(X_test)})

        if cv is not None:
            cv_scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "accuracy": "accuracy",
                    "balanced_accuracy": "balanced_accuracy",
                    "f1_macro": "f1_macro",
                    "f1_weighted": "f1_weighted",
                },
                n_jobs=1,
                error_score="raise",
            )
            for key, value in cv_scores.items():
                if not key.startswith("test_"):
                    continue
                metric_name = f"cv_{key[5:]}"
                metrics[f"{metric_name}_mean"] = float(np.mean(value))
                metrics[f"{metric_name}_std"] = float(np.std(value))

        metrics_rows.append(metrics)
        fitted_models[model_name] = model

        report = classification_report(
            y_test,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        (report_dir / f"classification_report_{model_name}.txt").write_text(
            report, encoding="utf-8"
        )
        save_confusion_matrix(y_test, y_pred, class_names, model_name, report_dir)
        save_feature_importance(model, feature_cols, model_name, report_dir, args.max_feature_plot)
        joblib.dump(model, model_dir / f"{model_name}.joblib")

        print(
            f"{model_name}: "
            f"acc={metrics['accuracy']:.4f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["f1_macro", "balanced_accuracy", "accuracy"], ascending=False
    )
    metrics_df.to_csv(report_dir / "model_comparison.csv", index=False, encoding="utf-8")
    write_json(report_dir / "model_comparison.json", metrics_df.to_dict(orient="records"))

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_payload = {
        "model": best_model,
        "label_encoder": label_encoder,
        "feature_columns": feature_cols,
        "target_col": args.target,
        "class_names": class_names,
        "model_name": best_model_name,
        "metrics": metrics_df.iloc[0].to_dict(),
        "metadata": metadata,
    }
    joblib.dump(best_payload, model_dir / "best_model.joblib")

    summary_lines = [
        f"# Classical ML Training Summary: {run_name}",
        "",
        f"- Dataset: `{data_path.name}`",
        f"- Rows used: {len(df)}",
        f"- Target: `{args.target}`",
        f"- Feature count: {len(feature_cols)}",
        f"- Best model: `{best_model_name}`",
        "",
        "## Class Distribution",
        "",
    ]
    for label, count in class_counts.items():
        summary_lines.append(f"- `{label}`: {count}")
    summary_lines.extend(
        [
            "",
            "## Model Comparison",
            "",
            metrics_df.to_markdown(index=False),
            "",
            "## Main Artifacts",
            "",
            f"- Best model bundle: `{model_dir / 'best_model.joblib'}`",
            f"- Metrics CSV: `{report_dir / 'model_comparison.csv'}`",
            f"- Dataset summary: `{report_dir / 'dataset_summary.json'}`",
        ]
    )
    (report_dir / "summary_report.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nDone.")
    print(f"Best model: {best_model_name}")
    print(f"Model bundle: {model_dir / 'best_model.joblib'}")
    print(f"Report: {report_dir / 'summary_report.md'}")


if __name__ == "__main__":
    main()
