from __future__ import annotations

import argparse
import json
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model2_risk_common import (
    RiskTargetConfig,
    build_proxy_risk_targets,
    model2_config_payload,
    risk_level_from_score,
    select_feature_columns,
)

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:  # pragma: no cover
    HistGradientBoostingRegressor = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Model 2 risk assessment from window features. "
            "This creates proxy risk targets from motion/rotation/posture/impact/PPG features."
        )
    )
    parser.add_argument("--data", default="windows_all.csv")
    parser.add_argument("--run-name", default="model2_risk_assessment")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=["group", "random"],
        default="group",
        help="Use group split by session_id when available for a stricter estimate.",
    )
    parser.add_argument(
        "--original-only",
        action="store_true",
        help="Train only on rows where source == original.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_preprocessor(feature_cols: list[str], scale: bool) -> ColumnTransformer:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(steps), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_regressors(feature_cols: list[str], random_state: int) -> dict[str, Pipeline]:
    regressors: dict[str, Pipeline] = {
        "random_forest_regressor": Pipeline(
            [
                ("prep", build_preprocessor(feature_cols, scale=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "ridge_regressor": Pipeline(
            [
                ("prep", build_preprocessor(feature_cols, scale=True)),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        ),
    }
    if HistGradientBoostingRegressor is not None:
        regressors["hist_gradient_boosting_regressor"] = Pipeline(
            [
                ("prep", build_preprocessor(feature_cols, scale=False)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=0.01,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return regressors


def build_classifiers(feature_cols: list[str], random_state: int) -> dict[str, Pipeline]:
    return {
        "random_forest_classifier": Pipeline(
            [
                ("prep", build_preprocessor(feature_cols, scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "logistic_regression_classifier": Pipeline(
            [
                ("prep", build_preprocessor(feature_cols, scale=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def split_data(
    df: pd.DataFrame,
    test_size: float,
    split_mode: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_indexes = np.arange(len(df))
    if split_mode == "group" and "session_id" in df.columns and df["session_id"].nunique() > 1:
        groups = df["session_id"].astype(str).to_numpy()
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(row_indexes, groups=groups))
        return train_idx, test_idx

    target = df["model2_risk_level"].astype(str)
    stratify = target if target.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        row_indexes,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_idx, test_idx


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    clipped = np.clip(y_pred, 0.0, 1.0)
    return {
        "mae": float(mean_absolute_error(y_true, clipped)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, clipped))),
        "r2": float(r2_score(y_true, clipped)),
    }


def classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def estimator_importance(model: Pipeline, feature_cols: list[str]) -> np.ndarray | None:
    estimator = model.named_steps.get("model")
    if estimator is None:
        return None
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        return np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    return None


def save_importance(model: Pipeline, feature_cols: list[str], name: str, report_dir: Path) -> None:
    importance = estimator_importance(model, feature_cols)
    if importance is None or len(importance) != len(feature_cols):
        return
    fi = (
        pd.DataFrame({"feature": feature_cols, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi.to_csv(report_dir / f"feature_importance_{name}.csv", index=False, encoding="utf-8")
    top = fi.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.3)))
    ax.barh(top["feature"], top["importance"], color="#0f766e")
    ax.set_title(f"Model 2 Feature Importance: {name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(report_dir / f"feature_importance_{name}.png", dpi=180)
    plt.close(fig)


def save_plots(enriched: pd.DataFrame, report_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(
        enriched,
        x="model2_risk_target",
        hue="model2_risk_level",
        bins=30,
        multiple="stack",
        ax=ax,
    )
    ax.set_title("Model 2 Proxy Risk Target Distribution")
    ax.set_xlabel("Risk target score")
    fig.tight_layout()
    fig.savefig(report_dir / "risk_target_distribution.png", dpi=180)
    plt.close(fig)

    if "class_en" in enriched.columns:
        order = (
            enriched.groupby("class_en")["model2_risk_target"]
            .median()
            .sort_values()
            .index
        )
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.boxplot(
            data=enriched,
            x="model2_risk_target",
            y="class_en",
            order=order,
            ax=ax,
            color="#99f6e4",
        )
        ax.set_title("Model 2 Risk Target by Class")
        ax.set_xlabel("Risk target score")
        ax.set_ylabel("Class")
        fig.tight_layout()
        fig.savefig(report_dir / "risk_target_by_class.png", dpi=180)
        plt.close(fig)

    component_cols = [c for c in enriched.columns if c.startswith("component_")]
    if component_cols:
        comp = enriched[component_cols + ["model2_risk_target"]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(comp, annot=True, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
        ax.set_title("Component Correlation With Risk Target")
        fig.tight_layout()
        fig.savefig(report_dir / "component_correlation.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    output_root = data_path.parent
    model_dir = output_root / "models" / args.run_name
    report_dir = output_root / "reports" / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if args.original_only:
        if "source" not in df.columns:
            raise ValueError("--original-only was requested but column 'source' was not found.")
        df = df[df["source"].astype(str) == "original"].copy()
    if df.empty:
        raise ValueError("No rows available after filtering.")

    feature_cols = select_feature_columns(df)
    target_config = RiskTargetConfig()
    enriched, normalizer = build_proxy_risk_targets(df, feature_cols, config=target_config)
    enriched.to_csv(report_dir / "model2_training_targets.csv", index=False, encoding="utf-8")

    train_idx, test_idx = split_data(
        enriched, args.test_size, args.split, args.random_state
    )
    X = enriched[feature_cols]
    y_reg = enriched["model2_risk_target"].astype(float).to_numpy()
    y_cls = enriched["model2_high_risk_target"].astype(int).to_numpy()

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]
    y_cls_train, y_cls_test = y_cls[train_idx], y_cls[test_idx]

    reg_rows: list[dict[str, Any]] = []
    regressors = build_regressors(feature_cols, args.random_state)
    fitted_regressors: dict[str, Pipeline] = {}
    for name, model in regressors.items():
        print(f"\n== Training Model 2 regressor: {name} ==")
        model.fit(X_train, y_reg_train)
        pred = model.predict(X_test)
        metrics = regression_metrics(y_reg_test, pred)
        metrics.update({"model": name, "train_rows": len(X_train), "test_rows": len(X_test)})
        reg_rows.append(metrics)
        fitted_regressors[name] = model
        joblib.dump(model, model_dir / f"{name}.joblib")
        save_importance(model, feature_cols, name, report_dir)
        print(f"{name}: mae={metrics['mae']:.4f}, rmse={metrics['rmse']:.4f}, r2={metrics['r2']:.4f}")

    reg_metrics = pd.DataFrame(reg_rows).sort_values(["mae", "rmse"], ascending=[True, True])
    reg_metrics.to_csv(report_dir / "regressor_comparison.csv", index=False, encoding="utf-8")
    best_reg_name = str(reg_metrics.iloc[0]["model"])
    best_regressor = fitted_regressors[best_reg_name]

    cls_rows: list[dict[str, Any]] = []
    classifiers = build_classifiers(feature_cols, args.random_state)
    fitted_classifiers: dict[str, Pipeline] = {}
    for name, model in classifiers.items():
        print(f"\n== Training Model 2 high-risk classifier: {name} ==")
        model.fit(X_train, y_cls_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = classifier_metrics(y_cls_test, pred, proba)
        metrics.update({"model": name, "train_rows": len(X_train), "test_rows": len(X_test)})
        cls_rows.append(metrics)
        fitted_classifiers[name] = model
        joblib.dump(model, model_dir / f"{name}.joblib")
        save_importance(model, feature_cols, name, report_dir)
        print(
            f"{name}: f1={metrics['f1']:.4f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}, "
            f"auc={metrics['roc_auc']:.4f}"
        )

    cls_metrics = pd.DataFrame(cls_rows).sort_values(
        ["f1", "balanced_accuracy", "roc_auc"], ascending=False
    )
    cls_metrics.to_csv(report_dir / "classifier_comparison.csv", index=False, encoding="utf-8")
    best_cls_name = str(cls_metrics.iloc[0]["model"])
    best_classifier = fitted_classifiers[best_cls_name]

    final_regressor = clone(best_regressor)
    final_regressor.fit(X, y_reg)
    final_classifier = clone(best_classifier)
    final_classifier.fit(X, y_cls)
    joblib.dump(final_regressor, model_dir / "final_risk_regressor.joblib")
    joblib.dump(final_classifier, model_dir / "final_high_risk_classifier.joblib")

    reg_pred = np.clip(best_regressor.predict(X_test), 0.0, 1.0)
    cls_proba = best_classifier.predict_proba(X_test)[:, 1]
    prediction_preview = enriched.iloc[test_idx][
        [
            c
            for c in [
                "window_start_ts",
                "window_end_ts",
                "session_id",
                "class_en",
                "category",
                "source",
                "model2_risk_target",
                "model2_risk_level",
            ]
            if c in enriched.columns
        ]
    ].copy()
    prediction_preview["predicted_risk_score"] = reg_pred
    prediction_preview["predicted_high_risk_probability"] = cls_proba
    prediction_preview["predicted_risk_level"] = [
        risk_level_from_score(float(v)) for v in reg_pred
    ]
    prediction_preview.to_csv(
        report_dir / "test_predictions_preview.csv", index=False, encoding="utf-8"
    )

    save_plots(enriched, report_dir)

    dataset_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "rows": int(len(enriched)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "split": args.split,
        "test_size": args.test_size,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "risk_level_counts": {
            str(k): int(v) for k, v in enriched["model2_risk_level"].value_counts().items()
        },
        "high_risk_counts": {
            str(k): int(v)
            for k, v in enriched["model2_high_risk_target"].value_counts().items()
        },
    }
    if "class_en" in enriched.columns:
        dataset_summary["risk_by_class_median"] = {
            str(k): float(v)
            for k, v in enriched.groupby("class_en")["model2_risk_target"].median().items()
        }

    config_payload = model2_config_payload(normalizer, target_config)
    write_json(report_dir / "dataset_summary.json", dataset_summary)
    write_json(report_dir / "risk_formula_config.json", config_payload)
    write_json(model_dir / "feature_columns.json", feature_cols)

    bundle = {
        "kind": "model2_risk_assessment",
        "note": (
            "This is a prototype risk assessment model trained on proxy labels, "
            "not clinical future-fall ground truth."
        ),
        "risk_regressor": final_regressor,
        "high_risk_classifier": final_classifier,
        "feature_columns": feature_cols,
        "normalizer": normalizer,
        "risk_config": config_payload,
        "best_regressor_name": best_reg_name,
        "best_classifier_name": best_cls_name,
        "final_fit_rows": int(len(enriched)),
        "regressor_metrics": reg_metrics.to_dict(orient="records"),
        "classifier_metrics": cls_metrics.to_dict(orient="records"),
        "dataset_summary": dataset_summary,
    }
    joblib.dump(bundle, model_dir / "model2_risk_bundle.joblib")

    summary_lines = [
        "# Model 2 Risk Assessment Summary",
        "",
        "Model 2 is a mobility risk assessment layer. It is not the Model 1 fall/no-fall detector.",
        "",
        "## Method",
        "",
        "- Input: numeric window features from `windows_all.csv`.",
        "- Target: domain-informed proxy risk score from motion, rotation, posture, impact, and PPG components.",
        "- Output: risk score 0.0-1.0, high-risk probability, and low/medium/high risk level.",
        "- Important limitation: this is not clinical future-fall probability because the dataset has no true longitudinal fall-risk labels.",
        "",
        "## Dataset",
        "",
        f"- Rows: {len(enriched)}",
        f"- Feature count: {len(feature_cols)}",
        f"- Split: {args.split}",
        f"- Train rows: {len(train_idx)}",
        f"- Test rows: {len(test_idx)}",
        "",
        "## Risk Level Counts",
        "",
    ]
    for label, count in enriched["model2_risk_level"].value_counts().items():
        summary_lines.append(f"- `{label}`: {count}")
    summary_lines.extend(
        [
            "",
            "## Best Models",
            "",
            f"- Best risk-score regressor: `{best_reg_name}`",
            f"- Best high-risk classifier: `{best_cls_name}`",
            "",
            "## Regressor Comparison",
            "",
            reg_metrics.to_markdown(index=False),
            "",
            "## Classifier Comparison",
            "",
            cls_metrics.to_markdown(index=False),
            "",
            "## Main Artifacts",
            "",
            f"- Model bundle: `{model_dir / 'model2_risk_bundle.joblib'}`",
            f"- Training targets: `{report_dir / 'model2_training_targets.csv'}`",
            f"- Formula config: `{report_dir / 'risk_formula_config.json'}`",
            f"- Test predictions: `{report_dir / 'test_predictions_preview.csv'}`",
        ]
    )
    (report_dir / "summary_report.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nDone.")
    print(f"Best risk regressor: {best_reg_name}")
    print(f"Best high-risk classifier: {best_cls_name}")
    print(f"Model bundle: {model_dir / 'model2_risk_bundle.joblib'}")
    print(f"Report: {report_dir / 'summary_report.md'}")


if __name__ == "__main__":
    main()
