from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_FILES = [
    "windows_all.csv",
    "windows_extracted.csv",
    "joined_imu_windows.csv",
    "merged_dataset_full.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dataset class imbalance and missing values for Fall Detection / Model 2."
    )
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--output-dir", default="reports/dataset_balance")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def count_column(df: pd.DataFrame, file_name: str, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(
            columns=["file", "column", "value", "count", "percent"]
        )
    counts = df[column].value_counts(dropna=False)
    out = counts.rename_axis("value").reset_index(name="count")
    out["file"] = file_name
    out["column"] = column
    out["percent"] = out["count"] / len(df) * 100.0
    out["value"] = out["value"].astype(str).replace({"nan": "NaN"})
    return out[["file", "column", "value", "count", "percent"]]


def imbalance_summary(df: pd.DataFrame, file_name: str, column: str) -> dict[str, Any]:
    if column not in df.columns:
        return {
            "file": file_name,
            "column": column,
            "available": False,
        }
    counts = df[column].value_counts(dropna=False)
    non_null_counts = df[column].dropna().value_counts()
    max_count = int(non_null_counts.max()) if not non_null_counts.empty else 0
    min_count = int(non_null_counts.min()) if not non_null_counts.empty else 0
    ratio = float(max_count / min_count) if min_count else None
    top_class = str(non_null_counts.idxmax()) if not non_null_counts.empty else None
    top_percent = float(max_count / len(df) * 100.0) if len(df) else 0.0
    return {
        "file": file_name,
        "column": column,
        "available": True,
        "rows": int(len(df)),
        "unique_non_null": int(df[column].nunique(dropna=True)),
        "null_count": int(df[column].isna().sum()),
        "max_class_count": max_count,
        "min_class_count": min_count,
        "max_min_ratio": ratio,
        "top_class": top_class,
        "top_class_percent": top_percent,
        "is_single_class": bool(df[column].nunique(dropna=True) == 1),
    }


def missing_summary(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    nulls = df.isna().sum()
    out = pd.DataFrame(
        {
            "file": file_name,
            "column": nulls.index,
            "missing_count": nulls.values,
            "missing_percent": nulls.values / len(df) * 100.0,
        }
    )
    return out[out["missing_count"] > 0].sort_values(
        ["missing_count", "column"], ascending=[False, True]
    )


def plot_windows_all_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    class_counts = (
        df["class_en"].value_counts().rename_axis("class_en").reset_index(name="count")
    )
    class_counts["percent"] = class_counts["count"] / len(df) * 100.0

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=class_counts, x="count", y="class_en", hue="class_en", palette="viridis", legend=False, ax=ax)
    ax.set_title("windows_all.csv: Class Distribution (class_en)")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Class")
    for i, row in class_counts.iterrows():
        ax.text(
            row["count"] + max(class_counts["count"]) * 0.01,
            i,
            f"{int(row['count'])} ({row['percent']:.1f}%)",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "windows_all_class_distribution.png", dpi=180)
    plt.close(fig)

    category_counts = (
        df["category"].value_counts().rename_axis("category").reset_index(name="count")
    )
    category_counts["percent"] = category_counts["count"] / len(df) * 100.0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=category_counts, x="category", y="count", hue="category", palette="mako", legend=False, ax=ax)
    ax.set_title("windows_all.csv: Category Distribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Rows")
    for i, row in category_counts.iterrows():
        ax.text(i, row["count"], f"{int(row['count'])}\n{row['percent']:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "windows_all_category_distribution.png", dpi=180)
    plt.close(fig)


def plot_file_class_comparison(class_counts: pd.DataFrame, output_dir: Path) -> None:
    data = class_counts[class_counts["column"] == "class_en"].copy()
    if data.empty:
        return
    data = data[data["value"] != "NaN"]
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=data, x="count", y="value", hue="file", ax=ax)
    ax.set_title("Class Distribution Comparison Across CSV Files")
    ax.set_xlabel("Rows")
    ax.set_ylabel("class_en")
    ax.legend(title="CSV file", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution_across_files.png", dpi=180)
    plt.close(fig)


def plot_missing_top(missing: pd.DataFrame, output_dir: Path) -> None:
    if missing.empty:
        return
    top = missing.sort_values("missing_percent", ascending=False).head(30).copy()
    top["file_column"] = top["file"] + " :: " + top["column"]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=top, x="missing_percent", y="file_column", hue="file", dodge=False, ax=ax)
    ax.set_title("Top Missing Columns Across Dataset Files")
    ax.set_xlabel("Missing percent")
    ax.set_ylabel("File / Column")
    ax.legend(title="CSV file", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "top_missing_columns.png", dpi=180)
    plt.close(fig)


def plot_windows_all_class_column_missing(df: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for label_col in ["class_en", "class", "class_th", "category", "label"]:
        if label_col in df.columns:
            rows.append(
                {
                    "column": label_col,
                    "missing_count": int(df[label_col].isna().sum()),
                    "present_count": int(df[label_col].notna().sum()),
                }
            )
    data = pd.DataFrame(rows)
    if data.empty:
        return
    long = data.melt(id_vars="column", value_vars=["present_count", "missing_count"], var_name="status", value_name="count")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=long, x="column", y="count", hue="status", palette=["#0f766e", "#f59e0b"], ax=ax)
    ax.set_title("windows_all.csv: Label Column Missingness")
    ax.set_xlabel("Column")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(output_dir / "windows_all_label_column_missingness.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = resolve_path(args.base_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframes: dict[str, pd.DataFrame] = {}
    for file_name in DATASET_FILES:
        path = base_dir / file_name
        if not path.exists():
            continue
        dataframes[file_name] = safe_read_csv(path)

    count_frames: list[pd.DataFrame] = []
    missing_frames: list[pd.DataFrame] = []
    imbalance_rows: list[dict[str, Any]] = []

    for file_name, df in dataframes.items():
        for column in ["class_en", "class", "category", "label", "source", "win_class", "win_label"]:
            count_frames.append(count_column(df, file_name, column))
            imbalance_rows.append(imbalance_summary(df, file_name, column))
        missing_frames.append(missing_summary(df, file_name))

    all_counts = pd.concat(count_frames, ignore_index=True) if count_frames else pd.DataFrame()
    all_missing = pd.concat(missing_frames, ignore_index=True) if missing_frames else pd.DataFrame()
    imbalance = pd.DataFrame(imbalance_rows)

    all_counts.to_csv(output_dir / "dataset_value_counts.csv", index=False, encoding="utf-8-sig")
    all_missing.to_csv(output_dir / "dataset_missing_values.csv", index=False, encoding="utf-8-sig")
    imbalance.to_csv(output_dir / "dataset_imbalance_summary.csv", index=False, encoding="utf-8-sig")

    if "windows_all.csv" in dataframes:
        plot_windows_all_distribution(dataframes["windows_all.csv"], output_dir)
        plot_windows_all_class_column_missing(dataframes["windows_all.csv"], output_dir)
    if not all_counts.empty:
        plot_file_class_comparison(all_counts, output_dir)
    if not all_missing.empty:
        plot_missing_top(all_missing, output_dir)

    summary: dict[str, Any] = {
        "files": {},
        "main_findings": [
            "windows_all.csv has all 11 class_en classes but is strongly imbalanced.",
            "joined_imu_windows.csv has only slow_collapse_fall, so it should not be used as a general multiclass training file.",
            "windows_extracted.csv contains only original windows and is missing slow_collapse_fall, sideways_fall, and backward_fall.",
            "The class column in windows_all.csv has many missing values, but class_en is complete and should be used as the canonical class column.",
        ],
    }
    for file_name, df in dataframes.items():
        file_info: dict[str, Any] = {"rows": int(len(df)), "columns": int(len(df.columns))}
        for column in ["class_en", "class", "category", "label"]:
            if column in df.columns:
                file_info[f"{column}_unique_non_null"] = int(df[column].nunique(dropna=True))
                file_info[f"{column}_missing"] = int(df[column].isna().sum())
                file_info[f"{column}_counts"] = {
                    str(k): int(v) for k, v in df[column].value_counts(dropna=False).items()
                }
        summary["files"][file_name] = file_info
    write_json(output_dir / "dataset_balance_summary.json", summary)

    windows_all = dataframes.get("windows_all.csv")
    lines = [
        "# Dataset Balance And Missingness Report",
        "",
        "This report explains whether the received dataset is imbalanced and where class/missing-value issues come from.",
        "",
        "## Short Answer",
        "",
        "- Yes, `windows_all.csv` is imbalanced.",
        "- No, `windows_all.csv` does not have only one class. It has 11 classes in `class_en`.",
        "- The file that really has only one class is `joined_imu_windows.csv` (`slow_collapse_fall` only).",
        "- The confusing missing values are mostly from the `class` column, but `class_en` is complete and should be used as the canonical class label.",
        "",
    ]

    if windows_all is not None:
        class_counts = windows_all["class_en"].value_counts()
        category_counts = windows_all["category"].value_counts()
        max_count = int(class_counts.max())
        min_count = int(class_counts.min())
        imbalance_ratio = max_count / min_count
        lines.extend(
            [
                "## windows_all.csv",
                "",
                f"- Rows: {len(windows_all)}",
                f"- `class_en` unique classes: {windows_all['class_en'].nunique(dropna=True)}",
                f"- Class imbalance ratio max/min: {imbalance_ratio:.1f}:1",
                f"- Largest class: `{class_counts.idxmax()}` = {max_count}",
                f"- Smallest class: `{class_counts.idxmin()}` = {min_count}",
                "",
                "Category counts:",
                "",
                category_counts.rename_axis("category").reset_index(name="count").to_markdown(index=False),
                "",
                "Class counts:",
                "",
                class_counts.rename_axis("class_en").reset_index(name="count").to_markdown(index=False),
                "",
                "Label/missing columns:",
                "",
                pd.DataFrame(
                    [
                        {
                            "column": col,
                            "missing": int(windows_all[col].isna().sum()),
                            "present": int(windows_all[col].notna().sum()),
                        }
                        for col in ["class_en", "class", "class_th", "category", "label"]
                        if col in windows_all.columns
                    ]
                ).to_markdown(index=False),
                "",
            ]
        )

    lines.extend(
        [
            "## File-Level Summary",
            "",
            imbalance[
                imbalance["column"].isin(["class_en", "class", "category", "label"])
            ][
                [
                    "file",
                    "column",
                    "available",
                    "rows",
                    "unique_non_null",
                    "null_count",
                    "max_min_ratio",
                    "top_class",
                    "top_class_percent",
                    "is_single_class",
                ]
            ].to_markdown(index=False),
            "",
            "## Recommended Use",
            "",
            "- Use `windows_all.csv` for Model 2 feature-window training/testing because it has the broadest class coverage.",
            "- Use `class_en`, `category`, and `label`; avoid using `class` as the main class label because it is missing for many rows.",
            "- Treat `joined_imu_windows.csv` as a sequence/hybrid demo file only, because it contains only `slow_collapse_fall`.",
            "- Treat `merged_dataset_full.csv` carefully because many columns are sparse due to merging raw rows and window rows with different schemas.",
            "",
            "## Generated Graphs",
            "",
            "- `windows_all_class_distribution.png`",
            "- `windows_all_category_distribution.png`",
            "- `class_distribution_across_files.png`",
            "- `windows_all_label_column_missingness.png`",
            "- `top_missing_columns.png`",
        ]
    )

    (output_dir / "dataset_balance_report.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )

    print(f"Report: {output_dir / 'dataset_balance_report.md'}")
    print(f"Counts: {output_dir / 'dataset_value_counts.csv'}")
    print(f"Missing: {output_dir / 'dataset_missing_values.csv'}")
    if windows_all is not None:
        print("windows_all class_en classes:", windows_all["class_en"].nunique(dropna=True))
        print("windows_all class max/min ratio:", f"{imbalance_ratio:.1f}:1")


if __name__ == "__main__":
    main()
