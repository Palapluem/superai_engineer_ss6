from pathlib import Path
import textwrap

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_fahmai_detailed_eda.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = [
    md(
        """
        # FahMai The Finale: Detailed Data EDA and OCR Readiness

        This notebook profiles the public FahMai enterprise bundle before building the OCR pipeline.
        It is intentionally broad enough for the agentic track, then narrows into the bank-statement
        surface used by the OCR sub-track.

        ## What the challenge brief establishes

        - The data lake covers fiscal years 2024-2025 and was released as of 2026-01-15.
        - The five source surfaces are tables, narrative documents, rendered artifacts, logs, and reports.
        - Data-quality artifacts and a mid-2025 sales schema cutover are part of the benchmark.
        - OCR evaluation focuses on bank-statement renders and structured extraction of `amount`, `date`,
          `account`, and `description`.
        - The intended OCR runtime is a single RTX 5090 with 32 GB VRAM.

        ## Notebook outputs

        The final export section writes reusable CSV summaries into `eda_outputs/`. No OCR model is run here.
        The recommended next implementation step is a GLM-OCR pipeline with page routing, structured parsing,
        and table-backed validation.
        """
    ),
    md(
        """
        ## 1. Setup

        The root finder makes the notebook portable: it works when launched from either the bundle root
        or the `notebooks/` directory.
        """
    ),
    code(
        """
        from pathlib import Path
        from collections import Counter
        import json
        import re
        import warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from PIL import Image
        from IPython.display import display

        warnings.filterwarnings("ignore")
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.max_colwidth", 140)
        sns.set_theme(style="whitegrid")

        candidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / "fah-mai-the-finale-enterprise-data-agentic-showdown"]
        DATA_ROOT = next(
            path.resolve()
            for path in candidates
            if (path / "tables").is_dir() and (path / "renders").is_dir() and (path / "docs").is_dir()
        )
        OUTPUT_DIR = DATA_ROOT / "eda_outputs"
        GENERATED_DIR_NAMES = {"notebooks", "eda_outputs", ".ipynb_checkpoints"}

        SCAN_ALL_RENDER_METADATA = True
        DOC_PROFILE_SAMPLE_PER_FAMILY = 1000

        print(f"DATA_ROOT = {DATA_ROOT}")
        print(f"OUTPUT_DIR = {OUTPUT_DIR}")
        """
    ),
    md(
        """
        ## 2. Bundle Inventory

        Start with a filesystem-level inventory. This is useful for capacity planning because image count
        and disk footprint are very different measures: a small number of rendered artifacts dominates bytes,
        while narrative documents dominate file count.
        """
    ),
    code(
        """
        def is_generated(relative_path: Path) -> bool:
            return any(part in GENERATED_DIR_NAMES for part in relative_path.parts)


        inventory_rows = []
        for path in DATA_ROOT.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(DATA_ROOT)
            if is_generated(relative_path):
                continue
            inventory_rows.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "surface": relative_path.parts[0] if len(relative_path.parts) > 1 else "<root>",
                    "extension": path.suffix.lower() or "<none>",
                    "size_bytes": path.stat().st_size,
                    "size_mb": path.stat().st_size / (1024**2),
                }
            )

        file_inventory = pd.DataFrame(inventory_rows)
        surface_summary = (
            file_inventory.groupby("surface", as_index=False)
            .agg(files=("relative_path", "size"), size_mb=("size_mb", "sum"))
            .sort_values("files", ascending=False)
        )
        extension_summary = (
            file_inventory.groupby("extension", as_index=False)
            .agg(files=("relative_path", "size"), size_mb=("size_mb", "sum"))
            .sort_values("files", ascending=False)
        )

        print(f"Source files: {len(file_inventory):,}")
        print(f"Source size:  {file_inventory['size_mb'].sum() / 1024:,.2f} GiB")
        display(surface_summary.round({"size_mb": 2}))
        display(extension_summary.round({"size_mb": 2}))
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.barplot(data=surface_summary, x="surface", y="files", ax=axes[0], color="#4472C4")
        axes[0].set_title("Files by source surface")
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].set_ylabel("File count")

        size_plot = surface_summary.sort_values("size_mb", ascending=False)
        sns.barplot(data=size_plot, x="surface", y="size_mb", ax=axes[1], color="#ED7D31")
        axes[1].set_title("Disk footprint by source surface")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].set_ylabel("MiB")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 3. Structured Tables

        All 31 CSV tables are loaded once. The profile captures row counts, schema width, memory footprint,
        full-row duplicates, probable primary-key duplicates, missingness, and common temporal coverage.
        """
    ),
    code(
        """
        table_files = sorted((DATA_ROOT / "tables").glob("*.csv"))
        CSV_DTYPE_OVERRIDES = {
            # This nullable long ID is otherwise inferred as float64 and loses precision.
            "FACT_RETURN": {"line_item_id": "string"},
            "FACT_SALES_LINE_ITEM": {"line_item_id": "string"},
            "T2_DOC_INVENTORY": {"source_pk": "string"},
        }
        tables = {
            path.stem: pd.read_csv(path, low_memory=False, dtype=CSV_DTYPE_OVERRIDES.get(path.stem))
            for path in table_files
        }

        table_profile_rows = []
        for path in table_files:
            name = path.stem
            frame = tables[name]
            probable_key = frame.columns[0]
            key_duplicates = int(frame[probable_key].dropna().duplicated().sum())
            table_profile_rows.append(
                {
                    "table": name,
                    "table_family": "FACT" if name.upper().startswith("FACT_") else "DIM" if name.upper().startswith("DIM_") else "OTHER",
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "disk_mb": path.stat().st_size / (1024**2),
                    "memory_mb": frame.memory_usage(deep=True).sum() / (1024**2),
                    "missing_cells_pct": frame.isna().mean().mean() * 100,
                    "full_row_duplicates": int(frame.duplicated().sum()),
                    "probable_key": probable_key,
                    "probable_key_duplicates": key_duplicates,
                }
            )

        table_profile = pd.DataFrame(table_profile_rows).sort_values("rows", ascending=False)
        print(f"Tables loaded: {len(table_profile):,}")
        print(f"Rows across all tables: {table_profile['rows'].sum():,}")
        display(table_profile.round({"disk_mb": 3, "memory_mb": 3, "missing_cells_pct": 2}))
        """
    ),
    code(
        """
        plt.figure(figsize=(12, 8))
        plot_frame = table_profile.sort_values("rows", ascending=True)
        sns.barplot(data=plot_frame, x="rows", y="table", hue="table_family", dodge=False)
        plt.xscale("log")
        plt.title("Table row counts (log scale)")
        plt.xlabel("Rows")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        schema_catalog = pd.DataFrame(
            [
                {"table": table_name, "column_position": position, "column": column, "dtype": str(frame[column].dtype)}
                for table_name, frame in tables.items()
                for position, column in enumerate(frame.columns, start=1)
            ]
        )
        display(schema_catalog)
        """
    ),
    md(
        """
        ### Missingness

        Nulls are not automatically errors. Several columns are conditional by business process, and the
        common `effective_date` field is present but empty in many fact tables. The following view isolates
        non-zero missingness so OCR validation can distinguish expected nulls from suspicious extraction gaps.
        """
    ),
    code(
        """
        missingness_rows = []
        for table_name, frame in tables.items():
            for column in frame.columns:
                missing_pct = frame[column].isna().mean() * 100
                if missing_pct > 0:
                    missingness_rows.append(
                        {
                            "table": table_name,
                            "column": column,
                            "missing_rows": int(frame[column].isna().sum()),
                            "missing_pct": missing_pct,
                        }
                    )

        missingness_profile = (
            pd.DataFrame(missingness_rows)
            .sort_values(["missing_pct", "table", "column"], ascending=[False, True, True])
            .reset_index(drop=True)
        )
        display(missingness_profile.round({"missing_pct": 2}))
        """
    ),
    md(
        """
        ### Temporal Coverage

        The challenge is date-aware and includes bitemporal behavior. This scan summarizes parseable date and
        timestamp columns instead of assuming that the latest row is the correct row.
        """
    ),
    code(
        """
        temporal_rows = []
        temporal_pattern = re.compile(r"(date|timestamp|period_start|period_end)$", flags=re.IGNORECASE)
        for table_name, frame in tables.items():
            for column in frame.columns:
                if not temporal_pattern.search(column):
                    continue
                parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
                parseable = int(parsed.notna().sum())
                temporal_rows.append(
                    {
                        "table": table_name,
                        "column": column,
                        "rows": len(frame),
                        "non_null_rows": int(frame[column].notna().sum()),
                        "parseable_rows": parseable,
                        "min_value": parsed.min().date().isoformat() if parseable else None,
                        "max_value": parsed.max().date().isoformat() if parseable else None,
                    }
                )

        temporal_profile = pd.DataFrame(temporal_rows).sort_values(["table", "column"])
        display(temporal_profile)
        """
    ),
    md(
        """
        ### FACT_SALES Schema Cutover

        The bundle notes explicitly call out a schema-version transition. The exact boundary matters for
        downstream joins and for questions that reconcile sales with logs or rendered evidence.
        """
    ),
    code(
        """
        sales = tables["FACT_SALES"].copy()
        sales["business_event_date"] = pd.to_datetime(sales["business_event_date"])
        sales["business_month"] = sales["business_event_date"].dt.to_period("M").astype(str)

        sales_schema_summary = (
            sales.groupby("schema_version", as_index=False)
            .agg(rows=("txn_id", "size"), min_date=("business_event_date", "min"), max_date=("business_event_date", "max"))
        )
        display(sales_schema_summary)

        sales_schema_monthly = pd.crosstab(sales["business_month"], sales["schema_version"])
        sales_schema_monthly.plot(kind="bar", stacked=True, figsize=(14, 4), color=["#4472C4", "#ED7D31"])
        plt.title("FACT_SALES rows by month and schema version")
        plt.xlabel("Business month")
        plt.ylabel("Rows")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Known Data-Quality Signals

        These are diagnostics, not cleanup instructions. The benchmark intentionally includes operational
        artifacts such as duplicate invoices and phantom redemptions. Preserve the raw rows and make any
        deduplication rule explicit in the question-specific logic.
        """
    ),
    code(
        """
        vendor_payments = tables["FACT_VENDOR_PAYMENT"]
        promo_redemptions = tables["FACT_PROMO_REDEMPTION"]

        duplicated_invoice_rows = vendor_payments[
            vendor_payments.duplicated("vendor_invoice_id", keep=False)
        ].sort_values(["vendor_invoice_id", "posting_date"])
        duplicated_redemption_rows = promo_redemptions[
            promo_redemptions.duplicated("txn_id", keep=False)
        ].sort_values(["txn_id", "channel"])

        dq_summary = pd.DataFrame(
            [
                {
                    "signal": "FACT_VENDOR_PAYMENT duplicate vendor_invoice_id rows",
                    "rows": len(duplicated_invoice_rows),
                    "distinct_business_keys": duplicated_invoice_rows["vendor_invoice_id"].nunique(),
                },
                {
                    "signal": "FACT_PROMO_REDEMPTION duplicate txn_id rows",
                    "rows": len(duplicated_redemption_rows),
                    "distinct_business_keys": duplicated_redemption_rows["txn_id"].nunique(),
                },
                {
                    "signal": "FACT_SALES non-null retry_idempotency_marker rows",
                    "rows": int(tables["FACT_SALES"]["retry_idempotency_marker"].notna().sum()),
                    "distinct_business_keys": int(tables["FACT_SALES"]["retry_idempotency_marker"].nunique(dropna=True)),
                },
            ]
        )
        display(dq_summary)
        display(duplicated_invoice_rows)
        display(duplicated_redemption_rows)
        """
    ),
    md(
        """
        ## 4. Narrative Documents

        Narrative files are indexed by family and sampled deterministically for a lightweight language profile.
        Increase `DOC_PROFILE_SAMPLE_PER_FAMILY` in the setup cell if a full text scan is needed.
        """
    ),
    code(
        """
        def extract_iso_date(text: str):
            match = re.search(r"(20\\d{2})[-_]?([01]\\d)[-_]?([0-3]\\d)", text)
            if not match:
                return None
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


        doc_rows = []
        for path in sorted((DATA_ROOT / "docs").rglob("*.md")):
            relative_to_docs = path.relative_to(DATA_ROOT / "docs")
            doc_rows.append(
                {
                    "relative_path": path.relative_to(DATA_ROOT).as_posix(),
                    "family": "/".join(relative_to_docs.parts[:-1]),
                    "size_kb": path.stat().st_size / 1024,
                    "filename_date": extract_iso_date(path.name),
                }
            )

        doc_inventory = pd.DataFrame(doc_rows)
        doc_family_summary = (
            doc_inventory.groupby("family", as_index=False)
            .agg(files=("relative_path", "size"), size_mb=("size_kb", lambda values: values.sum() / 1024))
            .sort_values("files", ascending=False)
        )
        display(doc_family_summary.round({"size_mb": 2}))
        """
    ),
    code(
        """
        def deterministic_sample(values, limit):
            values = sorted(values)
            if limit is None or len(values) <= limit:
                return values
            positions = np.linspace(0, len(values) - 1, num=limit, dtype=int)
            return [values[position] for position in positions]


        def classify_language(text: str):
            thai_count = len(re.findall(r"[\\u0E00-\\u0E7F]", text))
            latin_count = len(re.findall(r"[A-Za-z]", text))
            total = thai_count + latin_count
            if total < 10:
                label = "sparse"
            elif thai_count / total >= 0.80:
                label = "thai-dominant"
            elif latin_count / total >= 0.80:
                label = "english-dominant"
            else:
                label = "mixed"
            return label, thai_count, latin_count


        sampled_doc_rows = []
        for family, group in doc_inventory.groupby("family"):
            selected_paths = deterministic_sample(group["relative_path"].tolist(), DOC_PROFILE_SAMPLE_PER_FAMILY)
            for relative_path in selected_paths:
                text = (DATA_ROOT / relative_path).read_text(encoding="utf-8", errors="replace")
                language, thai_chars, latin_chars = classify_language(text)
                sampled_doc_rows.append(
                    {
                        "family": family,
                        "relative_path": relative_path,
                        "characters": len(text),
                        "language": language,
                        "thai_chars": thai_chars,
                        "latin_chars": latin_chars,
                    }
                )

        sampled_doc_profile = pd.DataFrame(sampled_doc_rows)
        doc_language_summary = (
            sampled_doc_profile.groupby(["family", "language"], as_index=False)
            .agg(sampled_files=("relative_path", "size"), characters=("characters", "sum"))
            .sort_values(["family", "sampled_files"], ascending=[True, False])
        )
        display(doc_language_summary)
        """
    ),
    md(
        """
        ## 5. Operational Logs

        Logs contain daily POS TSV files, daily web-order JSONL files, and selected PayWise fee CSVs.
        The inventory view below is enough to design ingestion without loading every raw log into memory.
        """
    ),
    code(
        """
        def classify_log(filename: str):
            if filename.startswith("pos_"):
                return "pos_tsv"
            if filename.startswith("web_"):
                return "web_jsonl"
            if filename.startswith("paywise_fee_log_"):
                return "paywise_fee_csv"
            return "other"


        log_rows = []
        for path in sorted((DATA_ROOT / "logs").iterdir()):
            if not path.is_file():
                continue
            kind = classify_log(path.name)
            date_match = re.search(r"(20\\d{2})([01]\\d)([0-3]\\d)", path.name)
            month_match = re.search(r"(20\\d{2})-([01]\\d)", path.name)
            branch_match = re.match(r"pos_(.+)_20\\d{6}\\.tsv$", path.name)
            file_date = (
                f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                if date_match
                else f"{month_match.group(1)}-{month_match.group(2)}"
                if month_match
                else None
            )
            log_rows.append(
                {
                    "filename": path.name,
                    "kind": kind,
                    "branch_code": branch_match.group(1) if branch_match else None,
                    "file_date": file_date,
                    "size_kb": path.stat().st_size / 1024,
                }
            )

        log_inventory = pd.DataFrame(log_rows)
        log_summary = (
            log_inventory.groupby("kind", as_index=False)
            .agg(files=("filename", "size"), size_mb=("size_kb", lambda values: values.sum() / 1024), min_file_date=("file_date", "min"), max_file_date=("file_date", "max"))
            .sort_values("files", ascending=False)
        )
        display(log_summary.round({"size_mb": 2}))

        pos_branch_coverage = (
            log_inventory[log_inventory["kind"] == "pos_tsv"]
            .groupby("branch_code", as_index=False)
            .agg(files=("filename", "size"), min_file_date=("file_date", "min"), max_file_date=("file_date", "max"))
            .sort_values("branch_code")
        )
        display(pos_branch_coverage)
        """
    ),
    code(
        """
        log_schema_samples = []
        for kind, group in log_inventory.groupby("kind"):
            sample_path = DATA_ROOT / "logs" / group.iloc[0]["filename"]
            if kind == "pos_tsv":
                columns = pd.read_csv(sample_path, sep="\\t", nrows=1).columns.tolist()
            elif kind == "paywise_fee_csv":
                columns = pd.read_csv(sample_path, nrows=1).columns.tolist()
            elif kind == "web_jsonl":
                with sample_path.open(encoding="utf-8") as handle:
                    columns = sorted(json.loads(handle.readline()).keys())
            else:
                columns = []
            log_schema_samples.append({"kind": kind, "sample_file": sample_path.name, "columns": columns})

        display(pd.DataFrame(log_schema_samples))
        """
    ),
    md(
        """
        ## 6. Rendered Artifacts

        Rendered artifacts are the multimodal surface. Their distribution matters for storage, routing,
        and batch-time estimates. Bank statements are analyzed in depth in the next section.
        """
    ),
    code(
        """
        render_rows = []
        for path in sorted((DATA_ROOT / "renders").rglob("*")):
            if not path.is_file():
                continue
            relative_to_renders = path.relative_to(DATA_ROOT / "renders")
            render_rows.append(
                {
                    "relative_path": path.relative_to(DATA_ROOT).as_posix(),
                    "render_type": relative_to_renders.parts[0],
                    "render_month": relative_to_renders.parts[1] if len(relative_to_renders.parts) > 2 else None,
                    "extension": path.suffix.lower(),
                    "stem": path.stem,
                    "size_mb": path.stat().st_size / (1024**2),
                }
            )

        render_inventory = pd.DataFrame(render_rows)
        render_summary = (
            render_inventory.groupby(["render_type", "extension"], as_index=False)
            .agg(files=("relative_path", "size"), size_mb=("size_mb", "sum"), min_month=("render_month", "min"), max_month=("render_month", "max"))
            .sort_values("files", ascending=False)
        )
        display(render_summary.round({"size_mb": 2}))
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        render_type_summary = (
            render_inventory.groupby("render_type", as_index=False)
            .agg(files=("relative_path", "size"), size_mb=("size_mb", "sum"))
            .sort_values("files", ascending=False)
        )
        sns.barplot(data=render_type_summary, x="render_type", y="files", ax=axes[0], color="#70AD47")
        axes[0].set_title("Rendered artifact counts")
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].set_xlabel("")

        size_plot = render_type_summary.sort_values("size_mb", ascending=False)
        sns.barplot(data=size_plot, x="render_type", y="size_mb", ax=axes[1], color="#A5A5A5")
        axes[1].set_title("Rendered artifact footprint")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("MiB")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### PNG Metadata Scan

        Pillow reads image headers to capture dimensions and mode. This avoids decoding full images while still
        surfacing the major layout families. Set `SCAN_ALL_RENDER_METADATA = False` during quick iterations to
        scan bank statements only.
        """
    ),
    code(
        """
        png_inventory = render_inventory[render_inventory["extension"] == ".png"].copy()
        if not SCAN_ALL_RENDER_METADATA:
            png_inventory = png_inventory[png_inventory["render_type"] == "bank_statement"].copy()

        image_metadata_rows = []
        image_errors = []
        for relative_path in png_inventory["relative_path"]:
            path = DATA_ROOT / relative_path
            try:
                with Image.open(path) as image:
                    image_metadata_rows.append(
                        {
                            "relative_path": relative_path,
                            "width": image.width,
                            "height": image.height,
                            "mode": image.mode,
                            "aspect_ratio": image.width / image.height,
                            "megapixels": image.width * image.height / 1_000_000,
                        }
                    )
            except Exception as exc:
                image_errors.append({"relative_path": relative_path, "error": repr(exc)})

        image_metadata = pd.DataFrame(image_metadata_rows)
        image_error_profile = pd.DataFrame(image_errors)
        print(f"PNG files scanned: {len(image_metadata):,}")
        print(f"Metadata errors:   {len(image_error_profile):,}")

        image_layout_summary = (
            png_inventory.merge(image_metadata, on="relative_path", how="left")
            .groupby(["render_type", "width", "height", "mode"], dropna=False, as_index=False)
            .agg(files=("relative_path", "size"))
            .sort_values(["render_type", "files"], ascending=[True, False])
        )
        display(image_layout_summary)
        """
    ),
    md(
        """
        ## 7. OCR Focus: Bank Statements

        Bank statements should not be treated as unrelated single images. Filenames encode statement identity,
        account, Buddhist Era year, month, role (`header` versus transaction page), and transaction page number.
        Preserve this hierarchy in the OCR pipeline.
        """
    ),
    code(
        """
        bank_pages = render_inventory[render_inventory["render_type"] == "bank_statement"].copy()
        bank_pattern = re.compile(
            r"^(?P<statement_id>BS-(?P<account_id>.+)-(?P<be_year>\\d{4})-(?P<month>\\d{2}))_"
            r"(?P<raw_page_role>header|transactions_p(?P<page_number>\\d+))$"
        )
        parsed = bank_pages["stem"].str.extract(bank_pattern)
        bank_pages = pd.concat([bank_pages.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
        bank_pages["page_role"] = np.where(bank_pages["raw_page_role"] == "header", "header", "transactions")
        bank_pages["page_number"] = pd.to_numeric(bank_pages["page_number"], errors="coerce").astype("Int64")

        print(f"Bank-statement pages: {len(bank_pages):,}")
        print(f"Filename parse failures: {bank_pages['statement_id'].isna().sum():,}")
        display(bank_pages["page_role"].value_counts().rename_axis("page_role").reset_index(name="pages"))
        """
    ),
    code(
        """
        def sequence_is_complete(values):
            page_numbers = sorted({int(value) for value in values.dropna()})
            return page_numbers == list(range(1, max(page_numbers) + 1)) if page_numbers else True


        statement_summary = (
            bank_pages.groupby(["statement_id", "account_id", "render_month", "be_year", "month"], as_index=False)
            .agg(
                pages=("relative_path", "size"),
                header_pages=("page_role", lambda values: int((values == "header").sum())),
                transaction_pages=("page_role", lambda values: int((values == "transactions").sum())),
                page_sequence_complete=("page_number", sequence_is_complete),
            )
        )
        statement_summary["page_sequence_complete"] = statement_summary["page_sequence_complete"].astype(bool)
        account_statement_summary = (
            statement_summary.groupby("account_id", as_index=False)
            .agg(
                statements=("statement_id", "size"),
                total_pages=("pages", "sum"),
                header_pages=("header_pages", "sum"),
                transaction_pages=("transaction_pages", "sum"),
                min_month=("render_month", "min"),
                max_month=("render_month", "max"),
            )
            .sort_values("transaction_pages", ascending=False)
        )

        print(f"Logical statements: {len(statement_summary):,}")
        print(f"Statements missing exactly one header: {(statement_summary['header_pages'] != 1).sum():,}")
        print(f"Statements with page-sequence gaps: {(~statement_summary['page_sequence_complete']).sum():,}")
        display(account_statement_summary)
        display(statement_summary["pages"].value_counts().sort_index().rename_axis("pages_per_statement").reset_index(name="statements"))
        """
    ),
    code(
        """
        bank_page_month = pd.crosstab(statement_summary["account_id"], statement_summary["render_month"])
        plt.figure(figsize=(16, 5))
        sns.heatmap(bank_page_month, cmap="Blues", cbar_kws={"label": "Statements"})
        plt.title("Bank-statement coverage: every account by month")
        plt.xlabel("Statement month")
        plt.ylabel("Account")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Bank-Statement Layout Families

        The page layouts differ by bank/template. Route header and transaction pages separately, and benchmark
        GLM-OCR on both resolution families before selecting resize or tiling settings.
        """
    ),
    code(
        """
        bank_image_metadata = bank_pages.merge(image_metadata, on="relative_path", how="left")
        bank_layout_summary = (
            bank_image_metadata.groupby(["page_role", "width", "height", "mode"], dropna=False, as_index=False)
            .agg(pages=("relative_path", "size"), accounts=("account_id", "nunique"))
            .sort_values(["page_role", "pages"], ascending=[True, False])
        )
        display(bank_layout_summary)
        """
    ),
    code(
        """
        unique_layout_examples = (
            bank_image_metadata.dropna(subset=["width", "height"])
            .sort_values(["page_role", "width", "height", "relative_path"])
            .drop_duplicates(["page_role", "width", "height"])
        )

        fig, axes = plt.subplots(len(unique_layout_examples), 1, figsize=(14, 4 * len(unique_layout_examples)))
        axes = np.atleast_1d(axes)
        for axis, (_, row) in zip(axes, unique_layout_examples.iterrows()):
            with Image.open(DATA_ROOT / row["relative_path"]) as image:
                thumbnail = image.copy()
                thumbnail.thumbnail((1400, 900))
            axis.imshow(thumbnail)
            axis.set_title(f"{row['page_role']} | {int(row['width'])}x{int(row['height'])} | {row['relative_path']}")
            axis.axis("off")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Reconcile Rendered Pages Against FACT_BANK_TRANSACTION

        The fact table provides a strong validation surface. The pipeline can check account identity, date range,
        amounts, descriptions, and expected density per account-month. Running balances still need to be read
        from the rendered surface because they are part of the image-specific evidence.
        """
    ),
    code(
        """
        bank_transactions = tables["FACT_BANK_TRANSACTION"].copy()
        bank_transactions["business_event_date"] = pd.to_datetime(bank_transactions["business_event_date"])
        bank_transactions["render_month"] = bank_transactions["business_event_date"].dt.to_period("M").astype(str)

        fact_account_month = (
            bank_transactions.groupby(["account_id", "render_month"], as_index=False)
            .agg(fact_transactions=("bank_txn_id", "size"), amount_total_thb=("amount_thb", "sum"))
        )
        render_account_month = (
            statement_summary.groupby(["account_id", "render_month"], as_index=False)
            .agg(statements=("statement_id", "size"), transaction_pages=("transaction_pages", "sum"))
        )
        bank_reconciliation = render_account_month.merge(
            fact_account_month, on=["account_id", "render_month"], how="outer", validate="one_to_one"
        ).fillna({"statements": 0, "transaction_pages": 0, "fact_transactions": 0})
        bank_reconciliation["fact_rows_per_transaction_page"] = (
            bank_reconciliation["fact_transactions"] / bank_reconciliation["transaction_pages"].replace(0, np.nan)
        )

        print(f"Account-month rows: {len(bank_reconciliation):,}")
        print(f"Missing statement rows: {(bank_reconciliation['statements'] == 0).sum():,}")
        print(f"Missing transaction-page rows: {(bank_reconciliation['transaction_pages'] == 0).sum():,}")
        print(f"Page-count correlation with FACT rows: {bank_reconciliation['transaction_pages'].corr(bank_reconciliation['fact_transactions']):.4f}")
        display(bank_reconciliation.describe(include="all").T)

        account_reconciliation = (
            bank_reconciliation.groupby("account_id", as_index=False)
            .agg(
                months=("render_month", "size"),
                statements=("statements", "sum"),
                transaction_pages=("transaction_pages", "sum"),
                fact_transactions=("fact_transactions", "sum"),
                median_fact_rows_per_page=("fact_rows_per_transaction_page", "median"),
            )
            .sort_values("fact_transactions", ascending=False)
        )
        display(account_reconciliation.round({"median_fact_rows_per_page": 2}))
        """
    ),
    code(
        """
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=bank_reconciliation,
            x="fact_transactions",
            y="transaction_pages",
            hue="account_id",
            legend=False,
            alpha=0.8,
        )
        plt.title("Rendered transaction pages scale with fact-table rows")
        plt.xlabel("FACT_BANK_TRANSACTION rows per account-month")
        plt.ylabel("Rendered transaction pages")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 8. Recommended GLM-OCR Preparation

        ### Page routing

        1. Parse the filename before inference and retain `statement_id`, `account_id`, `render_month`,
           `page_role`, and `page_number`.
        2. Route `header` pages and `transactions` pages through separate prompts or schemas.
        3. Keep original-resolution images. Benchmark a conservative resize and tiled inference for dense
           transaction pages rather than applying one resize rule to both layout families.
        4. Reassemble transaction pages in numeric order. Lexical sorting alone places `p10` before `p2`.

        ### Proposed internal extraction contract

        The exact Kaggle submission schema should be adapted from the competition sample submission when it is
        available. Internally, keep a richer record so parsing errors remain debuggable:

        ```json
        {
          "image_id": "BS-KBANK-OPER-2567-01_transactions_p1.png",
          "statement_id": "BS-KBANK-OPER-2567-01",
          "page_role": "transactions",
          "page_number": 1,
          "account": "KBANK-OPER",
          "rows": [
            {
              "date": "2024-01-01",
              "amount": "-29900.00",
              "description": "...",
              "balance_after": "9970100.00"
            }
          ],
          "raw_text": "...",
          "parse_warnings": []
        }
        ```

        ### Validation layers

        - Validate filename account against header OCR and `DIM_BANK_ACCOUNT`.
        - Validate transaction dates against the statement month, with careful Buddhist Era conversion.
        - Normalize THB numbers without dropping signs, commas, or decimal precision too early.
        - Compare extracted `(account, date, amount, description)` rows with `FACT_BANK_TRANSACTION`.
        - Preserve rendered running balance as OCR-derived evidence; it is useful for row-order and arithmetic checks.
        - Track edit distance and field-level structure accuracy separately by page role, layout family, account,
          and month.

        ### Efficiency telemetry

        Capture image dimensions, page role, prompt version, GLM-OCR model revision, latency, VRAM peak,
        generated tokens, parser warnings, and validation status for every image. Benchmark on a stratified
        sample before processing all pages.
        """
    ),
    md(
        """
        ## 9. Export Reusable EDA Tables

        These CSVs are small, stable inputs for the next notebook or service implementation.
        """
    ),
    code(
        """
        OUTPUT_DIR.mkdir(exist_ok=True)

        exports = {
            "file_inventory.csv": file_inventory,
            "surface_summary.csv": surface_summary,
            "table_profile.csv": table_profile,
            "table_schema_catalog.csv": schema_catalog,
            "table_missingness_profile.csv": missingness_profile,
            "table_temporal_profile.csv": temporal_profile,
            "doc_family_summary.csv": doc_family_summary,
            "doc_language_sample_profile.csv": doc_language_summary,
            "log_summary.csv": log_summary,
            "render_summary.csv": render_summary,
            "image_layout_summary.csv": image_layout_summary,
            "bank_statement_pages.csv": bank_pages,
            "bank_statement_summary.csv": statement_summary,
            "bank_statement_account_summary.csv": account_statement_summary,
            "bank_statement_layout_summary.csv": bank_layout_summary,
            "bank_statement_fact_reconciliation.csv": bank_reconciliation,
            "dq_summary.csv": dq_summary,
        }
        if not image_error_profile.empty:
            exports["image_metadata_errors.csv"] = image_error_profile

        for filename, frame in exports.items():
            frame.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

        export_manifest = pd.DataFrame(
            [{"filename": filename, "rows": len(frame), "columns": len(frame.columns)} for filename, frame in exports.items()]
        ).sort_values("filename")
        display(export_manifest)
        print(f"Exported {len(exports)} CSV files to {OUTPUT_DIR}")
        """
    ),
    md(
        """
        ## 10. Executive Summary

        The EDA establishes the dataset shape needed for the OCR implementation:

        - The public bundle is a multimodal enterprise lake, not only an image folder.
        - OCR work should prioritize `renders/bank_statement/` and retain links to structured bank transactions.
        - Bank statements form complete account-month groups with separate header and transaction-page layouts.
        - Dense pages and two major resolution families require stratified benchmarking.
        - Raw text, parsed fields, warnings, and reconciliation results should all be stored for auditability.

        The next notebook should implement GLM-OCR inference on a stratified bank-statement sample, normalize
        structured rows, and report field-level validation metrics against `FACT_BANK_TRANSACTION`.
        """
    ),
]

cells.extend(
    [
        md(
            """
            # Extended EDA Appendix

            This appendix expands the first-pass EDA into a working data catalog. It inventories nested folders,
            profiles table contents at column level, validates relational mappings against actual values, draws
            ER diagrams, measures cross-surface links, and deepens the bank-statement analysis for OCR planning.
            """
        ),
        md(
            """
            ## 11. Nested Folder Tree

            The bundle is deeply nested in `docs/`, `renders/`, and `reports/`. This rollup records each folder's
            depth, parent folder, direct file count, recursive file count, and recursive disk footprint.
            """
        ),
        code(
            """
            folder_rows = []
            folder_paths = sorted(
                {
                    "/".join(Path(relative_path).parts[:depth])
                    for relative_path in file_inventory["relative_path"]
                    for depth in range(1, len(Path(relative_path).parts))
                }
            )
            for folder in folder_paths:
                prefix = f"{folder}/"
                recursive_mask = file_inventory["relative_path"].str.startswith(prefix)
                direct_mask = file_inventory["relative_path"].map(
                    lambda value: "/".join(Path(value).parts[:-1]) == folder
                )
                folder_rows.append(
                    {
                        "folder": folder,
                        "depth": folder.count("/") + 1,
                        "parent_folder": folder.rsplit("/", 1)[0] if "/" in folder else "<root>",
                        "direct_files": int(direct_mask.sum()),
                        "recursive_files": int(recursive_mask.sum()),
                        "recursive_size_mb": file_inventory.loc[recursive_mask, "size_mb"].sum(),
                    }
                )

            folder_hierarchy = pd.DataFrame(folder_rows).sort_values(["depth", "folder"])
            display(folder_hierarchy.round({"recursive_size_mb": 2}))
            """
        ),
        code(
            """
            render_monthly_summary = (
                render_inventory.groupby(["render_type", "render_month"], dropna=False, as_index=False)
                .agg(files=("relative_path", "size"), size_mb=("size_mb", "sum"))
                .sort_values(["render_type", "render_month"])
            )
            display(render_monthly_summary.round({"size_mb": 2}))

            render_monthly_pivot = render_monthly_summary.pivot(
                index="render_month", columns="render_type", values="files"
            ).fillna(0)
            render_monthly_pivot.plot(kind="bar", stacked=True, figsize=(16, 5), colormap="tab20")
            plt.title("Rendered artifacts by month and type")
            plt.xlabel("Render month")
            plt.ylabel("Files")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 12. Detailed Data Dictionary

            The following catalog profiles every column in all 31 tables. Identifier-like fields are treated
            carefully: `FACT_RETURN.line_item_id` must be loaded as a string because nullable 12-13 digit IDs
            can lose precision when pandas infers `float64`.
            """
        ),
        code(
            """
            def compact_samples(series, limit=5):
                values = series.dropna().astype(str).drop_duplicates().head(limit).tolist()
                return " | ".join(values)


            def compact_top_values(series, limit=5):
                counts = series.fillna("<NULL>").astype(str).value_counts(dropna=False).head(limit)
                return " | ".join(f"{value}: {count:,}" for value, count in counts.items())


            column_profile_rows = []
            for table_name, frame in tables.items():
                for column in frame.columns:
                    series = frame[column]
                    non_null = int(series.notna().sum())
                    unique_values = int(series.nunique(dropna=True))
                    column_profile_rows.append(
                        {
                            "table": table_name,
                            "column": column,
                            "dtype": str(series.dtype),
                            "rows": len(series),
                            "non_null_rows": non_null,
                            "null_rows": int(series.isna().sum()),
                            "null_pct": series.isna().mean() * 100,
                            "unique_values": unique_values,
                            "unique_pct_of_non_null": unique_values / non_null * 100 if non_null else np.nan,
                            "sample_values": compact_samples(series),
                            "top_values": compact_top_values(series),
                        }
                    )

            table_column_profile = pd.DataFrame(column_profile_rows).sort_values(["table", "column"])
            display(table_column_profile)
            """
        ),
        code(
            """
            numeric_profile_rows = []
            for table_name, frame in tables.items():
                for column in frame.select_dtypes(include=np.number).columns:
                    series = frame[column]
                    numeric_profile_rows.append(
                        {
                            "table": table_name,
                            "column": column,
                            "non_null_rows": int(series.notna().sum()),
                            "mean": series.mean(),
                            "std": series.std(),
                            "min": series.min(),
                            "p25": series.quantile(0.25),
                            "median": series.median(),
                            "p75": series.quantile(0.75),
                            "max": series.max(),
                            "sum": series.sum(),
                        }
                    )

            numeric_column_profile = pd.DataFrame(numeric_profile_rows).sort_values(["table", "column"])
            display(numeric_column_profile.round(2))
            """
        ),
        code(
            """
            categorical_profile_rows = []
            for table_name, frame in tables.items():
                for column in frame.columns:
                    series = frame[column]
                    unique_values = series.nunique(dropna=True)
                    if unique_values <= 30:
                        categorical_profile_rows.append(
                            {
                                "table": table_name,
                                "column": column,
                                "unique_values": int(unique_values),
                                "null_rows": int(series.isna().sum()),
                                "top_values": compact_top_values(series, limit=15),
                            }
                        )

            categorical_column_profile = pd.DataFrame(categorical_profile_rows).sort_values(["table", "column"])
            display(categorical_column_profile)
            """
        ),
        md(
            """
            ## 13. Domain Map and Monthly Activity

            This map connects business domains to their structured, narrative, rendered, log, and report
            surfaces. The monthly fact view exposes spikes such as the July 2025 sales cohort.
            """
        ),
        code(
            """
            domain_map = pd.DataFrame(
                [
                    {"domain": "Sales", "tables": "FACT_SALES, FACT_SALES_LINE_ITEM", "docs": "LINE OA, OPS reports", "renders": "receipt", "logs": "pos_*.tsv, web_*.jsonl", "primary_use": "orders, basket totals, channels, payments"},
                    {"domain": "Finance", "tables": "FACT_BANK_TRANSACTION, FACT_VENDOR_PAYMENT, FACT_PAYROLL, DIM_BANK_ACCOUNT", "docs": "memo, LINE Works, FIN close reports", "renders": "bank_statement, vendor_invoice", "logs": "paywise_fee_log_*.csv", "primary_use": "cash flow, reconciliation, approvals"},
                    {"domain": "Customer Service", "tables": "FACT_CS_INTERACTION, DIM_CUSTOMER", "docs": "chat_line_oa", "renders": "warranty_form", "logs": "", "primary_use": "customer cases and chat evidence"},
                    {"domain": "Returns and Warranty", "tables": "FACT_RETURN, FACT_REFUND_PAID, FACT_WARRANTY_CLAIM", "docs": "policy KB, memo", "renders": "warranty_form, receipt", "logs": "", "primary_use": "refunds, claims, policy checks"},
                    {"domain": "Inventory", "tables": "FACT_INVENTORY_MOVEMENT, FACT_INVENTORY_MONTHLY_SNAPSHOT, DIM_PRODUCT", "docs": "product KB", "renders": "", "logs": "pos_*.tsv", "primary_use": "stock movement and month-end positions"},
                    {"domain": "Promotion", "tables": "DIM_PROMO_CAMPAIGN, dim_promo_mechanic, FACT_PROMO_REDEMPTION", "docs": "memo", "renders": "e7_banner", "logs": "web_*.jsonl", "primary_use": "campaign performance and phantom redemption analysis"},
                    {"domain": "Policy and Approval", "tables": "DIM_POLICY_VERSION, dim_signing_authority_ladder, DIM_POSITION_LEVEL", "docs": "policy KB, memo", "renders": "t2_doc, t3_doc", "logs": "", "primary_use": "date-aware policy and signing authority"},
                    {"domain": "Shipping", "tables": "FACT_SHIPPING, DIM_VENDOR", "docs": "shipping policy KB", "renders": "", "logs": "", "primary_use": "delivery fulfillment"},
                ]
            )
            display(domain_map)
            """
        ),
        code(
            """
            FACT_DATE_COLUMNS = {
                "FACT_SALES": "business_event_date",
                "FACT_SALES_LINE_ITEM": "business_event_date",
                "FACT_RETURN": "business_event_date",
                "FACT_REFUND_PAID": "business_event_date",
                "FACT_WARRANTY_CLAIM": "business_event_date",
                "FACT_CS_INTERACTION": "business_event_date",
                "FACT_PROMO_REDEMPTION": "business_event_date",
                "FACT_LOYALTY_LEDGER": "business_event_date",
                "FACT_BANK_TRANSACTION": "business_event_date",
                "FACT_VENDOR_PAYMENT": "posting_date",
                "FACT_PAYROLL": "business_event_date",
                "FACT_INVENTORY_MOVEMENT": "business_event_date",
                "FACT_INVENTORY_MONTHLY_SNAPSHOT": "month_end_date",
                "FACT_SHIPPING": "business_event_date",
            }

            monthly_fact_rows = []
            for table_name, date_column in FACT_DATE_COLUMNS.items():
                frame = tables[table_name]
                months = pd.to_datetime(frame[date_column], errors="coerce").dt.to_period("M").astype("string")
                counts = months.value_counts().sort_index()
                monthly_fact_rows.extend(
                    {"table": table_name, "month": month, "rows": int(count)}
                    for month, count in counts.items()
                    if pd.notna(month)
                )

            monthly_fact_activity = pd.DataFrame(monthly_fact_rows)
            monthly_activity_pivot = monthly_fact_activity.pivot(index="month", columns="table", values="rows").fillna(0)
            display(monthly_activity_pivot)

            selected_monthly_tables = [
                "FACT_SALES",
                "FACT_RETURN",
                "FACT_WARRANTY_CLAIM",
                "FACT_CS_INTERACTION",
                "FACT_BANK_TRANSACTION",
            ]
            normalized_activity = monthly_activity_pivot[selected_monthly_tables].apply(
                lambda series: series / series.median()
            )
            normalized_activity.plot(figsize=(16, 5), marker="o")
            plt.title("Monthly activity normalized by each table's median")
            plt.xlabel("Month")
            plt.ylabel("Rows / median rows")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 14. Relationship Catalog and ER Diagram

            The catalog below is validated against actual values, not inferred only from column names.
            Coverage below 100% can be legitimate when a field uses multiple namespaces. Empty optional
            relationships remain visible as `empty_optional`.
            """
        ),
        code(
            """
            RELATIONSHIPS = [
                ("sales", "FACT_SALES", "branch_code", "DIM_BRANCH", "branch_code"),
                ("sales", "FACT_SALES", "customer_id", "DIM_CUSTOMER", "customer_id"),
                ("sales", "FACT_SALES", "employee_id", "DIM_EMPLOYEE", "employee_id"),
                ("sales", "FACT_SALES", "promo_campaign_id", "DIM_PROMO_CAMPAIGN", "campaign_id"),
                ("sales", "FACT_SALES", "settlement_bank_txn_id", "FACT_BANK_TRANSACTION", "bank_txn_id"),
                ("sales", "FACT_SALES_LINE_ITEM", "txn_id", "FACT_SALES", "txn_id"),
                ("sales", "FACT_SALES_LINE_ITEM", "sku_id", "DIM_PRODUCT", "sku_id"),
                ("returns", "FACT_RETURN", "original_txn_id", "FACT_SALES", "txn_id"),
                ("returns", "FACT_RETURN", "line_item_id", "FACT_SALES_LINE_ITEM", "line_item_id"),
                ("returns", "FACT_RETURN", "sku_id", "DIM_PRODUCT", "sku_id"),
                ("returns", "FACT_RETURN", "branch_code", "DIM_BRANCH", "branch_code"),
                ("returns", "FACT_RETURN", "customer_id", "DIM_CUSTOMER", "customer_id"),
                ("returns", "FACT_RETURN", "approved_by_employee_id", "DIM_EMPLOYEE", "employee_id"),
                ("returns", "FACT_REFUND_PAID", "return_id", "FACT_RETURN", "return_id"),
                ("returns", "FACT_REFUND_PAID", "cs_interaction_id", "FACT_CS_INTERACTION", "cs_interaction_id"),
                ("returns", "FACT_REFUND_PAID", "customer_id", "DIM_CUSTOMER", "customer_id"),
                ("returns", "FACT_REFUND_PAID", "approver_employee_id", "DIM_EMPLOYEE", "employee_id"),
                ("returns", "FACT_REFUND_PAID", "cosig_employee_id", "DIM_EMPLOYEE", "employee_id"),
                ("returns", "FACT_REFUND_PAID", "bank_txn_id", "FACT_BANK_TRANSACTION", "bank_txn_id"),
                ("warranty", "FACT_WARRANTY_CLAIM", "customer_id", "DIM_CUSTOMER", "customer_id"),
                ("warranty", "FACT_WARRANTY_CLAIM", "sku_id", "DIM_PRODUCT", "sku_id"),
                ("warranty", "FACT_WARRANTY_CLAIM", "original_txn_id", "FACT_SALES", "txn_id"),
                ("customer_service", "FACT_CS_INTERACTION", "customer_id", "DIM_CUSTOMER", "customer_id"),
                ("customer_service", "FACT_CS_INTERACTION", "employee_id", "DIM_EMPLOYEE", "employee_id"),
                ("customer_service", "FACT_CS_INTERACTION", "branch_code", "DIM_BRANCH", "branch_code"),
                ("customer_service", "FACT_CS_INTERACTION", "related_refund_id", "FACT_REFUND_PAID", "refund_id"),
                ("customer_service", "FACT_CS_INTERACTION", "related_warranty_claim_id", "FACT_WARRANTY_CLAIM", "claim_id"),
                ("promotion", "FACT_PROMO_REDEMPTION", "txn_id", "FACT_SALES", "txn_id"),
                ("promotion", "FACT_PROMO_REDEMPTION", "customer_id", "DIM_CUSTOMER", "customer_id"),
                ("promotion", "FACT_PROMO_REDEMPTION", "campaign_id", "DIM_PROMO_CAMPAIGN", "campaign_id"),
                ("promotion", "dim_promo_mechanic", "campaign_id", "DIM_PROMO_CAMPAIGN", "campaign_id"),
                ("loyalty", "FACT_LOYALTY_LEDGER", "customer_id", "DIM_CUSTOMER", "customer_id"),
                ("loyalty", "FACT_LOYALTY_LEDGER", "txn_id", "FACT_SALES", "txn_id"),
                ("inventory", "FACT_INVENTORY_MOVEMENT", "sku_id", "DIM_PRODUCT", "sku_id"),
                ("inventory", "FACT_INVENTORY_MOVEMENT", "branch_code", "DIM_BRANCH", "branch_code"),
                ("inventory", "FACT_INVENTORY_MOVEMENT", "related_txn_id", "FACT_SALES", "txn_id"),
                ("inventory", "FACT_INVENTORY_MONTHLY_SNAPSHOT", "sku_id", "DIM_PRODUCT", "sku_id"),
                ("inventory", "FACT_INVENTORY_MONTHLY_SNAPSHOT", "branch_code", "DIM_BRANCH", "branch_code"),
                ("shipping", "FACT_SHIPPING", "txn_id", "FACT_SALES", "txn_id"),
                ("shipping", "FACT_SHIPPING", "vendor_id", "DIM_VENDOR", "vendor_id"),
                ("shipping", "FACT_SHIPPING", "origin_branch_code", "DIM_BRANCH", "branch_code"),
                ("finance", "FACT_BANK_TRANSACTION", "account_id", "DIM_BANK_ACCOUNT", "account_id"),
                ("finance", "FACT_PAYROLL", "employee_id", "DIM_EMPLOYEE", "employee_id"),
                ("finance", "FACT_PAYROLL", "bank_txn_id", "FACT_BANK_TRANSACTION", "bank_txn_id"),
                ("finance", "FACT_VENDOR_PAYMENT", "vendor_id", "DIM_VENDOR", "vendor_id"),
                ("finance", "FACT_VENDOR_PAYMENT", "vendor_contract_version_id", "DIM_VENDOR_CONTRACT_VERSION", "contract_version_id"),
                ("finance", "FACT_VENDOR_PAYMENT", "signing_employee_id", "DIM_EMPLOYEE", "employee_id"),
                ("finance", "FACT_VENDOR_PAYMENT", "cosig_employee_id", "DIM_EMPLOYEE", "employee_id"),
                ("finance", "FACT_VENDOR_PAYMENT", "bank_txn_id", "FACT_BANK_TRANSACTION", "bank_txn_id"),
                ("organization", "DIM_EMPLOYEE", "branch_code", "DIM_BRANCH", "branch_code"),
                ("organization", "DIM_EMPLOYEE", "dept_code", "DIM_DEPARTMENT", "dept_code"),
                ("organization", "DIM_EMPLOYEE", "reports_to_employee_id", "DIM_EMPLOYEE", "employee_id"),
                ("organization", "DIM_EMPLOYEE", "position_level", "DIM_POSITION_LEVEL", "position_level_code"),
                ("product", "DIM_PRODUCT", "dept_code", "DIM_DEPARTMENT", "dept_code"),
                ("product", "DIM_PRODUCT", "vendor_id", "DIM_VENDOR", "vendor_id"),
                ("product", "dim_product_recall_history", "sku_id", "DIM_PRODUCT", "sku_id"),
                ("vendor", "DIM_VENDOR_CONTRACT_VERSION", "vendor_id", "DIM_VENDOR", "vendor_id"),
                ("policy", "dim_signing_authority_ladder", "policy_version_id", "DIM_POLICY_VERSION", "policy_version_id"),
                ("policy", "dim_signing_authority_ladder", "position_level_code", "DIM_POSITION_LEVEL", "position_level_code"),
                ("policy", "dim_signing_authority_ladder", "dept_code", "DIM_DEPARTMENT", "dept_code"),
                ("policy", "dim_signing_authority_ladder", "co_signer_min_position_level_code", "DIM_POSITION_LEVEL", "position_level_code"),
                ("policy", "dim_care_plus_sku_tier", "policy_version_id", "DIM_POLICY_VERSION", "policy_version_id"),
                ("policy", "dim_care_plus_sku_tier", "sku_id", "DIM_PRODUCT", "sku_id"),
            ]

            relationship_rows = []
            for domain, child_table, child_column, parent_table, parent_column in RELATIONSHIPS:
                child = tables[child_table][child_column]
                parent = tables[parent_table][parent_column]
                child_non_null = child.dropna().astype(str)
                parent_keys = set(parent.dropna().astype(str))
                matched_rows = int(child_non_null.isin(parent_keys).sum())
                coverage_pct = matched_rows / len(child_non_null) * 100 if len(child_non_null) else np.nan
                if len(child_non_null) == 0:
                    status = "empty_optional"
                elif matched_rows == len(child_non_null):
                    status = "full_match"
                elif matched_rows == 0:
                    status = "no_match"
                else:
                    status = "partial_match"
                relationship_rows.append(
                    {
                        "domain": domain,
                        "child_table": child_table,
                        "child_column": child_column,
                        "parent_table": parent_table,
                        "parent_column": parent_column,
                        "child_rows": len(child),
                        "non_null_child_rows": len(child_non_null),
                        "matched_rows": matched_rows,
                        "unmatched_rows": len(child_non_null) - matched_rows,
                        "coverage_pct": coverage_pct,
                        "parent_key_unique": not parent.dropna().astype(str).duplicated().any(),
                        "status": status,
                    }
                )

            relationship_catalog = pd.DataFrame(relationship_rows).sort_values(
                ["status", "domain", "child_table", "child_column"]
            )
            display(relationship_catalog.round({"coverage_pct": 2}))

            relationship_diagnostics = relationship_catalog[
                relationship_catalog["status"].isin(["partial_match", "no_match"])
            ].copy()
            display(relationship_diagnostics.round({"coverage_pct": 2}))
            """
        ),
        code(
            """
            inventory_movements = tables["FACT_INVENTORY_MOVEMENT"]
            sales_ids = set(tables["FACT_SALES"]["txn_id"].astype(str))
            inventory_related = inventory_movements["related_txn_id"].dropna().astype(str)
            unmatched_movement_rows = inventory_movements.loc[inventory_related.index[~inventory_related.isin(sales_ids)]].copy()
            unmatched_movement_summary = (
                unmatched_movement_rows.groupby("movement_type", as_index=False)
                .agg(rows=("movement_id", "size"), unique_related_ids=("related_txn_id", "nunique"))
                .sort_values("rows", ascending=False)
            )
            display(unmatched_movement_summary)
            display(unmatched_movement_rows.head(20))
            """
        ),
        code(
            """
            import networkx as nx

            graph = nx.DiGraph()
            for _, row in relationship_catalog.iterrows():
                graph.add_edge(row["child_table"], row["parent_table"])

            node_colors = []
            for node in graph.nodes:
                if node.upper().startswith("FACT_"):
                    node_colors.append("#F4B183")
                elif node.upper().startswith("DIM_") or node.lower().startswith("dim_"):
                    node_colors.append("#9DC3E6")
                else:
                    node_colors.append("#C5E0B4")

            plt.figure(figsize=(24, 18))
            positions = nx.spring_layout(graph, seed=42, k=2.4, iterations=250)
            nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=3800, edgecolors="#555555")
            nx.draw_networkx_edges(graph, positions, arrows=True, arrowstyle="-|>", arrowsize=16, width=1.2, alpha=0.55)
            nx.draw_networkx_labels(graph, positions, font_size=8)
            plt.title("FahMai ER Diagram: validated child-to-parent relationships")
            plt.axis("off")
            plt.tight_layout()
            OUTPUT_DIR.mkdir(exist_ok=True)
            er_diagram_path = OUTPUT_DIR / "er_diagram_full.png"
            plt.savefig(er_diagram_path, dpi=180, bbox_inches="tight")
            plt.show()
            print(f"Saved {er_diagram_path}")

            mermaid_lines = ["flowchart LR"]
            for _, row in relationship_catalog.iterrows():
                child_node = re.sub(r"[^A-Za-z0-9_]", "_", row["child_table"])
                parent_node = re.sub(r"[^A-Za-z0-9_]", "_", row["parent_table"])
                edge_label = f"{row['child_column']} -> {row['parent_column']}"
                mermaid_lines.append(f"    {child_node}[{row['child_table']}] -->|{edge_label}| {parent_node}[{row['parent_table']}]")
            er_mermaid_path = OUTPUT_DIR / "er_diagram_full.mmd"
            er_mermaid_path.write_text("\\n".join(mermaid_lines), encoding="utf-8")
            print(f"Saved {er_mermaid_path}")
            """
        ),
        md(
            """
            ## 15. Cross-Surface Mapping Coverage

            Tables, renders, docs, and logs are not independent corpora. This section validates common joins:
            receipts to sales, invoices to payments, warranty forms to claims, T2 PDFs to inventory rows,
            LINE OA files to referenced sessions, and POS/web references to existing raw log files.
            """
        ),
        code(
            """
            sales = tables["FACT_SALES"]
            warranty_claims = tables["FACT_WARRANTY_CLAIM"].copy()
            vendor_payments = tables["FACT_VENDOR_PAYMENT"].copy()
            cs_interactions = tables["FACT_CS_INTERACTION"]
            line_items = tables["FACT_SALES_LINE_ITEM"]
            t2_inventory = tables["T2_DOC_INVENTORY"]

            receipt_ids = {path.stem.removeprefix("RC-") for path in (DATA_ROOT / "renders" / "receipt").rglob("*.png")}
            rendered_invoice_ids = {path.stem.removeprefix("VI-") for path in (DATA_ROOT / "renders" / "vendor_invoice").rglob("*.png")}
            t2_pdf_ids = {path.stem for path in (DATA_ROOT / "renders" / "t2_doc").rglob("*.pdf")}
            line_oa_doc_ids = {path.stem for path in (DATA_ROOT / "docs" / "chat_line_oa").glob("*.md")}
            existing_log_files = {path.name for path in (DATA_ROOT / "logs").iterdir() if path.is_file()}

            rendered_warranty_claim_ids = set()
            for path in (DATA_ROOT / "renders" / "warranty_form").rglob("*.png"):
                match = re.search(r"-(20\\d{4})-(\\d{8})$", path.stem)
                if match:
                    rendered_warranty_claim_ids.add(f"WC-{match.group(1)}-{match.group(2)}")

            pos_log_refs = line_items["pos_log_line_id"].dropna().astype(str)
            pos_log_files = pos_log_refs.str.split(":").str[0]
            web_log_refs = sales["web_log_line_id"].dropna().astype(str)
            web_log_files = web_log_refs.str.split(":").str[0]
            referenced_chat_ids = cs_interactions["chat_session_id"].dropna().astype(str)

            cross_surface_mapping = pd.DataFrame(
                [
                    {"mapping": "receipt render -> FACT_SALES.txn_id", "source_rows": len(receipt_ids), "matched_rows": sum(value in set(sales["txn_id"].astype(str)) for value in receipt_ids), "note": "RC-{txn_id}.png"},
                    {"mapping": "vendor_invoice render -> FACT_VENDOR_PAYMENT.vendor_invoice_id", "source_rows": len(rendered_invoice_ids), "matched_rows": sum(value in set(vendor_payments["vendor_invoice_id"].astype(str)) for value in rendered_invoice_ids), "note": "VI-{vendor_invoice_id}.png"},
                    {"mapping": "warranty_form render -> FACT_WARRANTY_CLAIM.claim_id", "source_rows": len(rendered_warranty_claim_ids), "matched_rows": sum(value in set(warranty_claims["claim_id"].astype(str)) for value in rendered_warranty_claim_ids), "note": "claim ID derived from filename date and suffix"},
                    {"mapping": "T2 PDF -> T2_DOC_INVENTORY.doc_id", "source_rows": len(t2_pdf_ids), "matched_rows": sum(value in set(t2_inventory["doc_id"].astype(str)) for value in t2_pdf_ids), "note": "{doc_id}.pdf"},
                    {"mapping": "FACT_CS_INTERACTION.chat_session_id -> LINE OA doc", "source_rows": len(referenced_chat_ids), "matched_rows": int(referenced_chat_ids.isin(line_oa_doc_ids).sum()), "note": "{chat_session_id}.md"},
                    {"mapping": "FACT_SALES_LINE_ITEM.pos_log_line_id -> POS file", "source_rows": len(pos_log_refs), "matched_rows": int(pos_log_files.isin(existing_log_files).sum()), "note": "filename:lineN"},
                    {"mapping": "FACT_SALES.web_log_line_id -> web JSONL file", "source_rows": len(web_log_refs), "matched_rows": int(web_log_files.isin(existing_log_files).sum()), "note": "filename:lineN"},
                    {"mapping": "bank statement account-month -> FACT_BANK_TRANSACTION account-month", "source_rows": len(render_account_month), "matched_rows": int(render_account_month.merge(fact_account_month, on=["account_id", "render_month"], how="inner").shape[0]), "note": "filename-derived account and month"},
                ]
            )
            cross_surface_mapping["coverage_pct"] = (
                cross_surface_mapping["matched_rows"] / cross_surface_mapping["source_rows"] * 100
            )
            display(cross_surface_mapping.round({"coverage_pct": 2}))
            """
        ),
        code(
            """
            warranty_claims["has_render"] = warranty_claims["claim_id"].astype(str).isin(rendered_warranty_claim_ids)
            warranty_render_coverage = (
                warranty_claims.groupby(["resolution_type", "routing_destination", "claim_reason"], dropna=False, as_index=False)
                .agg(claims=("claim_id", "size"), rendered_claims=("has_render", "sum"))
            )
            warranty_render_coverage["render_pct"] = (
                warranty_render_coverage["rendered_claims"] / warranty_render_coverage["claims"] * 100
            )
            display(warranty_render_coverage.round({"render_pct": 2}))

            vendor_payments["has_render"] = vendor_payments["vendor_invoice_id"].astype(str).isin(rendered_invoice_ids)
            vendor_invoice_render_coverage = (
                vendor_payments.groupby("vendor_id", as_index=False)
                .agg(payments=("payment_id", "size"), unique_invoice_ids=("vendor_invoice_id", "nunique"), rendered_payments=("has_render", "sum"))
            )
            vendor_invoice_render_coverage["render_pct"] = (
                vendor_invoice_render_coverage["rendered_payments"] / vendor_invoice_render_coverage["payments"] * 100
            )
            display(vendor_invoice_render_coverage.round({"render_pct": 2}))
            """
        ),
        md(
            """
            ## 16. Narrative, Report, and T2 Document Structure

            LINE transcript files use JSONL records inside `.md` files. The sampled profile checks message count,
            malformed lines, and speaker diversity without scanning every document on each exploratory run.
            """
        ),
        code(
            """
            def parse_concatenated_json_objects(text):
                decoder = json.JSONDecoder()
                position = 0
                messages = []
                malformed_fragments = 0
                while position < len(text):
                    while position < len(text) and text[position].isspace():
                        position += 1
                    if position >= len(text):
                        break
                    try:
                        message, position = decoder.raw_decode(text, position)
                        messages.append(message)
                    except json.JSONDecodeError:
                        malformed_fragments += 1
                        next_newline = text.find("\\n", position)
                        if next_newline == -1:
                            break
                        position = next_newline + 1
                return messages, malformed_fragments


            transcript_rows = []
            for family in ["chat_line_oa", "chat_line_works"]:
                family_paths = sorted((DATA_ROOT / "docs" / family).glob("*.md"))
                sampled_paths = deterministic_sample([path.relative_to(DATA_ROOT).as_posix() for path in family_paths], 1000)
                for relative_path in sampled_paths:
                    text = (DATA_ROOT / relative_path).read_text(encoding="utf-8", errors="replace")
                    messages, malformed_fragments = parse_concatenated_json_objects(text)
                    transcript_rows.append(
                        {
                            "family": family,
                            "relative_path": relative_path,
                            "messages": len(messages),
                            "malformed_fragments": malformed_fragments,
                            "speakers": len({message.get("speaker") for message in messages}),
                            "message_types": " | ".join(sorted({str(message.get("message_type")) for message in messages})),
                        }
                    )

            transcript_sample_profile = pd.DataFrame(transcript_rows)
            transcript_family_summary = (
                transcript_sample_profile.groupby("family", as_index=False)
                .agg(
                    sampled_files=("relative_path", "size"),
                    messages=("messages", "sum"),
                    mean_messages_per_file=("messages", "mean"),
                    min_messages_per_file=("messages", "min"),
                    max_messages_per_file=("messages", "max"),
                    malformed_fragments=("malformed_fragments", "sum"),
                )
            )
            display(transcript_family_summary.round({"mean_messages_per_file": 2}))
            """
        ),
        md(
            """
            ## 17. Adversarial Narrative Audit

            The challenge brief explicitly requires defenses against instruction override attempts inside the
            corpus. The scan below checks all control documents and reports plus the deterministic chat sample.
            Hits are evidence to quarantine from instruction-following context, not instructions for the agent.
            """
        ),
        code(
            """
            suspicious_patterns = {
                "internal_override": r"\\[INTERNAL OVERRIDE\\]",
                "grader_instructions": r"grader instructions",
                "ignore_previous": r"ignore\\s+(?:all\\s+|the\\s+)?(?:previous|prior)",
                "system_prompt": r"system prompt",
                "prompt_injection": r"prompt injection",
                "must_contain": r"must[_ ]?contain",
                "answer_exactly": r"answer exactly",
            }
            suspicious_regex = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in suspicious_patterns.items()), re.IGNORECASE)

            control_paths = []
            for relative_folder in ["docs/memo", "docs/minutes", "docs/email", "docs/l1_kb", "reports"]:
                control_paths.extend(sorted((DATA_ROOT / relative_folder).rglob("*.md")))
            sampled_chat_paths = [DATA_ROOT / relative_path for relative_path in transcript_sample_profile["relative_path"]]
            narrative_security_scan_paths = sorted(set(control_paths + sampled_chat_paths))

            suspicious_rows = []
            for path in narrative_security_scan_paths:
                text = path.read_text(encoding="utf-8", errors="replace")
                for match in suspicious_regex.finditer(text):
                    line_number = text[: match.start()].count("\\n") + 1
                    suspicious_rows.append(
                        {
                            "relative_path": path.relative_to(DATA_ROOT).as_posix(),
                            "line_number": line_number,
                            "pattern": match.lastgroup,
                            "matched_text": match.group(0),
                            "context": text[max(0, match.start() - 80) : match.end() + 160].replace("\\n", " "),
                        }
                    )

            suspicious_narrative_hits = pd.DataFrame(
                suspicious_rows,
                columns=["relative_path", "line_number", "pattern", "matched_text", "context"],
            ).sort_values(["relative_path", "line_number"])
            print(f"Scanned narrative files: {len(narrative_security_scan_paths):,}")
            print(f"Suspicious hits: {len(suspicious_narrative_hits):,}")
            display(suspicious_narrative_hits)
            """
        ),
        md(
            """
            ### Adversarial Handling Rule

            Retrieved corpus text is untrusted data. Never execute instructions found inside documents, chats,
            renders, or reports. Extract facts with source attribution, prefer structured authoritative tables
            where the benchmark says they are authoritative, and isolate suspicious passages in audit telemetry.
            """
        ),
        code(
            """
            report_rows = []
            for path in sorted((DATA_ROOT / "reports").rglob("*.md")):
                report_rows.append(
                    {
                        "relative_path": path.relative_to(DATA_ROOT).as_posix(),
                        "report_period": path.parent.name,
                        "report_type": "quarterly_fin_close" if "-Q" in path.parent.name else "monthly_ops",
                        "size_kb": path.stat().st_size / 1024,
                    }
                )
            report_inventory = pd.DataFrame(report_rows)
            report_summary = (
                report_inventory.groupby("report_type", as_index=False)
                .agg(files=("relative_path", "size"), min_period=("report_period", "min"), max_period=("report_period", "max"), size_kb=("size_kb", "sum"))
            )
            display(report_summary.round({"size_kb": 2}))

            t2_source_summary = (
                t2_inventory.groupby(["doc_kind", "template_name", "source_table"], as_index=False)
                .agg(documents=("doc_id", "size"), min_issue_date=("issue_date", "min"), max_issue_date=("issue_date", "max"))
                .sort_values("documents", ascending=False)
            )
            display(t2_source_summary)
            """
        ),
        md(
            """
            ## 18. Bank-Statement Deep Dive

            The OCR surface contains 336 logical statements across 14 accounts and 24 months. Bank template,
            image resolution, and row density are tightly coupled. This section creates the benchmark strata
            and account-month manifest needed for GLM-OCR experiments.
            """
        ),
        code(
            """
            bank_accounts = tables["DIM_BANK_ACCOUNT"].copy()
            bank_transactions = tables["FACT_BANK_TRANSACTION"].copy()
            bank_transactions["business_event_date"] = pd.to_datetime(bank_transactions["business_event_date"])
            bank_transactions["render_month"] = bank_transactions["business_event_date"].dt.to_period("M").astype(str)

            bank_fact_account_month = (
                bank_transactions.groupby(["account_id", "render_month"], as_index=False)
                .agg(
                    fact_transactions=("bank_txn_id", "size"),
                    net_amount_thb=("amount_thb", "sum"),
                    deposit_rows=("transaction_type", lambda values: int((values == "deposit").sum())),
                    withdrawal_rows=("transaction_type", lambda values: int((values == "withdrawal").sum())),
                    transfer_rows=("transaction_type", lambda values: int((values == "transfer").sum())),
                    fee_rows=("transaction_type", lambda values: int((values == "fee").sum())),
                    min_balance_after_thb=("balance_after_thb", "min"),
                    max_balance_after_thb=("balance_after_thb", "max"),
                )
            )

            transaction_statement_layout = (
                bank_image_metadata[bank_image_metadata["page_role"] == "transactions"]
                .groupby(["statement_id", "account_id", "render_month"], as_index=False)
                .agg(
                    transaction_pages=("relative_path", "size"),
                    transaction_width=("width", "first"),
                    transaction_height=("height", "first"),
                    transaction_megapixels=("megapixels", "first"),
                )
            )
            bank_statement_deep = (
                transaction_statement_layout.merge(bank_fact_account_month, on=["account_id", "render_month"], how="outer", validate="one_to_one")
                .merge(bank_accounts, on="account_id", how="left", validate="many_to_one")
            )
            bank_statement_deep["fact_rows_per_transaction_page"] = (
                bank_statement_deep["fact_transactions"] / bank_statement_deep["transaction_pages"]
            )
            display(bank_statement_deep)
            """
        ),
        code(
            """
            bank_account_deep_summary = (
                bank_statement_deep.groupby(["account_id", "bank", "account_number", "associated_branch_code"], dropna=False, as_index=False)
                .agg(
                    months=("render_month", "size"),
                    transaction_pages=("transaction_pages", "sum"),
                    fact_transactions=("fact_transactions", "sum"),
                    min_pages_per_month=("transaction_pages", "min"),
                    median_pages_per_month=("transaction_pages", "median"),
                    max_pages_per_month=("transaction_pages", "max"),
                    median_fact_rows_per_page=("fact_rows_per_transaction_page", "median"),
                    min_fact_rows_per_page=("fact_rows_per_transaction_page", "min"),
                    max_fact_rows_per_page=("fact_rows_per_transaction_page", "max"),
                )
                .sort_values("transaction_pages", ascending=False)
            )
            display(bank_account_deep_summary.round(2))
            """
        ),
        code(
            """
            bank_transaction_type_summary = (
                bank_transactions.groupby("transaction_type", as_index=False)
                .agg(
                    rows=("bank_txn_id", "size"),
                    amount_sum_thb=("amount_thb", "sum"),
                    amount_min_thb=("amount_thb", "min"),
                    amount_median_thb=("amount_thb", "median"),
                    amount_max_thb=("amount_thb", "max"),
                )
                .sort_values("rows", ascending=False)
            )
            display(bank_transaction_type_summary.round(2))

            bank_related_entity_summary = (
                bank_transactions.groupby("related_entity_table", dropna=False, as_index=False)
                .agg(rows=("bank_txn_id", "size"), amount_sum_thb=("amount_thb", "sum"))
                .sort_values("rows", ascending=False)
            )
            display(bank_related_entity_summary.round(2))
            """
        ),
        code(
            """
            bank_ocr_strata = (
                bank_image_metadata.merge(bank_accounts[["account_id", "bank"]], on="account_id", how="left")
                .groupby(["bank", "page_role", "width", "height"], as_index=False)
                .agg(
                    pages=("relative_path", "size"),
                    accounts=("account_id", "nunique"),
                    statements=("statement_id", "nunique"),
                    example_path=("relative_path", "first"),
                )
                .sort_values(["page_role", "bank"])
            )
            display(bank_ocr_strata)

            benchmark_sample_manifest = (
                bank_image_metadata.merge(bank_accounts[["account_id", "bank"]], on="account_id", how="left")
                .sort_values(["bank", "page_role", "account_id", "render_month", "page_number"], na_position="first")
                .groupby(["bank", "page_role", "width", "height"], as_index=False)
                .head(3)
                [["relative_path", "statement_id", "account_id", "bank", "render_month", "page_role", "page_number", "width", "height", "megapixels"]]
            )
            display(benchmark_sample_manifest)
            """
        ),
        code(
            """
            page_heatmap = bank_statement_deep.pivot(index="account_id", columns="render_month", values="transaction_pages")
            plt.figure(figsize=(17, 6))
            sns.heatmap(page_heatmap, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Transaction pages"})
            plt.title("Bank-statement transaction pages per account-month")
            plt.xlabel("Statement month")
            plt.ylabel("Account")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 5))
            sns.scatterplot(
                data=bank_statement_deep,
                x="fact_transactions",
                y="transaction_pages",
                hue="bank",
                style="bank",
                s=70,
            )
            plt.title("Page count versus FACT_BANK_TRANSACTION rows")
            plt.xlabel("Fact rows per account-month")
            plt.ylabel("Rendered transaction pages")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ### Bank OCR Interpretation Notes

            - KBank transaction pages use the larger `2968 x 2850` layout and often hold roughly 40-46 fact rows
              per page. SCB and BBL pages use the smaller `1240 x 1234` layout and usually hold roughly 14-16 rows.
            - `KBANK-OPER` is the dominant OCR workload and needs explicit benchmark coverage.
            - Header pages and transaction pages must use separate prompts and parsing contracts.
            - Running-balance validation should be applied to OCR row order from each statement page. The generic
              fact CSV order is not guaranteed to be statement order.
            - Filename parsing should happen before OCR. It provides account, statement month, role, and page
              number for routing and post-validation.
            """
        ),
        md(
            """
            ## 19. Render Gallery Beyond Bank Statements

            OCR is scored on bank statements, but the agentic track may still need to open receipts, invoices,
            warranty forms, banners, and T3 documents. This gallery keeps those layout families visible.
            """
        ),
        code(
            """
            gallery_types = ["receipt", "vendor_invoice", "warranty_form", "e7_banner", "t3_doc"]
            gallery_rows = []
            for render_type in gallery_types:
                row = (
                    image_metadata.merge(render_inventory[["relative_path", "render_type"]], on="relative_path", how="left")
                    .query("render_type == @render_type")
                    .sort_values("relative_path")
                    .head(1)
                )
                if not row.empty:
                    gallery_rows.append(row.iloc[0])

            fig, axes = plt.subplots(len(gallery_rows), 1, figsize=(14, 5 * len(gallery_rows)))
            axes = np.atleast_1d(axes)
            for axis, row in zip(axes, gallery_rows):
                with Image.open(DATA_ROOT / row["relative_path"]) as image:
                    thumbnail = image.copy()
                    thumbnail.thumbnail((1400, 1000))
                axis.imshow(thumbnail)
                axis.set_title(f"{row['render_type']} | {int(row['width'])}x{int(row['height'])} | {row['relative_path']}")
                axis.axis("off")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 20. Extended Export Manifest

            These outputs are small reusable catalogs for retrieval, pipeline routing, and OCR experiment design.
            """
        ),
        code(
            """
            extended_exports = {
                "folder_hierarchy.csv": folder_hierarchy,
                "render_monthly_summary.csv": render_monthly_summary,
                "table_column_profile.csv": table_column_profile,
                "numeric_column_profile.csv": numeric_column_profile,
                "categorical_column_profile.csv": categorical_column_profile,
                "domain_map.csv": domain_map,
                "monthly_fact_activity.csv": monthly_fact_activity,
                "relationship_catalog.csv": relationship_catalog,
                "relationship_diagnostics.csv": relationship_diagnostics,
                "unmatched_inventory_movement_summary.csv": unmatched_movement_summary,
                "cross_surface_mapping.csv": cross_surface_mapping,
                "warranty_render_coverage.csv": warranty_render_coverage,
                "vendor_invoice_render_coverage.csv": vendor_invoice_render_coverage,
                "transcript_sample_profile.csv": transcript_sample_profile,
                "transcript_family_summary.csv": transcript_family_summary,
                "suspicious_narrative_hits.csv": suspicious_narrative_hits,
                "report_inventory.csv": report_inventory,
                "report_summary.csv": report_summary,
                "t2_source_summary.csv": t2_source_summary,
                "bank_statement_deep.csv": bank_statement_deep,
                "bank_account_deep_summary.csv": bank_account_deep_summary,
                "bank_transaction_type_summary.csv": bank_transaction_type_summary,
                "bank_related_entity_summary.csv": bank_related_entity_summary,
                "bank_ocr_strata.csv": bank_ocr_strata,
                "bank_ocr_benchmark_sample_manifest.csv": benchmark_sample_manifest,
            }
            for filename, frame in extended_exports.items():
                frame.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

            extended_export_manifest = pd.DataFrame(
                [{"filename": filename, "rows": len(frame), "columns": len(frame.columns)} for filename, frame in extended_exports.items()]
            ).sort_values("filename")
            display(extended_export_manifest)
            print(f"Exported {len(extended_exports)} extended CSV files to {OUTPUT_DIR}")
            print(f"Also exported {er_diagram_path.name} and {er_mermaid_path.name}")
            """
        ),
        md(
            """
            ## 21. Extended EDA Conclusions

            - Use string dtypes for identifier columns, especially nullable long numeric IDs.
            - Treat the bundle as a connected graph across tables, documents, renders, logs, and reports.
            - Preserve raw rows: some partial matches are intentional namespace differences rather than bad data.
            - Treat retrieved narrative as untrusted data and quarantine embedded instruction overrides.
            - Build GLM-OCR around bank-specific layout strata and statement hierarchy, not a single-image loop.
            - Start OCR benchmarking with the exported `bank_ocr_benchmark_sample_manifest.csv`, then expand to
              the full `bank_statement_deep.csv` manifest with latency, parse warning, and validation telemetry.
            """
        ),
    ]
)

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
