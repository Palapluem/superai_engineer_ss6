# Super AI Engineer Season 6 Workspace

Personal study and competition workspace for Super AI Engineer Season 6. The repository keeps source code, notebooks, notes, small assets, reproducibility notes, and selected final artifacts that are useful when moving the work to another machine.

Private CV files were removed from this repository on 2026-06-14.

## Main Structure

```text
superai_engineer_ss6/
├── Certificate (SS5)/                  # Prior certificate/reference material
├── Level 1/                            # Foundation quizzes and Level 1 hackathons
├── Level 2/                            # Level 2 hackathons, final projects, and preparation
├── .gitignore                          # Local-only data, virtualenvs, caches, renders, and archives
└── README.md                           # This transfer and usage guide
```

## Level 1 Coverage

- Foundation AI quiz preparation and extracted study material.
- Form Data to Insight / public transport EDA work.
- OCR election-document challenge.
- FahMai RAG challenge.
- 5-domain hackathon preparation, including heart disease prediction, sleep-stage classification, Thai image captioning, house recognition, and word segmentation.

## Level 2 Coverage

- FahMai Telephone Directory challenge.
- Demand Forecasting Coffee Chain hackathon.
- WellSense AIoT and system product work, including fall detection, localization simulation, and dashboard-related material.
- FahMai The Finale OCR and enterprise data-agent work.
- 5 Domains Hackathon preparation and reference solutions from previous seasons.

## What Is Tracked

The repository is meant to keep material that is practical to clone and reuse:

- Source code, notebooks, scripts, SQL, and configuration files.
- Markdown notes, README files, study guides, and project explanations.
- Small diagrams, selected result files, and final documentation.
- Reproducibility packages that are small enough for normal Git use.

## What Stays Local-Only

The following are intentionally ignored and should not be forced into Git:

- `.venv/`, `venv/`, `env/`, `__pycache__/`, and `*.pyc`.
- Personal CV or resume files.
- Raw competition datasets that are distributed externally.
- Large generated render folders from OCR/document pipelines.
- Heavy zip archives and model/data bundles that are too large for normal GitHub use.
- Notebook checkpoints and temporary analysis outputs.

Several ignored files in this workspace are very large, including multi-GB ASR/test archives and generated document render sets. Keep those on local storage or external backup rather than adding them to this repository.

## Move To A Main Machine

1. Clone the repository.
2. Install Python 3.11 or 3.12.
3. Create a virtual environment in the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

4. Install dependencies inside the specific hackathon folder you want to run. If a folder has `requirements.txt`, use:

```powershell
python -m pip install -r requirements.txt
```

5. Restore any required raw datasets from the original competition platform or private backup into the same local path expected by the notebook/script.
6. Run notebooks or scripts from their own challenge folder so relative paths continue to work.

## Data And Artifact Policy

Use this rule of thumb before adding files:

- Commit source, notes, notebooks, and small reproducible outputs.
- Do not commit virtual environments, cache folders, generated render folders, or private documents.
- Do not commit files above normal GitHub limits. If a file is important and still reasonably sized, consider Git LFS intentionally; if it is hundreds of MB or multiple GB, keep it local and document how to regenerate or retrieve it.

## Maintenance Checklist

- Keep each hackathon self-contained.
- Add or update a README in each major project folder when the project becomes useful outside the original machine.
- Keep final scripts and notebooks runnable from a fresh clone.
- Record external dataset requirements in README files instead of committing the raw private/heavy data.
- Remove personal documents before pushing.

## Author

Wisit Suwannao

Last updated: 2026-06-14
