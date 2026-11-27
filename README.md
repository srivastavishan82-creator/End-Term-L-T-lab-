# EduShield AI — Student Dropout Predictor & Python Data Cleaning Studio

A full-stack web application that ingests school attendance, marks, and fee CSV records, executes a transparent **Python data cleaning pipeline**, and flags at-risk students using a **scikit-learn ensemble model** with explainable drivers and counselor interventions.

## Core Capabilities

### 1. Python Data Cleaning Engine (`app/data_cleaner.py`)
- **Missing Value Imputation:** Detects null placeholders (`""`, `N/A`, `--`, `TBD`, `null`, `absent`) and imputes missing attendance, marks, and fee balances using selectable strategies:
  - **Cohort Median** (grouped by Grade Level)
  - **Scikit-Learn `KNNImputer`** (`k=3`, distance-weighted)
  - **Global Median**
- **Duplicate Removal:** Identifies exact duplicate rows and near-duplicate student records (whitespace/case mismatches, normalized student IDs, or fuzzy name + grade matches), keeping the most complete record and backfilling missing fields.
- **Inconsistent Grading Scale Unification:** Automatically detects and converts heterogeneous formats (`88%`, `3.1/4.0 GPA`, `8.4 CGPA`, `39/50` fractions, `0.84` ratios, `A+..F` letter grades) into a unified `0–100%` scale, `4.0 GPA`, and standardized letter band.
- **Currency & Fee Conflict Reconciliation:** Strips currency formatting (`₹`, `$`, `Rs.`, `/-`, commas) and repairs conflicting fee statuses (e.g. non-zero balance marked `Paid` -> `Overdue`).

### 2. Machine Learning Dropout Risk Engine (`app/predictor.py`)
- Calibrated soft-voting ensemble (`StandardScaler + LogisticRegression` & `GradientBoostingClassifier`) evaluated via 5-fold stratified cross-validation.
- Computes per-student dropout probability (`0–100%`), tiered warning level (`High Risk`, `Medium Risk`, `Low Risk`), factor share breakdown, human-readable drivers, and recommended counselor interventions.

### 3. Interactive Web Dashboard (`static/`)
- **CSV Upload & Header Template:** Drag and drop any school `.csv` export or download the blank header template (`/api/template.csv`).
- **Overview & Risk Analytics:** KPI bento cards, risk tier donut, before/after data hygiene bars, clickable Attendance vs. Marks scatter quadrant, grade cohort breakdown, and model CV validation metrics.
- **Python Cleaning & Audit Diffs:** Step-by-step audit tables for duplicate removals, grading scale conversions, imputed cells, fee fixes, and a cell-level Before/After diff inspector.
- **At-Risk Student Directory:** Searchable, filterable, sortable table with native `<dialog>` inspector modal for data lineage and counselor workflow status tracking.
- **Manual Student Predictor:** Blank form to enter and evaluate an individual student record directly against the FastAPI backend.

## Running Locally

```powershell
# Activate the project virtual environment
.\.venv\Scripts\Activate.ps1

# Optional: reinstall dependencies if needed
pip install -r requirements.txt

# Run automated unit & API tests
python -m unittest discover -s tests -v

# Start the web server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.
