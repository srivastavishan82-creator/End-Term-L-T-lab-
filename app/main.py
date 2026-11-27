# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""FastAPI backend serving the Student Dropout Predictor web app and cleaning APIs."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.data_cleaner import clean_school_dataset, parse_messy_csv_rows
from app.predictor import MODEL_BUNDLE, predict_dropout_risk

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

CSV_TEMPLATE_HEADER = (
    "student_id,student_name,grade_level,attendance,marks_grade,fee_due,fee_status\n"
)

app = FastAPI(
    title="Student Dropout Predictor & Data Cleaning Studio",
    description="Cleans messy school attendance, marks, and fee records in Python to predict at-risk students.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if (STATIC_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _json_safe(obj: Any) -> Any:
    """Recursively convert NumPy scalars and guard non-finite floats so JSON serialization never crashes."""
    if isinstance(obj, dict):
        typed_dict = cast(dict[Any, Any], obj)
        return {str(k): _json_safe(v) for k, v in typed_dict.items()}
    if isinstance(obj, (list, tuple)):
        typed_seq = cast(Iterable[Any], obj)
        return [_json_safe(v) for v in typed_seq]
    if isinstance(obj, np.bool_):
        return bool(cast(Any, obj))
    if isinstance(obj, np.integer):
        return int(cast(Any, obj))
    if isinstance(obj, (float, np.floating)):
        val = float(cast(Any, obj))
        return val if math.isfinite(val) else 0.0
    return obj


class SingleStudentRequest(BaseModel):
    student_id: str = Field(min_length=1)
    student_name: str = Field(min_length=1)
    grade_level: str = Field(min_length=1)
    attendance: str = Field(default="")
    marks_raw: str = Field(default="")
    fee_due: str = Field(default="")
    fee_status: str = Field(default="")
    imputation_method: Literal["cohort_median", "global_median", "knn"] = "cohort_median"
    reference_csv: str | None = Field(default=None)


def _rows_to_csv(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    """Serialize a list of dictionaries to CSV text without DataFrame overload ambiguity."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in fieldnames})
    return buf.getvalue()


def _build_full_response(
    csv_text: str,
    imputation_method: str,
    dedup_mode: str,
    source_label: str,
) -> dict[str, Any]:
    audit = clean_school_dataset(
        csv_text=csv_text,
        imputation_method=imputation_method,
        dedup_mode=dedup_mode,
    )
    ml_output = predict_dropout_risk(audit.cleaned_df, raw_csv_hint=csv_text)

    predictions = cast(list[dict[str, Any]], ml_output["predictions"])
    lineage_by_id: dict[str, dict[str, Any]] = {
        str(r["student_id"]): r for r in audit.row_lineages
    }
    enriched_students: list[dict[str, Any]] = [
        {**pred, "lineage": lineage_by_id.get(str(pred["student_id"]), {})}
        for pred in predictions
    ]

    export_cols = [
        "student_id",
        "student_name",
        "grade_level",
        "attendance_pct",
        "marks_pct",
        "gpa4",
        "letter_grade",
        "detected_scale",
        "fee_due",
        "fee_status",
        "attendance_imputed",
        "marks_imputed",
        "fee_imputed",
        "dropout_probability_pct",
        "risk_tier",
        "primary_driver",
    ]
    cleaned_csv = _rows_to_csv(predictions, export_cols)

    return cast(
        dict[str, Any],
        _json_safe(
            {
                "source_label": source_label,
                "cleaning_summary": audit.summary,
                "raw_preview": audit.raw_preview,
                "duplicates_removed": audit.duplicates_removed,
                "imputed_cells": audit.imputed_cells,
                "scale_conversions": audit.scale_conversions,
                "fee_fixes": audit.fee_fixes,
                "row_lineages": audit.row_lineages,
                "students": enriched_students,
                "cohort_analytics": ml_output["cohort_analytics"],
                "model_metrics": ml_output["model_metrics"],
                "feature_importances": ml_output["feature_importances"],
                "cleaned_csv": cleaned_csv,
            }
        ),
    )


@app.get("/", include_in_schema=False)
@app.get("/login", include_in_schema=False)
async def serve_login() -> FileResponse:
    login_path = STATIC_DIR / "login.html"
    if login_path.exists():
        return FileResponse(str(login_path))
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_path))


@app.get("/dashboard", include_in_schema=False)
@app.get("/main", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_path))


@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon() -> FileResponse:
    favicon_path = STATIC_DIR / "favicon.ico"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found.")
    return FileResponse(str(favicon_path), media_type="image/x-icon")


@app.get("/styles.css", include_in_schema=False)
@app.get("/dashboard/styles.css", include_in_schema=False)
async def serve_styles() -> FileResponse:
    css_path = STATIC_DIR / "styles.css"
    if not css_path.exists():
        raise HTTPException(status_code=404, detail="styles.css not found.")
    return FileResponse(str(css_path), media_type="text/css")


@app.get("/app.js", include_in_schema=False)
@app.get("/dashboard/app.js", include_in_schema=False)
async def serve_app_js() -> FileResponse:
    js_path = STATIC_DIR / "app.js"
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="app.js not found.")
    return FileResponse(str(js_path), media_type="application/javascript")


@app.get("/favicon.svg", include_in_schema=False)
@app.get("/dashboard/favicon.svg", include_in_schema=False)
async def serve_svg_favicon() -> FileResponse:
    svg_path = STATIC_DIR / "favicon.svg"
    if not svg_path.exists():
        raise HTTPException(status_code=404, detail="favicon.svg not found.")
    return FileResponse(str(svg_path), media_type="image/svg+xml")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_BUNDLE.cv_metrics,
        "feature_importances": MODEL_BUNDLE.feature_importances,
    }


@app.get("/api/template.csv")
async def download_template_csv() -> PlainTextResponse:
    return PlainTextResponse(
        content=CSV_TEMPLATE_HEADER,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="student_records_template.csv"'},
    )


@app.post("/api/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    imputation_method: str = Form("cohort_median"),
    dedup_mode: str = Form("smart"),
) -> dict[str, Any]:
    if imputation_method not in {"cohort_median", "global_median", "knn"}:
        raise HTTPException(status_code=400, detail="Invalid imputation_method.")
    if dedup_mode not in {"smart", "exact_id"}:
        raise HTTPException(status_code=400, detail="Invalid dedup_mode.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    try:
        csv_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        csv_text = raw_bytes.decode("latin-1")

    try:
        return _build_full_response(
            csv_text=csv_text,
            imputation_method=imputation_method,
            dedup_mode=dedup_mode,
            source_label=f"upload:{file.filename or 'custom.csv'}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse or clean CSV: {exc}") from exc


@app.post("/api/analyze/single")
async def analyze_single_student(payload: SingleStudentRequest) -> dict[str, Any]:
    """Clean and score a single manually entered student record (optionally using uploaded CSV cohort)."""
    sid_clean = payload.student_id.strip()
    sname_clean = payload.student_name.strip()
    grade_clean = payload.grade_level.strip()
    if not sid_clean or not sname_clean or not grade_clean:
        raise HTTPException(
            status_code=400,
            detail="student_id, student_name, and grade_level are required.",
        )

    single_row: dict[str, Any] = {
        "student_id": sid_clean,
        "student_name": sname_clean,
        "grade_level": grade_clean,
        "attendance": payload.attendance,
        "marks_grade": payload.marks_raw,
        "fee_due": payload.fee_due,
        "fee_status": payload.fee_status,
    }

    if payload.reference_csv and payload.reference_csv.strip():
        base_cols, raw_base_rows = parse_messy_csv_rows(payload.reference_csv)
        base_rows: list[dict[str, Any]] = [dict(r) for r in raw_base_rows]
        row_dict: dict[str, Any] = {col: "" for col in base_cols}
        col_targets: dict[str, list[str]] = {
            "student_id": ["student_id", "studentid", "id", "roll_no", "rollno", "roll", "admission_no", "reg_no"],
            "student_name": ["student_name", "name", "full_name", "student", "learner_name"],
            "grade_level": ["grade_level", "grade", "class", "standard", "cohort", "year", "section"],
            "attendance": ["attendance", "attendance_pct", "attendance_rate", "att_pct", "att", "present_pct"],
            "marks_grade": ["marks_grade", "marks_raw", "marks", "score", "academic_score", "grade_score", "gpa", "cgpa", "result", "percentage"],
            "fee_due": ["fee_due", "fees_due", "fee_balance", "pending_fee", "pending_fees", "fees_pending", "balance", "dues", "unpaid_fee"],
            "fee_status": ["fee_status", "payment_status", "fees_status", "status"],
        }
        for canonical, candidates in col_targets.items():
            for col in base_cols:
                if col.strip().lower().replace(" ", "_") in candidates:
                    row_dict[col] = single_row[canonical]
                    break
        combined_csv = _rows_to_csv([*base_rows, row_dict], base_cols)
    else:
        combined_csv = _rows_to_csv([single_row], list(single_row.keys()))

    audit = clean_school_dataset(
        csv_text=combined_csv,
        imputation_method=payload.imputation_method,
        dedup_mode="smart",
    )
    ml_output = predict_dropout_risk(audit.cleaned_df, raw_csv_hint=combined_csv)
    predictions = cast(list[dict[str, Any]], ml_output["predictions"])

    target_id = sid_clean.upper().replace(" ", "")
    matched_pred = next(
        (p for p in predictions if str(p["student_id"]) == target_id),
        None,
    )
    matched_lineage = next(
        (r for r in audit.row_lineages if str(r["student_id"]) == target_id),
        None,
    )
    dup_hit = next(
        (d for d in audit.duplicates_removed if str(d["student_id"]) == target_id),
        None,
    )

    return cast(
        dict[str, Any],
        _json_safe(
            {
                "student": matched_pred,
                "lineage": matched_lineage,
                "duplicate_merged": dup_hit,
                "imputation_method": payload.imputation_method,
            }
        ),
    )
