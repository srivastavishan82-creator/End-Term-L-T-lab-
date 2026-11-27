# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Python Data Cleaning Pipeline for school attendance, marks, and fee records.

Performs:
1. Flexible schema & semantic column alias normalization with value inspection
2. Safe duplicate detection & merging (exact IDs, near-IDs, backfill, protecting distinct students)
3. Heterogeneous grading scale parsing & standardization (CBSE A1-E2, College O-F, GPA 4.0, CGPA 10.0, fractions, decimals, 0-100%)
4. Robust attendance parsing (percentages, ratios, fractions, days attended, categorical)
5. Currency & fee status reconciliation (₹, $, Rs, lakhs, k, advance fees, conflict resolution)
6. Missing value imputation (Cohort Median, Global Median, or Scikit-Learn KNNImputer)
7. Cell-level audit logging, before/after diff tracking, and data health scoring
"""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

NULL_TOKENS = {
    "",
    "na",
    "n/a",
    "null",
    "none",
    "nil",
    "nan",
    "--",
    "-",
    "---",
    "missing",
    "unknown",
    "tbd",
    "?",
    "pending_entry",
    "undefined",
    "not available",
    "not_available",
    "empty",
    "void",
}

ZERO_FEE_TOKENS = {
    "",
    "0",
    "0.0",
    "0.00",
    "0/-",
    "nil",
    "none",
    "paid",
    "settled",
    "cleared",
    "waived",
    "zero",
    "no dues",
    "no due",
    "nodues",
    "no_dues",
    "clear",
    "free",
    "scholarship",
    "exempt",
    "fully paid",
    "full paid",
    "full",
    "nill",
}

# Standard letter grades + CBSE Indian school scales + University bands
LETTER_GRADE_MAP: dict[str, float] = {
    # Traditional US / International Letter Grades
    "A+": 96.0,
    "A": 92.0,
    "A-": 88.0,
    "B+": 84.0,
    "B": 80.0,
    "B-": 76.0,
    "C+": 72.0,
    "C": 67.0,
    "C-": 62.0,
    "D+": 57.0,
    "D": 52.0,
    "D-": 47.0,
    "E": 38.0,
    "F": 25.0,
    # Indian CBSE Secondary & Higher Secondary Grade Points
    "A1": 95.0,
    "A2": 85.0,
    "B1": 75.0,
    "B2": 65.0,
    "C1": 55.0,
    "C2": 45.0,
    "D1": 40.0,
    "D2": 35.0,
    "E1": 25.0,
    "E2": 20.0,
    # University / Board Classifications
    "O": 95.0,  # Outstanding
    "OUTSTANDING": 95.0,
    "EXCELLENT": 90.0,
    "DISTINCTION": 88.0,
    "VERY GOOD": 82.0,
    "GOOD": 72.0,
    "MERIT": 75.0,
    "ABOVE AVERAGE": 65.0,
    "PASS": 60.0,
    "PASSED": 60.0,
    "AVERAGE": 55.0,
    "SATISFACTORY": 55.0,
    "P": 50.0,
    "FAIL": 25.0,
    "FAILED": 25.0,
    "REAPPEAR": 25.0,
    "COMPARTMENT": 30.0,
    "ESSENTIAL REPEAT": 20.0,
}

COLUMN_ALIASES: dict[str, list[str]] = {
    "student_id": [
        "student_id",
        "studentid",
        "stud_id",
        "student_code",
        "sid",
        "id",
        "sr_no",
        "srno",
        "sl_no",
        "slno",
        "serial_no",
        "roll_no",
        "rollno",
        "roll_number",
        "roll",
        "admission_no",
        "admission_number",
        "adm_no",
        "admno",
        "reg_no",
        "regno",
        "registration_no",
        "enrollment_no",
        "enrollment_number",
        "enrolment_no",
        "enrollment",
        "scholar_no",
        "scholar_number",
        "learner_id",
        "student_registration_no",
        "uid",
        "urn",
    ],
    "student_name": [
        "student_name",
        "studentname",
        "stud_name",
        "name_of_student",
        "name_of_the_student",
        "name",
        "full_name",
        "fullname",
        "student",
        "learner_name",
        "learner",
        "candidate_name",
        "candidate",
        "pupil_name",
        "pupil",
        "first_name",
        "first_and_last_name",
    ],
    "grade_level": [
        "grade_level",
        "gradelevel",
        "grade",
        "class",
        "class_name",
        "class_level",
        "class_sec",
        "class_section",
        "std",
        "standard",
        "cohort",
        "year",
        "section",
        "sec",
        "semester",
        "sem",
        "batch",
        "stream",
        "branch",
        "course",
    ],
    "attendance": [
        "attendance",
        "attendence",
        "attendance_pct",
        "attendance_percentage",
        "attendance_percent",
        "attendance_rate",
        "att_pct",
        "att_percent",
        "att",
        "total_attendance",
        "present_pct",
        "present_percentage",
        "present_days",
        "days_attended",
        "classes_attended",
        "total_present",
        "attendance_days",
        "attendance_score",
        "attendance_ratio",
    ],
    "marks_raw": [
        "marks_grade",
        "marks_raw",
        "marks",
        "mark",
        "total_marks",
        "marks_obtained",
        "marks_secured",
        "obtained_marks",
        "marks_pct",
        "score",
        "scores",
        "exam_score",
        "test_score",
        "final_score",
        "academic_score",
        "academic_performance",
        "grade_score",
        "final_grade",
        "overall_grade",
        "overall_score",
        "gpa",
        "cgpa",
        "sgpa",
        "result",
        "percentage",
        "percent",
        "average_marks",
        "exam_percentage",
        "academic_marks",
        "marks_percent",
        "grade_point_average",
    ],
    "fee_due": [
        "fee_due",
        "fees_due",
        "fee_dues",
        "fees_dues",
        "fee_balance",
        "fees_balance",
        "balance_fee",
        "pending_fee",
        "pending_fees",
        "fees_pending",
        "fee_pending",
        "pending_amount",
        "due_amount",
        "amount_due",
        "total_due",
        "outstanding_fee",
        "outstanding_fees",
        "outstanding_balance",
        "balance_due",
        "balance",
        "dues",
        "pending_dues",
        "fee",
        "fees",
        "unpaid_fee",
        "unpaid_fees",
        "tuition_fee",
        "tuition_due",
        "fee_amount",
    ],
    "fee_status": [
        "fee_status",
        "fees_status",
        "payment_status",
        "payment_state",
        "fee_payment_status",
        "fee_state",
        "fee_clearance",
        "dues_status",
        "paid_status",
        "status",
        "fee_flag",
    ],
}


def _clean_col_key(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def is_null_like(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    s = str(val).strip().lower()
    clean_s = re.sub(r"^[\s\-.,_?#]+$", "", s)
    return clean_s in NULL_TOKENS or s in NULL_TOKENS


ROMAN_TO_NUM: dict[str, str] = {
    "I": "1",
    "II": "2",
    "III": "3",
    "IV": "4",
    "V": "5",
    "VI": "6",
    "VII": "7",
    "VIII": "8",
    "IX": "9",
    "X": "10",
    "XI": "11",
    "XII": "12",
}


def normalize_grade_level(val: Any) -> str:
    if is_null_like(val):
        return "Grade 10"  # Sensible institutional cohort default
    s = re.sub(r"\s+", " ", str(val).strip())

    # Check for Roman Numerals (e.g. Class X-A, Std XII, Grade IX)
    m_roman = re.match(r"^(?:grade|class|std|standard|year)?\s*(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\s*(?:-|\s)?\s*([a-z]?)$", s, re.IGNORECASE)
    if m_roman:
        r_num = ROMAN_TO_NUM.get(m_roman.group(1).upper(), m_roman.group(1).upper())
        sec = m_roman.group(2).upper()
        return f"Grade {r_num}{('-' + sec) if sec else ''}"

    # Check for standard digits (e.g. Grade 10, Class 9B, 11th)
    m = re.match(r"^(?:grade|class|std|standard|year)?\s*(\d{1,2})(?:st|nd|rd|th)?\s*(?:-|\s)?\s*([a-z]?)$", s, re.IGNORECASE)
    if m:
        num = m.group(1)
        sec = m.group(2).upper()
        return f"Grade {num}{('-' + sec) if sec else ''}"
    return s.title()


def normalize_name(val: Any) -> str:
    if is_null_like(val):
        return "Unknown Student"
    cleaned = re.sub(r"[^a-zA-Z\s.'-]", "", str(val).strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned else "Unknown Student"


def normalize_student_id(val: Any, fallback_idx: int) -> str:
    if is_null_like(val):
        return f"AUTO-{fallback_idx + 1:03d}"
    s = str(val).strip().upper()
    cleaned = re.sub(r"\s+", "", s)
    return cleaned if cleaned else f"AUTO-{fallback_idx + 1:03d}"


def parse_attendance(
    val: Any,
    column_scale: str = "PERCENTAGE_100",
) -> tuple[float | None, str | None]:
    """Parse raw attendance into 0-100 percentage and audit note."""
    if is_null_like(val):
        return None, None

    raw = str(val).strip()
    lower = raw.lower()

    # Categorical attendance descriptors
    cat_map: dict[str, float] = {
        "present": 100.0,
        "full": 100.0,
        "regular": 95.0,
        "very good": 90.0,
        "good": 85.0,
        "average": 75.0,
        "moderate": 75.0,
        "normal": 75.0,
        "poor": 50.0,
        "low": 45.0,
        "irregular": 40.0,
        "critical": 35.0,
    }
    if lower in cat_map:
        pct = cat_map[lower]
        return pct, f"Categorical attendance '{raw}' -> {pct:.1f}%"

    # Fraction e.g. 42/50, 160/200, 175 out of 200
    frac = re.match(r"^(-?\d+(?:\.\d+)?)\s*(?:/|OUT\s+OF)\s*(\d+(?:\.\d+)?)(?:\s*days)?$", raw, re.IGNORECASE)
    if frac:
        num, den = float(frac.group(1)), float(frac.group(2))
        if den > 0 and 0 <= num <= den * 1.25:
            pct = round(min(100.0, max(0.0, (num / den) * 100.0)), 1)
            return pct, f"Parsed fractional attendance {raw} -> {pct}%"

    cleaned = re.sub(r"[%\s,]", "", lower).replace("pct", "").replace("days", "")
    try:
        num = float(cleaned)
    except ValueError:
        return None, f"Unrecognized attendance token '{raw}' marked missing"

    if not math.isfinite(num):
        return None, f"Non-finite attendance '{raw}' marked missing"
    if num < 0:
        return None, f"Negative attendance '{raw}' invalidated for imputation"

    # Ratio e.g. 0.84 -> 84%
    if 0 < num < 1.0:
        pct = round(num * 100.0, 1)
        return pct, f"Scaled ratio {raw} -> {pct}%"

    # Standard 0-100%
    if num <= 100.0:
        pct = round(num, 1)
        note = f"Stripped formatting '{raw}' -> {pct}%" if (raw != str(pct) and "%" in raw) else None
        return pct, note

    # Days attended out of standard school calendar (e.g. 160-220 days)
    if 100.0 < num <= 240.0:
        assumed_den = 200.0
        pct = round(min(100.0, (num / assumed_den) * 100.0), 1)
        return pct, f"Days attended '{raw}' -> {pct:.1f}% (relative to {int(assumed_den)} school days)"

    return 100.0, f"Clamped out-of-range attendance {raw} -> 100.0%"


def parse_grading_scale(
    val: Any,
    column_scale: str = "PERCENTAGE_100",
) -> tuple[float | None, str, str | None]:
    """Normalize heterogeneous grading formats into a standardized 0-100 percentage score."""
    if is_null_like(val):
        return None, "Missing", None

    raw = str(val).strip()
    upper = re.sub(r"\s+", " ", raw.upper())

    if upper in {"ABSENT", "AB", "INC", "INCOMPLETE", "PENDING", "WITHHELD"}:
        return None, "Absent/Incomplete", f"Flagged '{raw}' as missing score for imputation"

    # Strip prefixes like "Grade: A+", "Marks: 85", "Score: 78"
    upper_stripped = re.sub(r"^(?:GRADE|MARKS?|SCORE|BAND)[:\s]+", "", upper).strip()

    # Letter Grade Map
    if upper_stripped in LETTER_GRADE_MAP:
        pct = LETTER_GRADE_MAP[upper_stripped]
        return pct, "Letter Grade", f"Letter Grade '{raw}' -> {pct:.1f}%"

    # Explicit Fraction e.g. 39/50, 7.5/10, 3.2/4.0, 18 out of 25
    frac = re.match(r"^(-?\d+(?:\.\d+)?)\s*(?:/|OUT\s+OF)\s*(\d+(?:\.\d+)?)$", upper)
    if frac:
        num = float(frac.group(1))
        den = float(frac.group(2))
        if den <= 0 or num < 0:
            return None, "Invalid Fraction", f"Invalid fraction '{raw}' marked missing"
        pct = round(min(100.0, max(0.0, (num / den) * 100.0)), 1)
        if abs(den - 4.0) < 1e-6:
            scale = "4.0 GPA Scale"
        elif abs(den - 10.0) < 1e-6:
            scale = "10.0 CGPA Scale"
        elif abs(den - 100.0) < 1e-6:
            scale = "Raw /100"
        else:
            scale = f"Fraction (/{int(den) if den.is_integer() else den})"
        return pct, scale, f"{scale} '{raw}' -> {pct:.1f}%"

    # Explicit CGPA keyword (e.g. 8.4 CGPA, CGPA 8.4)
    if "CGPA" in upper or "SGPA" in upper:
        m = re.search(r"(\d+(?:\.\d+)?)", upper)
        if m:
            num = float(m.group(1))
            if 0 <= num <= 10.0:
                pct = round(min(100.0, num * 10.0), 1)
                return pct, "10.0 CGPA Scale", f"10-pt CGPA '{raw}' -> {pct:.1f}%"

    # Explicit GPA keyword (e.g. 3.4 GPA, GPA 3.4)
    if "GPA" in upper:
        m = re.search(r"(\d+(?:\.\d+)?)", upper)
        if m:
            num = float(m.group(1))
            if 0 <= num <= 4.3:
                pct = round(min(100.0, (num / 4.0) * 100.0), 1)
                return pct, "4.0 GPA Scale", f"4.0 GPA '{raw}' -> {pct:.1f}%"

    # Explicit Percentage indicator (e.g. 84%, 84.5 %)
    if "%" in upper or "PCT" in upper or "PERCENT" in upper:
        cleaned = re.sub(r"[%\sA-Z]", "", upper)
        try:
            num = float(cleaned)
            if math.isfinite(num):
                pct = round(min(100.0, max(0.0, num)), 1)
                return pct, "Percentage (0-100)", None
        except ValueError:
            pass

    # Pure numeric score
    try:
        num = float(upper_stripped.replace(",", ""))
    except ValueError:
        return None, "Unrecognized", f"Unparseable grade '{raw}' marked missing"

    if not math.isfinite(num):
        return None, "Invalid", f"Non-finite mark '{raw}' marked missing"
    if num < 0:
        return None, "Invalid Negative", f"Negative mark '{raw}' marked missing"

    # Ratio 0-1 (e.g. 0.85 -> 85%)
    if 0 < num < 1.0:
        pct = round(num * 100.0, 1)
        return pct, "Decimal Ratio (0-1)", f"Ratio '{raw}' -> {pct:.1f}%"

    # Evaluate based on column scale context
    if column_scale == "4.0_GPA" and num <= 4.3:
        pct = round(min(100.0, (num / 4.0) * 100.0), 1)
        return pct, "4.0 GPA Scale", f"4.0 GPA '{raw}' -> {pct:.1f}%"

    if column_scale == "10.0_CGPA" and num <= 10.0:
        pct = round(min(100.0, num * 10.0), 1)
        return pct, "10.0 CGPA Scale", f"10-pt CGPA '{raw}' -> {pct:.1f}%"

    if column_scale == "OUT_OF_500" and num > 100.0:
        pct = round(min(100.0, (num / 500.0) * 100.0), 1)
        return pct, "Total /500", f"Total /500 '{raw}' -> {pct:.1f}%"

    if column_scale == "OUT_OF_1000" and num > 100.0:
        pct = round(min(100.0, (num / 1000.0) * 100.0), 1)
        return pct, "Total /1000", f"Total /1000 '{raw}' -> {pct:.1f}%"

    # Standard 0-100% scale
    if num <= 100.0:
        pct = round(num, 1)
        return pct, "Percentage (0-100)", None

    # Fallback for values > 100
    if 100.0 < num <= 200.0:
        pct = round((num / 200.0) * 100.0, 1)
        return pct, "Raw /200", f"Inferred /200 total '{raw}' -> {pct:.1f}%"
    if 200.0 < num <= 500.0:
        pct = round((num / 500.0) * 100.0, 1)
        return pct, "Raw /500", f"Inferred /500 total '{raw}' -> {pct:.1f}%"

    return 100.0, "Outlier Clamped", f"Clamped score '{raw}' -> 100.0%"


def pct_to_letter(pct: float) -> str:
    if pct >= 93:
        return "A+"
    if pct >= 89:
        return "A"
    if pct >= 85:
        return "A-"
    if pct >= 81:
        return "B+"
    if pct >= 77:
        return "B"
    if pct >= 73:
        return "B-"
    if pct >= 69:
        return "C+"
    if pct >= 64:
        return "C"
    if pct >= 60:
        return "C-"
    if pct >= 55:
        return "D+"
    if pct >= 50:
        return "D"
    return "F"


def pct_to_gpa4(pct: float) -> float:
    return round(max(0.0, min(4.0, (pct / 100.0) * 4.0)), 2)


def parse_fee_due(val: Any) -> tuple[float | None, str | None]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None, None

    raw = str(val).strip()
    lower = raw.lower()

    if lower in ZERO_FEE_TOKENS:
        return 0.0, f"Normalized fee token '{raw}' -> 0"
    if lower in NULL_TOKENS:
        return None, None

    # Handle negative / advance fee e.g. -500, (500)
    if re.match(r"^\(?-\d+(?:\.\d+)?\)?$", lower) or re.match(r"^\(\d+(?:\.\d+)?\)$", lower):
        return 0.0, f"Advance fee credit '{raw}' normalized to balance 0"

    # Multipliers: e.g. 15k -> 15000, 1.5L / 1.5 Lakh -> 150000
    m_k = re.match(r"^(?:rs\.?|inr|[₹$€£])?\s*(\d+(?:\.\d+)?)\s*k$", lower)
    if m_k:
        amount = float(m_k.group(1)) * 1000.0
        return round(amount, 2), f"Parsed multiplier '{raw}' -> {round(amount):,}"

    m_lakh = re.match(r"^(?:rs\.?|inr|[₹$€£])?\s*(\d+(?:\.\d+)?)\s*(?:l|lakh|lac)s?$", lower)
    if m_lakh:
        amount = float(m_lakh.group(1)) * 100000.0
        return round(amount, 2), f"Parsed Lakh multiplier '{raw}' -> {round(amount):,}"

    cleaned = re.sub(r"(rs\.?|inr|usd|eur|gbp|/-|[₹$€£,\s])", "", lower)
    try:
        amount = float(cleaned)
    except ValueError:
        return None, f"Unrecognized fee token '{raw}' marked missing"

    if not math.isfinite(amount):
        return None, f"Non-finite fee '{raw}' marked missing"
    if amount < 0:
        return 0.0, f"Negative fee '{raw}' clamped to 0"

    had_formatting = any(sym in lower for sym in ("₹", "$", "€", "rs", "inr", "usd", "/-", ","))
    note = f"Parsed currency '{raw}' -> {round(amount):,}" if had_formatting else None
    return round(amount, 2), note


def reconcile_fee_status(raw_status: Any, fee_due: float | None) -> tuple[str, str | None]:
    status_str = "" if is_null_like(raw_status) else str(raw_status).strip().title()

    if fee_due is None or math.isnan(fee_due):
        return (status_str if status_str else "Pending"), None

    if fee_due <= 0:
        if status_str in {"Waived", "Scholarship", "Exempt"}:
            return "Waived", None
        if status_str in {"Advance", "Advance Paid", "Credit"}:
            return "Advance Paid", None
        if status_str and status_str not in {"Paid", "Settled", "Cleared", "No Dues"}:
            return "Paid", f"Reconciled status '{status_str}' -> 'Paid' (balance is 0)"
        return "Paid", ("Inferred 'Paid' from zero balance" if not status_str else None)

    # Balance is strictly > 0
    if status_str == "Paid":
        return "Overdue", f"Conflict fixed: balance {round(fee_due):,} > 0 but marked 'Paid' -> 'Overdue'"
    if status_str in {"Partial", "Partially Paid", "Installment", "Half Paid"}:
        return "Partial", None
    if status_str in {"Unpaid", "Due", "Pending Dues"}:
        return "Overdue", f"Standardized '{status_str}' -> 'Overdue'"
    if not status_str:
        inferred = "Overdue" if fee_due >= 10000 else "Pending"
        return inferred, f"Inferred '{inferred}' from fee balance {round(fee_due):,}"
    return status_str, None


@dataclass
class CleaningAuditResult:
    cleaned_df: pd.DataFrame
    raw_preview: list[dict[str, Any]]
    duplicates_removed: list[dict[str, Any]]
    imputed_cells: list[dict[str, Any]]
    scale_conversions: list[dict[str, Any]]
    fee_fixes: list[dict[str, Any]]
    row_lineages: list[dict[str, Any]]
    summary: dict[str, Any]


def _infer_column_role(col_name: str, sample_vals: list[str], unmapped_roles: list[str]) -> str | None:
    """Infer the canonical schema role of an unknown column header using regex and value heuristics."""
    clean_k = _clean_col_key(col_name)

    # 1. Header substring heuristics
    if "student_id" in unmapped_roles and any(x in clean_k for x in ("student_id", "roll", "sr_no", "srno", "adm", "reg", "enroll", "scholar", "uid")):
        return "student_id"
    if "student_name" in unmapped_roles and any(x in clean_k for x in ("name", "student", "learner", "candidate", "pupil")):
        return "student_name"
    if "grade_level" in unmapped_roles and any(x in clean_k for x in ("grade", "class", "std", "standard", "cohort", "year", "stream", "sem", "section")):
        return "grade_level"
    if "attendance" in unmapped_roles and any(x in clean_k for x in ("attend", "present", "absence")):
        return "attendance"
    if "marks_raw" in unmapped_roles and any(x in clean_k for x in ("mark", "score", "percent", "gpa", "cgpa", "sgpa", "result", "exam", "eval")):
        return "marks_raw"
    if "fee_status" in unmapped_roles and any(x in clean_k for x in ("status", "state", "clearance")):
        return "fee_status"
    if "fee_due" in unmapped_roles and any(x in clean_k for x in ("fee", "dues", "balance", "amount", "tuition", "outstanding", "payable")):
        return "fee_due"

    # 2. Corroborate with non-empty cell sample values
    non_empty = [v.strip() for v in sample_vals if v and not is_null_like(v)][:10]
    if not non_empty:
        return None

    # Fee status values
    if "fee_status" in unmapped_roles and any(v.lower() in ("paid", "pending", "overdue", "unpaid", "cleared", "waived") for v in non_empty):
        return "fee_status"

    # Fee currency values (₹, $, Rs, numbers with commas > 100)
    if "fee_due" in unmapped_roles and any(any(s in v.lower() for s in ("₹", "$", "rs", "inr", "/-")) for v in non_empty):
        return "fee_due"

    # Attendance percentages or marks
    has_pct = any("%" in v for v in non_empty)
    if has_pct:
        if "attendance" in unmapped_roles and "marks_raw" not in unmapped_roles:
            return "attendance"
        if "marks_raw" in unmapped_roles and "attendance" not in unmapped_roles:
            return "marks_raw"

    return None


def _find_subject_columns(headers: list[str]) -> list[str]:
    """Find all subject or evaluation score columns in the CSV headers."""
    known_subject_words = {
        "math", "maths", "mathematics", "science", "physics", "chem", "chemistry",
        "bio", "biology", "english", "history", "geography", "social", "social_studies",
        "social_science", "sst", "hindi", "computer", "computer_science", "cs", "it",
        "lang", "language", "economics", "accounts", "accountancy", "business",
        "algebra", "geometry", "calculus", "literature", "reading", "writing",
    }
    non_subject = {
        "student_id", "student_name", "grade_level", "attendance", "fee_due", "fee_status",
        "days_present", "total_school_days", "dropout_label", "dropout", "target", "label",
        "sr_no", "roll_no", "id", "name", "grade", "class", "status", "fee", "fees"
    }
    found: list[str] = []
    for h in headers:
        ck = _clean_col_key(h)
        if ck in non_subject:
            continue
        if any(w == ck or ck.startswith(w + "_") or ck.endswith("_" + w) for w in known_subject_words) or re.match(r"^sub(?:ject)?_?\d+$", ck) or re.match(r"^course_?\d+$", ck) or re.search(r"_(?:marks?|score|grade|pct|percent)$", ck) and not any(x in ck for x in ("fee", "att", "class", "std")):
            found.append(h)
    return found


def _map_columns(
    headers: list[str],
    raw_records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Map messy CSV column names to canonical schema names with multi-pass matching."""
    rename_map: dict[str, str] = {}
    cleaned_headers: dict[str, str] = {_clean_col_key(c): c for c in headers}
    mapped_canonicals: set[str] = set()

    # Pass 1: Canonical or exact alias match
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in headers:
            rename_map[canonical] = canonical
            mapped_canonicals.add(canonical)
            continue
        for alias in aliases:
            if alias in cleaned_headers:
                orig = cleaned_headers[alias]
                if orig not in rename_map:
                    rename_map[orig] = canonical
                    mapped_canonicals.add(canonical)
                break

    # Pass 2: Semantic substring & sample value inference for unmapped fields
    required_cols = [
        "student_id",
        "student_name",
        "grade_level",
        "attendance",
        "marks_raw",
        "fee_due",
        "fee_status",
    ]
    unmapped_roles = [c for c in required_cols if c not in mapped_canonicals]

    for h in headers:
        if h in rename_map:
            continue
        sample_vals = [r.get(h, "") for r in raw_records[:15]]
        inferred = _infer_column_role(h, sample_vals, unmapped_roles)
        if inferred and inferred in unmapped_roles and inferred not in mapped_canonicals:
            rename_map[h] = inferred
            mapped_canonicals.add(inferred)
            unmapped_roles.remove(inferred)

    # Pass 3: Assemble mapped records and backfill missing required columns
    mapped_records: list[dict[str, str]] = []
    for pos, rec in enumerate(raw_records):
        mapped_row: dict[str, str] = {
            rename_map.get(col, col): val for col, val in rec.items()
        }
        for req in required_cols:
            if req not in mapped_row:
                mapped_row[req] = ""
        # Auto-fill missing ID/name if not present in the CSV
        if not mapped_row["student_id"]:
            mapped_row["student_id"] = f"STU-{pos + 1:03d}"
        if not mapped_row["student_name"]:
            mapped_row["student_name"] = f"Student {pos + 1}"
        mapped_records.append(mapped_row)

    return mapped_records, rename_map


def _detect_marks_column_scale(raw_marks: list[str], header_name: str) -> str:
    """Detect whether the marks column scale is 4.0 GPA, 10.0 CGPA, 0-100%, or large total."""
    h_lower = header_name.lower()
    if "gpa" in h_lower and "cgpa" not in h_lower and "sgpa" not in h_lower:
        return "4.0_GPA"
    if "cgpa" in h_lower or "sgpa" in h_lower:
        return "10.0_CGPA"

    nums: list[float] = []
    for m in raw_marks:
        if is_null_like(m) or "%" in m or "/" in m:
            continue
        cleaned = re.sub(r"[^\d.]", "", m)
        try:
            val = float(cleaned)
            if math.isfinite(val):
                nums.append(val)
        except ValueError:
            pass

    if not nums:
        return "PERCENTAGE_100"

    max_val = max(nums)
    if max_val <= 4.3 and len([x for x in nums if x <= 4.0]) / len(nums) >= 0.75:
        return "4.0_GPA"
    if max_val <= 10.0 and len([x for x in nums if x <= 10.0]) / len(nums) >= 0.8:
        return "10.0_CGPA"
    if max_val > 100.0:
        if max_val <= 500.0:
            return "OUT_OF_500"
        if max_val <= 1000.0:
            return "OUT_OF_1000"

    return "PERCENTAGE_100"


def parse_messy_csv_rows(csv_text: str) -> tuple[list[str], list[dict[str, str]]]:
    """Parse messy CSV text cleanly into headers and row dictionaries."""
    cleaned_text = csv_text.lstrip("\ufeff").strip()
    if not cleaned_text:
        raise ValueError("Uploaded CSV contains no data.")

    lines = [ln for ln in cleaned_text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Uploaded CSV contains no data.")

    # Determine delimiter
    sample_text = "\n".join(lines[:10])
    delimiter = ","
    try:
        sniffed = csv.Sniffer().sniff(sample_text, delimiters=";,|\t,")
        delimiter = sniffed.delimiter
    except (csv.Error, ValueError, IndexError, TypeError):
        header_candidate = lines[0]
        counts = {d: header_candidate.count(d) for d in (",", ";", "\t", "|")}
        best = max(counts, key=lambda d: counts[d])
        delimiter = best if counts[best] > 0 else ","

    all_parsed_rows = [
        row
        for row in csv.reader(io.StringIO(cleaned_text), delimiter=delimiter)
        if any(cell.strip() for cell in row)
    ]
    if len(all_parsed_rows) < 2:
        raise ValueError("Uploaded CSV contains no data rows.")

    # Detect true header row (skip report preamble/title lines if present)
    header_idx = 0
    school_keywords = {"student", "id", "name", "roll", "grade", "class", "att", "mark", "score", "fee", "status", "result", "exam"}
    for idx, r in enumerate(all_parsed_rows[:5]):
        row_str = " ".join(r).lower()
        matches = sum(1 for kw in school_keywords if kw in row_str)
        if matches >= 2 or (len(r) >= 3 and idx == 0):
            header_idx = idx
            break

    raw_headers = [h.strip() or f"col_{i + 1}" for i, h in enumerate(all_parsed_rows[header_idx])]
    seen_headers: dict[str, int] = {}
    headers: list[str] = []
    for h in raw_headers:
        if h in seen_headers:
            seen_headers[h] += 1
            headers.append(f"{h}_{seen_headers[h]}")
        else:
            seen_headers[h] = 1
            headers.append(h)

    width = len(headers)
    records: list[dict[str, str]] = []
    for row in all_parsed_rows[header_idx + 1:]:
        curr_row = list(row)
        # Heal unquoted comma splits in numbers (e.g. '12' and '000' or '₹12' and '500')
        while len(curr_row) > width:
            healed = False
            for j in range(len(curr_row) - 1):
                part1 = curr_row[j].strip()
                part2 = curr_row[j + 1].strip()
                # Check if part1 is currency/prefix+digits and part2 is 3-digit thousand chunk or 2-digit decimal
                if re.search(r"\d+$", part1) and re.match(r"^\d{3}$", part2):
                    curr_row[j] = f"{part1}{part2}"
                    curr_row.pop(j + 1)
                    healed = True
                    break
            if not healed:
                curr_row = curr_row[:width]
                break

        if len(curr_row) < width:
            padded = curr_row + [""] * (width - len(curr_row))
        else:
            padded = curr_row[:width]
        records.append({headers[i]: padded[i] for i in range(width)})

    return headers, records


def read_messy_csv(csv_text: str) -> pd.DataFrame:
    """Parse messy CSV text cleanly into a string DataFrame."""
    headers, records = parse_messy_csv_rows(csv_text)
    return pd.DataFrame(records, columns=headers, dtype=str)


def clean_school_dataset(
    csv_text: str,
    imputation_method: str = "cohort_median",
    dedup_mode: str = "smart",
) -> CleaningAuditResult:
    """Execute the full Python cleaning pipeline on messy school CSV records."""
    headers, raw_records = parse_messy_csv_rows(csv_text)
    if not raw_records:
        raise ValueError("Uploaded CSV contains no data rows.")

    mapped_records, column_mapping = _map_columns(headers, raw_records)

    raw_row_count = len(mapped_records)
    raw_missing_attendance = sum(1 for r in mapped_records if is_null_like(r.get("attendance")))
    raw_missing_marks = sum(1 for r in mapped_records if is_null_like(r.get("marks_raw")))
    raw_missing_fees = sum(1 for r in mapped_records if is_null_like(r.get("fee_due")))
    raw_missing_total = raw_missing_attendance + raw_missing_marks + raw_missing_fees

    raw_preview: list[dict[str, Any]] = [dict(r) for r in raw_records[:50]]

    # Detect all subject columns and attendance log columns in raw headers
    subject_cols = _find_subject_columns(headers)
    col_days_present = next((h for h in headers if _clean_col_key(h) in (
        "days_present", "present_days", "days_attended", "attended_days",
        "classes_attended", "total_present", "attended", "present"
    )), None)

    col_total_days = next((h for h in headers if _clean_col_key(h) in (
        "total_school_days", "school_days", "total_days", "total_classes",
        "working_days", "total_working_days", "max_days"
    )), None)

    cohort_max_days = 220.0
    if col_total_days:
        td_nums: list[float] = []
        for r in raw_records:
            val_clean = re.sub(r"[^\d.]", "", r.get(col_total_days, ""))
            if val_clean:
                try:
                    td_nums.append(float(val_clean))
                except ValueError:
                    pass
        if td_nums:
            cohort_max_days = float(statistics.median(td_nums))
    elif col_days_present:
        dp_nums: list[float] = []
        for r in raw_records:
            val_clean = re.sub(r"[^\d.]", "", r.get(col_days_present, ""))
            if val_clean:
                try:
                    dp_nums.append(float(val_clean))
                except ValueError:
                    pass
        if dp_nums and max(dp_nums) > 100.0:
            cohort_max_days = float(max(dp_nums))

    # Detect global column scale for marks to prevent misinterpreting failing scores
    marks_col_header = next((k for k, v in column_mapping.items() if v == "marks_raw"), "marks_raw")
    marks_scale_mode = _detect_marks_column_scale(
        [r.get("marks_raw", "") for r in mapped_records],
        marks_col_header,
    )

    # Compute genuine missing counts
    raw_missing_attendance = 0
    raw_missing_marks = 0
    raw_missing_fees = 0

    for pos, r in enumerate(mapped_records):
        raw_snap = raw_records[pos]
        has_att = not is_null_like(r.get("attendance"))
        has_dp = bool(col_days_present and not is_null_like(raw_snap.get(col_days_present, "")))
        if not (has_att or has_dp):
            raw_missing_attendance += 1

        has_direct_mark = not is_null_like(r.get("marks_raw"))
        has_subj = any(not is_null_like(raw_snap.get(sc, "")) for sc in subject_cols)
        if not (has_direct_mark or has_subj):
            raw_missing_marks += 1

        st_clean = str(r.get("fee_status", "")).strip().title()
        is_paid = st_clean in {"Paid", "Cleared", "Waived", "Settled", "No Dues"}
        if is_null_like(r.get("fee_due")) and not is_paid:
            raw_missing_fees += 1

    raw_missing_total = raw_missing_attendance + raw_missing_marks + raw_missing_fees

    # Stage 1: Parse fields & compute completeness before deduplication
    staged_rows: list[dict[str, Any]] = []
    scale_types_found: dict[str, int] = {}

    for pos, row in enumerate(mapped_records):
        raw_snapshot = raw_records[pos]
        sid = normalize_student_id(row.get("student_id"), pos)
        sname = normalize_name(row.get("student_name"))
        grade = normalize_grade_level(row.get("grade_level"))

        # 1. Attendance: Check attendance log columns (days_present / total_school_days) first, or parse direct attendance
        att_val = None
        att_note = None

        if col_days_present:
            dp_str = str(raw_snapshot.get(col_days_present, "")).strip()
            if not is_null_like(dp_str):
                if "/" in dp_str or "out of" in dp_str.lower() or "%" in dp_str:
                    att_val, att_note = parse_attendance(dp_str)
                else:
                    try:
                        num_days = float(re.sub(r"[^\d.]", "", dp_str))
                        den_days = cohort_max_days
                        if col_total_days:
                            td_str = raw_snapshot.get(col_total_days, "")
                            if not is_null_like(td_str):
                                den_days = float(re.sub(r"[^\d.]", "", td_str))
                        if den_days > 0 and 0 <= num_days <= den_days * 1.25:
                            att_val = round(min(100.0, max(0.0, (num_days / den_days) * 100.0)), 1)
                            att_note = f"Calculated from attendance log ({int(num_days)}/{int(den_days)} days) -> {att_val}%"
                    except (ValueError, TypeError):
                        pass

        if att_val is None:
            att_val, att_note = parse_attendance(row.get("attendance"))

        # 2. Marks: Parse direct marks or aggregate all subject columns
        direct_raw_mark = row.get("marks_raw", "")
        marks_val, scale_label, marks_note = parse_grading_scale(
            direct_raw_mark,
            column_scale=marks_scale_mode,
        )

        if (marks_val is None or scale_label in {"Missing", "Absent/Incomplete"}) and subject_cols:
            subject_scores: list[float] = []
            subject_scales: list[str] = []
            for sc in subject_cols:
                sc_val = raw_snapshot.get(sc, "")
                if not is_null_like(sc_val):
                    sc_pct, sc_scale, _ = parse_grading_scale(sc_val, column_scale=marks_scale_mode)
                    if sc_pct is not None:
                        subject_scores.append(sc_pct)
                        subject_scales.append(sc_scale)

            if subject_scores:
                marks_val = round(float(statistics.mean(subject_scores)), 1)
                scale_label = subject_scales[0] if len(set(subject_scales)) == 1 else "Subject Composite"
                marks_note = f"Averaged {len(subject_scores)} subject scores -> {marks_val}%"

        # 3. Fees: Reconcile verified Paid status with missing fee_due (never impute debt on paid students)
        fee_raw = row.get("fee_due", "")
        fee_val, fee_note = parse_fee_due(fee_raw)
        raw_status = row.get("fee_status", "")
        status_clean = str(raw_status).strip().title()

        is_paid = status_clean in {"Paid", "Cleared", "Waived", "Settled", "No Dues", "Advance", "Advance Paid"}
        if is_paid and (fee_val is None or is_null_like(fee_raw)):
            fee_val = 0.0
            fee_note = f"Verified zero balance from verified '{status_clean}' status"

        fee_status, status_note = reconcile_fee_status(raw_status, fee_val)

        scale_types_found[scale_label] = scale_types_found.get(scale_label, 0) + 1

        completeness = (
            int(att_val is not None)
            + int(marks_val is not None)
            + int(fee_val is not None)
            + int(not is_null_like(row.get("fee_status")))
        )

        staged_rows.append(
            {
                "_orig_row": pos + 1,
                "_raw": raw_snapshot,
                "_raw_id": str(row.get("student_id", "")),
                "_raw_name": str(row.get("student_name", "")),
                "_raw_grade": str(row.get("grade_level", "")),
                "_raw_attendance": str(row.get("attendance", "")),
                "_raw_marks": str(row.get("marks_raw", "")),
                "_raw_fee_due": str(row.get("fee_due", "")),
                "_raw_fee_status": str(row.get("fee_status", "")),
                "student_id": sid,
                "student_name": sname,
                "grade_level": grade,
                "attendance_pct": att_val,
                "marks_pct": marks_val,
                "detected_scale": scale_label,
                "fee_due": fee_val,
                "fee_status": fee_status,
                "_att_note": att_note,
                "_marks_note": marks_note,
                "_fee_note": fee_note,
                "_status_note": status_note,
                "_completeness": completeness,
                "_norm_name_key": re.sub(r"[^a-z0-9]", "", sname.lower()),
                "_clean_id_key": re.sub(r"[^A-Z0-9]", "", sid),
            }
        )

    # Stage 2: Duplicate detection & removal (smart + safe: NEVER drop distinct student IDs)
    duplicates_removed: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    seen_ids: dict[str, int] = {}
    seen_name_grade: dict[tuple[str, str], int] = {}
    merged_counts: dict[int, int] = {}

    order = sorted(
        range(len(staged_rows)),
        key=lambda idx: (-int(staged_rows[idx]["_completeness"]), int(staged_rows[idx]["_orig_row"])),
    )

    for i in order:
        rec = staged_rows[i]
        id_key = str(rec["_clean_id_key"])
        ng_key = (str(rec["_norm_name_key"]), str(rec["grade_level"]))

        dup_target: int | None = None
        dup_reason = ""

        if id_key in seen_ids:
            dup_target = seen_ids[id_key]
            keeper = staged_rows[dup_target]
            same_raw = rec["_raw"] == keeper["_raw"]
            dup_reason = (
                f"Exact duplicate row of {rec['student_id']} (Row #{keeper['_orig_row']})"
                if same_raw
                else f"Duplicate ID '{rec['_raw_id']}' matched {keeper['student_id']} (kept Row #{keeper['_orig_row']})"
            )
        elif dedup_mode == "smart" and ng_key in seen_name_grade and str(rec["_norm_name_key"]) not in {"unknownstudent", "student"}:
            candidate_target = seen_name_grade[ng_key]
            candidate_keeper = staged_rows[candidate_target]

            # ONLY merge name duplicates if at least ONE ID is an auto-generated fallback!
            # Distinct explicit IDs (e.g. STU1016 and STU1027) are DIFFERENT students and MUST NOT be dropped!
            is_rec_auto = str(rec["student_id"]).startswith("AUTO-")
            is_keeper_auto = str(candidate_keeper["student_id"]).startswith("AUTO-")
            if is_rec_auto or is_keeper_auto:
                dup_target = candidate_target
                dup_reason = (
                    f"Matched unassigned learner '{rec['student_name']}' ({rec['grade_level']}) "
                    f"with {candidate_keeper['student_id']}"
                )

        if dup_target is not None:
            keeper = staged_rows[dup_target]
            for col in ("attendance_pct", "marks_pct", "fee_due"):
                if keeper[col] is None and rec[col] is not None:
                    keeper[col] = rec[col]
            merged_counts[dup_target] = merged_counts.get(dup_target, 0) + 1
            duplicates_removed.append(
                {
                    "dropped_row": int(rec["_orig_row"]),
                    "kept_row": int(keeper["_orig_row"]),
                    "student_id": str(rec["student_id"]),
                    "student_name": str(rec["student_name"]),
                    "grade_level": str(rec["grade_level"]),
                    "reason": dup_reason,
                    "raw_attendance": str(rec["_raw_attendance"]) or "∅",
                    "raw_marks": str(rec["_raw_marks"]) or "∅",
                    "raw_fee_due": str(rec["_raw_fee_due"]) or "∅",
                }
            )
        else:
            seen_ids[id_key] = i
            seen_name_grade[ng_key] = i
            kept_indices.append(i)

    kept_indices.sort(key=lambda idx: int(staged_rows[idx]["_orig_row"]))
    clean_records: list[dict[str, Any]] = [dict(staged_rows[idx]) for idx in kept_indices]

    # Stage 3: Missing Value Imputation
    for rec in clean_records:
        rec["attendance_imputed"] = rec["attendance_pct"] is None
        rec["marks_imputed"] = rec["marks_pct"] is None
        rec["fee_imputed"] = rec["fee_due"] is None

    imputed_cells: list[dict[str, Any]] = []
    defaults: list[tuple[str, str, float]] = [
        ("attendance_pct", "_raw_attendance", 75.0),
        ("marks_pct", "_raw_marks", 65.0),
        ("fee_due", "_raw_fee_due", 0.0),
    ]

    if imputation_method == "knn" and len(clean_records) >= 3:
        numeric_cols = ["attendance_pct", "marks_pct", "fee_due"]
        col_defaults = {"attendance_pct": 75.0, "marks_pct": 65.0, "fee_due": 0.0}
        raw_matrix: list[list[float]] = []
        for col in numeric_cols:
            vals = [rec[col] for rec in clean_records if rec[col] is not None]
            if not vals:
                for rec in clean_records:
                    rec["_knn_tmp_" + col] = col_defaults[col]
            else:
                for rec in clean_records:
                    rec["_knn_tmp_" + col] = float(rec[col]) if rec[col] is not None else float("nan")

        for rec in clean_records:
            raw_matrix.append([float(rec["_knn_tmp_" + col]) for col in numeric_cols])

        mat_arr = np.array(raw_matrix, dtype=float)
        n_neighbors = max(1, min(3, len(clean_records) - 1))
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        fit_transform_fn = cast(Any, imputer).fit_transform
        imputed_arr = np.asarray(fit_transform_fn(mat_arr), dtype=float)

        for col_idx, (col, raw_field, default_fallback) in enumerate(defaults):
            for r_idx, rec in enumerate(clean_records):
                if rec[col] is None:
                    raw_imputed = float(imputed_arr[r_idx, col_idx])
                    if not math.isfinite(raw_imputed):
                        raw_imputed = default_fallback
                    val = round(raw_imputed, 1 if col != "fee_due" else 0)
                    if col in ("attendance_pct", "marks_pct"):
                        val = min(100.0, max(0.0, val))
                    else:
                        val = max(0.0, val)
                    rec[col] = val
                    imputed_cells.append(
                        {
                            "student_id": str(rec["student_id"]),
                            "student_name": str(rec["student_name"]),
                            "column": col,
                            "raw_value": str(rec[raw_field]) or "∅",
                            "imputed_value": val,
                            "method": f"KNNImputer (k={n_neighbors}, distance-weighted)",
                        }
                    )
        for rec in clean_records:
            for col in numeric_cols:
                rec.pop("_knn_tmp_" + col, None)
    else:
        # Cohort Median (by grade level) or Global Median
        for col, raw_field, default_fallback in defaults:
            all_vals = [float(rec[col]) for rec in clean_records if rec[col] is not None]
            global_med = float(statistics.median(all_vals)) if all_vals else default_fallback

            by_grade: dict[str, list[float]] = {}
            for rec in clean_records:
                if rec[col] is not None:
                    by_grade.setdefault(str(rec["grade_level"]), []).append(float(rec[col]))

            cohort_medians: dict[str, float] = {
                g: float(statistics.median(vs)) for g, vs in by_grade.items() if vs
            }

            for rec in clean_records:
                if rec[col] is None:
                    grade = str(rec["grade_level"])
                    if imputation_method == "cohort_median" and grade in cohort_medians:
                        val = cohort_medians[grade]
                        method_desc = f"{grade} cohort median"
                    else:
                        val = global_med
                        method_desc = "Global cohort median"
                    rounded_val = round(val, 1 if col != "fee_due" else 0)
                    if not math.isfinite(rounded_val):
                        rounded_val = default_fallback
                    rec[col] = rounded_val
                    imputed_cells.append(
                        {
                            "student_id": str(rec["student_id"]),
                            "student_name": str(rec["student_name"]),
                            "column": col,
                            "raw_value": str(rec[raw_field]) or "∅",
                            "imputed_value": rounded_val,
                            "method": method_desc,
                        }
                    )

    # Stage 4: Final standardization, lineage & quality scoring
    scale_conversions: list[dict[str, Any]] = []
    fee_fixes: list[dict[str, Any]] = []
    row_lineages: list[dict[str, Any]] = []

    orig_to_staged_idx = {
        int(staged_rows[i]["_orig_row"]): i for i in range(len(staged_rows))
    }

    for rec in clean_records:
        att_num = float(rec["attendance_pct"])
        marks_num = float(rec["marks_pct"])
        fee_num = float(rec["fee_due"])

        rec["normalized_letter"] = pct_to_letter(marks_num)
        rec["normalized_gpa4"] = pct_to_gpa4(marks_num)

        final_status, extra_status_note = reconcile_fee_status(rec["_raw_fee_status"], fee_num)
        rec["fee_status"] = final_status

        transformations: list[str] = []
        if rec["_att_note"]:
            transformations.append(str(rec["_att_note"]))
        if bool(rec["attendance_imputed"]):
            transformations.append(f"Imputed missing attendance -> {att_num:.1f}%")
        if rec["_marks_note"]:
            transformations.append(str(rec["_marks_note"]))
            if str(rec["detected_scale"]) != "Percentage (0-100)":
                scale_conversions.append(
                    {
                        "student_id": str(rec["student_id"]),
                        "student_name": str(rec["student_name"]),
                        "raw_marks": str(rec["_raw_marks"]),
                        "detected_scale": str(rec["detected_scale"]),
                        "normalized_pct": round(marks_num, 1),
                        "normalized_gpa4": float(rec["normalized_gpa4"]),
                        "normalized_letter": str(rec["normalized_letter"]),
                    }
                )
        if bool(rec["marks_imputed"]):
            transformations.append(f"Imputed missing marks -> {marks_num:.1f}%")
        if rec["_fee_note"]:
            transformations.append(str(rec["_fee_note"]))
            fee_fixes.append(
                {
                    "student_id": str(rec["student_id"]),
                    "student_name": str(rec["student_name"]),
                    "field": "fee_due",
                    "raw_value": str(rec["_raw_fee_due"]),
                    "cleaned_value": f"{round(fee_num):,}",
                    "note": str(rec["_fee_note"]),
                }
            )
        if bool(rec["fee_imputed"]):
            transformations.append(f"Imputed missing fee balance -> {round(fee_num):,}")

        status_note = rec["_status_note"] or extra_status_note
        if status_note:
            transformations.append(str(status_note))
            fee_fixes.append(
                {
                    "student_id": str(rec["student_id"]),
                    "student_name": str(rec["student_name"]),
                    "field": "fee_status",
                    "raw_value": str(rec["_raw_fee_status"]) or "∅",
                    "cleaned_value": final_status,
                    "note": str(status_note),
                }
            )

        staged_idx = orig_to_staged_idx.get(int(rec["_orig_row"]))
        if staged_idx is not None and merged_counts.get(staged_idx, 0) > 0:
            transformations.append(f"Deduplicated ({merged_counts[staged_idx]} duplicate row removed)")

        row_lineages.append(
            {
                "student_id": str(rec["student_id"]),
                "student_name": str(rec["student_name"]),
                "grade_level": str(rec["grade_level"]),
                "raw_attendance": str(rec["_raw_attendance"]) or "∅",
                "clean_attendance_pct": round(att_num, 1),
                "attendance_imputed": bool(rec["attendance_imputed"]),
                "raw_marks": str(rec["_raw_marks"]) or "∅",
                "detected_scale": str(rec["detected_scale"]),
                "clean_marks_pct": round(marks_num, 1),
                "clean_gpa4": float(rec["normalized_gpa4"]),
                "clean_letter": str(rec["normalized_letter"]),
                "marks_imputed": bool(rec["marks_imputed"]),
                "raw_fee_due": str(rec["_raw_fee_due"]) or "∅",
                "clean_fee_due": round(fee_num, 2),
                "fee_imputed": bool(rec["fee_imputed"]),
                "raw_fee_status": str(rec["_raw_fee_status"]) or "∅",
                "clean_fee_status": final_status,
                "was_modified": len(transformations) > 0,
                "transformations": transformations,
            }
        )

    clean_df = pd.DataFrame(clean_records)

    total_core_cells = max(1, raw_row_count * 3)
    non_standard_scales = sum(
        count
        for scale, count in scale_types_found.items()
        if scale not in {"Percentage (0-100)", "Missing"}
    )
    raw_issues = raw_missing_total + len(duplicates_removed) * 2 + non_standard_scales
    quality_before = round(max(12.0, 100.0 - (raw_issues / total_core_cells) * 100.0), 1)
    quality_after = 100.0

    summary: dict[str, Any] = {
        "raw_rows": raw_row_count,
        "clean_rows": len(clean_records),
        "duplicates_removed_count": len(duplicates_removed),
        "missing_values_before": {
            "attendance": raw_missing_attendance,
            "marks": raw_missing_marks,
            "fee_due": raw_missing_fees,
            "total": raw_missing_total,
        },
        "missing_values_after": 0,
        "cells_imputed_count": len(imputed_cells),
        "scale_conversions_count": len(scale_conversions),
        "scale_distribution": scale_types_found,
        "fee_fixes_count": len(fee_fixes),
        "imputation_method": imputation_method,
        "dedup_mode": dedup_mode,
        "data_quality_before_pct": quality_before,
        "data_quality_after_pct": quality_after,
        "column_mapping": column_mapping,
    }

    return CleaningAuditResult(
        cleaned_df=clean_df,
        raw_preview=raw_preview,
        duplicates_removed=duplicates_removed,
        imputed_cells=imputed_cells,
        scale_conversions=scale_conversions,
        fee_fixes=fee_fixes,
        row_lineages=row_lineages,
        summary=summary,
    )
