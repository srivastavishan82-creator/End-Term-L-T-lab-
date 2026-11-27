# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Automated tests for the Python data cleaning pipeline, ML risk predictor, and FastAPI routes."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.data_cleaner import (
    clean_school_dataset,
    parse_attendance,
    parse_fee_due,
    parse_grading_scale,
    reconcile_fee_status,
)
from app.main import app
from app.predictor import predict_dropout_risk

SYNTHETIC_TEST_CSV = """student_id,student_name,grade_level,attendance,marks_grade,fee_due,fee_status
T-101,Test Learner One,Grade 10,92%,88%,0,Paid
T-102,Test Learner Two,Grade 10,54%,3.1/4.0,₹18500,Overdue
t-102,test learner two,Grade 10,54%,3.1/4.0,,Overdue
T-103,Test Learner Three,Grade 11,N/A,7.8/10,"12,000",Pending
T-104,Test Learner Four,Grade 12,38%,F,34000,Overdue
T-105,Test Learner Five,Grade 9,86%,0.84,nil,Paid
T-106,Test Learner Six,Grade 11,81%,39/50,0,Paid
T-101,Test Learner One,Grade 10,92%,88%,0,Paid
"""


class TestDataCleaning(unittest.TestCase):
    def test_grading_scale_normalization(self) -> None:
        pct, scale, _ = parse_grading_scale("3.2/4.0")
        assert pct is not None
        self.assertAlmostEqual(pct, 80.0, places=1)
        self.assertIn("4.0", scale)

        pct2, scale2, _ = parse_grading_scale("8.4 CGPA")
        assert pct2 is not None
        self.assertAlmostEqual(pct2, 84.0, places=1)
        self.assertIn("10.0", scale2)

        pct3, _, _ = parse_grading_scale("39/50")
        assert pct3 is not None
        self.assertAlmostEqual(pct3, 78.0, places=1)

        pct4, scale4, _ = parse_grading_scale("B+")
        assert pct4 is not None
        self.assertAlmostEqual(pct4, 84.0, places=1)
        self.assertEqual(scale4, "Letter Grade")

        pct5, _, _ = parse_grading_scale("0.84")
        assert pct5 is not None
        self.assertAlmostEqual(pct5, 84.0, places=1)

    def test_cbse_and_indian_grades(self) -> None:
        # CBSE A1, B2, C1, D, O
        pct_a1, _, _ = parse_grading_scale("A1")
        assert pct_a1 is not None
        self.assertEqual(pct_a1, 95.0)

        pct_b2, _, _ = parse_grading_scale("B2")
        assert pct_b2 is not None
        self.assertEqual(pct_b2, 65.0)

        pct_o, _, _ = parse_grading_scale("O")
        assert pct_o is not None
        self.assertEqual(pct_o, 95.0)

        pct_pass, _, _ = parse_grading_scale("Passed")
        assert pct_pass is not None
        self.assertEqual(pct_pass, 60.0)

    def test_pure_numeric_low_marks_not_confused_with_gpa(self) -> None:
        # In a 0-100 column, a score of 4 or 8 is 4% or 8%, NOT 100% or 80%
        pct_4, scale_4, _ = parse_grading_scale("4", column_scale="PERCENTAGE_100")
        assert pct_4 is not None
        self.assertEqual(pct_4, 4.0)
        self.assertEqual(scale_4, "Percentage (0-100)")

        pct_8, scale_8, _ = parse_grading_scale("8", column_scale="PERCENTAGE_100")
        assert pct_8 is not None
        self.assertEqual(pct_8, 8.0)
        self.assertEqual(scale_8, "Percentage (0-100)")

        # In a 4.0 GPA column, 3.5 is 87.5%
        pct_gpa, scale_gpa, _ = parse_grading_scale("3.5", column_scale="4.0_GPA")
        assert pct_gpa is not None
        self.assertAlmostEqual(pct_gpa, 87.5, places=1)
        self.assertEqual(scale_gpa, "4.0 GPA Scale")

    def test_attendance_and_fee_parsing(self) -> None:
        att, _ = parse_attendance("42/50")
        assert att is not None
        self.assertAlmostEqual(att, 84.0, places=1)

        att_frac, _ = parse_attendance("160 out of 200")
        assert att_frac is not None
        self.assertEqual(att_frac, 80.0)

        att_days, _ = parse_attendance("180 days")
        assert att_days is not None
        self.assertEqual(att_days, 90.0)

        att_cat, _ = parse_attendance("Good")
        assert att_cat is not None
        self.assertEqual(att_cat, 85.0)

        fee, _ = parse_fee_due("₹18,500")
        assert fee is not None
        self.assertAlmostEqual(fee, 18500.0, places=1)

        fee_nil, _ = parse_fee_due("nil")
        self.assertEqual(fee_nil, 0.0)

        fee_k, _ = parse_fee_due("15k")
        assert fee_k is not None
        self.assertEqual(fee_k, 15000.0)

        fee_lakh, _ = parse_fee_due("1.5 Lakh")
        assert fee_lakh is not None
        self.assertEqual(fee_lakh, 150000.0)

        fee_neg, _ = parse_fee_due("-500")
        assert fee_neg is not None
        self.assertEqual(fee_neg, 0.0)

    def test_fee_status_reconciliation(self) -> None:
        # Zero fee due -> Paid
        st1, note1 = reconcile_fee_status("Overdue", 0.0)
        self.assertEqual(st1, "Paid")
        self.assertIsNotNone(note1)

        # Non-zero fee due with Paid status -> conflict reconciled to Overdue
        st2, note2 = reconcile_fee_status("Paid", 15000.0)
        self.assertEqual(st2, "Overdue")
        self.assertIn("Conflict fixed", str(note2))

        # Blank status with fee due -> Inferred Overdue / Pending
        st3, _ = reconcile_fee_status("", 15000.0)
        self.assertEqual(st3, "Overdue")

    def test_dedup_preserves_different_students_with_same_name(self) -> None:
        # Two students named Rahul Sharma with different IDs (S-1 and S-2) must NOT be dropped
        csv_data = (
            "student_id,student_name,grade_level,attendance,marks_grade,fee_due,fee_status\n"
            "S-1,Rahul Sharma,Grade 10,85%,78%,0,Paid\n"
            "S-2,Rahul Sharma,Grade 10,65%,52%,15000,Overdue\n"
        )
        audit = clean_school_dataset(csv_data, dedup_mode="smart")
        self.assertEqual(audit.summary["clean_rows"], 2)
        self.assertEqual(audit.summary["duplicates_removed_count"], 0)

    def test_arbitrary_column_names_mapping(self) -> None:
        # Test messy Indian school column headers with brackets, symbols, and alternate names
        messy_headers_csv = (
            "Adm No,Candidate Name,Class / Sec,Present Days,Total Marks (100),Pending Dues (INR),Fee Clearance\n"
            "ADM-01,Aarav Patel,Class X-A,170/200,82,0,Cleared\n"
            "ADM-02,Diya Singh,Class X-B,140/200,68,Rs 12500/-,Unpaid\n"
            "ADM-03,Karan Verma,Class XII,110/200,42,25k,Pending\n"
        )
        audit = clean_school_dataset(messy_headers_csv)
        self.assertEqual(audit.summary["clean_rows"], 3)
        self.assertEqual(audit.summary["missing_values_after"], 0)
        # Check that values were parsed and not blanked out
        df_records = audit.cleaned_df.to_dict(orient="records")
        self.assertAlmostEqual(float(df_records[0]["attendance_pct"]), 85.0, places=1)
        self.assertAlmostEqual(float(df_records[1]["fee_due"]), 12500.0, places=1)
        self.assertAlmostEqual(float(df_records[2]["fee_due"]), 25000.0, places=1)

    def test_dedup_and_imputation_methods(self) -> None:
        for method in ("cohort_median", "global_median", "knn"):
            audit = clean_school_dataset(
                SYNTHETIC_TEST_CSV, imputation_method=method, dedup_mode="smart"
            )
            self.assertEqual(audit.summary["raw_rows"], 8)
            self.assertEqual(audit.summary["clean_rows"], 6)
            self.assertEqual(audit.summary["duplicates_removed_count"], 2)
            self.assertEqual(audit.summary["missing_values_after"], 0)
            records = audit.cleaned_df.to_dict(orient="records")
            self.assertFalse(any(r["attendance_pct"] is None for r in records))
            self.assertFalse(any(r["marks_pct"] is None for r in records))
            self.assertFalse(any(r["fee_due"] is None for r in records))

    def test_prediction_ordering_and_tiers(self) -> None:
        audit = clean_school_dataset(
            SYNTHETIC_TEST_CSV, imputation_method="cohort_median", dedup_mode="smart"
        )
        result = predict_dropout_risk(audit.cleaned_df, raw_csv_hint=SYNTHETIC_TEST_CSV)
        preds = result["predictions"]
        self.assertEqual(len(preds), 6)
        by_id = {p["student_id"]: p for p in preds}
        self.assertEqual(by_id["T-104"]["risk_tier"], "High Risk")
        self.assertEqual(by_id["T-101"]["risk_tier"], "Low Risk")
        self.assertGreater(
            by_id["T-104"]["dropout_probability_pct"],
            by_id["T-101"]["dropout_probability_pct"],
        )

    def test_multi_subject_scores_recovery(self) -> None:
        # CSV with separate subject columns and NO single marks_raw column
        csv_subjects = (
            "student_id,student_name,grade_level,attendance,Mathematics,Science,English,fee_due,fee_status\n"
            "STU-01,Alice Smith,Grade 10,95%,80,90,70,0,Paid\n"
            "STU-02,Bob Jones,Grade 10,85%,A,B+,A-,0,Paid\n"
        )
        audit = clean_school_dataset(csv_subjects)
        recs = audit.cleaned_df.to_dict(orient="records")
        # Alice: average of 80, 90, 70 is 80.0
        self.assertAlmostEqual(float(recs[0]["marks_pct"]), 80.0, places=1)
        self.assertFalse(recs[0]["marks_imputed"])
        # Bob: average of A (95), B+ (84), A- (90) is ~89.7
        self.assertGreater(float(recs[1]["marks_pct"]), 80.0)
        self.assertFalse(recs[1]["marks_imputed"])

    def test_attendance_calculated_from_days_present_and_total_days(self) -> None:
        # Attendance log columns days_present and total_school_days
        csv_att = (
            "student_id,student_name,grade_level,days_present,total_school_days,marks,fee_due,fee_status\n"
            "STU1233,Test Student,Grade 10,205,220,85%,0,Paid\n"
        )
        audit = clean_school_dataset(csv_att)
        recs = audit.cleaned_df.to_dict(orient="records")
        # 205 / 220 = 93.18% -> 93.2%
        self.assertAlmostEqual(float(recs[0]["attendance_pct"]), 93.2, places=1)
        self.assertFalse(recs[0]["attendance_imputed"])

    def test_paid_fee_status_with_missing_fee_due_preserved(self) -> None:
        # Students with Paid status but missing fee_due must not have debt imputed or be marked Overdue
        csv_fee = (
            "student_id,student_name,grade_level,attendance,marks,fee_due,fee_status\n"
            "STU1046,Student One,Grade 10,90%,75%,,Paid\n"
            "STU1233,Student Two,Grade 10,88%,82%,nil,Cleared\n"
        )
        audit = clean_school_dataset(csv_fee)
        recs = audit.cleaned_df.to_dict(orient="records")
        self.assertEqual(float(recs[0]["fee_due"]), 0.0)
        self.assertEqual(recs[0]["fee_status"], "Paid")
        self.assertFalse(recs[0]["fee_imputed"])
        self.assertEqual(float(recs[1]["fee_due"]), 0.0)
        self.assertEqual(recs[1]["fee_status"], "Paid")
        self.assertFalse(recs[1]["fee_imputed"])

    def test_distinct_student_ids_not_dropped(self) -> None:
        # 10 distinct students who may share similar default stats must NOT be dropped
        student_rows = [
            f"STU{sid},Student {sid},Grade 10,85%,75%,0,Paid"
            for sid in [1016, 1027, 1028, 1059, 1098, 1120, 1128, 1179, 1217, 1242]
        ]
        csv_data = "student_id,student_name,grade_level,attendance,marks,fee_due,fee_status\n" + "\n".join(student_rows)
        audit = clean_school_dataset(csv_data, dedup_mode="smart")
        self.assertEqual(audit.summary["clean_rows"], 10)
        self.assertEqual(audit.summary["duplicates_removed_count"], 0)

    def test_calibrated_cohort_risk_distribution(self) -> None:
        # Generate a cohort of 50 students with realistic distribution
        rows = []
        for i in range(35):
            rows.append(f"STU{i:03d},Good Student {i},Grade 10,90%,85%,0,Paid")
        for i in range(35, 45):
            rows.append(f"STU{i:03d},Borderline Student {i},Grade 10,72%,60%,15000,Pending")
        for i in range(45, 50):
            rows.append(f"STU{i:03d},At Risk Student {i},Grade 10,45%,35%,45000,Overdue")

        csv_cohort = "student_id,student_name,grade_level,attendance,marks,fee_due,fee_status\n" + "\n".join(rows)
        audit = clean_school_dataset(csv_cohort)
        result = predict_dropout_risk(audit.cleaned_df, raw_csv_hint=csv_cohort)
        avg_risk = result["cohort_analytics"]["average_dropout_probability_pct"]
        high_cnt = result["cohort_analytics"]["high_risk_count"]
        # Cohort average risk should be calibrated around 20-30%, NOT 77%
        self.assertLess(avg_risk, 35.0)
        # High risk count should be small (~5-10 out of 50), NOT majority
        self.assertLessEqual(high_cnt, 12)


class TestApiEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_and_template(self) -> None:
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

        tpl = self.client.get("/api/template.csv")
        self.assertEqual(tpl.status_code, 200)
        self.assertEqual(
            tpl.text.strip(),
            "student_id,student_name,grade_level,attendance,marks_grade,fee_due,fee_status",
        )

    def test_upload_and_standalone_single(self) -> None:
        up = self.client.post(
            "/api/analyze/upload",
            files={"file": ("school.csv", SYNTHETIC_TEST_CSV.encode("utf-8"), "text/csv")},
            data={"imputation_method": "knn", "dedup_mode": "smart"},
        )
        self.assertEqual(up.status_code, 200)
        body = up.json()
        self.assertEqual(body["cleaning_summary"]["clean_rows"], 6)
        self.assertEqual(body["cleaning_summary"]["duplicates_removed_count"], 2)

        for method in ("cohort_median", "knn", "global_median"):
            single = self.client.post(
                "/api/analyze/single",
                json={
                    "student_id": "X-901",
                    "student_name": "test student",
                    "grade_level": "Grade 10",
                    "attendance": "",
                    "marks_raw": "1.9 GPA",
                    "fee_due": "₹22,500",
                    "fee_status": "Paid",
                    "imputation_method": method,
                },
            )
            self.assertEqual(single.status_code, 200)
            sdata = single.json()
            self.assertTrue(sdata["student"]["attendance_imputed"])
            self.assertEqual(sdata["student"]["fee_status"], "Overdue")
            self.assertIn(sdata["student"]["risk_tier"], ("High Risk", "Medium Risk"))

    def test_homepage_has_no_pseudo_defaults(self) -> None:
        # Home page serves dedicated login.html
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("EduShield", home.text)
        self.assertIn("login-portal", home.text)
        self.assertIn("Sign In to Retention Portal", home.text)
        self.assertIn("quick-demo-btn", home.text)
        self.assertNotIn("preset-select", home.text)
        self.assertNotIn("STU-901", home.text)
        self.assertNotIn("Oakridge", home.text)
        self.assertNotIn("rihan", home.text.lower())

        # /login serves dedicated login.html
        login_res = self.client.get("/login")
        self.assertEqual(login_res.status_code, 200)
        self.assertIn("login-portal", login_res.text)

        # /dashboard serves main application workspace
        dash_res = self.client.get("/dashboard")
        self.assertEqual(dash_res.status_code, 200)
        self.assertIn("main-workspace", dash_res.text)
        self.assertIn("sign-out-btn", dash_res.text)

        css = self.client.get("/static/styles.css")
        self.assertEqual(css.status_code, 200)
        js = self.client.get("/static/app.js")
        self.assertEqual(js.status_code, 200)
        img = self.client.get("/static/assets/hero-illustration.webp")
        self.assertEqual(img.status_code, 200)
        ico = self.client.get("/favicon.ico")
        self.assertEqual(ico.status_code, 200)
        svg = self.client.get("/static/favicon.svg")
        self.assertEqual(svg.status_code, 200)

    def test_messy_delimiters_bom_ragged_and_non_finite_csv(self) -> None:
        messy_csv = (
            "\ufeffRollNo;Name;Class;Attendence;Total Marks;Pending Fees;Status\n"
            "M-1;Learner A;Grade 10;88%;3.5/4.0;0;Paid;extra_trailing\n"
            "M-2;Learner B;Grade 11;inf;1e400;inf;Overdue\n"
            "M-3;Learner C;Grade 9;N/A;B+;₹14,000;Paid\n"
        )
        res = self.client.post(
            "/api/analyze/upload",
            files={"file": ("semicolon.csv", messy_csv.encode("utf-8"), "text/csv")},
            data={"imputation_method": "cohort_median", "dedup_mode": "smart"},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["cleaning_summary"]["clean_rows"], 3)
        self.assertEqual(body["cleaning_summary"]["missing_values_after"], 0)


if __name__ == "__main__":
    unittest.main()
