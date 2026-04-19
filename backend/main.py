"""
PS1 - Placement Risk Modeling
Step 4: FastAPI Backend — prediction + LLM explanation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle, os, numpy as np
from typing import Optional

# ── Load models ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "../models/saved")

with open(f"{MODEL_DIR}/classifier.pkl", "rb") as f:  clf = pickle.load(f)
with open(f"{MODEL_DIR}/regressor.pkl", "rb") as f:   reg = pickle.load(f)
with open(f"{MODEL_DIR}/explainer.pkl", "rb") as f:   explainer = pickle.load(f)
with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f: le_course = pickle.load(f)
with open(f"{MODEL_DIR}/features.pkl", "rb") as f:    FEATURES = pickle.load(f)

COURSE_DEMAND = {
    "Engineering": 0.75, "MBA": 0.70, "Data Science": 0.85,
    "Nursing": 0.60, "Law": 0.45, "Arts": 0.35
}
INSTITUTE_TIER_SCORE = {1: 1.0, 2: 0.7, 3: 0.45}
LABEL_MAP = {0: "placed_3m", 1: "placed_6m", 2: "placed_12m", 3: "delayed"}
RISK_MAP = {
    "placed_3m": "low", "placed_6m": "low",
    "placed_12m": "medium", "delayed": "high"
}

app = FastAPI(title="PS1 — Placement Risk Modeling API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Request / Response models ──────────────────────────────────────────────
class StudentInput(BaseModel):
    student_name: str = Field(..., example="Arjun Sharma")
    course: str = Field(..., example="Engineering")
    institute_tier: int = Field(..., ge=1, le=3, example=2)
    cgpa: float = Field(..., ge=0, le=10, example=7.8)
    internships: int = Field(..., ge=0, le=10, example=2)
    certifications: int = Field(..., ge=0, le=20, example=3)
    job_portal_active: int = Field(..., ge=0, le=1, example=1)
    loan_amount: Optional[float] = Field(None, example=1500000)

class PredictionResponse(BaseModel):
    student_name: str
    placement_prediction: str
    placement_probabilities: dict
    risk_level: str
    salary_range: dict
    employability_score: float
    risk_explanation: str
    recommended_actions: list
    loan_repayment_signal: str
    top_risk_factors: list

# ── Helper: build feature vector ───────────────────────────────────────────
def build_features(inp: StudentInput):
    if inp.course not in COURSE_DEMAND:
        raise HTTPException(400, f"Unknown course. Valid: {list(COURSE_DEMAND)}")
    ind_demand   = COURSE_DEMAND[inp.course]
    inst_score   = INSTITUTE_TIER_SCORE[inp.institute_tier]
    employability = (
        0.35 * (inp.cgpa / 10.0) +
        0.25 * min(inp.internships / 3.0, 1.0) +
        0.20 * inst_score +
        0.15 * ind_demand +
        0.05 * min(inp.certifications / 5.0, 1.0)
    )
    engagement = (
        0.6 * inp.job_portal_active +
        0.4 * min(inp.certifications / 5.0, 1.0)
    )
    try:
        course_enc = le_course.transform([inp.course])[0]
    except Exception:
        course_enc = 0

    return np.array([[
        inp.cgpa, inp.internships, inp.certifications,
        inp.job_portal_active, inp.institute_tier, ind_demand,
        inst_score, employability, engagement, course_enc
    ]]), round(employability, 3)

# ── Helper: rule-based explanation ────────────────────────────────────────
def generate_explanation(inp: StudentInput, risk: str, probs: dict, sal_mid: int):
    factors, actions = [], []

    if inp.cgpa < 6.5:
        factors.append(f"Low CGPA ({inp.cgpa}) — below 6.5 threshold preferred by recruiters")
        actions.append("Focus on improving academic performance or highlight project work")
    if inp.internships == 0:
        factors.append("No internship experience — major gap for employers")
        actions.append("Complete at least 1 internship before graduation (even virtual)")
    elif inp.internships == 1:
        factors.append("Only 1 internship — competitive candidates have 2+")
        actions.append("Pursue a second internship or live project")
    if inp.institute_tier == 3:
        factors.append("Tier-3 institute — lower recruiter footfall, self-sourcing needed")
        actions.append("Leverage LinkedIn, Naukri, and off-campus drives actively")
    if COURSE_DEMAND.get(inp.course, 0.5) < 0.5:
        factors.append(f"{inp.course} has below-average industry hiring demand")
        actions.append("Consider upskilling in adjacent high-demand areas")
    if inp.job_portal_active == 0:
        factors.append("Not active on job portals — lower visibility to recruiters")
        actions.append("Update LinkedIn and Naukri profiles immediately")
    if inp.certifications < 2:
        factors.append("Few certifications — skills validation weak")
        actions.append("Add 1–2 recognized certifications (Coursera, Google, AWS)")

    if not factors:
        factors.append("Strong overall profile — low risk of placement delay")
        actions.append("Keep networking and applying proactively")

    risk_text = {
        "low":    f"Low risk: Likely placed within 3–6 months. Estimated salary Rs {sal_mid:,}/yr.",
        "medium": f"Medium risk: Placement expected within 6–12 months. Salary Rs {sal_mid:,}/yr.",
        "high":   f"High risk: May face 12+ month delay. Early intervention recommended. Est. salary Rs {sal_mid:,}/yr."
    }
    explanation = risk_text.get(risk, "")

    loan_signal = {
        "low":    "Green — Student likely to repay on time. Proceed with standard loan terms.",
        "medium": "Amber — Monitor quarterly. Consider grace period of 3 months.",
        "high":   "Red — High delinquency risk. Recommend co-applicant verification and skill intervention."
    }.get(risk, "Unknown")

    return explanation, actions[:3], factors[:3], loan_signal

# ── Main prediction endpoint ───────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(inp: StudentInput):
    X, emp_score = build_features(inp)

    # Classification
    probs_arr = clf.predict_proba(X)[0]
    pred_idx  = int(np.argmax(probs_arr))
    pred_label = LABEL_MAP[pred_idx]
    risk_level = RISK_MAP[pred_label]
    probs_dict = {LABEL_MAP[i]: round(float(p), 3) for i, p in enumerate(probs_arr)}

    # Salary prediction
    sal_pred = float(reg.predict(X)[0])
    sal_low  = int(sal_pred * 0.88)
    sal_high = int(sal_pred * 1.12)
    sal_mid  = int(sal_pred)

    # Explanations
    explanation, actions, factors, loan_signal = generate_explanation(
        inp, risk_level, probs_dict, sal_mid
    )

    return PredictionResponse(
        student_name=inp.student_name,
        placement_prediction=pred_label,
        placement_probabilities=probs_dict,
        risk_level=risk_level,
        salary_range={"low": sal_low, "mid": sal_mid, "high": sal_high},
        employability_score=emp_score,
        risk_explanation=explanation,
        recommended_actions=actions,
        loan_repayment_signal=loan_signal,
        top_risk_factors=factors
    )

@app.get("/courses")
def get_courses():
    return {"courses": list(COURSE_DEMAND.keys())}

@app.get("/health")
def health():
    return {"status": "ok", "model": "PS1 Placement Risk v1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
