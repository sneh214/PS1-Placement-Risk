# PS1 — Placement Risk Modeling System
## TenzorX Hackathon | Complete Implementation Guide

---

## What This System Does

Predicts how quickly an education loan borrower (student) will get a job after graduation, estimates their starting salary, and flags high-risk students whose delayed placement could affect loan repayment.

**Output for each student:**
- Placement timeline prediction (≤3 / ≤6 / ≤12 / delayed months)
- Predicted salary range (min / expected / max)
- Risk level (Low / Medium / High)
- Loan repayment signal (Green / Amber / Red)
- Top risk factors + recommended actions

---

## Project Structure

```
ps1_placement_risk/
├── data/
│   ├── generate_data.py        # Step 1: Creates synthetic student dataset
│   └── student_data.csv        # Generated dataset (2000 students)
├── models/
│   ├── train_model.py          # Steps 2-3: Feature engineering + XGBoost training
│   └── saved/                  # Saved model files (.pkl)
│       ├── classifier.pkl
│       ├── regressor.pkl
│       ├── explainer.pkl
│       ├── label_encoder.pkl
│       └── features.pkl
├── backend/
│   └── main.py                 # Step 4: FastAPI REST API
├── frontend/
│   └── app.py                  # Step 5: Streamlit dashboard UI
├── requirements.txt
└── README.md
```

---

## Step-by-Step Implementation Guide

---

### STEP 1 — Generate Synthetic Dataset (`data/generate_data.py`)

**Why?** Real student placement data is private. We create realistic synthetic data based on known patterns.

**How it works:**
- Creates 2000 student profiles with: CGPA, internships, institute tier, course, certifications
- Placement timeline is computed using a weighted formula:
  ```
  score = 0.35 × (CGPA/10) + 0.25 × (internships/3) + 0.25 × institute_score + 0.15 × industry_demand
  ```
- Higher score → more likely placed in 3 months
- Lower score → more likely delayed (12+ months)
- Salary is derived from course base salary × multipliers for CGPA, internships, tier

**Run it:**
```bash
cd data/
python generate_data.py
```

**Output:** `student_data.csv` with 2000 rows and columns:
`student_id, course, institute_tier, cgpa, internships, certifications, job_portal_active, placement_months, placement_label, starting_salary`

---

### STEP 2 — Feature Engineering (`models/train_model.py`)

**Why?** Raw features like CGPA and internships need to be combined into meaningful signals.

**Key engineered features:**
```python
# Combines all profile signals into one 0-1 score
employability_score = (
    0.35 × (cgpa / 10)              # Academic strength
  + 0.25 × min(internships / 3, 1)  # Practical experience
  + 0.20 × institute_score          # Tier 1=1.0, Tier 2=0.7, Tier 3=0.45
  + 0.15 × industry_demand          # Data Science=0.85, Arts=0.35
  + 0.05 × min(certifications / 5, 1) # Extra skills
)

# Job search activity score
engagement_score = 0.6 × job_portal_active + 0.4 × (certifications / 5)
```

**Also:** Course name → encoded number using `LabelEncoder` (ML models need numbers)

---

### STEP 3 — Train XGBoost Models (`models/train_model.py`)

**Two separate models:**

**Model 1: Placement Classifier (XGBClassifier)**
- Input: student features
- Output: which bucket? [placed_3m, placed_6m, placed_12m, delayed]
- Algorithm: XGBoost (gradient boosting — best for tabular data)
- Saved as: `models/saved/classifier.pkl`

**Model 2: Salary Regressor (XGBRegressor)**
- Input: same student features
- Output: predicted starting salary (Rs)
- Saved as: `models/saved/regressor.pkl`

**SHAP Explainer:**
- SHAP = SHapley Additive exPlanations
- Tells you WHICH features drove the prediction
- e.g., "CGPA contributed +0.15 to delayed placement probability"
- Saved as: `models/saved/explainer.pkl`

**Run it:**
```bash
cd models/
python train_model.py
```

---

### STEP 4 — FastAPI Backend (`backend/main.py`)

**Why?** Expose the ML model as a REST API so any frontend can call it.

**Endpoints:**
```
POST /predict    → Main prediction endpoint
GET  /courses    → List of valid courses
GET  /health     → API health check
```

**Sample API call:**
```python
import requests
response = requests.post("http://localhost:8000/predict", json={
    "student_name": "Arjun Sharma",
    "course": "Engineering",
    "institute_tier": 2,
    "cgpa": 7.8,
    "internships": 2,
    "certifications": 3,
    "job_portal_active": 1,
    "loan_amount": 1500000
})
print(response.json())
```

**Sample Response:**
```json
{
  "student_name": "Arjun Sharma",
  "placement_prediction": "placed_6m",
  "risk_level": "low",
  "salary_range": {"low": 528000, "mid": 600000, "high": 672000},
  "employability_score": 0.712,
  "risk_explanation": "Low risk: Likely placed within 3–6 months.",
  "recommended_actions": ["Keep networking and applying proactively"],
  "loan_repayment_signal": "Green — Proceed with standard loan terms.",
  "top_risk_factors": ["Strong overall profile — low risk indicators found"]
}
```

**Run it:**
```bash
cd backend/
uvicorn main:app --reload --port 8000
# Visit: http://localhost:8000/docs  (auto Swagger UI)
```

---

### STEP 5 — Streamlit Dashboard (`frontend/app.py`)

**Why?** Loan officers need a simple UI — not JSON APIs.

**3 tabs:**
1. **Single Student Analysis** — Input one student, get full risk report
2. **Batch Analysis** — Upload CSV of many students, get risk table
3. **Dataset Insights** — Charts of the training data distribution

**Run it:**
```bash
cd frontend/
streamlit run app.py
# Opens at: http://localhost:8501
```

---

## How to Run the Full Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
cd data && python generate_data.py && cd ..

# 3. Train models
cd models && python train_model.py && cd ..

# 4. (Optional) Start FastAPI backend
cd backend && uvicorn main:app --reload &

# 5. Launch Streamlit UI
cd frontend && streamlit run app.py
```

---

## Judging Criteria — How This Project Scores

| Criterion | What We Built | Score |
|-----------|--------------|-------|
| Accuracy of placement predictions | XGBoost with SHAP, 3/6/12m buckets | ✅ |
| Explainability of risk drivers | SHAP values + human-readable factors | ✅ |
| Usefulness for lenders | Green/Amber/Red loan signal | ✅ |
| Scalability | Batch CSV analysis tab | ✅ |
| Impact potential | Recommended actions per student | ✅ |
| Robustness | Handles all courses, tiers, markets | ✅ |

---

## Key Design Decisions

1. **XGBoost over Deep Learning** — tabular data, small dataset, needs explainability
2. **Range outputs not point estimates** — salary as low/mid/high, not a single number
3. **Employability Score** — single composite metric loan officers can understand immediately
4. **Rule-based explanations** — more reliable and auditable than pure LLM output
5. **Streamlit over React** — faster to build, judges can run locally without setup

---

*Built for TenzorX Hackathon — PS1: Linking Education Loan to Career Success using AI*
