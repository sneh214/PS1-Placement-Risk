"""
PS1 - Placement Risk Modeling
Streamlit Dashboard — Loan Officer UI (FIXED VERSION)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os

st.set_page_config(
    page_title="PS1 — Placement Risk Modeling",
    page_icon="🎓",
    layout="wide"
)

# ── FIXED PATHS ────────────────────────────────────────────────────────────
MODEL_DIR = "D:\\PS1\\models\\saved"
DATA_DIR  = "D:\\PS1\\data"

@st.cache_resource
def load_models():
    clf = pickle.load(open(os.path.join(MODEL_DIR, "classifier.pkl"),    "rb"))
    reg = pickle.load(open(os.path.join(MODEL_DIR, "regressor.pkl"),     "rb"))
    le  = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
    return clf, reg, le

clf, reg, le = load_models()

COURSE_DEMAND = {
    "Engineering": 0.75, "MBA": 0.70, "Data Science": 0.85,
    "Nursing": 0.60, "Law": 0.45, "Arts": 0.35
}
TIER_SCORE = {1: 1.0, 2: 0.7, 3: 0.45}
TIER_NAME  = {1: "Tier 1 (IIT/IIM/NIT)", 2: "Tier 2 (State/Private)", 3: "Tier 3 (Local)"}
LABEL_MAP  = {0: "Placed <= 3 months", 1: "Placed <= 6 months",
              2: "Placed <= 12 months", 3: "Delayed (12+ months)"}

# ── Prediction ─────────────────────────────────────────────────────────────
def predict_student(course, tier, cgpa, internships, certs, active):
    ind  = COURSE_DEMAND[course]
    inst = TIER_SCORE[tier]
    emp  = (0.35*(cgpa/10) + 0.25*min(internships/3,1) +
            0.20*inst + 0.15*ind + 0.05*min(certs/5,1))
    eng  = 0.6*active + 0.4*min(certs/5,1)
    ce   = le.transform([course])[0]
    X    = np.array([[cgpa, internships, certs, active, tier,
                      ind, inst, emp, eng, ce]])
    probs = clf.predict_proba(X)[0]
    pred  = int(np.argmax(probs))
    sal   = float(reg.predict(X)[0])
    risk  = ["low","low","medium","high"][pred]
    return probs, pred, sal, emp, risk

def get_risk_factors(course, tier, cgpa, internships, certs, active):
    factors, actions = [], []
    if cgpa < 6.5:
        factors.append(f"Low CGPA ({cgpa}) — below recruiter threshold of 6.5")
        actions.append("Highlight strong projects to compensate for CGPA")
    if internships == 0:
        factors.append("Zero internship experience — critical gap")
        actions.append("Complete at least 1 internship immediately")
    elif internships == 1:
        factors.append("Only 1 internship — competitive candidates have 2+")
        actions.append("Pursue a second internship or freelance project")
    if tier == 3:
        factors.append("Tier-3 institute — limited on-campus placement drives")
        actions.append("Use LinkedIn, Naukri, off-campus drives aggressively")
    if COURSE_DEMAND.get(course, 0.5) < 0.55:
        factors.append(f"{course} — below-average industry hiring demand")
        actions.append("Upskill in adjacent high-demand areas (Data, Tech, Finance)")
    if active == 0:
        factors.append("Inactive on job portals — low recruiter visibility")
        actions.append("Update LinkedIn and Naukri profile and apply daily")
    if certs < 2:
        factors.append("Fewer than 2 certifications — skills not validated")
        actions.append("Add Google/AWS/Coursera certifications")
    if not factors:
        factors.append("Strong overall profile — minimal risk indicators found")
        actions.append("Continue active networking and job applications")
    return factors[:4], actions[:4]

# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown("## Placement Risk Modeling System")
st.markdown("*PS1 — TenzorX Hackathon | Education Loan Risk Intelligence*")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "Single Student Analysis",
    "Batch Analysis",
    "Dataset Insights"
])

# ── Tab 1 ──────────────────────────────────────────────────────────────────
with tab1:
    col_form, col_result = st.columns([1, 1.4], gap="large")

    with col_form:
        st.markdown("#### Student Profile Input")
        name        = st.text_input("Student Name", "Arjun Sharma")
        loan_amt    = st.number_input("Loan Amount (Rs)", 200000, 5000000, 1200000, step=50000)
        course      = st.selectbox("Course / Field", list(COURSE_DEMAND.keys()))
        tier        = st.selectbox("Institute Tier", [1,2,3], format_func=lambda x: TIER_NAME[x])
        cgpa        = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
        internships = st.slider("Number of Internships", 0, 5, 1)
        certs       = st.slider("Certifications Completed", 0, 10, 2)
        active      = st.radio("Active on Job Portals?", [1, 0],
                                format_func=lambda x: "Yes" if x else "No",
                                horizontal=True)
        predict_btn = st.button("Analyse Placement Risk", use_container_width=True, type="primary")

    with col_result:
        if predict_btn:
            probs, pred, sal, emp, risk = predict_student(
                course, tier, cgpa, internships, certs, active
            )
            factors, actions = get_risk_factors(course, tier, cgpa, internships, certs, active)

            sal_low  = int(sal * 0.88)
            sal_high = int(sal * 1.12)

            risk_labels = {
                "low":    ("LOW RISK",    "GREEN"),
                "medium": ("MEDIUM RISK", "YELLOW"),
                "high":   ("HIGH RISK",   "RED")
            }
            rl, ri = risk_labels[risk]

            st.markdown(f"#### Results for {name}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Placement Outlook", LABEL_MAP[pred])
            m2.metric("Risk Level", rl)
            m3.metric("Employability Score", f"{emp:.2f} / 1.0")

            st.divider()
            st.markdown("**Predicted Salary Range**")
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Minimum",  f"Rs {sal_low:,}")
            col_s2.metric("Expected", f"Rs {int(sal):,}")
            col_s3.metric("Maximum",  f"Rs {sal_high:,}")

            st.divider()
            st.markdown("**Placement Probability Breakdown**")
            prob_df = pd.DataFrame({
                "Outcome": list(LABEL_MAP.values()),
                "Score":   probs
            })
            st.bar_chart(prob_df.set_index("Outcome")["Score"])

            st.divider()
            if risk == "low":
                st.success("Loan Repayment Signal: GREEN — Proceed with standard loan terms. Low delinquency risk.")
            elif risk == "medium":
                st.warning("Loan Repayment Signal: AMBER — Monitor quarterly. Consider 3-month EMI grace period.")
            else:
                st.error("Loan Repayment Signal: RED — High delinquency risk. Verify co-applicant and recommend skill intervention.")

            st.divider()
            col_f, col_a = st.columns(2)
            with col_f:
                st.markdown("**Top Risk Factors**")
                for f in factors:
                    st.markdown(f"- {f}")
            with col_a:
                st.markdown("**Recommended Actions**")
                for a in actions:
                    st.markdown(f"- {a}")
        else:
            st.info("Fill in the student profile on the left and click Analyse Placement Risk.")

# ── Tab 2 ──────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Batch Student Risk Analysis")

    sample_data = pd.DataFrame({
        "name":             ["Riya Patel","Mohan Kumar","Sneha Gupta","Rohit Singh","Priya Nair"],
        "course":           ["Data Science","Arts","MBA","Engineering","Nursing"],
        "institute_tier":   [2,3,1,2,3],
        "cgpa":             [8.2,5.8,9.1,7.0,6.5],
        "internships":      [2,0,3,1,0],
        "certifications":   [4,0,5,2,1],
        "job_portal_active":[1,0,1,1,0],
        "loan_amount":      [1200000,800000,2000000,1000000,600000]
    })

    use_sample = st.checkbox("Use sample dataset (5 students)", value=True)
    if use_sample:
        batch_df = sample_data.copy()
    else:
        uploaded = st.file_uploader("Upload CSV", type="csv")
        batch_df = pd.read_csv(uploaded) if uploaded else sample_data.copy()

    if st.button("Run Batch Analysis", type="primary"):
        results = []
        for _, row in batch_df.iterrows():
            probs, pred, sal, emp, risk = predict_student(
                row["course"], int(row["institute_tier"]), float(row["cgpa"]),
                int(row["internships"]), int(row["certifications"]),
                int(row["job_portal_active"])
            )
            results.append({
                "Name":             row["name"],
                "Course":           row["course"],
                "CGPA":             row["cgpa"],
                "Placement Outlook":LABEL_MAP[pred],
                "Risk Level":       risk.upper(),
                "Employability":    f"{emp:.2f}",
                "Salary Expected":  f"Rs {int(sal):,}",
                "Loan Signal":      {"low":"Green","medium":"Amber","high":"Red"}[risk]
            })

        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        rc = pd.Series([r["Risk Level"] for r in results]).value_counts()
        c1.metric("Low Risk Students",    rc.get("LOW",0))
        c2.metric("Medium Risk Students", rc.get("MEDIUM",0))
        c3.metric("High Risk Students",   rc.get("HIGH",0))

# ── Tab 3 ──────────────────────────────────────────────────────────────────
with tab3:
    data_path = os.path.join(DATA_DIR, "student_data_features.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.markdown("#### Training Dataset Insights")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students",    len(df))
        c2.metric("Avg CGPA",          f"{df['cgpa'].mean():.2f}")
        c3.metric("Avg Internships",   f"{df['internships'].mean():.1f}")
        c4.metric("Avg Starting Salary",f"Rs {df['starting_salary'].mean():,.0f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Placement Distribution**")
            st.bar_chart(df["placement_label"].value_counts())
        with col_b:
            st.markdown("**Avg Placement Months by Course**")
            st.bar_chart(df.groupby("course")["placement_months"].mean().sort_values())
    else:
        st.warning("Run python run.py first to generate data and train models.")

st.divider()
st.caption("PS1 · TenzorX Hackathon · Placement Risk Modeling System · Built with XGBoost + Streamlit")
