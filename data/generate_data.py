"""
PS1 - Placement Risk Modeling
Step 1: Generate synthetic student dataset
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 2000

COURSE_DEMAND = {"Engineering": 0.75, "MBA": 0.70, "Data Science": 0.85,
                 "Nursing": 0.60, "Law": 0.45, "Arts": 0.35}

INSTITUTE_TIER_SCORE = {1: 1.0, 2: 0.7, 3: 0.45}

SALARY_BASE = {
    "Engineering": 600000, "MBA": 750000, "Data Science": 900000,
    "Nursing": 380000, "Law": 500000, "Arts": 280000
}

def generate_placement_months(cgpa, internships, institute_tier, course, noise=True):
    score = (
        0.35 * (cgpa / 10.0) +
        0.25 * min(internships / 3.0, 1.0) +
        0.25 * INSTITUTE_TIER_SCORE[institute_tier] +
        0.15 * COURSE_DEMAND[course]
    )
    if noise:
        score += np.random.normal(0, 0.08)
    score = np.clip(score, 0, 1)
    if score >= 0.75:
        return np.random.choice([3, 6], p=[0.75, 0.25])
    elif score >= 0.55:
        return np.random.choice([3, 6, 12], p=[0.20, 0.55, 0.25])
    elif score >= 0.35:
        return np.random.choice([6, 12, 24], p=[0.20, 0.50, 0.30])
    else:
        return np.random.choice([12, 24], p=[0.35, 0.65])

def generate_salary(cgpa, internships, institute_tier, course, placement_months):
    base = SALARY_BASE[course]
    multiplier = (
        1.0 +
        0.15 * (cgpa - 7.0) / 3.0 +
        0.10 * min(internships / 3.0, 1.0) +
        0.20 * INSTITUTE_TIER_SCORE[institute_tier] -
        0.05 * (placement_months / 24.0)
    )
    salary = base * max(multiplier, 0.7)
    salary += np.random.normal(0, base * 0.08)
    return max(int(salary), 180000)

courses = np.random.choice(list(COURSE_DEMAND.keys()), N,
                           p=[0.30, 0.20, 0.20, 0.10, 0.10, 0.10])
institute_tiers = np.random.choice([1, 2, 3], N, p=[0.20, 0.45, 0.35])
cgpas = np.clip(np.random.normal(7.2, 1.1, N), 4.0, 10.0)
internships = np.random.choice([0, 1, 2, 3], N, p=[0.25, 0.35, 0.28, 0.12])
certs = np.random.randint(0, 5, N)
job_portal_active = np.random.choice([0, 1], N, p=[0.40, 0.60])

placement_months = [
    generate_placement_months(cgpas[i], internships[i], institute_tiers[i], courses[i])
    for i in range(N)
]
salaries = [
    generate_salary(cgpas[i], internships[i], institute_tiers[i],
                    courses[i], placement_months[i])
    for i in range(N)
]

def months_to_label(m):
    if m <= 3:   return "placed_3m"
    elif m <= 6:  return "placed_6m"
    elif m <= 12: return "placed_12m"
    else:         return "delayed"

df = pd.DataFrame({
    "student_id": [f"STU{i:04d}" for i in range(N)],
    "course": courses,
    "institute_tier": institute_tiers,
    "cgpa": np.round(cgpas, 2),
    "internships": internships,
    "certifications": certs,
    "job_portal_active": job_portal_active,
    "industry_demand": [COURSE_DEMAND[c] for c in courses],
    "institute_score": [INSTITUTE_TIER_SCORE[t] for t in institute_tiers],
    "placement_months": placement_months,
    "placement_label": [months_to_label(m) for m in placement_months],
    "starting_salary": salaries
})

df.to_csv("student_data.csv", index=False)
print(f"Dataset created: {len(df)} students")
print(df["placement_label"].value_counts())
print(f"\nAvg salary: Rs {df['starting_salary'].mean():,.0f}")
