import os, sys, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
import shap

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
SAVED_DIR = os.path.join(BASE, "models", "saved")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVED_DIR, exist_ok=True)
print("Step 1: Generating data...")
np.random.seed(42)
N = 2000
COURSE_DEMAND = {"Engineering":0.75,"MBA":0.70,"Data Science":0.85,"Nursing":0.60,"Law":0.45,"Arts":0.35}
TIER_SCORE = {1:1.0,2:0.7,3:0.45}
SALARY_BASE = {"Engineering":600000,"MBA":750000,"Data Science":900000,"Nursing":380000,"Law":500000,"Arts":280000}
def gen_months(cgpa,intern,tier,course):
    s=0.35*(cgpa/10)+0.25*min(intern/3,1)+0.25*TIER_SCORE[tier]+0.15*COURSE_DEMAND[course]
    s+=np.random.normal(0,0.08)
    s=np.clip(s,0,1)
    if s>=0.75: return np.random.choice([3,6],p=[0.75,0.25])
    elif s>=0.55: return np.random.choice([3,6,12],p=[0.20,0.55,0.25])
    elif s>=0.35: return np.random.choice([6,12,24],p=[0.20,0.50,0.30])
    else: return np.random.choice([12,24],p=[0.35,0.65])
def gen_salary(cgpa,intern,tier,course,months):
    base=SALARY_BASE[course]
    m=1.0+0.15*(cgpa-7)/3+0.10*min(intern/3,1)+0.20*TIER_SCORE[tier]-0.05*(months/24)
    sal=base*max(m,0.7)+np.random.normal(0,base*0.08)
    return max(int(sal),180000)
courses=np.random.choice(list(COURSE_DEMAND.keys()),N,p=[0.30,0.20,0.20,0.10,0.10,0.10])
tiers=np.random.choice([1,2,3],N,p=[0.20,0.45,0.35])
cgpas=np.clip(np.random.normal(7.2,1.1,N),4.0,10.0)
interns=np.random.choice([0,1,2,3],N,p=[0.25,0.35,0.28,0.12])
certs=np.random.randint(0,5,N)
portal=np.random.choice([0,1],N,p=[0.40,0.60])
pm=[gen_months(cgpas[i],interns[i],tiers[i],courses[i]) for i in range(N)]
sal=[gen_salary(cgpas[i],interns[i],tiers[i],courses[i],pm[i]) for i in range(N)]
def label(m):
    if m<=3: return "placed_3m"
    elif m<=6: return "placed_6m"
    elif m<=12: return "placed_12m"
    else: return "delayed"
df=pd.DataFrame({"student_id":[f"STU{i:04d}" for i in range(N)],"course":courses,"institute_tier":tiers,"cgpa":np.round(cgpas,2),"internships":interns,"certifications":certs,"job_portal_active":portal,"industry_demand":[COURSE_DEMAND[c] for c in courses],"institute_score":[TIER_SCORE[t] for t in tiers],"placement_months":pm,"placement_label":[label(m) for m in pm],"starting_salary":sal})
df.to_csv(os.path.join(DATA_DIR,"student_data.csv"),index=False)
print(f"Data done: {len(df)} students")
print("Step 2: Training models...")
df["employability_score"]=0.35*(df["cgpa"]/10)+0.25*(df["internships"]/3).clip(0,1)+0.20*df["institute_score"]+0.15*df["industry_demand"]+0.05*(df["certifications"]/5).clip(0,1)
df["engagement_score"]=0.6*df["job_portal_active"]+0.4*(df["certifications"]/5).clip(0,1)
le=LabelEncoder()
df["course_enc"]=le.fit_transform(df["course"])
FEATURES=["cgpa","internships","certifications","job_portal_active","institute_tier","industry_demand","institute_score","employability_score","engagement_score","course_enc"]
LABELS={"placed_3m":0,"placed_6m":1,"placed_12m":2,"delayed":3}
df["label_enc"]=df["placement_label"].map(LABELS)
X=df[FEATURES]; y=df["label_enc"]; ys=df["starting_salary"]
Xtr,Xte,ytr,yte,ystr,yste=train_test_split(X,y,ys,test_size=0.2,random_state=42,stratify=y)
clf=XGBClassifier(n_estimators=150,max_depth=5,learning_rate=0.08,subsample=0.85,colsample_bytree=0.85,eval_metric="mlogloss",random_state=42)
clf.fit(Xtr,ytr,eval_set=[(Xte,yte)],verbose=False)
reg=XGBRegressor(n_estimators=150,max_depth=4,learning_rate=0.08,subsample=0.85,colsample_bytree=0.85,random_state=42)
reg.fit(Xtr,ystr,eval_set=[(Xte,yste)],verbose=False)
exp=shap.TreeExplainer(clf)
pickle.dump(clf,open(os.path.join(SAVED_DIR,"classifier.pkl"),"wb"))
pickle.dump(reg,open(os.path.join(SAVED_DIR,"regressor.pkl"),"wb"))
pickle.dump(exp,open(os.path.join(SAVED_DIR,"explainer.pkl"),"wb"))
pickle.dump(le,open(os.path.join(SAVED_DIR,"label_encoder.pkl"),"wb"))
pickle.dump(FEATURES,open(os.path.join(SAVED_DIR,"features.pkl"),"wb"))
df.to_csv(os.path.join(DATA_DIR,"student_data_features.csv"),index=False)
print("Models saved!")
print("")
print("NOW RUN THIS COMMAND:")
print("streamlit run frontend\\app.py")