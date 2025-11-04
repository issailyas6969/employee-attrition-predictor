# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os


st.set_page_config(page_title="💼 Employee Attrition Predictor", layout="centered")


MODEL_PATH = "model/adaboost_pipeline.pkl"


FEATURE_COLUMNS = [
    'Age', 'Department', 'DistanceFromHome', 'EducationField', 'JobInvolvement',
    'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"⚠️ Model file not found at: {path}")
        return None
    return joblib.load(path)

model = load_model(MODEL_PATH)

# -------------------------
# Page Header
# -------------------------
st.markdown("""
    <style>
        h1 {
            text-align: center;
            color: #1f77b4;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            font-weight: bold;
            padding: 0.6em 1.5em;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #155d8b;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Employee Attrition Predictor")
st.write("Predict whether an employee is likely to leave using key HR metrics.")

# -------------------------
# Mode Selection
# -------------------------
mode = st.radio("Choose Input Mode", ("🧍 Single Entry", "📂 Batch Upload"))

# -------------------------
# Prediction Helper
# -------------------------
def predict_dataframe(df: pd.DataFrame):
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(df))
    return pd.DataFrame({
        "Prediction": preds,
        "Attrition_Probability": probs
    })

# -------------------------
# Single Input Mode
# -------------------------
if mode == "🧍 Single Entry":
    if model is None:
        st.stop()

    st.subheader("Enter Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", 18, 60, 30)
        Department = st.selectbox("Department", ["Sales", "R&D", "HR"])
        DistanceFromHome = st.slider("Distance From Home (km)", 0, 50, 10)
        EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
        JobInvolvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
        JobLevel = st.slider("Job Level", 1, 5, 2)
        JobRole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Manager", "Technician", "HR", "Others"])
        JobSatisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with col2:
        MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
        NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 10, 2)
        PercentSalaryHike = st.slider("Percent Salary Hike (%)", 0, 30, 15)
        PerformanceRating = st.slider("Performance Rating (1-4)", 1, 4, 3)
        TotalWorkingYears = st.slider("Total Working Years", 0, 40, 5)
        TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 10, 2)
        WorkLifeBalance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
        YearsAtCompany = st.slider("Years at Company", 0, 20, 5)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
        YearsWithCurrManager = st.slider("Years With Current Manager", 0, 15, 3)

    if st.button("🔍 Predict Attrition"):
        input_df = pd.DataFrame([[
            Age, Department, DistanceFromHome, EducationField, JobInvolvement,
            JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome,
            NumCompaniesWorked, PercentSalaryHike, PerformanceRating, TotalWorkingYears,
            TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany,
            YearsSinceLastPromotion, YearsWithCurrManager
        ]], columns=FEATURE_COLUMNS)

        result = predict_dataframe(input_df)
        prob = result.iloc[0]["Attrition_Probability"]
        label = "🚨 Likely to Leave" if result.iloc[0]["Prediction"] == 1 else "✅ Likely to Stay"

        st.markdown(f"""
            <div style="background-color:#f0f9ff;border-radius:10px;padding:20px;text-align:center;">
                <h3>{label}</h3>
                <p><b>Attrition Probability:</b> {prob:.2%}</p>
            </div>
        """, unsafe_allow_html=True)

# -------------------------
# Batch Mode
# -------------------------
else:
    st.info("Upload a CSV with columns matching your training features.")
    uploaded = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            if st.button("⚡ Run Batch Prediction"):
                results = predict_dataframe(df)
                output = pd.concat([df, results], axis=1)
                st.success("✅ Predictions complete!")
                st.dataframe(output.head(10))

                # Download option
                csv = output.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions CSV",
                    data=csv,
                    file_name="attrition_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.markdown("---")
st.caption("Made using Streamlit and AdaBoost Classifier.")
