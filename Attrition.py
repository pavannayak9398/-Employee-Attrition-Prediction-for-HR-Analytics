
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from lightgbm import LGBMClassifier

# Config
st.set_page_config(layout="wide", page_title="Attrition Predictor")

# Inject CSS for background and layout
st.markdown("""
    <style>
        /* Background Gradient */
        .stApp {
            background: linear-gradient(135deg, #e0f7fa, #fff);
            font-family: 'Segoe UI', sans-serif;
        }

        /* Headings */
        h1, h2, h3, .stMarkdown {
            color: #004d40;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #b2dfdb;
            padding-top: 20px;
        }

        /* DataFrame Preview */
        .stDataFrame {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }

        /* Button Styling */
        div.stButton > button:first-child {
            background-color: #00796b;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            font-size: 16px;
        }
        div.stButton > button:hover {
            background-color: #004d40;
        }

        /* Result Section */
        .stMarkdown, .stSubheader {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)


st.title("Employee Attrition Predictor")
st.markdown("Enter employee details to predict attrition risk and download a report.")

# Slidebar Inputs
st.sidebar.header("Categorical Inputs")

business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
department = st.sidebar.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
education_field = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
job_role = st.sidebar.selectbox("Job Role", [
    'Sales Executive', 'Research Scientist', 'Laboratory Technician',
    'Manufacturing Director', 'Healthcare Representative', 'Manager',
    'Sales Representative', 'Research Director', 'Human Resources'
])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

# Numerical Inputs
st.subheader("Numerical Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 60, 35)
    education = st.number_input("Education Level (1-5)", 1, 5, 3)
    job_level = st.number_input("Job Level", 1, 5, 2)
    percent_hike = st.number_input("Percent Salary Hike", 10, 25, 15)
    total_years = st.number_input("Total Working Years", 0, 40, 10)
    years_company = st.number_input("Years At Company", 0, 30, 5)
    years_since_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
    daily_rate = st.number_input("Daily Rate", 100, 1500, 1100)

with col2:
    env_satisfaction = st.number_input("Environment Satisfaction", 1, 4, 3)
    job_satisfaction = st.number_input("Job Satisfaction", 1, 4, 4)
    performance_rating = st.number_input("Performance Rating", 1, 4, 3)
    training_times = st.number_input("Training Times Last Year", 0, 10, 3)
    years_current_role = st.number_input("Years in Current Role", 0, 20, 3)
    relationship_satisfaction = st.number_input("Relationship Satisfaction", 1, 4, 3)
    distance = st.number_input("Distance From Home", 1, 30, 10)

with col3:
    hourly_rate = st.number_input("Hourly Rate", 30, 150, 85)
    job_involvement = st.number_input("Job Involvement", 1, 4, 3)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 7000)
    monthly_rate = st.number_input("Monthly Rate", 2000, 20000, 14000)
    stock_option = st.number_input("Stock Option Level", 0, 3, 1)
    years_with_manager = st.number_input("Years With Current Manager", 0, 20, 2)
    num_companies = st.number_input("Num Companies Worked", 0, 10, 2)
    work_life = st.number_input("WorkLife Balance", 1, 4, 3)

# Creating Input DataFrame
df=pd.DataFrame([{
    'Age': age,
    'BusinessTravel': business_travel,
    'DailyRate': daily_rate,
    'Department': department,
    'DistanceFromHome': distance,
    'Education': education,
    'EducationField': education_field,
    'EnvironmentSatisfaction': env_satisfaction,
    'HourlyRate': hourly_rate,
    'JobInvolvement': job_involvement,
    'JobLevel': job_level,
    'JobRole': job_role,
    'JobSatisfaction': job_satisfaction,
    'MaritalStatus': marital_status,
    'MonthlyIncome': monthly_income,
    'MonthlyRate': monthly_rate,
    'NumCompaniesWorked': num_companies,
    'OverTime': overtime,
    'PercentSalaryHike': percent_hike,
    'PerformanceRating': performance_rating,
    'RelationshipSatisfaction': relationship_satisfaction,
    'StockOptionLevel': stock_option,
    'TotalWorkingYears': total_years,
    'TrainingTimesLastYear': training_times,
    'WorkLifeBalance': work_life,
    'YearsAtCompany': years_company,
    'YearsInCurrentRole': years_current_role,
    'YearsSinceLastPromotion': years_since_promotion,
    'YearsWithCurrManager': years_with_manager
}])

# Data Preview
st.subheader("Preview of Input Data")
st.dataframe(df, use_container_width=True)

# Preprocessing
def preprocess_employee_data(df):
    df = df.copy()
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Income_per_Level'] = df['MonthlyIncome'] / df['JobLevel']
    df.drop(['MonthlyIncome', 'JobLevel'], axis=1, inplace=True)

    numerical_cols = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobSatisfaction', 'MonthlyRate',
        'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]

    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    scaler = StandardScaler()
    for col in numerical_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])

    return df

# Load Models and columns
model = joblib.load("lgbm_tuned_model.pkl")

expected_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
       'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
       'Department_Research & Development', 'Department_Sales',
       'EducationField_Life Sciences', 'EducationField_Marketing',
       'EducationField_Medical', 'EducationField_Other',
       'EducationField_Technical Degree', 'JobRole_Human Resources',
       'JobRole_Laboratory Technician', 'JobRole_Manager',
       'JobRole_Manufacturing Director', 'JobRole_Research Director',
       'JobRole_Research Scientist', 'JobRole_Sales Executive',
       'JobRole_Sales Representative', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'Income_per_Level']


# Predict button
if st.button("Predict Attrition"):
    processed = preprocess_employee_data(df)

    # Ensure all expected columns are present
    for col in expected_cols:
        if col not in processed.columns:
            processed[col] = 0
    processed = processed[expected_cols]

    # Predict
    pred = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0][1]

    # Risk level logic
    def get_risk_level(prob):
        if prob < 0.25:
            st.success(f"Low Risk (Attrition Probability: {prob:.2f})")
            return "No immediate action required. Maintain current employee engagement levels."
        elif prob < 0.60:
            st.warning(f"Medium Risk (Attrition Probability: {prob:.2f})")
            return "Monitor engagement and review career growth opportunities."
        else:
            st.error(f"High Risk (Attrition Probability: {prob:.2f})")
            return "Consider discussing role satisfaction and possible internal mobility options."

    risk_level = get_risk_level(proba)

    # Display Results
    st.subheader("Prediction Results")
    st.markdown(f"**Prediction:** {'Yes' if pred == 1 else 'No'}")
    st.markdown(f"**Probability of Attrition:** `{proba:.2f}`")
    st.markdown(f"**Advice/Suggestion:** `{risk_level}`")
