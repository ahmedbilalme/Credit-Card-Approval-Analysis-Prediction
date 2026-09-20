"""
Credit Card Approval Analysis and Prediction
----------------------------------------------
A beginner-friendly Streamlit app that lets a user explore the dataset
and get a machine-learning-based Good Credit / Risky Credit prediction.

Author: Ahmed
University Data Science Project
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Approval Analysis and Prediction",
    page_icon="💳",
    layout="wide"
)

# ----------------------------------------------------------------------
# Load the trained model pipeline (preprocessing + Random Forest)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/credit_card_model.pkl")

model = load_model()

# ----------------------------------------------------------------------
# Load the processed dataset (used only for the Data Analysis page)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/final_features.csv")

data = load_data()

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Data Analysis", "Prediction", "About"]
)

# ========================================================================
# PAGE 1: HOME
# ========================================================================
if page == "Home":
    st.title("💳 Credit Card Approval Analysis and Prediction")
    st.markdown("### A Data Science & Machine Learning Project")

    st.write(
        """
        This project analyzes credit card applicant information and credit
        history in order to build a machine learning model that classifies
        applicants as either **Good Credit** or **Risky Credit**.

        The system also provides an interactive interface where a user can
        enter applicant details and receive a model-based prediction.
        """
    )

    st.warning(
        "⚠️ **Important:** This prediction is a machine-learning "
        "classification based on a historical dataset. It is **not** a "
        "guaranteed real-world bank approval decision. This is an academic "
        "project, not a real credit-scoring system."
    )

    st.subheader("Project Objectives")
    st.markdown(
        """
        - Analyze applicant demographic and financial information
        - Build a meaningful credit-risk target from repayment history
        - Compare multiple classification models
        - Provide an easy-to-use prediction interface
        """
    )

    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applicants Analyzed", f"{data.shape[0]:,}")
    col2.metric("Features Used", f"{data.shape[1] - 1}")
    col3.metric(
        "Risky Credit Rate",
        f"{(data['TARGET'].mean() * 100):.2f}%"
    )

    st.write(
        """
        **Source:** [Credit Card Approval Prediction — Kaggle]
        (https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
        """
    )

# ========================================================================
# PAGE 2: DATA ANALYSIS
# ========================================================================
elif page == "Data Analysis":
    st.title("📊 Data Analysis")
    st.write(
        "Below are the key visualizations and statistics generated during "
        "the exploratory data analysis (EDA) stage of this project."
    )

    st.subheader("Summary Statistics")
    st.dataframe(data.describe())

    st.subheader("Target Class Distribution")
    st.image("images/target_distribution.png",
              caption="Good Credit vs Risky Credit — note the strong class imbalance")
    st.write(
        f"Out of {data.shape[0]:,} applicants with credit history, only "
        f"{int(data['TARGET'].sum())} ({data['TARGET'].mean()*100:.2f}%) "
        "were classified as Risky Credit. This class imbalance is an "
        "important factor in how the models were trained and evaluated."
    )

    st.subheader("Age Distribution")
    st.image("images/age_distribution.png")

    st.subheader("Income Distribution")
    st.image("images/income_distribution.png")

    st.subheader("Gender Distribution")
    st.image("images/gender_distribution.png")

    st.subheader("Education Distribution")
    st.image("images/education_distribution.png")

    st.subheader("Income Type Distribution")
    st.image("images/income_type_distribution.png")

    st.subheader("Housing Type Distribution")
    st.image("images/housing_type_distribution.png")

    st.subheader("Annual Income vs Target Class")
    st.image("images/income_vs_target.png")
    st.write(
        "The analysis shows very little difference in income levels between "
        "the Good Credit and Risky Credit groups — income alone does not "
        "strongly separate the two classes in this dataset."
    )

    st.subheader("Risky Credit Rate by Education Level")
    st.image("images/education_vs_target.png")
    st.write(
        "The analysis shows a mild association between lower education "
        "levels and a higher risky-credit rate, though the differences "
        "are modest."
    )

    st.subheader("Years Employed vs Target Class")
    st.image("images/employment_vs_target.png")
    st.write(
        "The analysis shows applicants with shorter employment history "
        "have some association with a higher risky-credit rate, though "
        "the distributions overlap considerably."
    )

# ========================================================================
# PAGE 3: PREDICTION
# ========================================================================
elif page == "Prediction":
    st.title("🔮 Credit Risk Prediction")
    st.write(
        "Enter applicant information below to get a model-based prediction."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["M", "F"])
            own_car = st.selectbox("Owns a Car?", ["Y", "N"])
            own_realty = st.selectbox("Owns Property?", ["Y", "N"])
            children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
            annual_income = st.number_input(
                "Annual Income", min_value=0, max_value=2_000_000, value=180000, step=5000
            )
            age = st.slider("Age", min_value=18, max_value=75, value=35)
            years_employed = st.slider(
                "Years Employed (0 if unemployed/retired)",
                min_value=0.0, max_value=45.0, value=5.0, step=0.5
            )

        with col2:
            income_type = st.selectbox(
                "Income Type",
                sorted(data["Income_Type"].unique())
            )
            education = st.selectbox(
                "Education Level",
                sorted(data["Education"].unique())
            )
            marital_status = st.selectbox(
                "Marital Status",
                sorted(data["Marital_Status"].unique())
            )
            housing_type = st.selectbox(
                "Housing Type",
                sorted(data["Housing_Type"].unique())
            )
            family_size = st.number_input(
                "Family Size", min_value=1, max_value=15, value=2
            )
            occupation_type = st.selectbox(
                "Occupation Type",
                sorted(data["Occupation_Type"].unique())
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Own_Car": own_car,
            "Own_Realty": own_realty,
            "Children": children,
            "Annual_Income": annual_income,
            "Income_Type": income_type,
            "Education": education,
            "Marital_Status": marital_status,
            "Housing_Type": housing_type,
            "Age": age,
            "Years_Employed": years_employed,
            "Family_Size": family_size,
            "Occupation_Type": occupation_type
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.subheader("Prediction Result")

        if prediction == 0:
            st.success("✅ Good Credit Profile")
        else:
            st.error("⚠️ Risky Credit Profile")

        col1, col2 = st.columns(2)
        col1.metric("Model-Estimated P(Good Credit)", f"{probability[0]*100:.1f}%")
        col2.metric("Model-Estimated P(Risky Credit)", f"{probability[1]*100:.1f}%")

        st.caption(
            "These are model-estimated probabilities based on historical "
            "patterns in the dataset, not a guaranteed real-world approval "
            "probability."
        )

# ========================================================================
# PAGE 4: ABOUT
# ========================================================================
elif page == "About":
    st.title("ℹ️ About This Project")

    st.subheader("Project Information")
    st.write(
        """
        **Title:** Credit Card Approval Analysis and Prediction
        **Type:** University Data Science Project
        **Dataset:** [Credit Card Approval Prediction — Kaggle]
        (https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
        """
    )

    st.subheader("Technologies Used")
    st.markdown(
        """
        - Python
        - Pandas & NumPy — data handling
        - Matplotlib & Seaborn — visualization
        - Scikit-learn — preprocessing and machine learning
        - Joblib — model persistence
        - Streamlit — web application interface
        """
    )

    st.subheader("Machine Learning Models Compared")
    st.markdown(
        """
        - Logistic Regression
        - Decision Tree Classifier
        - **Random Forest Classifier (final selected model, based on F1 score)**
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - The dataset is historical and does not reflect current banking policy.
        - Only applicants with recorded credit history could be used for
          training (~36,000 of the ~438,000 total applicant records).
        - The target classes are highly imbalanced (about 98.3% Good Credit
          vs 1.7% Risky Credit), which limits how precisely the minority
          class can be predicted.
        - This is a credit-risk classification proxy, not an actual bank
          approval decision.
        - This is an academic project and should not be used for real
          financial decision-making.
        """
    )

    st.subheader("Future Improvements")
    st.markdown(
        """
        - Hyperparameter tuning (e.g., GridSearchCV)
        - Cross-validation for more robust evaluation
        - Techniques specifically for imbalanced data (e.g., SMOTE)
        - Model explainability (e.g., feature importance, SHAP values)
        - Testing with a larger or more recent dataset
        """
    )
