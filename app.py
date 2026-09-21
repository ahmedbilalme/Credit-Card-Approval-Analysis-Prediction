"""
Credit Card Approval Analysis and Prediction
----------------------------------------------
A Streamlit app that lets a user explore the dataset and get a
machine-learning-based Good Credit / Risky Credit prediction.

Author: Maniha Munawar
Internship Project

Note on design: all styling below is done with CSS injected through
Streamlit's supported st.markdown(unsafe_allow_html=True) mechanism.
No JavaScript is used, keeping the project pure Python + Streamlit.
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
# Design system: fonts, color palette, and component styling (CSS only)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --navy: #0F172A;
        --navy-light: #1E293B;
        --teal: #14B8A6;
        --teal-dark: #0D9488;
        --coral: #F43F5E;
        --coral-dark: #E11D48;
        --bg: #F8FAFC;
        --card: #FFFFFF;
        --border: #E2E8F0;
        --text-muted: #64748B;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif !important;
        color: var(--navy) !important;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .main .block-container {
        animation: fadeInUp 0.5s ease-out;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, var(--navy) 0%, var(--teal-dark) 100%);
        border-radius: 16px;
        padding: 40px 36px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
    }
    .hero h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        margin: 0 0 8px 0;
    }
    .hero p {
        color: #E2E8F0;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Stat / metric cards */
    .stat-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.10);
    }
    .stat-card .stat-value {
        font-family: 'Poppins', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--teal-dark);
    }
    .stat-card .stat-label {
        color: var(--text-muted);
        font-size: 0.88rem;
        margin-top: 4px;
    }

    /* Native metric widgets (Data Analysis / Prediction pages) */
    div[data-testid="stMetric"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricValue"] {
        color: var(--teal-dark);
    }

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3);
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(20, 184, 166, 0.4);
        color: white;
    }

    /* Result badges */
    .result-badge {
        border-radius: 14px;
        padding: 22px 24px;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 18px;
        animation: fadeInUp 0.5s ease-out;
    }
    .badge-good {
        background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
        box-shadow: 0 8px 20px rgba(20, 184, 166, 0.35);
    }
    .badge-risky {
        background: linear-gradient(135deg, var(--coral) 0%, var(--coral-dark) 100%);
        box-shadow: 0 8px 20px rgba(244, 63, 94, 0.35);
    }

    /* Probability bars */
    .prob-row { margin-bottom: 14px; }
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.92rem;
        color: var(--navy);
        margin-bottom: 4px;
        font-weight: 500;
    }
    .prob-track {
        background: var(--border);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
    }
    .prob-fill-good {
        height: 100%;
        background: linear-gradient(90deg, var(--teal) 0%, var(--teal-dark) 100%);
        border-radius: 8px;
    }
    .prob-fill-risky {
        height: 100%;
        background: linear-gradient(90deg, var(--coral) 0%, var(--coral-dark) 100%);
        border-radius: 8px;
    }

    /* Images (charts) */
    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--navy);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }
    section[data-testid="stSidebar"] .brand {
        font-family: 'Poppins', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        background: linear-gradient(90deg, #5EEAD4, #2DD4BF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    div[role="radiogroup"] label {
        background: var(--navy-light);
        border-radius: 8px;
        padding: 6px 10px;
        margin-bottom: 4px;
        transition: background 0.15s ease;
    }
    div[role="radiogroup"] label:hover {
        background: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True
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
st.sidebar.markdown('<div class="brand">💳 CreditIQ</div>', unsafe_allow_html=True)
st.sidebar.caption("Internship Project — Maniha Munawar")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate to:",
    ["Home", "Data Analysis", "Prediction", "About"]
)

# ========================================================================
# PAGE 1: HOME
# ========================================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>💳 Credit Card Approval Analysis and Prediction</h1>
            <p>An internship data science project that classifies applicants
            as Good Credit or Risky Credit using machine learning.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

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
        "guaranteed real-world bank approval decision. This is an internship "
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
    with col1:
        st.markdown(
            f"""<div class="stat-card">
                    <div class="stat-value">{data.shape[0]:,}</div>
                    <div class="stat-label">Total Applicants Analyzed</div>
                </div>""",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""<div class="stat-card">
                    <div class="stat-value">{data.shape[1] - 1}</div>
                    <div class="stat-label">Features Used</div>
                </div>""",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""<div class="stat-card">
                    <div class="stat-value">{data['TARGET'].mean() * 100:.2f}%</div>
                    <div class="stat-label">Risky Credit Rate</div>
                </div>""",
            unsafe_allow_html=True
        )

    st.write("")
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
        good_pct = probability[0] * 100
        risky_pct = probability[1] * 100

        st.subheader("Prediction Result")

        if prediction == 0:
            st.markdown(
                '<div class="result-badge badge-good">✅ Good Credit Profile</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-badge badge-risky">⚠️ Risky Credit Profile</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-label"><span>Model-Estimated P(Good Credit)</span><span>{good_pct:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill-good" style="width:{good_pct}%;"></div></div>
            </div>
            <div class="prob-row">
                <div class="prob-label"><span>Model-Estimated P(Risky Credit)</span><span>{risky_pct:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill-risky" style="width:{risky_pct}%;"></div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

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
        **Type:** Internship Project
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
        - Streamlit — web application interface (styled with custom CSS)
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
        - This is an internship project and should not be used for real
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
