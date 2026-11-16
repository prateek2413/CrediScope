
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="CrediScope | Loan Risk Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained Random Forest pipeline
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "random_forest.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()


# Floating Home Button
st.markdown(
    """
    <style>
    .floating-home {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 999999;
        background-color: #4F46E5;
        color: white !important;
        padding: 12px 18px;
        border-radius: 30px;
        text-decoration: none !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }

    .floating-home:hover {
        background-color: #3730A3;
    }
    </style>

    <a href="#home" class="floating-home">⌂ Home</a>
    """,
    unsafe_allow_html=True
)



# Sidebar
st.sidebar.title("📊 CrediScope")

st.sidebar.caption(
    "Loan Default Risk Analytics"
)

st.sidebar.divider()

st.sidebar.subheader("Project Information")

st.sidebar.write("**Primary Model:**")
st.sidebar.success("Random Forest")

st.sidebar.write("**Dataset:**")
st.sidebar.write("LendingClub Historical Data")

st.sidebar.write("**Evaluation Period:**")
st.sidebar.write("2007–2013")

st.sidebar.divider()

st.sidebar.subheader("Model Performance")

st.sidebar.metric(
    "Test ROC-AUC",
    "0.6783"
)

st.sidebar.metric(
    "Test F1 Score",
    "0.3437"
)

st.sidebar.divider()

st.sidebar.info(
    "CrediScope is an educational machine learning "
    "research project. It is not a real loan approval system."
)

st.sidebar.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)


# Main heading
st.markdown(
    '<div id="home" style="scroll-margin-top: 5rem;"></div>',
    unsafe_allow_html=True
)
# Professional Hero Section

st.markdown("""
<style>
.hero-section {
    padding: 30px 32px;
    border-radius: 16px;
    background: linear-gradient(120deg, #111827 0%, #1e3a5f 55%, #263f70 100%);
    border: 1px solid rgba(148, 163, 184, 0.25);
    margin: 10px 0 22px 0;
}

.hero-eyebrow {
    color: #93c5fd;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    color: #ffffff;
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.15;
    margin: 0;
}

.hero-subtitle {
    color: #bfdbfe;
    font-size: 21px;
    font-weight: 500;
    margin-top: 9px;
}

.hero-description {
    color: #d1d5db;
    font-size: 15px;
    line-height: 1.7;
    max-width: 750px;
    margin-top: 15px;
    margin-bottom: 0;
}

.hero-tag {
    display: inline-block;
    margin-top: 20px;
    padding: 7px 12px;
    border-radius: 20px;
    background: rgba(147, 197, 253, 0.14);
    border: 1px solid rgba(147, 197, 253, 0.35);
    color: #bfdbfe;
    font-size: 12px;
    font-weight: 600;
}
</style>

<div class="hero-section">
    <div class="hero-eyebrow">Machine Learning • Credit Risk Analytics</div>
    <div class="hero-title">CrediScope</div>
    <div class="hero-subtitle">Loan Default Risk Prediction</div>
    <p class="hero-description">
        An interactive machine learning dashboard that explores
        historical lending patterns, compares predictive models,
        and estimates loan default risk using borrower and loan attributes.
    </p>
    <span class="hero-tag">RESEARCH & EDUCATIONAL PROJECT</span>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Historical US LendingClub data | Not intended for real-world loan approval."
)



# Top Navigation Bar
st.markdown("""
<style>
.cs-nav {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
    padding: 10px 12px;
    margin: -1rem 0 20px 0;
    background: var(--background-color);
    border-bottom: 1px solid var(--border-color);
}

.cs-nav a {
    padding: 8px 12px;
    border-radius: 7px;
    background: var(--secondary-background-color);
    color: var(--text-color) !important;
    text-decoration: none !important;
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
}

.cs-nav a:hover {
    background: #4F46E5;
    color: white !important;
}

.cs-section {
    scroll-margin-top: 5rem;
}
</style>

<div class="cs-nav">
    <a href="#home">⌂ Home</a>
    <a href="#overview">📊 Overview</a>
    <a href="#performance">📈 Performance</a>
    <a href="#confusion">🎯 Confusion Matrices</a>
    <a href="#importance">🔍 Feature Importance</a>
    <a href="#evaluation">📉 ROC & PR Curves</a>
    <a href="#methodology">📚 Methodology</a>
    <a href="#prediction">🤖 Risk Prediction</a>
</div>
""", unsafe_allow_html=True)

st.divider()


# --------------------------------------------------
# 1. PROJECT OVERVIEW
# --------------------------------------------------

st.markdown(
    '<div id="overview" class="cs-section"></div>',
    unsafe_allow_html=True
)

st.header("Project Overview")

st.write(
    "CrediScope analyzes historical LendingClub loan data "
    "to identify patterns associated with loan defaults "
    "using machine learning."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Historical Loans",
        value="227,957"
    )

with col2:
    st.metric(
        label="Default Rate",
        value="15.50%"
    )

with col3:
    st.metric(
        label="Fully Paid Loans",
        value="192,619"
    )

with col4:
    st.metric(
        label="ML Models Evaluated",
        value="3"
    )

st.caption(
    "Dataset period: June 2007 – December 2013 | "
    "Source: LendingClub historical loan data"
)

st.divider()


# --------------------------------------------------
# 2. MODEL PERFORMANCE
# --------------------------------------------------

st.markdown(
    '<div id="performance" class="cs-section"></div>',
    unsafe_allow_html=True
)

st.header("Model Performance")

st.write(
    "Evaluation results on the chronological test dataset "
    "(loans issued during 2013)."
)

performance_data = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [0.5482, 0.5613, 0.6153],
    "Precision": [0.2187, 0.2004, 0.2342],
    "Recall": [0.7374, 0.6064, 0.6458],
    "F1 Score": [0.3374, 0.3012, 0.3437],
    "ROC-AUC": [0.6749, 0.6048, 0.6783],
    "PR-AUC": [0.2606, 0.2157, 0.2665]
})

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Best Test Accuracy", "61.53%")

with col2:
    st.metric("Best Test F1 Score", "0.3437")

with col3:
    st.metric("Best Test ROC-AUC", "0.6783")

st.subheader("Model Comparison")

st.dataframe(
    performance_data.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.4f}",
        "ROC-AUC": "{:.4f}",
        "PR-AUC": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.subheader("Visual Model Comparison")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("Classification Metrics")

    classification_metrics = performance_data.set_index(
        "Model"
    )[["Accuracy", "Precision", "Recall", "F1 Score"]]

    st.bar_chart(classification_metrics)

with chart_col2:
    st.write("Ranking Metrics")

    ranking_metrics = performance_data.set_index(
        "Model"
    )[["ROC-AUC", "PR-AUC"]]

    st.bar_chart(ranking_metrics)

st.info(
    "Random Forest achieved the best overall test performance. "
    "Logistic Regression achieved the highest recall (73.74%), "
    "meaning it identified a larger proportion of actual defaults."
)

st.caption(
    "All metrics use a classification threshold of 0.50. "
    "The test dataset was not used for model selection."
)

st.divider()


# --------------------------------------------------
# 3. CONFUSION MATRIX ANALYSIS
# --------------------------------------------------

st.markdown(
    '<div id="confusion" class="cs-section"></div>',
    unsafe_allow_html=True
)

st.header("Confusion Matrix Analysis")

st.write(
    "Confusion matrices show the classification performance "
    "of each model on the 2013 test dataset."
)

confusion_matrices = {
    "Logistic Regression": [
        [58400, 55380],
        [5521, 15503]
    ],
    "Decision Tree": [
        [62912, 50868],
        [8275, 12749]
    ],
    "Random Forest": [
        [69371, 44409],
        [7446, 13578]
    ]
}

matrix_cols = st.columns(3)

for col, (model_name, matrix) in zip(
    matrix_cols, confusion_matrices.items()
):
    with col:
        st.subheader(model_name)

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual Fully Paid", "Actual Default"],
            columns=["Predicted Fully Paid", "Predicted Default"]
        )

        st.dataframe(
            matrix_df,
            use_container_width=True
        )

st.caption(
    "Rows represent actual classes and columns represent "
    "predicted classes. Class 0 = Fully Paid; Class 1 = Default."
)

st.info(
    "False negatives represent actual defaults incorrectly "
    "classified as fully paid. False positives represent "
    "fully paid loans incorrectly classified as defaults."
)

st.divider()


# --------------------------------------------------
# 4. FEATURE IMPORTANCE
# --------------------------------------------------

st.markdown(
    '<div id="importance" class="cs-section"></div>',
    unsafe_allow_html=True
)

st.header("Feature Importance Analysis")

st.write(
    "The following features had the highest impurity-based "
    "importance in the trained Random Forest model."
)

importance_path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "random_forest_feature_importance.csv"
)

importance_df = pd.read_csv(importance_path)

importance_df.columns = (
    importance_df.columns.str.strip().str.lower()
)

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
).head(15).copy()

importance_df["feature"] = (
    importance_df["feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

importance_df["feature"] = (
    importance_df["feature"]
    .str.replace("grade_", "Grade ", regex=False)
    .str.replace("purpose_", "Purpose: ", regex=False)
    .str.replace("term_", "Term:", regex=False)
    .str.replace("emp_length_", "Employment: ", regex=False)
    .str.replace("home_ownership_", "Home: ", regex=False)
    .str.replace("verification_status_", "Verification: ", regex=False)
    .str.strip()
)

feature_names = {
    "int_rate": "Interest Rate",
    "annual_inc": "Annual Income",
    "revol_util": "Revolving Utilization",
    "dti": "Debt-to-Income Ratio",
    "revol_bal": "Revolving Balance",
    "installment": "Monthly Installment",
    "total_acc": "Total Credit Accounts",
    "loan_amnt": "Loan Amount",
    "open_acc": "Open Credit Accounts",
    "inq_last_6mths": "Recent Credit Inquiries",
    "pub_rec": "Public Records",
    "delinq_2yrs": "Recent Delinquencies"
}

importance_df["feature"] = (
    importance_df["feature"]
    .map(feature_names)
    .fillna(importance_df["feature"])
)

st.bar_chart(
    importance_df.set_index("feature")["importance"],
    horizontal=True
)

st.caption(
    "Feature importance represents the relative contribution "
    "of features to tree-splitting decisions. It does not "
    "establish causation."
)

st.divider()


# --------------------------------------------------
# 5. ADVANCED MODEL EVALUATION
# --------------------------------------------------

st.markdown(
    '<div id="evaluation" class="cs-section"></div>',
    unsafe_allow_html=True
)

st.header("Advanced Model Evaluation")

st.write(
    "ROC and Precision-Recall curves provide additional "
    "insights into the classification performance of our models."
)

research_path = (
    Path(__file__).resolve().parents[1]
    / "research"
)

roc_path = research_path / "roc_curve.png"
pr_path = research_path / "precision_recall_curve.png"

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("ROC Curve")

    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)
    else:
        st.warning("ROC curve image not found.")

    st.caption(
        "Random Forest ROC-AUC: 0.6783"
    )

with chart_col2:
    st.subheader("Precision-Recall Curve")

    if pr_path.exists():
        st.image(str(pr_path), use_container_width=True)
    else:
        st.warning("Precision-Recall curve image not found.")

    st.caption(
        "Random Forest PR-AUC: 0.2665"
    )

st.info(
    "ROC-AUC measures the model's ability to distinguish "
    "between classes across classification thresholds. "
    "PR-AUC focuses on the trade-off between precision and recall, "
    "which is especially useful for default detection."
)

st.divider()


# --------------------------------------------------
# 6. DATASET AND RESEARCH METHODOLOGY
# --------------------------------------------------

st.markdown(
    '<div id="methodology" class="cs-section"></div>',
    unsafe_allow_html=True
)

st.header("Dataset & Research Methodology")

st.write(
    "CrediScope uses historical LendingClub loan records "
    "to develop and evaluate machine learning models "
    "for binary loan default classification."
)

st.subheader("Dataset Summary")

dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

with dataset_col1:
    st.metric("Original Dataset", "2.26 Million")

with dataset_col2:
    st.metric("Eligible Records", "227,957")

with dataset_col3:
    st.metric("Selected Period", "2007–2013")

st.subheader("Chronological Data Splitting")

split_data = pd.DataFrame({
    "Dataset": ["Training", "Validation", "Testing"],
    "Period": ["2007–2011", "2012", "2013"],
    "Records": [39786, 53367, 134804],
    "Default Rate": ["14.25%", "16.20%", "15.60%"]
})

st.dataframe(
    split_data,
    use_container_width=True,
    hide_index=True
)

st.subheader("Research Workflow")

st.markdown("""
1. **Data Collection:** Historical LendingClub loan dataset.
2. **Data Cleaning:** Selected completed loan outcomes and excluded incomplete or ongoing records.
3. **Feature Engineering:** Prepared numerical and categorical borrower and loan attributes.
4. **Preprocessing:** Median imputation, standardization and one-hot encoding.
5. **Model Training:** Logistic Regression, Decision Tree and Random Forest.
6. **Model Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC and PR-AUC.
7. **Model Selection:** Random Forest selected as the primary demonstration model.
""")

st.warning(
    "The chronological split uses loans issued from 2007 to 2011 "
    "for training, 2012 for validation and 2013 for testing. "
    "The relatively small training set and historical nature "
    "of the data are limitations of this experiment."
)

st.divider()


# --------------------------------------------------
# 7. LOAN DEFAULT RISK PREDICTION
# --------------------------------------------------

st.markdown(
    '<div id="prediction" class="cs-section"></div>',
    unsafe_allow_html=True
)

st.header("Enter Loan Details")

with st.form("loan_prediction_form"):

    st.subheader("1. Loan Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt = st.number_input(
            "Loan Amount (USD)",
            min_value=725.0,
            value=15000.0,
            step=500.0
        )

        term = st.selectbox(
            "Loan Term",
            ["36 months", "60 months"]
        )

    with col2:
        int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            value=12.5,
            step=0.1
        )

        installment = st.number_input(
            "Monthly Installment (USD)",
            min_value=0.0,
            value=500.0,
            step=10.0
        )

    with col3:
        grade = st.selectbox(
            "Loan Grade",
            ["A", "B", "C", "D", "E", "F", "G"]
        )

        purpose = st.selectbox(
            "Loan Purpose",
            [
                "debt_consolidation", "credit_card",
                "home_improvement", "major_purchase",
                "small_business", "car", "medical",
                "moving", "vacation", "house",
                "wedding", "renewable_energy",
                "educational", "other"
            ]
        )

    st.divider()
    st.subheader("2. Borrower Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        annual_inc = st.number_input(
            "Annual Income (USD)",
            min_value=0.0,
            value=65000.0,
            step=1000.0
        )

        emp_length = st.selectbox(
            "Employment Length",
            [
                "Not provided", "< 1 year", "1 year",
                "2 years", "3 years", "4 years",
                "5 years", "6 years", "7 years",
                "8 years", "9 years", "10+ years"
            ]
        )

    with col2:
        home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "MORTGAGE", "OWN", "OTHER", "ANY", "NONE"]
        )

        verification_status = st.selectbox(
            "Income Verification",
            ["Not Verified", "Verified", "Source Verified"]
        )

    with col3:
        dti = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            value=18.0,
            step=0.5
        )

        revol_util = st.number_input(
            "Revolving Utilization (%)",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    st.divider()
    st.subheader("3. Credit History")

    col1, col2, col3 = st.columns(3)

    with col1:
        delinq_2yrs = st.number_input(
            "Delinquencies (Last 2 Years)",
            min_value=0,
            value=0,
            step=1
        )

        inq_last_6mths = st.number_input(
            "Credit Inquiries (Last 6 Months)",
            min_value=0,
            value=0,
            step=1
        )

    with col2:
        open_acc = st.number_input(
            "Open Credit Accounts",
            min_value=0,
            value=10,
            step=1
        )

        total_acc = st.number_input(
            "Total Credit Accounts",
            min_value=0,
            value=20,
            step=1
        )

    with col3:
        revol_bal = st.number_input(
            "Revolving Balance (USD)",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

        pub_rec = st.number_input(
            "Public Records",
            min_value=0,
            value=0,
            step=1
        )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Loan Default Risk",
        type="primary",
        use_container_width=True
    )


# Prediction Results
if submitted:

    input_data = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "term": " " + term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "emp_length": (
            None if emp_length == "Not provided"
            else emp_length
        ),
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "pub_rec": pub_rec
    }])

    prediction = model.predict(input_data)[0]
    score = model.predict_proba(input_data)[0][1]

    st.divider()
    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("Higher Predicted Default Risk")
        else:
            st.success("Lower Predicted Default Risk")

    with col2:
        st.metric(
            "Model Default Score",
            f"{score * 100:.2f}%"
        )

    st.caption(
        "The score is an uncalibrated model output, not a "
        "verified real-world probability of default."
    )

    st.write(
        "**Prediction:**",
        "Default" if prediction == 1 else "Fully Paid"
    )

    st.warning(
        "This prediction is for educational and research purposes "
        "only. It should not be used as the sole basis for lending decisions."
    )
