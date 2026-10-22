from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Machine Learning Imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern UI Design
st.markdown(
    """
    <style>
    /* Global Styles */
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Card Component Styling */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    
    /* Custom Risk Status Badges */
    .badge-approved {
        background-color: #065F46;
        color: #34D399;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }
    
    .badge-rejected {
        background-color: #991B1B;
        color: #FCA5A5;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }
    
    /* Section Headers */
    .section-header {
        border-left: 4px solid #3B82F6;
        padding-left: 12px;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #60A5FA;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. MODEL PIPELINE
# ==========================================
@st.cache_resource
def load_or_train_pipeline():
    """
    Loads saved model bundle if present.
    If not found, trains exact pipeline from notebook parameters and saves it.
    """
    model_bundle_path = Path("credit_risk_model_bundle.joblib")
    data_path = Path("credit_risk_dataset.csv")

    if model_bundle_path.is_file():
        return joblib.load(model_bundle_path)

    # Re-train using notebook exact steps if model file isn't present
    if not data_path.is_file():
        st.error(
            f"Dataset '{data_path}' not found! Please upload or make dataset available."
        )
        st.stop()

    df = pd.read_csv(data_path)

    # Data Cleaning (Notebook Exact Logic)
    df_model = df.drop_duplicates().copy()
    valid_age = df_model["person_age"].between(18, 100)
    valid_employment = df_model["person_emp_length"].isna() | (
        df_model["person_emp_length"] <= df_model["person_age"]
    )
    valid_loan_amount = df_model["loan_amnt"] > 0
    valid_target = df_model["loan_status"].isin([0, 1])
    valid_rows = (
        valid_age & valid_employment & valid_loan_amount & valid_target
    )
    df_model = df_model.loc[valid_rows].reset_index(drop=True)

    X = df_model.drop(columns="loan_status")
    y = df_model["loan_status"].astype("int8")

    # Split
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.50, random_state=42, stratify=y_remaining
    )

    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    # Exact Tuned XGBoost Hyperparameters from Notebook
    tuned_xgb = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", tuned_xgb)]
    )
    pipeline.fit(X_train, y_train)

    # Bundle details
    selected_threshold = 0.27
    bundle = {
        "pipeline": pipeline,
        "threshold": selected_threshold,
        "feature_columns": X.columns.tolist(),
        "target_column": "loan_status",
        "selected_model": "Tuned XGBoost",
        "X_test": X_test,
        "y_test": y_test,
    }
    joblib.dump(bundle, model_bundle_path)
    return bundle


# Initialize Pipeline Bundle
bundle = load_or_train_pipeline()
pipeline = bundle["pipeline"]
selected_threshold = bundle["threshold"]


# ==========================================
# 3. INTERFACE HEADER
# ==========================================
st.title("💳 Credit Risk Assessment & Loan Default Predictor")



# ==========================================
# 4. SIDEBAR INPUT FOR INDIVIDUAL PREDICTION
# ==========================================
st.sidebar.header("📋 Borrower Application Input")

person_age = st.sidebar.number_input(
    "Applicant Age", min_value=18, max_value=100, value=25, step=1
)
person_income = st.sidebar.number_input(
    "Annual Income ($)", min_value=1000, max_value=5000000, value=60000, step=1000
)
person_home_ownership = st.sidebar.selectbox(
    "Home Ownership Status", ["RENT", "OWN", "MORTGAGE", "OTHER"]
)
person_emp_length = st.sidebar.number_input(
    "Employment Length (Years)",
    min_value=0.0,
    max_value=60.0,
    value=3.0,
    step=0.5,
)

st.sidebar.markdown("---")
loan_intent = st.sidebar.selectbox(
    "Loan Intent",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
)
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input(
    "Loan Amount Requested ($)",
    min_value=500,
    max_value=100000,
    value=10000,
    step=500,
)
loan_int_rate = st.sidebar.number_input(
    "Interest Rate (%)", min_value=1.0, max_value=35.0, value=11.0, step=0.1
)

# Automated Feature Calculation
loan_percent_income = round(loan_amnt / max(person_income, 1), 2)
st.sidebar.info(f"💡 Calculated Loan/Income Ratio: **{loan_percent_income:.2f}**")

st.sidebar.markdown("---")
cb_person_default_on_file = st.sidebar.selectbox(
    "Historical Default on Record?", ["N", "Y"]
)
cb_person_cred_hist_length = st.sidebar.number_input(
    "Credit History Length (Years)",
    min_value=0,
    max_value=50,
    value=4,
    step=1,
)


# ==========================================
# 5. TABBED INTERFACE DESIGN
# ==========================================
tab1, tab2, tab3 = st.tabs(
    ["🎯 Single Risk Assessment", "📁 Batch File Evaluation", "📊 Model Metrics & Performance"]
)

# ------------------------------------------
# TAB 1: INDIVIDUAL PREDICTION
# ------------------------------------------
with tab1:
    st.markdown(
        "<h3 class='section-header'>Real-Time Assessment Panel</h3>",
        unsafe_allow_html=True,
    )

    # Input Dataframe Construction
    input_df = pd.DataFrame(
        [
            {
                "person_age": person_age,
                "person_income": person_income,
                "person_home_ownership": person_home_ownership,
                "person_emp_length": person_emp_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_default_on_file": cb_person_default_on_file,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
            }
        ]
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Application Data Review")
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    with col2:
        st.subheader("Decision Output")
        if st.button(
            "🚀 Execute Risk Evaluation", use_container_width=True, type="primary"
        ):
            risk_prob = float(
                pipeline.predict_proba(input_df)[:, 1][0]
            )
            prediction = 1 if risk_prob >= selected_threshold else 0

            st.markdown("---")
            st.metric(
                label="Predicted Default Probability",
                value=f"{risk_prob:.2%}",
                delta=f"Threshold: {selected_threshold:.2f}",
                delta_color="inverse",
            )

            st.progress(risk_prob)

            if prediction == 1:
                st.markdown(
                    "<div class='badge-rejected'>⚠️ HIGH RISK - LOAN REJECTED</div>",
                    unsafe_allow_html=True,
                )
                st.error(
                    f"Risk Probability ({risk_prob:.2%}) exceeds the risk threshold of {selected_threshold:.2f}."
                )
            else:
                st.markdown(
                    "<div class='badge-approved'>✅ LOW RISK - LOAN APPROVED</div>",
                    unsafe_allow_html=True,
                )
                st.success(
                    f"Risk Probability ({risk_prob:.2%}) is within the acceptable safety threshold of {selected_threshold:.2f}."
                )


# ------------------------------------------
# TAB 2: BATCH EVALUATION
# ------------------------------------------
with tab2:
    st.markdown(
        "<h3 class='section-header'>Bulk Applicant Processing</h3>",
        unsafe_allow_html=True,
    )
    st.write(
        "Upload a `.csv` file containing applicant features to run batch risk classifications."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV Data File", type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Sample:")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("⚡ Run Batch Risk Evaluation"):
            try:
                # Calculate probability and predicted status
                probs = pipeline.predict_proba(batch_df)[:, 1]
                preds = (probs >= selected_threshold).astype(int)

                results_df = batch_df.copy()
                results_df["risk_probability"] = np.round(probs, 4)
                results_df["predicted_loan_status"] = preds
                results_df["decision"] = np.where(
                    preds == 1, "REJECTED (High Risk)", "APPROVED (Low Risk)"
                )

                st.subheader("Evaluation Results")
                st.dataframe(results_df, use_container_width=True)

                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_data,
                    file_name="credit_risk_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error evaluating batch data: {str(e)}")


# ------------------------------------------
# TAB 3: MODEL METRICS & EVALUATION
# ------------------------------------------
with tab3:
    st.markdown(
        "<h3 class='section-header'>Notebook-Validated Performance Metrics</h3>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC Score", "0.9497")
    m2.metric("Overall Accuracy", "92.06%")
    m3.metric("Precision (Class 1)", "83.22%")
    m4.metric("Recall (Class 1)", "79.77%")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Confusion Matrix (Held-out Test Set)")
        conf_matrix = np.array([[3628, 171], [215, 848]])
        conf_df = pd.DataFrame(
            conf_matrix,
            columns=["Pred: Non-Default (0)", "Pred: Default (1)"],
            index=["Actual: Non-Default (0)", "Actual: Default (1)"],
        )
        st.table(conf_df)

    with c2:
        st.subheader("Model Parameter Configuration")
        st.json(
            {
                "Algorithm": "Tuned XGBoost Pipeline",
                "Decision Threshold": selected_threshold,
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            }
        )

st.caption("Made by Aryan Singh")