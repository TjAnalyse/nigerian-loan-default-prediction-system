
# ============================================================
# NIGERIAN LOAN DEFAULT PREDICTION SYSTEM
# Random Forest Classifier
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# ============================================================
# LOAD TRAINED FILES
# ============================================================

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nigerian Loan Default Prediction System",
    page_icon="🏦",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================

st.title("🏦 Nigerian Loan Default Prediction System")

st.markdown("""
This application predicts the likelihood that a loan applicant will default before
loan approval using a **Random Forest Machine Learning Model** trained on a
Nigerian loan dataset.

The prediction is intended to support loan officers during the credit assessment
process.
""")

st.divider()

# ============================================================
# APPLICANT INFORMATION
# ============================================================

left_col, right_col = st.columns(2)

with left_col:

    st.subheader("👤 Applicant Information")

    Age = st.slider(
        "Age",
        min_value=18,
        max_value=80,
        value=35
    )

    employment_status_clients = st.selectbox(
        "Employment Status",
        label_encoders["employment_status_clients"].classes_
    )

    bank_account_type = st.selectbox(
        "Bank Account Type",
        label_encoders["bank_account_type"].classes_
    )

with right_col:

    st.subheader("💰 Loan Information")

    loanamount = st.number_input(
        "Loan Amount (₦)",
        min_value=1000.0,
        value=10000.0,
        step=1000.0
    )

    termdays = st.selectbox(
        "Loan Term (Days)",
        [15, 30, 60, 90]
    )

    loannumber = st.number_input(
        "Current Loan Number",
        min_value=1,
        value=2
    )

st.divider()

# ============================================================
# PREVIOUS LOAN HISTORY
# ============================================================

st.subheader("📊 Previous Loan History")

first_time = st.checkbox(
    "This applicant is a First-time Borrower",
    value=False
)

if first_time:

    total_previous_loans = 0
    avg_previous_loan = 0.0
    max_previous_loan = 0.0
    avg_previous_term = 0.0
    avg_repayment_delay = 0.0
    max_repayment_delay = 0.0
    late_payment_rate = 0.0

    st.info(
        "Previous loan history has been automatically set to zero because this applicant is a first-time borrower."
    )

else:

    col1, col2 = st.columns(2)

    with col1:

        total_previous_loans = st.number_input(
            "Total Previous Loans",
            min_value=0,
            value=1
        )

        avg_previous_loan = st.number_input(
            "Average Previous Loan Amount (₦)",
            min_value=0.0,
            value=10000.0
        )

        max_previous_loan = st.number_input(
            "Maximum Previous Loan Amount (₦)",
            min_value=0.0,
            value=10000.0
        )

        avg_previous_term = st.number_input(
            "Average Previous Loan Term (Days)",
            min_value=0.0,
            value=30.0
        )

    with col2:

        avg_repayment_delay = st.number_input(
            "Average Repayment Delay (Days)",
            value=0.0
        )

        max_repayment_delay = st.number_input(
            "Maximum Repayment Delay (Days)",
            value=0.0
        )

        late_payment_rate = st.slider(
            "Late Payment Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.0
        )

st.divider()

# ============================================================
# PREDICT BUTTON
# ============================================================

predict = st.button(
    "🔍 Predict Loan Risk",
    use_container_width=True
)

if predict:

    # Save original inputs for summary later
    original_data = {
        "Loan Amount (₦)": loanamount,
        "Loan Term (Days)": termdays,
        "Current Loan Number": loannumber,
        "Age": Age,
        "Employment Status": employment_status_clients,
        "Bank Account Type": bank_account_type,
        "First-time Borrower": "Yes" if first_time else "No",
        "Total Previous Loans": total_previous_loans,
        "Average Previous Loan": avg_previous_loan,
        "Maximum Previous Loan": max_previous_loan,
        "Average Previous Loan Term": avg_previous_term,
        "Average Repayment Delay": avg_repayment_delay,
        "Maximum Repayment Delay": max_repayment_delay,
        "Late Payment Rate": late_payment_rate
    }

    # Encode categorical variables
    bank_account_type_encoded = label_encoders[
        "bank_account_type"
    ].transform([bank_account_type])[0]

    employment_status_encoded = label_encoders[
        "employment_status_clients"
    ].transform([employment_status_clients])[0]

    # Create dataframe
    input_data = pd.DataFrame({

        "loanamount": [loanamount],
        "termdays": [termdays],
        "loannumber": [loannumber],
        "bank_account_type": [bank_account_type_encoded],
        "employment_status_clients": [employment_status_encoded],
        "Age": [Age],
        "total_previous_loans": [total_previous_loans],
        "avg_previous_loan": [avg_previous_loan],
        "max_previous_loan": [max_previous_loan],
        "avg_previous_term": [avg_previous_term],
        "avg_repayment_delay": [avg_repayment_delay],
        "max_repayment_delay": [max_repayment_delay],
        "late_payment_rate": [late_payment_rate]

    })

    # Ensure column order matches training
    input_data = input_data[columns]

    # ============================================================
    # SCALE ONLY THE NUMERICAL FEATURES
    # ============================================================

    numeric_columns = [
        "loanamount",
        "termdays",
        "loannumber",
        "Age",
        "total_previous_loans",
        "avg_previous_loan",
        "max_previous_loan",
        "avg_previous_term",
        "avg_repayment_delay",
        "max_repayment_delay",
        "late_payment_rate"
    ]

    # Make a copy of the dataframe
    input_scaled = input_data.copy()

    # Scale ONLY the numerical columns
    input_scaled[numeric_columns] = scaler.transform(
        input_scaled[numeric_columns]
    )

    # ============================================================
    # MODEL PREDICTION
    # ============================================================

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    probability_good = probability[0]
    probability_default = probability[1]

    # ============================================================
    # RISK CLASSIFICATION
    # ============================================================

    if probability_default < 0.30:

        risk = "🟢 LOW RISK"

        recommendation = (
            "The applicant presents a low probability of default. "
            "The loan may be approved under the institution's normal "
            "credit approval procedures."
        )

        risk_color = "green"

    elif probability_default < 0.60:

        risk = "🟡 MODERATE RISK"

        recommendation = (
            "The applicant presents a moderate level of credit risk. "
            "Additional income verification, guarantors or collateral "
            "should be considered before loan approval."
        )

        risk_color = "orange"

    else:

        risk = "🔴 HIGH RISK"

        recommendation = (
            "The applicant presents a high probability of default. "
            "The loan should only be considered after comprehensive "
            "credit assessment and additional financial verification."
        )

        risk_color = "red"

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    st.success("Prediction Completed Successfully")

    st.divider()

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            label="Repayment Probability",
            value=f"{probability_good * 100:.2f}%"
        )

    with metric2:

        st.metric(
            label="Default Probability",
            value=f"{probability_default * 100:.2f}%"
        )

    with metric3:

        if prediction == 0:
            st.success("✅ GOOD CUSTOMER")

        else:
            st.error("❌ LIKELY TO DEFAULT")

    st.divider()

    st.subheader("Risk Assessment")

    if risk_color == "green":
        st.success(risk)

    elif risk_color == "orange":
        st.warning(risk)

    else:
        st.error(risk)

    st.info(recommendation)

        # ============================================================
    # APPLICANT SUMMARY
    # ============================================================

    st.divider()

    st.subheader("Applicant Summary")

    summary_df = pd.DataFrame(
        list(original_data.items()),
        columns=["Feature", "Value"]
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    # ============================================================
    # LOAN DECISION SUMMARY
    # ============================================================

    st.divider()

    st.subheader("Loan Decision Summary")

    decision = pd.DataFrame({

        "Prediction": [
            "Good Customer"
            if prediction == 0
            else "Likely to Default"
        ],

        "Risk Level": [risk],

        "Repayment Probability (%)": [
            round(probability_good * 100, 2)
        ],

        "Default Probability (%)": [
            round(probability_default * 100, 2)
        ]

    })

    st.dataframe(
        decision,
        use_container_width=True,
        hide_index=True
    )

    # ============================================================
    # DOWNLOAD RESULTS
    # ============================================================

    csv = decision.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="loan_prediction_result.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    with st.expander("Model Information"):

        st.markdown("""
### Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Target Variable:** Loan Default (Good / Bad)
- **Prediction Method:** Probability Estimation (`predict_proba`)
- **Feature Scaling:** StandardScaler (Numerical Features Only)
- **Categorical Encoding:** Label Encoding

### Input Features

- Loan Amount
- Loan Term
- Current Loan Number
- Age
- Employment Status
- Bank Account Type
- Total Previous Loans
- Average Previous Loan Amount
- Maximum Previous Loan Amount
- Average Previous Loan Term
- Average Repayment Delay
- Maximum Repayment Delay
- Late Payment Rate

The model estimates the probability that an applicant will default on a loan before approval.
""")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
"""
**Nigerian Loan Default Prediction System**

Developed as an undergraduate research project using Machine Learning.

**Model:** Random Forest Classifier

This application provides decision support for loan approval by estimating an applicant's probability of default. Predictions should complement professional credit assessment and should not be used as the sole basis for lending decisions.
"""
)   
