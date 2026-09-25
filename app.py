import streamlit as st
import pandas as pd
import os
from ml.fraud_model import predict_fraud
from ml.credit_model import predict_credit_risk
from ml.identity_model import analyze_identity_risk

# Page Configuration
st.set_page_config(
    page_title="AI Fraud Intelligence Platform",
    page_icon="🛡️",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("🛡️ AI Fraud Platform")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Chat Assistant", "Fraud Detection", "Credit Risk", "Identity Analytics", "Evaluation"]
)

# --- Dashboard Page ---
if page == "Dashboard":
    st.title("🚀 AI Fraud Intelligence Dashboard")
    st.markdown("""
    Welcome to the **AI-Powered Fraud, Credit Risk & Identity Intelligence Platform**.
    This platform integrates RAG-based document intelligence and ML-based predictive analytics to help analysts investigate financial crimes.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", "4", "PDFs")
    with col2:
        st.metric("ML Models", "3", "Active")
    with col3:
        st.metric("Avg Retrieval Accuracy", "92%", "+2%")

    st.divider()
    st.subheader("Quick Access")
    st.info("Use the sidebar to navigate between the RAG Chat Assistant and the predictive ML models.")

# --- Chat Assistant Page ---
elif page == "Chat Assistant":
    st.title("💬 RAG Chat Assistant")
    st.markdown("Ask questions about fraud policies, KYC, and AML guidelines.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("What is synthetic identity fraud?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Placeholder for RAG logic (Phase 3 & 4)
            response = f"This is a placeholder response. In Phase 3/4, I will retrieve context from ChromaDB and generate an answer for: '{prompt}'"
            st.markdown(response)
            st.caption("Sources: [Fraud_Policy.pdf, Page 2] | Similarity Score: 0.89")
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Fraud Detection Page ---
elif page == "Fraud Detection":
    st.title("🚩 Fraud Detection")
    st.markdown("Predict the probability of a transaction being fraudulent.")
    
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
            transaction_type = st.selectbox("Transaction Type", ['transfer', 'payment', 'cash_out', 'cash_in'])
            country = st.selectbox("Country", ['US', 'UK', 'IN', 'CA', 'DE'])
        with col2:
            merchant = st.selectbox("Merchant", ['Amazon', 'Walmart', 'Apple', 'Target', 'Netflix'])
            device = st.selectbox("Device", ['Mobile', 'Desktop', 'Tablet'])
            prev_fraud = st.number_input("Previous Fraud Count", min_value=0, value=0)
        
        submit = st.form_submit_button("Predict Fraud")
        
        if submit:
            # Integration with trained ML model
            input_data = {
                'amount': amount,
                'transaction_type': transaction_type,
                'country': country,
                'merchant': merchant,
                'device': device,
                'previous_fraud_count': prev_fraud
            }
            prob = predict_fraud(input_data)
            
            st.subheader("Prediction Result")
            if prob > 0.7:
                st.warning(f"Fraud Probability: {prob:.0%}")
                st.progress(prob)
                st.write("⚠️ **High Risk Detected**. This transaction shows patterns similar to known fraudulent activities.")
            elif prob > 0.3:
                st.info(f"Fraud Probability: {prob:.0%}")
                st.progress(prob)
                st.write("🟡 **Medium Risk**. Further verification is recommended.")
            else:
                st.success(f"Fraud Probability: {prob:.0%}")
                st.progress(prob)
                st.write("✅ **Low Risk**. Transaction appears normal.")

# --- Credit Risk Page ---
elif page == "Credit Risk":
    st.title("💳 Credit Risk Prediction")
    st.markdown("Assess the creditworthiness of a loan applicant.")
    
    with st.form("credit_form"):
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("Annual Income ($)", min_value=0, value=50000)
            loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, value=10000)
        with col2:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            employment_years = st.number_input("Employment Years", min_value=0, value=5)
            debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
        
        submit = st.form_submit_button("Analyze Risk")
        
        if submit:
            # Integration with trained ML model
            input_data = {
                'income': income,
                'loan_amount': loan_amount,
                'credit_score': credit_score,
                'employment_years': employment_years,
                'debt_ratio': debt_ratio
            }
            risk = predict_credit_risk(input_data)
            
            st.subheader("Risk Assessment")
            if risk == 'High Risk':
                st.error(f"Risk Level: {risk}")
                st.write("Recommendation: Loan application requires further manual review or a higher down payment.")
            elif risk == 'Medium Risk':
                st.warning(f"Risk Level: {risk}")
                st.write("Recommendation: Approved with conditions (e.g., higher interest rate).")
            else:
                st.success(f"Risk Level: {risk}")
                st.write("Recommendation: Low risk. Application recommended for approval.")

# --- Identity Analytics Page ---
elif page == "Identity Analytics":
    st.title("🆔 Identity Risk Analytics")
    st.markdown("Upload user activity logs to detect identity anomalies.")
    
    uploaded_file = st.file_uploader("Upload User Activity CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())
        
        if st.button("Analyze Identity Risk"):
            # Integration with trained ML model
            analyzed_df = analyze_identity_risk(df)
            
            st.subheader("Analysis Results")
            st.success("Analysis Complete!")
            
            suspicious_df = analyzed_df[analyzed_df['is_suspicious'] == 1]
            normal_df = analyzed_df[analyzed_df['is_suspicious'] == 0]
            
            st.warning(f"Detected {len(suspicious_df)} suspicious identity records.")
            
            st.write("### Suspicious Records")
            st.dataframe(suspicious_df)
            
            st.write("### Normal Records")
            st.dataframe(normal_df)
    else:
        st.info("Please upload a CSV file to begin analysis. You can use `data/identity.csv` for testing.")

# --- Evaluation Page ---
elif page == "Evaluation":
    st.title("📊 TruLens Evaluation Dashboard")
    st.markdown("Monitor the performance of the RAG pipeline using TruLens metrics.")
    
    st.info("Evaluation data will be populated after implementing the RAG pipeline in Phase 3.")
    
    # Mock evaluation table
    eval_data = {
        "Question": ["What is AML?", "How to handle fraud?", "KYC Process?"],
        "Answer": ["Anti-Money Laundering...", "Report to...", "Identify customer..."],
        "Groundedness": [0.91, 0.85, 0.98],
        "Context Relevance": [0.95, 0.88, 0.92],
        "Answer Relevance": [0.89, 0.91, 0.94],
        "Latency (s)": [1.2, 1.5, 1.1]
    }
    st.table(pd.DataFrame(eval_data))
