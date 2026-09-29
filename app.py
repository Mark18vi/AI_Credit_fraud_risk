import streamlit as st
import pandas as pd
import os
from ml.fraud_model import predict_fraud
from ml.credit_model import predict_credit_risk
from ml.identity_model import analyze_identity_risk
from rag.rag_pipeline import get_relevant_context
from rag.llm import generate_answer
from evaluation.trulens_eval import evaluate_rag_response, get_evaluation_report

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
    
    # Document Upload Section
    st.subheader("📄 Knowledge Base Management")
    uploaded_pdfs = st.file_uploader("Upload Policy Documents (PDF)", type="pdf", accept_multiple_files=True)
    
    if uploaded_pdfs:
        if st.button("Process and Index Documents"):
            with st.spinner("Ingesting documents into ChromaDB..."):
                try:
                    # Save uploaded files to the documents folder first
                    for pdf in uploaded_pdfs:
                        with open(os.path.join('documents', pdf.name), "wb") as f:
                            f.write(pdf.getbuffer())
                    
                    # Run the ingestion pipeline
                    from ingestion.pipeline import run_ingestion_pipeline
                    run_ingestion_pipeline()
                    
                    st.success(f"Successfully indexed {len(uploaded_pdfs)} documents!")
                except Exception as e:
                    st.error(f"Ingestion Error: {e}")

    st.divider()
    
    # Dynamic Metrics from CSVs
    try:
        fraud_df = pd.read_csv('data/fraud.csv')
        credit_df = pd.read_csv('data/credit.csv')
        identity_df = pd.read_csv('data/identity.csv')
        
        # Count PDFs in documents folder
        doc_count = len([f for f in os.listdir('documents') if f.endswith('.pdf')])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", doc_count, "PDFs")
        with col2:
            st.metric("Total Transactions", len(fraud_df), "Records")
        with col3:
            st.metric("High Risk Apps", len(credit_df[credit_df['risk_level'] == 'High Risk']), "Applicants")
    except Exception as e:
        st.error(f"Could not load metrics: {e}")
        st.info("Please ensure datasets exist in the data/ folder.")

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
            with st.spinner("Searching documents and generating answer..."):
                try:
                    # 1. Retrieve context from ChromaDB
                    context, docs = get_relevant_context(prompt)
                    
                    # 2. Generate answer using LLM
                    response = generate_answer(prompt, context)
                    
                    # 3. Evaluate the response using TruLens logic
                    scores = evaluate_rag_response(prompt, context, response)
                    
                    # 4. Display response
                    st.markdown(response)
                    
                    # 5. Display Evaluation Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Groundedness", f"{scores['groundedness']:.2f}")
                    col2.metric("Ans Relevance", f"{scores['answer_relevance']:.2f}")
                    col3.metric("Ctx Relevance", f"{scores['context_relevance']:.2f}")
                    
                    # 6. Display sources
                    if docs:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(docs):
                                source = doc.metadata.get('source', 'Unknown')
                                page = doc.metadata.get('page', 'Unknown')
                                st.write(f"Source {i+1}: {os.path.basename(source)} (Page {page})")
                except Exception as e:
                    st.error(f"Error: {e}")
                    response = "I'm sorry, I encountered an error while retrieving the answer. Please ensure the API key is configured."
                    st.markdown(response)
        
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
    st.markdown("Analyze user activity logs to detect identity anomalies.")
    
    # Dynamic Option: Load existing dataset or upload new one
    load_option = st.radio("Data Source", ["Use Default Dataset", "Upload New CSV"])
    
    if load_option == "Use Default Dataset":
        try:
            df = pd.read_csv('data/identity.csv')
            st.success("Loaded default identity dataset.")
        except Exception as e:
            st.error(f"Default dataset not found: {e}")
            df = None
    else:
        uploaded_file = st.file_uploader("Upload User Activity CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = None

    if df is not None:
        st.write("### Preview of Data")
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
    elif load_option == "Upload New CSV":
        st.info("Please upload a CSV file to begin analysis.")

# --- Evaluation Page ---
elif page == "Evaluation":
    st.title("📊 TruLens Evaluation Dashboard")
    st.markdown("Monitor the performance of the RAG pipeline using TruLens metrics.")
    
    st.info("The dashboard now displays real-time evaluation metrics for each query in the chat assistant.")
    
    st.subheader("Historical Performance Report")
    report_df = get_evaluation_report()
    st.table(report_df)
    
    st.markdown("""
    ### Metric Definitions:
    - **Groundedness**: Does the answer only use information present in the retrieved context? (Prevents Hallucinations)
    - **Answer Relevance**: Does the answer actually solve the user's query?
    - **Context Relevance**: Was the retrieved context actually helpful in answering the question?
    """)
