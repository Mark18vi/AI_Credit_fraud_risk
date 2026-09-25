import streamlit as st

st.title("Credit Card Fraud Detection")
st.set_page_config(
    page_title="AI Fraud Intelligence platform", 
    page_icon=":credit_card:", 
    layout="wide",
    initial_sidebar_state="expanded"
    )

st.markdown("""
    <style>

    .main{background-color: #f7f9fC;}

    .big-title {
        font-size: 50px;
        font-weight: bold;
        color: #2C3E50;
        }

    .subtitle {
        font-size: 20px;
        color: #34495E;
        text-align: center;
        }

    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin: 10px;
        text-align: center;
        }

    .feature-card {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin: 10px;
        height: 200px;
        }

    .footer {
        color: grey;
        padding-top: 20px;
        text-align: center;
        }

    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("AI Credit Fraud Intelligence Platform")
st.sidebar.markdown("---")
# st.sidebar.markdown("This platform is designed to detect fraudulent credit card transactions using advanced machine learning algorithms. It provides insights and visualizations to help understand the patterns of fraud and improve decision-making.")
st.sidebar.markdown("""
### Navigation

-> Dashboard

-> AI Assistant

-> Fraud Detection

-> Credit Risk

-> Indentity Analytics

-> Trulens evals

-> Reports

-> Settings

""")

st.sidebar.markdown("---")

# Header
st.markdown('<h1 class="big-title"> AI - Powered Credit Fraud Intelligence Platform</h1>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">Enterprise-Generative AI, ML, Fraud Analytics</p>', unsafe_allow_html=True)

st.divider()

left, right = st.columns([2, 1])

with left:
    st.markdown('<h2 class="subtitle">Welcome to the AI Credit Fraud Intelligence Platform</h2>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">This platform is designed to detect fraudulent credit card transactions using advanced machine learning algorithms. It provides insights and visualizations to help understand the patterns of fraud and improve decision-making.</p>', unsafe_allow_html=True)
    st.success("This application will evolve into a comprehensive AI-powered credit fraud intelligence platform, providing real-time insights and analytics to combat fraudulent activities effectively.")

with right:
    st.info("""
    - Python
    - Streamlit
""")

st.divider()

st.subheader("Platform Features")

col1, col2, col3, col4 = st.columns(4)

metrics = [
    ("Transaction Volume", "1,000,000+"),
    ("Fraud Alerts", "250+"),
    ("High-Risk Transactions", "92"),
    ("Knowledge Base Articles", "150+"),
]

for col, (title, value) in zip([col1, col2, col3, col4], metrics):
    with col:
        st.markdown(f'<div class="metric-card"><h3>{title}</h3><p>{value}</p></div>', unsafe_allow_html=True)

st.write("")

