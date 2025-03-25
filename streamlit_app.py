import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the model
model = joblib.load('model.pkl')

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #FAFAFA;
            color: #333;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stSuccess {
            color: #28a745;
            font-weight: bold;
        }
        .stError {
            color: #dc3545;
            font-weight: bold;
        }
        .report-container {
            padding: 20px;
            border-radius: 10px;
            background-color: #F8F9FA;
            border: 1px solid #DEE2E6;
            margin-bottom: 20px;
        }
        .header-style {
            font-size: 20px;
            font-weight: bold;
            color: #343A40;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title('🚀 Customer Churn Prediction')

# -------------------------------
# ✅ INPUT SECTION
# -------------------------------
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    balance = st.number_input('Balance', min_value=0, value=10000)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)

with col2:
    estimated_salary = st.number_input('Estimated Salary', min_value=0, value=50000)
    products_number = st.number_input('Number of Products', min_value=0, max_value=5, value=1)
    active_member = st.selectbox('Active Member', [0, 1])

credit_card = st.selectbox('Has Credit Card', [0, 1])
country = st.selectbox('Country', ['France', 'Germany', 'Spain'])

# Encode country
country_france = 1 if country == 'France' else 0
country_germany = 1 if country == 'Germany' else 0
country_spain = 1 if country == 'Spain' else 0

# Create the input array
input_data = np.array([[
    age, balance, credit_score, estimated_salary, products_number,
    active_member, credit_card, country_germany, country_spain,
    0, 0  # Missing features placeholder
]])

# -------------------------------
# 🚀 PREDICTION SECTION
# -------------------------------
if st.button('Predict Churn'):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("<div class='report-container'>", unsafe_allow_html=True)
    
    # Display churn prediction
    if prediction == 1:
        st.error(f'🚨 High Churn Risk! (Probability: {probability:.2%})')
    else:
        st.success(f'✅ Low Churn Risk! (Probability: {probability:.2%})')

    # Display recommendations based on input
    if credit_score < 500:
        st.warning("💡 Improve credit score to qualify for better products.")
    if balance < 1000:
        st.info("💡 Offer financial incentives to increase balance.")
    if products_number < 2:
        st.info("💡 Encourage customer to try more products.")
    if active_member == 0:
        st.warning("💡 Engage inactive members with special offers.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 📊 FEATURE IMPORTANCE
# -------------------------------
st.subheader("📊 Feature Importance")

feature_importance = model.feature_importances_
feature_names = [
    'Age', 'Balance', 'Credit Score', 'Estimated Salary', 'Products',
    'Active Member', 'Credit Card', 'Germany', 'Spain',
    'Missing Feature 1', 'Missing Feature 2'
]

fig = px.bar(
    x=feature_importance,
    y=feature_names,
    orientation='h',
    title="Feature Importance",
    labels={'x': 'Importance', 'y': 'Feature'}
)
st.plotly_chart(fig)

# -------------------------------
# 🧠 CLUSTER SEGMENTATION
# -------------------------------
st.subheader("🧠 Customer Segmentation")

# Generate sample data
sample_data = np.random.rand(100, 11) * 1000
scaler = StandardScaler()
sample_data_scaled = scaler.fit_transform(sample_data)

# Fit KMeans
num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=8, value=3)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(sample_data_scaled)

# Plot clusters using Plotly
cluster_fig = px.scatter(
    x=sample_data_scaled[:, 0], 
    y=sample_data_scaled[:, 1],
    color=clusters,
    title=f"Customer Segmentation with {num_clusters} Clusters",
    labels={'x': 'Feature 1', 'y': 'Feature 2'}
)
st.plotly_chart(cluster_fig)

# -------------------------------
# 📥 DOWNLOAD REPORT
# -------------------------------
st.subheader("📥 Download Report")

report = pd.DataFrame({
    'Cluster': clusters,
    'Recommendation': ["Improve product range" if i == 1 else "Retention incentives" for i in clusters]
})

csv = report.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Report (CSV)",
    data=csv,
    file_name='customer_churn_report.csv',
    mime='text/csv'
)

# Footer
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit & Random Forest")

