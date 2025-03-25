import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model = joblib.load('model.pkl')

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

# Custom CSS for better styling
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #F5F5F5;
            color: #333;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #D63230;
        }
        .stSuccess {
            color: #4CAF50;
            font-weight: bold;
        }
        .stError {
            color: #FF4B4B;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title('🚀 Customer Churn Prediction')

# Input fields
st.subheader("Customer Information")
age = st.number_input('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance', min_value=0, value=10000)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0, value=50000)
products_number = st.number_input('Number of Products', min_value=0, max_value=5, value=1)
active_member = st.selectbox('Active Member', [0, 1])
credit_card = st.selectbox('Has Credit Card', [0, 1])
country = st.selectbox('Country', ['France', 'Germany', 'Spain'])

# Encode country
country_france = 1 if country == 'France' else 0
country_germany = 1 if country == 'Germany' else 0
country_spain = 1 if country == 'Spain' else 0

# Placeholder for any missing features (11 total features)
missing_feature_1 = 0
missing_feature_2 = 0

# Create the input array
input_data = np.array([[
    age, balance, credit_score, estimated_salary, products_number,
    active_member, credit_card, country_germany, country_spain,
    missing_feature_1, missing_feature_2
]])

# Predict button
if st.button('Predict Churn'):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f'🚨 High Churn Risk! (Probability: {probability:.2%})')
    else:
        st.success(f'✅ Low Churn Risk! (Probability: {probability:.2%})')

    # -------------------------------
    # 📊 FEATURE IMPORTANCE CHART
    # -------------------------------
    st.subheader("📊 Feature Importance")
    feature_importance = model.feature_importances_
    feature_names = [
        'Age', 'Balance', 'Credit Score', 'Estimated Salary', 'Products',
        'Active Member', 'Credit Card', 'Germany', 'Spain',
        'Missing Feature 1', 'Missing Feature 2'
    ]

    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_names, palette="viridis", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # -------------------------------
    # 📈 MODEL CONFIDENCE CHART
    # -------------------------------
    st.subheader("📈 Model Confidence")
    fig, ax = plt.subplots(figsize=(4, 1))
    ax.barh(['Not Churn', 'Churn'], [1 - probability, probability], color=['green', 'red'])
    ax.set_xlim(0, 1)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit & Random Forest")
