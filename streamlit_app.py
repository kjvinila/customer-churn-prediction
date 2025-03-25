import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('model.pkl')

# Title
st.title('Customer Churn Prediction')

# Input fields
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

# Add placeholders for any missing features (11 total features)
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

# Footer
st.markdown('---')
st.markdown('Built with ❤️ using Streamlit & Random Forest')
