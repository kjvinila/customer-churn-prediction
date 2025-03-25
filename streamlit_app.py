import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

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
# 🧠 CLUSTER CUSTOMERS USING K-MEANS
# -------------------------------
st.subheader("🧠 Customer Segmentation")

sample_data = np.random.rand(100, 11) * 1000
scaler = StandardScaler()
sample_data_scaled = scaler.fit_transform(sample_data)

# Elbow Method to find optimal clusters
st.subheader("🎯 Optimal Number of Clusters (Elbow Method)")
inertia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(sample_data_scaled)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(K, inertia, marker='o')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method')
st.pyplot(fig)

num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=8, value=3)

# Fit KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(sample_data_scaled)

# -------------------------------
# 🎯 RECOMMENDATIONS ENGINE
# -------------------------------
st.subheader("🚀 Recommendations")

recommendations = {
    0: "Consider offering discounts or incentives to reduce churn risk.",
    1: "Encourage product upselling and cross-selling opportunities.",
    2: "Reward loyal customers with loyalty points or better product offerings."
}

cluster_names = ["High Churn Risk", "Potential Upsell", "Loyal Customers"]

for i in range(num_clusters):
    segment_size = (clusters == i).sum()
    st.markdown(f"**{cluster_names[i]}:** {segment_size} customers")
    
    if cluster_names[i] == "High Churn Risk":
        st.warning(f"🚨 {segment_size} customers are at high risk of churn!")
    elif cluster_names[i] == "Potential Upsell":
        st.info(f"💡 {segment_size} customers could be upsold!")
    elif cluster_names[i] == "Loyal Customers":
        st.success(f"✅ {segment_size} loyal customers identified.")

    # Show recommendation
    st.markdown(f"**💡 Recommendation:** {recommendations.get(i)}")

# -------------------------------
# 📥 DOWNLOAD REPORT
# -------------------------------
st.subheader("📥 Download Report")

report = pd.DataFrame({
    'Cluster': clusters,
    'Recommendation': [recommendations.get(i) for i in clusters]
})

csv = report.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Report (CSV)",
    data=csv,
    file_name='churn_report.csv',
    mime='text/csv'
)

# Footer
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit & Random Forest")
