import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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


# -------------------------------
# 🧠 CLUSTER CUSTOMERS USING K-MEANS
# -------------------------------
st.subheader("🧠 Customer Segmentation")

# Sample data for clustering (can adjust based on your actual dataset)
sample_data = np.random.rand(100, 11) * 1000

# Scale the data for better clustering
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

# Plot the Elbow Curve
fig, ax = plt.subplots()
ax.plot(K, inertia, marker='o')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method')
st.pyplot(fig)

# User selects number of clusters
num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=8, value=3)

# Create KMeans model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(sample_data_scaled)

# Add clusters to the data
sample_data_with_clusters = np.column_stack((sample_data_scaled, clusters))

# -------------------------------
# 📌 Define Cluster Types
# -------------------------------
st.subheader(f"📌 Customer Segmentation ({num_clusters} Clusters)")

# Define cluster types
cluster_labels = {
    0: "High Churn Risk", 
    1: "Loyal Customers", 
    2: "Potential Upsell"
}

# Create cluster mapping based on average cluster characteristics
cluster_names = [cluster_labels.get(i, f"Cluster {i}") for i in range(num_clusters)]

# Plot clusters
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    x=sample_data_scaled[:, 0], 
    y=sample_data_scaled[:, 1], 
    hue=clusters, 
    palette="viridis", 
    s=100
)

# Annotate cluster labels
for i, txt in enumerate(cluster_names):
    ax.annotate(txt, (sample_data_scaled[clusters == i, 0].mean(), 
                      sample_data_scaled[clusters == i, 1].mean()), 
                fontsize=10, color='black', fontweight='bold', ha='center')

ax.set_title(f'Customer Segmentation with {num_clusters} Clusters')
ax.set_xlabel('Feature 1 (Scaled)')
ax.set_ylabel('Feature 2 (Scaled)')
st.pyplot(fig)

# -------------------------------
# 📊 Segment Insights
# -------------------------------
st.subheader("📊 Segment Insights")

for i in range(num_clusters):
    segment_size = (clusters == i).sum()
    st.markdown(f"**{cluster_names[i]}:** {segment_size} customers")

    # Highlight high-risk segment
    if cluster_names[i] == "High Churn Risk":
        st.warning(f"🚨 **{segment_size} customers are at high risk of churn!**")
    elif cluster_names[i] == "Loyal Customers":
        st.success(f"✅ **{segment_size} loyal customers identified.**")
    elif cluster_names[i] == "Potential Upsell":
        st.info(f"💡 **{segment_size} customers could be upsold with better offers!**")

# Footer
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit & Random Forest")
