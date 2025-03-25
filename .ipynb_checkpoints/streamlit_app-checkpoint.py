{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9ba09b30",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'streamlit'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[6], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mstreamlit\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mst\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mjoblib\u001b[39;00m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnumpy\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnp\u001b[39;00m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load the model\n",
    "model = joblib.load('model.pkl')\n",
    "\n",
    "# Title\n",
    "st.title('Customer Churn Prediction')\n",
    "\n",
    "# Input fields\n",
    "age = st.number_input('Age', min_value=18, max_value=100, value=30)\n",
    "balance = st.number_input('Balance', min_value=0, value=10000)\n",
    "credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)\n",
    "estimated_salary = st.number_input('Estimated Salary', min_value=0, value=50000)\n",
    "products_number = st.number_input('Number of Products', min_value=0, max_value=5, value=1)\n",
    "active_member = st.selectbox('Active Member', [0, 1])\n",
    "credit_card = st.selectbox('Has Credit Card', [0, 1])\n",
    "country = st.selectbox('Country', ['France', 'Germany', 'Spain'])\n",
    "\n",
    "# Encode country\n",
    "country_france = 1 if country == 'France' else 0\n",
    "country_germany = 1 if country == 'Germany' else 0\n",
    "country_spain = 1 if country == 'Spain' else 0\n",
    "\n",
    "# Create the input array\n",
    "input_data = np.array([[\n",
    "    age, balance, credit_score, estimated_salary, products_number,\n",
    "    active_member, credit_card, country_germany, country_spain\n",
    "]])\n",
    "\n",
    "# Predict button\n",
    "if st.button('Predict Churn'):\n",
    "    prediction = model.predict(input_data)[0]\n",
    "    probability = model.predict_proba(input_data)[0][1]\n",
    "\n",
    "    if prediction == 1:\n",
    "        st.error(f'🚨 High Churn Risk! (Probability: {probability:.2%})')\n",
    "    else:\n",
    "        st.success(f'✅ Low Churn Risk! (Probability: {probability:.2%})')\n",
    "\n",
    "# Footer\n",
    "st.markdown('---')\n",
    "st.markdown('Built with ❤️ using Streamlit & Random Forest')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b469621a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
