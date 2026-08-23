
import streamlit as st
import pandas as pd
import joblib

st.title('Tourism Package Purchase Predictor')

model_path = 'tourism_project/deployment/model.joblib'
try:
  model = joblib.load(model_path)
except Exception as e:
  st.error(f"Model not found at {model_path}. Run the training cell first.")
  st.stop()

# Collect a few example inputs from the user
age = st.number_input('Age', min_value=18, max_value=100, value=35)
monthly_income = st.number_input('MonthlyIncome', min_value=0.0, value=20000.0)
num_persons = st.number_input('NumberOfPersonVisiting', min_value=1, max_value=10, value=2)
preferred_star = st.selectbox('PreferredPropertyStar', [1,2,3,4,5], index=2)
type_contact = st.selectbox('TypeofContact', ['Self Enquiry','Company Invited'])

input_df = pd.DataFrame([{
  'Age': age,
  'MonthlyIncome': monthly_income,
  'NumberOfPersonVisiting': num_persons,
  'PreferredPropertyStar': preferred_star,
  'TypeofContact': type_contact
}])

if st.button('Predict'):
  pred = model.predict(input_df)[0]
  prob = model.predict_proba(input_df)[0,1] if hasattr(model, 'predict_proba') else None
  st.write('Prediction (1 = will purchase):', int(pred))
  if prob is not None:
    st.write(f'Predicted probability: {prob:.3f}')
