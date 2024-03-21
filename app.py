import numpy as np
import pandas as pd
import streamlit as st
import joblib
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

st.title('🍷 Wine Quality Predictor 🔎📊')

# Create a grid layout for sliders
col_01, col_02 = st.columns(2)

with col_01:
  st.subheader('Select Wine Type:', divider='rainbow')
  data_options = st.radio("Data Type", ("Red Wine", "White Wine"), label_visibility  = 'collapsed')

with col_02:
  st.subheader('Select Classifier Model:', divider='rainbow')
  model_options = st.radio("Model Type", ("Random Forest", "Support Vector Machine", "Neural Network"), label_visibility  = 'collapsed')

# Data Paths
wine_data_path = {'Red Wine': 'Data/red_wine_processed_data.csv',
                  'White Wine': 'Data/white_wine_processed_data.csv'}[data_options]

# Model Paths
if data_options == 'Red Wine':
   model_path = {'Random Forest': 'Models/red_RF_model.joblib',
                  'Support Vector Machine': 'Models/red_SVM_model.joblib',
                  'Neural Network': 'Models/red_NN_model.joblib'}[model_options]
else:
   model_path = {'Random Forest': 'Models/white_RF_model.joblib',
                  'Support Vector Machine': 'Models/white_SVM_model.joblib',
                  'Neural Network': 'Models/white_NN_model.joblib'}[model_options]

# Scaling Information
wine_df = pd.read_csv(wine_data_path)
scalar = MinMaxScaler()
scalar.fit(wine_df)

st.subheader('Slide the bars to slect value of features:', divider='rainbow')

# Create a grid layout for sliders
col_11, col_12, col_13 = st.columns(3)

# Create sliders for each variable
with col_11:

    alcohol_f = st.slider('Alcohol', min_value=0.0, max_value=20.0, value=10.0)
    total_sulfur_dioxide_f = st.slider('Total Sulfur Dioxide', min_value=0.0, max_value=300.0, value=100.0)
    volatile_acidity_f = st.slider('Volatile Acidity', min_value=0.0, max_value=3.0, value=0.5)
    free_sulfur_dioxide_f = st.slider('Free Sulfur Dioxide', min_value=0.0, max_value=100.0, value=50.0)
    
with col_12:
    
    pH_f = st.slider('pH', min_value=0.0, max_value=7.0, value=3.0)
    sulphates_f = st.slider('Sulphates', min_value=0.0, max_value=3.0, value=1.0)
    citric_acid_f = st.slider('Citric Acid', min_value=0.0, max_value=1.0, value=0.2)
    density_f = st.slider('Density', min_value=0.8, max_value=1.2, value=0.9)

with col_13:
    fixed_acidity_f = st.slider('Fixed Acidity', min_value=0.0, max_value=20.0, value=10.0)
    residual_sugar_f = st.slider('Residual Sugar', min_value=0.0, max_value=20.0, value=10.0)
    chlorides_f = st.slider('Free Chlorides', min_value=0.0, max_value=1.0, value=0.2)

# User input to test
test_input_dict = {'alcohol': [alcohol_f], 'total sulfur dioxide': [total_sulfur_dioxide_f], 'volatile acidity': [volatile_acidity_f],
                   'fixed acidity': [fixed_acidity_f], 'residual sugar': [residual_sugar_f], 'chlorides': [chlorides_f],
                   'free sulfur dioxide': [free_sulfur_dioxide_f], 'pH': [pH_f], 'sulphates': [sulphates_f],
                   'citric acid': [citric_acid_f], 'density': [density_f]}

test_input = pd.DataFrame.from_dict(test_input_dict, orient='columns')              
test_input = scalar.transform(test_input)

# Load model
wine_classifier = load(model_path)

st.subheader('Predicted Wine Quality:',  divider='rainbow')

col_21, col_22 = st.columns(2)
with col_22:
  out_class = [0]               # Before calculating
  if st.button('Calculate'):
     out_class = wine_classifier.predict(test_input)

with col_21:
  txt = st.text_area(
    f'Prediction using {model_options} Classifier:', 'Wine Quality = ' + str(int(out_class[0]) + 3))
