# Import required libraries 
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

# Display Title
st.title('🍷 Wine Quality Predictor 🔎📊')

# Create a grid layout for data and model type options
col_01, col_02 = st.columns(2)

# 1st column for Data type options
with col_01:
  st.subheader('Select Wine Type:', divider='rainbow')
  # radio button allows to select only one of the options
  data_options = st.radio("Data Type", ("Red Wine", "White Wine"), label_visibility  = 'collapsed')

# 2nd column for Model type options
with col_02:
  st.subheader('Select Classifier Model:', divider='rainbow')
  # radio button allows to select only one of the options
  model_options = st.radio("Model Type", ("Random Forest", "Support Vector Machine", "Neural Network"), label_visibility  = 'collapsed')

# Data Paths: Data is stored at these paths. Data is used for scaling information.
wine_data_path = {'Red Wine': 'Data/red_wine_processed_data.csv',
                  'White Wine': 'Data/white_wine_processed_data.csv'}[data_options]

# Model Paths: All 6 trained models are stored at these paths
if data_options == 'Red Wine':
   # If red wine data is selected, the models trained on red wine data are loaded
   model_path = {'Random Forest': 'Models/red_RF_model.joblib',
                  'Support Vector Machine': 'Models/red_SVM_model.joblib',
                  'Neural Network': 'Models/red_NN_model.joblib'}[model_options]
else:
   # If white wine data is selected, the models trained on white wine data are loaded
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

# Adding sliders for each variable
# Column 1
with col_11:
    # Column 1
    alcohol_f = st.slider('Alcohol', min_value=0.0, max_value=20.0, value=10.0)
    total_sulfur_dioxide_f = st.slider('Total Sulfur Dioxide', min_value=0.0, max_value=300.0, value=100.0)
    volatile_acidity_f = st.slider('Volatile Acidity', min_value=0.0, max_value=3.0, value=0.5)
    free_sulfur_dioxide_f = st.slider('Free Sulfur Dioxide', min_value=0.0, max_value=100.0, value=50.0)

# Column 2    
with col_12:
    
    pH_f = st.slider('pH', min_value=0.0, max_value=7.0, value=3.0)
    sulphates_f = st.slider('Sulphates', min_value=0.0, max_value=3.0, value=1.0)
    citric_acid_f = st.slider('Citric Acid', min_value=0.0, max_value=1.0, value=0.2)
    density_f = st.slider('Density', min_value=0.8, max_value=1.2, value=0.9)

# Column 3
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
# Scale input
test_input = scalar.transform(test_input)

# Load model
wine_classifier = load(model_path)

st.subheader('Predicted Wine Quality:',  divider='rainbow')

col_21, col_22 = st.columns(2)
with col_22:
  out_class = [0]               # To display some default value before calculating
  # Button for calculating (making prediction)
  if st.button('Calculate'):
     # Make prediction
     out_class = wine_classifier.predict(test_input)

with col_21:
  # Display prediction
  txt = st.text_area(
    f'Prediction using {model_options} Classifier:', 'Wine Quality = ' + str(int(out_class[0]) + 3))
