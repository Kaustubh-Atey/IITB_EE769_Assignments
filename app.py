import numpy as np
import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

st.title('Wine Quality Predictor')

# Create a grid layout for sliders
col_01, col_02 = st.columns(2)

with col_01:
  st.subheader('Select Wine Type:', divider='rainbow')
  data_options = st.radio("", ("Red Wine", "White Wine"))

with col_02:
  st.subheader('Select Classifier Model:', divider='rainbow')
  model_options = st.radio("", ("Random Forest", "Support Vector Machine", "Neural Network"))

# Data Paths
wine_data_path = {'Red Wine': 'Data/red_wine_processed_data.csv',
                  'White Wine': 'Data/white_wine_processed_data.csv'}[data_options]

# Model Paths
if data_options == 'Red Wine':
   model_path = {'Random Forest': 'Models/red_RF_model.joblib',
                  'Support Vector Machine': 'Models/red_SVM_model.joblib',
                  ' Network': 'Models/red_NN_model.joblib'}[model_options]
else:
   model_path = {'Random Forest': 'Models/white_RF_model.joblib',
                  'Support Vector Machine': 'Models/white_SVM_model.joblib',
                  ' Network': 'Models/white_NN_model.joblib'}[model_options]

st.write('Selected = ', wine_data_path, model_path)

st.subheader('Slide the bars to slect value of features:')

# Create a grid layout for sliders
col_11, col_12, col_13 = st.columns(3)

# Create sliders for each variable
with col_11:

    alcohol_f = st.slider('Alcohol', min_value=0.0, max_value=20.0, value=10.0)
    total_sulfur_dioxide_f = st.slider('total sulfur dioxide', min_value=0.0, max_value=300.0, value=100.0)
    volatile_acidity_f = st.slider('volatile acidity', min_value=0.0, max_value=3.0, value=0.5)
    free_sulfur_dioxide_f = st.slider('free sulfur dioxide', min_value=0.0, max_value=100.0, value=50.0)
    
with col_12:
    
    pH_f = st.slider('pH', min_value=0.0, max_value=7.0, value=3.0)
    sulphates_f = st.slider('sulphates', min_value=0.0, max_value=3.0, value=1.0)
    citric_acid_f = st.slider('citric acid', min_value=0.0, max_value=1.0, value=0.2)
    density_f = st.slider('density', min_value=0.8, max_value=1.2, value=0.9)

with col_13:
    fixed_acidity_f = st.slider('fixed acidity', min_value=0.0, max_value=20.0, value=10.0)
    residual_sugar_f = st.slider('residual sugar', min_value=0.0, max_value=20.0, value=10.0)
    chlorides_f = st.slider('free chlorides', min_value=0.0, max_value=1.0, value=0.2)

# User input to test
test_input = np.array([alcohol_f, total_sulfur_dioxide_f, volatile_acidity_f, fixed_acidity_f, residual_sugar_f,
              chlorides_f, free_sulfur_dioxide_f, pH_f, sulphates_f, citric_acid_f, density_f]).reshape(1, 11)

# Scaling Information
wine_df = pd.read_csv(wine_data_path)
scalar = MinMaxScaler()
scalar.fit(wine_df)
test_input = scalar.transform(test_input)
st.write('input: ', test_input)

# Load model
clf = load(model_path)

if st.button('Calculate'):

  # Prediction
  out_class = clf.predict(test_input)

  st.subheader('Predicted Wine Quality (Model)')
  st.write('Wine Quality', out_class+3)
