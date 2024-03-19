import numpy as np
import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

st.title('ML Demo app')
st.subheader('Slide the bars to slect value of features:')

# Create a grid layout for sliders
col1, col2, col3 = st.columns(3)

# Create sliders for each variable
with col1:
    alcohol_f = st.slider('Alcohol', min_value=0.0, max_value=20.0, value=10.0)
    total_sulfur_dioxide_f = st.slider('total sulfur dioxide', min_value=0.0, max_value=300.0, value=100.0)
    volatile_acidity_f = st.slider('volatile acidity', min_value=0.0, max_value=3.0, value=0.5)
    free_sulfur_dioxide_f = st.slider('free sulfur dioxide', min_value=0.0, max_value=100.0, value=50.0)
    
with col2:
    pH_f = st.slider('pH', min_value=0.0, max_value=7.0, value=3.0)
    sulphates_f = st.slider('sulphates', min_value=0.0, max_value=3.0, value=1.0)
    citric_acid_f = st.slider('citric acid', min_value=0.0, max_value=1.0, value=0.2)
    density_f = st.slider('density', min_value=0.8, max_value=1.2, value=0.9)

with col3:
    fixed_acidity_f = st.slider('fixed acidity', min_value=0.0, max_value=20.0, value=10.0)
    residual_sugar_f = st.slider('residual sugar', min_value=0.0, max_value=20.0, value=10.0)
    chlorides_f = st.slider('free chlorides', min_value=0.0, max_value=1.0, value=0.2)
