import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np

html_temp = """
<div style="background-color:Blue;padding:1.5px">
<h1 style="color:white;text-align:center;">Select Fish Features</h1>
</div><br>"""
st.sidebar.markdown(html_temp,unsafe_allow_html=True)

html_temp2 = """
<div style="background-color:blue;padding:1.5px">
<h1 style="color:white;text-align:center;">Fish Wight Prediction App</h1>
</div><br>"""
st.markdown(html_temp2,unsafe_allow_html=True)

data = pd.read_csv("Fish.csv")
#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data[:5]


inp_species = st.sidebar.selectbox('Kind of the fish:', np.unique(data['Species']))
input_Length1 = st.sidebar.number_input('Vertical length(cm)', 0.0, max(data["Length1"]), .0,)
input_Length2 = st.sidebar.number_input('Diagonal length(cm)', 0.0, max(data["Length2"]), .0)
input_Length3 = st.sidebar.number_input('Cross length(cm)', 0.0, max(data["Length3"]), .0)
input_Height = st.sidebar.number_input('Height(cm)', 0.0, max(data["Height"]), .0)
input_Width = st.sidebar.number_input('Diagonal width(cm)', 0.0, max(data["Width"]), .0)


if st.button('Make Prediction'):
    input_species = encoder.transform(np.expand_dims(inp_species, -1))
    inputs = np.expand_dims(
        [int(input_species), input_Length1, input_Length2, input_Length3, input_Height, input_Width], 0)
    prediction = best_xgboost_model.predict(inputs)
    st.success(f"Your fish weight is: {np.squeeze(prediction, -1):.2f} g")



