import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np

st.header("Fish Weight Prediction App")
data = pd.read_csv("Fish.csv")
#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data

st.sidebar.subheader("Please select relevant features of your fish!")
inp_species = st.sidebar.radio(
        'Name of the fish:',
        np.unique(data['Species']))


input_Length1 = st.sidebar.slider('Vertical length(cm)', 0.0, max(data["Length1"]), 1.0)
input_Length2 = st.sidebar.slider('Diagonal length(cm)', 0.0, max(data["Length2"]), 1.0)
input_Length3 = st.sidebar.slider('Cross length(cm)', 0.0, max(data["Length3"]), 1.0)
input_Height = st.sidebar.slider('Height(cm)', 0.0, max(data["Height"]), 1.0)
input_Width = st.sidebar.slider('Diagonal width(cm)', 0.0, max(data["Width"]), 1.0)


if st.button('Make Prediction'):
    input_species = encoder.transform(np.expand_dims(inp_species, -1))
    inputs = np.expand_dims(
        [int(input_species), input_Length1, input_Length2, input_Length3, input_Height, input_Width], 0)
    prediction = best_xgboost_model.predict(inputs)
    st.write(f"Your fish weight is: {np.squeeze(prediction, -1):.2f}g")



