import pandas as pd
import pickle
import streamlit as st




html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">Select Car Features</h1>
</div><br>"""
st.sidebar.markdown(html_temp,unsafe_allow_html=True)

html_temp2 = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">Car Price Prediction App</h1>
</div><br>"""
st.markdown(html_temp2,unsafe_allow_html=True)
st.info("**Click the arrow in the upper left\
    corner to open the sidebar and select features of the car.**")


make_model = st.sidebar.selectbox("Make and Model",('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia',
                          'Renault Clio', 'Renault Duster', 'Renault Espace'))
gearing_type = st.sidebar.selectbox("Gear Type", ('Automatic', 'Manual', 'Semi-automatic'))
hp_kw = st.sidebar.number_input("Horse Power", 40, 239)
age = st.sidebar.number_input("Age", 0, 3)
km = st.sidebar.number_input("Kilometers", 0, 317000)
gears = st.sidebar.number_input("Gears", 5, 8)


new_data = {"hp_kW" : hp_kw,
            "age" : age,
            "km" : km,
            "Gears" : gears,
            "make_model" : make_model,
            "Gearing_Type" : gearing_type}

df = pd.DataFrame([new_data])

st.write(df)

final_model = pickle.load(open("final_model_auto_scout.pickle", "rb"))
st.info("**Check the features you selected from the table above. If correct, press the Predict button.**")
if st.button("Predict"):
    prediction = final_model.predict(df)
    st.success(f"The Model Prediction is : **€ {round(prediction[0])}**")