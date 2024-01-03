import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
html_temp = """
    <div style ="background-color:#FFA500; padding:13px">
    <h1 style ="color:#f0f0f5; text-align:center; ">Autoscout Project </h1>
    </div>
    """
# this line allows us to display the front end aspects we have defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)
# İmages of car
image = Image.open("2d_car.jpg")
st.image(image, use_column_width=True)
# Display Iris Dataset
st.header("_Autoscout_")
df = pd.read_csv('for_cat_boost_eda.csv')
st.write(df.sample(5))
# Loading the models to make predictions
car_model = pickle.load(open("my_model", "rb"))
# User input variables that will be used on predictions
st.sidebar.title("_Please Enter the Features informations to Predict the Car Price_")
make_model = st.sidebar.selectbox('Select the make_model', (df["make_model"].unique()))
hp_kw = st.sidebar.number_input("hp_kw", 0, 500, 75, 1)
age = st.sidebar.number_input("age", 0, 30, 3, 1)
gearing_type = st.sidebar.selectbox('Select the gearing type', (df["gearing_type"].unique()))
km = st.sidebar.slider("km", 0, 500000, 10000, 1000)
type = st.sidebar.selectbox('Select the type', (df["type"].unique()))
# Converting user inputs to dataframe
my_dict = {
    "make_model": make_model,
    "km":km,
    "type":type,
    "age": age,
    "hp_kw": hp_kw,
    "gearing_type":gearing_type,
    }
my_list = list(my_dict.values())
df_input = pd.DataFrame.from_dict([my_dict])
st.header("The values you selected is below")
st.table(df_input)
# defining the function which will make the prediction using the data
def prediction(model, input_data):
    prediction = model.predict(input_data)
    return prediction
# "Predict" button
if st.button("Predict"):
    # Make prediction
    predicted_price = prediction(car_model, df_input)
    # Show the prediction result
    st.success(f"Predicted Price: {predicted_price[0]:.2f}")