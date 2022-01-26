# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:56:22 2022

@author: harish
"""

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import pickle 


st.image("./car.jpg")
st.markdown("<h2 style='text-align:center;'>Used Car price Predictor</h2>",unsafe_allow_html=True)
   

st.sidebar.write("# Created by Harish Borse")
st.sidebar.write("""
                 # Car Specifications:
                 """)

def UserInputs():
    
    year = st.sidebar.selectbox("Purchase Year",[i for i in range(1990,2023,1)])
    km_driven = st.sidebar.number_input("Distance Driven in KMS:")
    fuel = st.sidebar.selectbox("Fuel Type",["Diesel","Petrol","LPG","CNG"])	
    seller_type	= st.sidebar.selectbox("Seller Type",['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.sidebar.selectbox("Transmission",['Manual', 'Automatic'])
    owner = st.sidebar.selectbox("Owner",['First Owner', 'Second Owner', 'Third Owner','Fourth & Above Owner', 'Test Drive Car'])
    mileage = st.sidebar.slider("Mileage",min_value=0,max_value=100,step=1)
    engine = st.sidebar.slider("Engine Capacity",min_value=500,max_value=4000,step=100)
    max_power = st.sidebar.slider("Max Power",min_value=30,max_value=500,step=10)
    seats = st.sidebar.selectbox("No of Seats",[2,3,4,5,6])
    age = 2022-year
    data = {'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission':transmission,
            'owner':owner,
            'mileage': mileage,
            'engine':engine,
            'max_power':max_power,
            'seats':seats,
            'age':age}
    
    df = pd.DataFrame(data,index=[0])
    return df


inputdf = UserInputs()

## Data Transformation
df = pd.read_csv("./AppData.csv")
df = pd.concat([df,inputdf])
inputdf = pd.get_dummies(df,columns=['fuel','seller_type','transmission','owner'],drop_first=True)

inputdf = inputdf.tail(1)

 
model = pickle.load(open("./CarPricePrediction_rf.pkl","rb"))

result = model.predict(inputdf)


st.markdown("<h5 style='text-align:center;'>Price of Car for given Specifications:</h5>",unsafe_allow_html=True)

re = int(result[0])

st.markdown("<h5 style='text-align:center; color:Tomato'>"+str(re)+"</h5>",unsafe_allow_html=True)



    
