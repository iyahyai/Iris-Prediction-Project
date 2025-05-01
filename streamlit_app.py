import streamlit as  st
import pickle 
import numpy as np


model=pickle.load(open('trained_model.sav','rb'))

st.title("Iris Prediciton")

sp_l = st.number_input('Sepal Length')
sp_w = st.number_input('Sepal Width')
pet_l = st.number_input('Petal Length')
pet_w = st.number_input('Petal Wength')

input_data = [sp_l,	sp_w, pet_l, pet_w]

input_data = np.asarray(input_data).reshape(1,-1)

prediction = model.predict(input_data)
print(prediction[0])

if st.button('Result'):
    prediction = model.predict(input_data)
    st.success(prediction[0])
