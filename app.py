import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image


pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"


def predict_impact(opened_by, location, ID_caller, Category_id, ID):
    """Let's find the prediction
    This is using docstrings for specifications.
    ---
    parameters:
      - name: opened_by
        in: query
        type: number
        required: true
      - name: location
        in: query
        type: number
        required: true
      - name: ID_caller
        in: query
        type: number
        required: true
      - name: Category_id
        in: query
        type: number
        required: true
      -name: ID
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """
    prediction = classifier.predict([[opened_by, location, ID_caller, Category_id, ID]])
    print(prediction)
    return prediction

def main():

    html_temp ="""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Incident Impact Prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    opened_by= st.text_input("Opened_By","Type Here")
    location = st.text_input("Location","Type Here")
    ID_caller = st.text_input("ID_Caller","Type Here")
    Category_id = st.text_input("Category_ID","Type Here")
    ID = st.text_input("ID","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_impact(opened_by,location,ID_caller,Category_id,ID)
    st.success('The output is {}'.format(result))
    

if __name__ == '__main__':
    main()









