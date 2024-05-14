import requests
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime
from datetime import datetime, date, time
# Set page configuration
detective_icon = "🕵️‍♂️"
from datetime import datetime

st.set_page_config(
    page_title='fruad prediction',
    page_icon=detective_icon,
    initial_sidebar_state='collapsed'  # Collapsed sidebar
)

# Load the trained model
joblib_file_xgb = 'xgb_model.sav'
joblib_file_rf = 'rf_model.sav'
joblib_file_DT= 'DT_model.sav' 
joblib_file_log = 'logistic_model.sav'


def prdict(amount, merchant, category, time_of_day, model):
    # Create a DataFrame with the input features


    # date_str = "14/5/2024 16:45"
    time_of_day = datetime.strptime(time_of_day, "%d/%m/%Y %H:%M")

    features = pd.DataFrame({
        'merchant': [merchant],
        'category': [category],
        'Time': [time_of_day],
        'Amount': [amount]
    })

    features['Time of Day']=pd.to_datetime(features['Time'], errors='coerce', format='%m/%d/%Y %I:%M:%S %p')
    loaded_scaler = joblib.load('scaler.sav')
    # One-hot encode categorical features
    loaded_encoder = joblib.load('encoder.sav')
    encoded_features = loaded_encoder.transform(features[['merchant', 'category', 'Time of Day']])
    #encoded_features = encoder.transform(features[['Merchant', 'Category', 'Time of Day']])
    
    # Transform 'Amount' column
    scaled_new_data = loaded_scaler.transform(features[['Amount']])
   # amount_scaled = scaler.transform(features[['Amount']])
    #amount_scaled = scaler.transform(features[['Amount']])
    
    # Combine encoded features and scaled 'Amount' column
    encoded_array = np.array((encoded_features.toarray(),scaled_new_data))
    
    # Make prediction
    prediction = model.predict(encoded_array.reshape(1, -1))
    
    return prediction

# Sidebar menu options
menu_options = ["Home", "Graphs", "About", "Contact"]
model_option=["rf model","xgb model","logistic model","DT model"]
# Sidebar
with st.sidebar:
    # Display sidebar menu
    choose = st.selectbox("Navigation", menu_options)

# Main content based on selected menu option
if choose == 'Home':
    st.title('fruad prediction')
    st.write('---')
    st.subheader('Enter your info')
    
    # User inputs
    fristName=st.text_input("enter your first name :")
    lastName=st.text_input("enter your last name :")
    cardNum = st.text_input("Enter your Card Number:")
    trans_num=st.text_input("Enter your trans Number: ")
    amount = st.number_input("Enter the amount: ", min_value=0)
    merchant = st.text_input("Enter your merchant:")
    category = st.text_input("Enter your category value:")
    time_of_day = st.text_input("Enter your time of day value:")
    #st.title("Date and Time Input")

    # Date input
   # date_value = st.date_input("Select Date")
    #time_value = st.time_input("Select Time")
    #time_of_day = datetime.combine(date_value, time_value).strftime("%m/%d/%Y %H:%M")
    # Time input

    choice_model=st.selectbox("select model",model_option)
    encoder = OneHotEncoder()
    scaler = MinMaxScaler()
    if choice_model=="rf model": model = joblib.load(joblib_file_rf) 
    elif choice_model=="xgb model":model = joblib.load(joblib_file_xgb) 
    elif choice_model=="logistic model":model = joblib.load(joblib_file_log) 
    elif choice_model=="DT model":model = joblib.load(joblib_file_DT) 
    
    

    if st.button("Predict"):
        # if fristName and lastName and cardNum and trans_num and amount and merchant and category and time_of_day and model:
            try:  
                sample_prediction = prdict(amount, merchant, category, time_of_day,model,encoder,scaler)
             
                if sample_prediction == 0:
                    st.warning("this is not fruad ")
                    st.write("this indicates a non fruad process.")
                    st.balloons()
                elif sample_prediction == 1:
                    st.success("this is fruad ")
                    st.write("This indicates fruad process .")
            except Exception as e :
                st.error(f"An error occurred: {e}")  
        # else :st.warning("please fill all data")    

elif choose == 'About':
    st.title('About Page')
    st.write('---')
    st.write("""
    
    """)
    #st.image("5355919-removebg-preview.png")

elif choose == "Contact":
    st.title('Contact Us')
    st.write('---')
    with st.form(key='contact_form'):
        st.write('## Please help us improve!')
        name = st.text_input(label='Name')
        email = st.text_input(label='Email')
        message = st.text_area(label='Message')
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for contacting us! We will respond to your questions or inquiries as soon as possible!')

#elif choose == 'Graphs':
   # st.title('Salary Classifier Graphs')
    #st.write('---')
   # st.set_option('deprecation.showPyplotGlobalUse', False)
    # Display your graphs here
