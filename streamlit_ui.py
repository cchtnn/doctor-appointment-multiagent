import streamlit as st
import requests

API_URL = "http://127.0.0.1:8002/execute" 

st.title("🩺 Doctor Appointment System")

user_id = st.text_input("Enter your ID number:", "1000082")
query = st.text_area("Enter your query:", "Can you check and make a booking if general dentist available on 8 august 2024 at 8 pm?")

if st.button("Submit Query"):
    if user_id and query:
        try:
            response = requests.post(API_URL, json={'messages': query, 'id_number': int(user_id)},verify=False)
            if response.status_code == 200:
                st.success("Response Received:")
                print("**********my response******************")
                print(response.json())
                st.write(response.json()["messages"])
            else:
                st.error(f"Error {response.status_code}: Could not process the request.")
        except Exception as e:
            st.error(f"Exception occurred: {e}")
    else:
        st.warning("Please enter both ID and query.")
