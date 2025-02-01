import streamlit as st
import pandas as pd
import requests

# Load scheme names from Excel file
file_path = '/home/rajkamal/Desktop/New_ERA/MTechProj/Details_Of_Schemes/schemes_formatted_info.xlsx'
df = pd.read_excel(file_path)

# Extract and clean scheme names
if 'Name' in df.columns:
    scheme_names = df['Name'].dropna().apply(lambda x: x.strip()).unique().tolist()
else:
    scheme_names = []

# Streamlit App
st.title("Primitive Formbot(Tailored for Schemes Dataset)")

# Scheme Name Dropdown
scheme_name = st.selectbox("Select Scheme Name:", scheme_names)

# Form Entry (Currently set to empty string)
form_entry = ""
st.text_area("Form Entry (OCR text can go here, currently passing it as empty):", value=form_entry, disabled=True, height=100)

# Query Input
query = st.text_input("Enter your query:")

# API Endpoint
api_url = "http://127.0.0.1:8020/get_llm_response_schemes"

# Submit Button
if st.button("Get Response"):
    if scheme_name and query:
        data = {
            "form_entry": form_entry,
            "voice_query": query,
            "scheme_name": scheme_name
        }

        # Send POST request to the API
        response = requests.post(api_url, json=data)

        # Display API Response
        if response.status_code == 200:
            st.success("Response:")
            st.write(response.json().get("response"))
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    else:
        st.warning("Please select a scheme name and enter your query.")
