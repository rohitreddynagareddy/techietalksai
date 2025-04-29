import streamlit as st
import pandas as pd
import os
from app1.app import pandas_agent_executor
from app2.app import csv_agent_executor

st.title("LangChain Data Analysis")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Save to shared volume
    file_path = f"/app/data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    question = st.text_input("Enter your question about the data:")

    if question:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Use Pandas Agent"):
                try:
                    response = pandas_agent_executor(df, question)
                    st.success(f"Pandas Agent: {response}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with col2:
            if st.button("Use CSV Agent"):
                try:
                    response = csv_agent_executor(file_path, question)
                    st.success(f"CSV Agent: {response}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
