import streamlit as st

st.title("PUCSD by Shubhada palwe")

name = st.text_input("Enter your name")

if name:
    st.success(f"Welcome {name}")
