import streamlit as st

st.title("Marvellous Infosystems by Piyush Manohar Khairnar")

name = st.text_input("Enter your name")

if name:
    st.success(f"Welcome {name}")