import streamlit as st

st.title("Marvellous Infosystems by Piyush Manohar Khairnar")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success("PDF Uploaded Successfully")