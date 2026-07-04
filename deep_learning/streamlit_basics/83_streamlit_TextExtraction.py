import streamlit as st
import PyPDF2

st.title("PUCSD by Shubhada palwe")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    st.text_area("Extracted Text", text)
