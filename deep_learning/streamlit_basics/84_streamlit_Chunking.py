import streamlit as st

st.title("Marvellous Infosystems by Piyush Manohar Khairnar")

text = st.text_area("Enter Text")

if text:

    chunk_size = 100

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    st.write("Total Chunks:", len(chunks))

    for index, chunk in enumerate(chunks):
        st.write(f"Chunk {index+1}")
        st.info(chunk)