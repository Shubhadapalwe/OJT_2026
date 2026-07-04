# Streamlit Basics

**Marvellous Infosystems — OJT 2026**
**Trainee : Shubhada A. Palwe**

---

## About This Folder

Streamlit is a Python library that lets us build web applications without knowing HTML, CSS, or JavaScript. We just write Python and Streamlit turns it into an interactive webpage. This is super useful for deploying machine learning models with a proper UI.

There are 6 files here, each adding one new Streamlit feature:

| File | Feature |
|------|---------|
| 79 | Hello world — `st.title` + `st.write` |
| 80 | Text input — `st.text_input` + `st.success` |
| 81 | Button — `st.button` |
| 82 | File upload — `st.file_uploader` for PDFs |
| 83 | Text extraction — read PDF and show text |
| 84 | Chunking — split text into 100-character pieces |

Files 82–84 together form the first steps of a RAG (Retrieval-Augmented Generation) pipeline — upload a PDF, extract text, break it into chunks for processing.

---

## How to Run Any Streamlit File

Streamlit apps are NOT run with `python filename.py`. They use their own command:

```bash
pip install streamlit
streamlit run 79_streamlit_JayGanesh.py
```

This opens a browser tab automatically at `http://localhost:8501`. Every time you save the file, the page refreshes with your changes.

To stop: press `Ctrl+C` in the terminal.

---

## File 79 — Hello World

```python
import streamlit as st

st.title("Marvellous Infosystems by Piyush Manohar Khairnar")
st.write("Jay Ganesh...")
```

The simplest possible Streamlit app. Two lines of actual code.

`st.title()` — shows the text as a large heading at the top of the page.

`st.write()` — Streamlit's most flexible function. It can display strings, numbers, dataframes, matplotlib plots, and more. It figures out the format automatically.

---

## File 80 — Text Input

```python
name = st.text_input("Enter your name")

if name:
    st.success(f"Welcome {name}")
```

`st.text_input("label")` — shows a text box on the page. The label ("Enter your name") appears above the box. Returns whatever the user typed as a Python string. If nothing is typed yet, returns `""` (empty string).

`if name:` — empty string is falsy in Python. So this only runs after the user types something.

`st.success()` — shows text inside a green box. Streamlit has colour-coded message types:
- `st.success()` → green (good result)
- `st.error()` → red (something went wrong)
- `st.warning()` → yellow (caution)
- `st.info()` → blue (information)

Note: Streamlit reruns the **entire script** from top to bottom every time the user types a character or clicks anything. There are no callbacks like in tkinter. The current value of `st.text_input()` is just returned directly.

---

## File 81 — Button

```python
if st.button("Marvellous_Python"):
    st.success("Jay Ganesh")
```

`st.button("label")` — shows a clickable button. Returns `True` only when clicked, `False` otherwise.

The `if` block runs only on the rerun that happens immediately after the click. On the next user action, `st.button()` returns `False` again — so the success message disappears. This is how Streamlit's stateless model works.

If you want the message to stay after clicking, you'd use `st.session_state` (a more advanced Streamlit feature not covered here).

---

## File 82 — File Upload

```python
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF Uploaded Successfully")
```

`st.file_uploader("label", type=["pdf"])` — shows a drag-and-drop file upload area. `type=["pdf"]` restricts it to PDF files only. If we wanted images too we'd write `type=["pdf", "png", "jpg"]`.

`uploaded_file` is a file-like object (similar to what `open()` returns). We can pass it directly to libraries like `PyPDF2`. If no file is uploaded yet, it's `None`.

`if uploaded_file:` — `None` is falsy, so this only runs after a file is chosen.

---

## File 83 — PDF Text Extraction

```python
import PyPDF2

if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    st.text_area("Extracted Text", text)
```

`PyPDF2.PdfReader(uploaded_file)` — creates a reader object from the uploaded file. Streamlit's uploaded file object is directly compatible — no need to save it to disk first.

`pdf_reader.pages` — a list of page objects. We loop through all pages and extract text from each.

`page.extract_text()` — extracts all text from one page as a string. Some PDFs (scanned images) may return empty strings because the content is an image, not actual text. For those, OCR (Optical Character Recognition) tools would be needed.

`st.text_area("label", value)` — shows a multi-line text box with the extracted text pre-filled. The user can scroll through it or copy from it.

---

## File 84 — Text Chunking

```python
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
```

Here the user types (or pastes) text directly. Then we split it into chunks of 100 characters.

`range(0, len(text), chunk_size)` — generates indices 0, 100, 200, 300... `text[i:i+100]` slices 100 characters starting at each index. The last chunk might be shorter if the text length isn't divisible by 100.

`st.info(chunk)` — shows each chunk in a blue info box. This makes it visually easy to see where one chunk ends and another begins.

**Why chunking?** In RAG systems, we can't feed an entire 50-page PDF into an LLM at once (too many tokens). We break the text into small chunks, convert each chunk to a vector (embedding), store in a vector database, and retrieve only the relevant chunks when a user asks a question. This folder lays the groundwork for that.

---

## The RAG Pipeline This Is Building Toward

```
Upload PDF (File 82)
    ↓
Extract text from all pages (File 83)
    ↓
Split into chunks (File 84)
    ↓
[Next steps — not in these files:]
Convert chunks to embeddings (e.g. using SentenceTransformers)
    ↓
Store in vector database (e.g. FAISS, ChromaDB)
    ↓
User asks question → find closest chunks → send to LLM → get answer
```

---

## Streamlit Functions Used — Quick Reference

| Function | What it shows |
|----------|--------------|
| `st.title(text)` | Large heading |
| `st.write(text)` | Any content (text, numbers, dataframes, plots) |
| `st.text_input(label)` | Single-line text box |
| `st.text_area(label)` | Multi-line text box |
| `st.button(label)` | Clickable button, returns True when clicked |
| `st.file_uploader(label, type=[...])` | Drag-and-drop file upload |
| `st.success(text)` | Green message box |
| `st.info(text)` | Blue message box |
| `st.error(text)` | Red message box |
| `st.warning(text)` | Yellow message box |

---

## How to Run Each File

```bash
pip install streamlit PyPDF2

streamlit run 79_streamlit_JayGanesh.py
streamlit run 80_streamlit_Input.py
streamlit run 81_streamlit_Button.py
streamlit run 82_streamlit_fileUpload.py
streamlit run 83_streamlit_TextExtraction.py   # needs a PDF to test
streamlit run 84_streamlit_Chunking.py
```

Run one file at a time. The browser opens automatically at `http://localhost:8501`.

---

*Marvellous Infosystems — Building ML web apps with Streamlit, from hello world to PDF chunking.*
