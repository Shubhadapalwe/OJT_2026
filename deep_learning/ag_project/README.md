# RAG Project — Intelligent Document Question Answering System

**Marvellous Infosystems | Deep Learning & AI Portfolio**
**Trainee: Shubhada A. Palwe**

---

## What's in This Folder

| File | What it does |
|------|-------------|
| `86_Marvellous_Intelligent_Document_Question_Answering_System.py` | Full RAG pipeline: PDF upload → text extraction → chunking → vector search → LLM answer |

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. It's a technique where instead of asking a language model questions from its own training knowledge, we:

1. Upload our own documents
2. Search those documents for relevant pieces
3. Feed those pieces to the LLM as context
4. Get an answer grounded in *our* documents — not just generic knowledge

This is how tools like "Chat with your PDF" work. It's one of the most practical AI applications right now.

---

## The Full Pipeline

```
User uploads PDF
        ↓
Extract text (PyPDF2)
        ↓
Split into overlapping chunks (500 chars, 100 overlap)
        ↓
Embed each chunk → dense vector (SentenceTransformers)
        ↓
Store vectors in FAISS index
        ↓
User asks a question
        ↓
Embed the question → vector
        ↓
FAISS finds top-3 nearest chunk vectors (semantic search)
        ↓
Combine chunks into context prompt
        ↓
Send prompt to Ollama/Llama3 (local LLM)
        ↓
Display answer in Streamlit UI
```

---

## File 86 — Full RAG Streamlit App

### Step 1: Extract Text from PDF

```python
import PyPDF2

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
```

`PdfReader` accepts the file-like object from `st.file_uploader` directly — no need to save it to disk first. We loop through every page and concatenate the text.

### Step 2: Split into Overlapping Chunks

```python
def split_text_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap   # step back by overlap before moving forward
    return chunks
```

**Why overlap?** If we split text into clean non-overlapping blocks, important sentences might get cut in half between two chunks. By overlapping 100 characters, we ensure no key idea is lost at a boundary. Each chunk shares its last 100 characters with the start of the next chunk.

Example with chunk_size=10, overlap=3:
```
Text:  "ABCDEFGHIJKLMNOP"
Chunk 1: "ABCDEFGHIJ"    (0 to 10)
Chunk 2: "HIJKLMNOP..."  (7 to 17)  ← starts 3 chars before end of chunk 1
```

### Step 3: Load the Embedding Model

```python
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
```

`@st.cache_resource` is crucial here. Streamlit reruns the entire script on every user interaction. Without caching, it would reload the 80MB model on every button click — which would be very slow. With `@st.cache_resource`, the model loads once and stays in memory.

`all-MiniLM-L6-v2` is a lightweight but effective model that converts text into 384-dimensional vectors. Similar text → similar vectors.

### Step 4: Build the FAISS Vector Database

```python
import faiss
import numpy as np

def create_vector_database(chunks, model):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]   # 384 for MiniLM
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    return index, chunks
```

- `model.encode(chunks)` converts all chunks to embedding vectors at once — shape `(num_chunks, 384)`
- `faiss.IndexFlatL2(384)` creates a FAISS index using L2 (Euclidean) distance
- `index.add(...)` stores all the vectors — now we can search them instantly

**Note:** FAISS (Facebook AI Similarity Search) is a library specifically designed for fast vector similarity search. Even with millions of vectors, it finds the nearest neighbors in milliseconds.

### Step 5: Semantic Search

```python
def search_relevant_chunks(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype("float32"), top_k)
    return [chunks[i] for i in indices[0]]
```

- We embed the user's question the same way we embedded the chunks
- `index.search(query_vec, 3)` returns the 3 nearest chunk vectors by L2 distance
- `indices[0]` gives us the chunk positions; we return the actual text

**This is semantic search, not keyword search.** If the user asks "What is the revenue?", it will find chunks talking about "earnings", "income", or "profit" — even if the word "revenue" never appears in those chunks — because semantically similar concepts produce similar vectors.

### Step 6: Ask the LLM

```python
import requests
import json

def ask_llm(question, context):
    prompt = f"""You are a helpful assistant. Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    result = response.json()
    return result.get("response", "No response received")
```

We're using **Ollama** — a tool that lets you run LLMs locally on your own machine. It exposes a REST API at `localhost:11434`. We POST our prompt and get back a JSON response with the model's answer.

**Why local LLM?** No API costs, no data leaving your machine, no internet required. For enterprise document Q&A (legal docs, medical records, confidential reports), this is very important.

### The Streamlit UI

```python
import streamlit as st

# Dark custom CSS
st.markdown("""
<style>
.stApp { background-color: #0f172a; color: white; }
...
</style>
""", unsafe_allow_html=True)

st.title("Intelligent Document Q&A System")

# 3 tabs
tab1, tab2, tab3 = st.tabs(["Upload Document", "Ask Questions", "System Info"])
```

The UI has 3 tabs:

**Tab 1 — Upload Document:**
```python
with tab1:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing document..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = split_text_into_chunks(text)
            model = load_embedding_model()
            index, stored_chunks = create_vector_database(chunks, model)
            st.session_state["index"] = index
            st.session_state["chunks"] = stored_chunks
            st.session_state["model"] = model
        st.success(f"Document processed! {len(chunks)} chunks created.")
        st.metric("Total Chunks", len(chunks))
```

We store the index, chunks, and model in `st.session_state` so they persist across reruns. Otherwise each tab click would reset everything.

**Tab 2 — Ask Questions:**
```python
with tab2:
    question = st.text_input("Ask a question about your document")
    if st.button("Get Answer") and question:
        relevant = search_relevant_chunks(question, model, index, stored_chunks)
        context = "\n\n".join(relevant)
        with st.spinner("Thinking..."):
            answer = ask_llm(question, context)
        st.write("**Answer:**", answer)
        with st.expander("View source chunks"):
            for i, chunk in enumerate(relevant):
                st.info(f"Chunk {i+1}: {chunk}")
```

**Tab 3 — System Info:** Shows model name, index type, chunk count as `st.metric` cards.

The sidebar shows quick stats and usage instructions.

---

## How to Run

```bash
# 1. Install dependencies
pip install streamlit sentence-transformers faiss-cpu PyPDF2

# 2. Install and start Ollama (for local LLM)
# Download from https://ollama.ai
ollama pull llama3
ollama serve

# 3. Run the app
streamlit run 86_Marvellous_Intelligent_Document_Question_Answering_System.py
```

**Note:** Ollama must be running in the background before you launch the Streamlit app. If it's not running, the Q&A step will fail (you'll get a connection error to `localhost:11434`).

---

## Component Summary

| Component | Library | Role |
|-----------|---------|------|
| PDF text extraction | PyPDF2 | Read PDF pages |
| Overlapping chunking | Pure Python | Split text safely |
| Text embedding | SentenceTransformers | Convert text → vectors |
| Vector storage & search | FAISS | Fast nearest-neighbor search |
| Local LLM | Ollama + Llama3 | Generate answers |
| Web UI | Streamlit | Interactive interface |

---

## What We Learned

This project ties together almost everything from the course:
- Text preprocessing (chunking, cleaning)
- Embeddings (the same concept as in RNN/Transformer — dense vector representations)
- Similarity search (vector math in action)
- LLM prompting (how to give a model context)
- Streamlit UI (tabs, session state, spinner, expander, metric)

RAG is one of the most widely deployed AI patterns in industry right now. Almost every enterprise chatbot, document search tool, or knowledge base assistant uses some version of this pipeline.
