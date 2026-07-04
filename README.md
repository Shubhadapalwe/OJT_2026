<div align="center">

#  OJT 2026 — AI & Deep Learning Portfolio

### Shubhada A. Palwe
**On-the-Job Training**
*Data Science · Machine Learning · Deep Learning · Generative AI*

---

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?style=flat&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?style=flat&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)
![Files](https://img.shields.io/badge/Programs-90%2B-blueviolet?style=flat)

</div>

---

##  Hello, I'm Shubhada

This repository is my complete learning journal from the OJT 2026 program at **Marvellous Infosystems**.

When I started this program, I knew basic Python. By the end of it, I had built a working AI agent, a Retrieval-Augmented Generation (RAG) system, and trained Transformer models from scratch. This repo has every program I wrote along the way — more than **90 Python files** across **Machine Learning and Deep Learning** — kept in the exact order I wrote them.

Every folder has a README I wrote myself, explaining what I understood, what confused me at first, and how I eventually figured it out. If you're a student learning AI/ML, I hope this helps you too.

---

##  What's in This Repository

```
OJT_2026/
├── 📁 machine_learning/        → Where the journey started
└── 📁 deep_learning/           → Where things got exciting
```

---

##  Machine Learning

> *"I didn't understand why we needed math. Then I wrote my first gradient descent loop and it all clicked."*

| Folder | What I learned |
|--------|---------------|
| `linear_regression/` | How a straight line can predict things — slope, intercept, cost function |
| `irisclassification/` | My first real classifier — Logistic Regression on the famous Iris dataset |
| `ballclassification/` | A problem I designed myself: can a computer tell a tennis ball from a cricket ball? |
| `knn/` | K-Nearest Neighbors — sometimes the simplest idea works best |
| `decision_tree/` | How computers make decisions using Gini impurity and entropy |
| `advertising_casestudy/` | Multiple linear regression on real advertising data (TV, Radio, Newspaper → Sales) |
| `iris_casestudy_complete/` | End-to-end pipeline — preprocessing, training, evaluation, confusion matrix |
| `titanic_casestudy/` | The classic survival prediction problem |
| `wine_classifier_knn/` | KNN applied to wine quality classification |
| `userdefined_knn/` | I built KNN from scratch without sklearn — just NumPy and math |
| `pandas_dataframe/` | Wrangling real data with Pandas |
| `pandas_and_confusionmatrix/` | Precision, Recall, F1 — understanding what "accuracy" really means |
| `matplotlib_visualizations/` | Making data speak through charts |
| `rsquare/` | How to measure if a regression model is actually good |

---

##  Deep Learning

> *"The moment I saw a neural network learn on its own — even a tiny one — I was hooked."*

###  ANN — Artificial Neural Networks (Foundations)

| Folder | What I learned |
|--------|---------------|
| `ann_fundamentals/` | Built a neuron from scratch. Sigmoid, ReLU, forward pass, training loop. Even animated one. |
| `loss_and_backprop/` | Why MSE penalizes big errors harder than MAE. Traced backprop chain rule by hand. |
| `fnn_case_studies/` | MLPRegressor for salary prediction, MLPClassifier for pass/fail. Learned *why* scaling matters. |

###  CNN — Convolutional Neural Networks

| Folder | What I learned |
|--------|---------------|
| `image_fundamentals/` | What a pixel actually is. RGB channels. How computers see images. |
| `cnn_edge_detection/` | Sobel filter, manual convolution — edges are just math. |
| `tensorflow_fundamentals/` | TensorFlow tensors, constants, variables, operations from scratch. |
| `tensorflow_neural_networks/` | Keras Sequential API. model.compile, model.fit, model.evaluate. |
| `cnn_pipeline/` | The full CNN: Conv2D → ReLU → MaxPool → Flatten → Dense → Dropout. |
| `realtime_cnn_classification/` | Hooked up a webcam with OpenCV and ran live inference. That was fun. |

###  RNN — Recurrent Neural Networks

| Folder | What I learned |
|--------|---------------|
| `rnn_fundamentals/` | Character prediction, SimpleRNN, sequence input shape, Tokenizer. |
| `rnn_text_preprocessing/` | The full pipeline: split → vocab → sequences → padding. Wrote it 7 ways to understand it deeply. |
| `rnn_embedding_internals/` | How word embeddings work — from a manual dict lookup to Keras Embedding layer. |
| `rnn_sentiment_model/` | Built a complete sentiment classifier: Embedding → SimpleRNN → Dense → sigmoid. |
| `lstm_time_series/` | LSTM for stock price forecasting. EarlyStopping, MinMaxScaler, inverse_transform. |
| `industrial_projects/` | Wisconsin Breast Cancer dataset — sklearn Pipeline, RandomForest, joblib model saving. |

###  Transformer

| Folder | What I learned |
|--------|---------------|
| `transformer_fundamentals/` | Built self-attention from Q·K·V math, multi-head attention, positional encoding, full encoder and decoder. Then built BERT-style, NMT, and GPT-style models. |

###  Deployment & Generative AI

| Folder | What I learned |
|--------|---------------|
| `streamlit_basics/` | Built interactive web apps in Python. st.button, st.file_uploader, PDF extraction, text chunking. |
| `rag_project/` | **The capstone.** PDF → chunks → embeddings → FAISS → Llama3. A full working RAG system. |
| `tkinter_projects/` | Built a real desktop GUI calculator in Python using tkinter. |
| `ai_agents/` | Four agents: rule-based → calculator tool → multi-tool → memory. The building blocks of LangChain. |

---

##  Tech Stack

| What | Tools |
|------|-------|
| Language | Python 3.x |
| Data | NumPy, Pandas, Matplotlib |
| ML | Scikit-learn, joblib |
| Deep Learning | TensorFlow 2.x, Keras |
| Computer Vision | OpenCV |
| NLP | NLTK, SentenceTransformers |
| Vector Search | FAISS |
| Web App | Streamlit |
| Desktop GUI | tkinter |
| Local LLM | Ollama + Llama3 |
| Version Control | Git + GitHub |

---

##  How to Use This Repo

Every folder follows the same pattern:

```
topic_folder/
├── README.md        ← Start here. I explain what the code does and why.
├── 01_program.py
├── 02_program.py
└── ...
```

**Read the README first**, then go through the Python files in order. I wrote the READMEs to explain concepts the way I wish someone had explained them to me — in plain language, with the "why" not just the "what".

---

##  How to Run Any Program

```bash
# Most programs — just run directly
python filename.py

# Streamlit apps only
streamlit run filename.py

# RAG project — needs Ollama running first
ollama serve
streamlit run 86_Marvellous_Intelligent_Document_Question_Answering_System.py
```

**Requirements:**
```bash
pip install tensorflow keras scikit-learn numpy pandas matplotlib opencv-python streamlit sentence-transformers faiss-cpu PyPDF2
```

---

##  My Biggest Takeaways

After writing 90+ programs, these are the things that actually stuck with me:

1. **Scaling matters more than the model.** I learned this the hard way — one unscaled feature broke my entire ANN.
2. **Backpropagation is just the chain rule.** Once I traced it manually for a 2-layer network, I stopped being afraid of it.
3. **Embeddings are everywhere.** Word2Vec, Transformer attention, FAISS search — they all come down to "similar things should have similar vectors."
4. **RAG is just search + prompting.** It sounds fancy but the idea is simple: find the relevant text, then ask the LLM about it.
5. **Agents are just routers.** Rule-based or LLM-powered, an agent is just deciding which tool to call.

---

##  Contact

**Shubhada A. Palwe**
OJT Trainee —  Pune (2026)
 monapalwe0@gmail.com
 [GitHub](https://github.com/Shubhadapalwe/OJT_2026)

---

<div align="center">

*"The best way to learn AI is to build things that don't work, figure out why, and fix them."*

 If this repo helped you understand something, give it a star!

</div>
