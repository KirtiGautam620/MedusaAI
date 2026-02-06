# Intelligent Research Topic Analysis System

## Project Overview
This project focuses on building an **AI-based research topic analysis system** using **traditional Natural Language Processing (NLP) and Machine Learning techniques**.  
The system processes research papers related to a user-specified topic, identifies key themes and keywords, applies topic modeling or clustering, and generates **extractive summaries**.

The application includes a **user-friendly interface** and is designed using **only open-source libraries and free tools**, without the use of large language models (LLMs).

---

##  Project Objective
The objective of this project is to:
- Analyze research documents related to a given topic
- Extract important keywords and themes
- Discover latent topics or clusters within documents
- Generate structured **extractive summaries**
- Present analytical insights through a simple UI

This system demonstrates the strengths and limitations of **traditional NLP-based research analysis**.

---

##  Dataset
- **arXiv Research Papers Dataset**
- Source: https://www.kaggle.com/datasets/Cornell-University/arxiv

The dataset contains metadata and abstracts of academic research papers across multiple domains and is used for topic analysis and summarization.

---


##  Workflow (Traditional NLP Pipeline)

Research Papers / Uploaded Documents

↓

Text Preprocessing
(Tokenization, Stopword Removal, Lemmatization)

↓


Feature Extraction
(TF-IDF / Bag of Words)

↓

Topic Modeling / Clustering
(LDA / NMF / KMeans)

↓

Extractive Summarization
(TF-IDF-based Sentence Scoring)

↓

UI Display
(Keywords, Topics, Summary)



---

##  Tech Stack

### Programming Language
- Python

### NLP & Machine Learning
- NLTK
- spaCy
- scikit-learn
- NumPy

### Feature Extraction & Modeling
- TF-IDF
- Bag of Words
- LDA / NMF
- KMeans Clustering

### User Interface
- Streamlit / Gradio

### Visualization 
- matplotlib
- seaborn




