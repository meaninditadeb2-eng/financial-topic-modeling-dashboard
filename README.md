# 📊 Financial Intelligence Dashboard using Topic Modeling

## 🚀 Overview

This project builds an end-to-end **Financial News Topic Modeling Dashboard** to extract meaningful insights from unstructured business news data.

It combines **traditional NLP models (LDA, NMF)** with a **transformer-based model (BERTopic)** to demonstrate how modern techniques improve topic quality and business understanding.

---

## 🎯 Objectives

* Extract hidden topics from financial news
* Compare traditional vs transformer-based models
* Generate actionable business insights
* Build an interactive dashboard using Streamlit

---

## 🧠 Models Used

### 🔹 LDA (Latent Dirichlet Allocation)

* Based on word frequency
* Serves as baseline model

### 🔹 NMF (Non-negative Matrix Factorization)

* Uses TF-IDF representation
* Produces more interpretable topics than LDA

### 🔹 BERTopic (Transformer-Based)

* Uses BERT embeddings
* Captures semantic meaning
* Produces highly coherent topics

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* BERTopic
* Sentence Transformers
* Streamlit
* Plotly

---

## 🔄 Project Pipeline

```text
Data Collection (API) → Data Preprocessing → 
LDA / NMF Modeling → BERTopic Modeling → 
Topic Labeling → Visualization → Dashboard
```

---

## 📊 Key Features

✔ NLP preprocessing pipeline
✔ Comparative topic modeling (LDA vs NMF vs BERTopic)
✔ Transformer-based semantic analysis
✔ Interactive Streamlit dashboard
✔ Topic visualization and distribution

---

## 📈 Business Insights

* Financial news is dominated by market-related topics
* Economic indicators like inflation show distinct topic clusters
* Transformer-based models provide more meaningful insights compared to traditional methods

---

## 🖥️ Dashboard Features

* Topic overview and clustering
* Interactive topic visualization
* Topic distribution analysis
* Filterable data exploration

---

## 📁 Project Structure

```text
financial-topic-modeling-dashboard
│
├── financial_news_Data.csv
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

The application is deployed using
Streamlit Community Cloud

---

## 💡 Key Learnings

* Importance of preprocessing in NLP pipelines
* Differences between classical and transformer-based models
* Handling real-world deployment challenges
* Trade-offs between interpretability and performance

---

## 💼 Author

**Anindita Deb**

---

## 🚀 Future Improvements

* Real-time API integration in dashboard
* Automated topic labeling
* Sentiment analysis integration
* Advanced UI enhancements
