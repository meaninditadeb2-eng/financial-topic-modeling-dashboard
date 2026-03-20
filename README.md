# 📊 Financial Intelligence Dashboard using Transformer-Based Topic Modeling

## 🚀 Project Overview

This project builds an end-to-end **Financial News Topic Modeling System** using both traditional and modern NLP techniques.

It leverages real-time data from the NewsAPI to extract meaningful business insights from unstructured financial news.

---

## 🎯 Objectives

* Extract hidden topics from financial news
* Compare traditional vs transformer-based topic models
* Generate business insights from textual data
* Build an interactive dashboard for visualization

---

## 🧠 Models Used

### 🔹 1. LDA (Latent Dirichlet Allocation)

* Probabilistic model based on word frequency
* Baseline for topic modeling

### 🔹 2. NMF (Non-negative Matrix Factorization)

* Uses TF-IDF representation
* Produces more interpretable topics than LDA

### 🔹 3. BERTopic (Transformer-Based)

* Uses BERT embeddings
* Clustering + semantic understanding
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
NewsAPI → Data Collection → NLP Preprocessing → 
LDA / NMF → BERTopic → Topic Labeling → 
Sentiment Analysis → Streamlit Dashboard
```

---

## 📊 Key Features

✔ Real-time data collection using API
✔ NLP preprocessing pipeline
✔ Comparative analysis (LDA vs NMF vs BERTopic)
✔ Topic labeling for business understanding
✔ Sentiment analysis integration
✔ Interactive Streamlit dashboard

---

## 📈 Business Insights

* Stock market dominates financial news coverage
* Inflation-related topics show negative sentiment trends
* M&A topics indicate corporate growth activity
* Transformer-based models outperform traditional methods in topic coherence

---

## 🖥️ Dashboard Features

* Topic visualization (interactive)
* Topic distribution analysis
* Sentiment analysis by topic
* Data filtering and exploration

---

## 📁 Project Structure

```text
financial-topic-modeling
│
├── data/
│   └── processed_data.csv
│
├── model/
│   └── topic_model/
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 🌐 Deployment

The application can be deployed using
Streamlit Community Cloud

---

## 💡 Key Learnings

* Importance of NLP preprocessing in traditional models
* Transformer models improve semantic understanding
* Trade-off between model complexity and interpretability
* Real-world challenges in topic modeling (data size, clustering issues)

---

## 💼 Author

**Anindita Deb**

---

## 📌 Future Improvements

* Real-time API integration in dashboard
* Automated topic labeling using LLMs
* Time-series trend analysis
* Advanced UI enhancements
