import streamlit as st
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Financial Dashboard", layout="wide")

st.title("📊 Financial Topic Modeling Dashboard")

# Load data
df = pd.read_csv("https://github.com/meaninditadeb2-eng/financial-topic-modeling-dashboard/blob/main/financial_news_Data.csv")

docs = df["clean_text"].tolist()

# Train model LIVE (important)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic(
    embedding_model=embedding_model,
    min_topic_size=3
)

topics, probs = topic_model.fit_transform(docs)

# Show topics
st.subheader("📌 Topics")
st.dataframe(topic_model.get_topic_info())

# Visualization
st.subheader("📈 Visualization")
try:
    st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)
except:
    st.plotly_chart(topic_model.visualize_barchart(), use_container_width=True)

# Distribution
st.subheader("📊 Topic Distribution")
df["topic"] = topics
st.bar_chart(df["topic"].value_counts())

# Data preview
st.subheader("📰 Data")
st.dataframe(df.head(20))
