import streamlit as st
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Financial Intelligence Dashboard",
    layout="wide"
)

# ---------------- TITLE ----------------
st.markdown("""
# 📊 Financial Intelligence Dashboard
### AI-powered Topic Modeling & Business Insights
""")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("financial_news_Data.csv")

if "clean_text" not in df.columns:
    df["clean_text"] = df["text"]

docs = df["clean_text"].astype(str).tolist()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model(docs):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=3
    )
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics

topic_model, topics = load_model(docs)

df["topic"] = topics

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Filters")

selected_topic = st.sidebar.selectbox(
    "Select Topic",
    ["All"] + sorted(df["topic"].unique())
)

# Filter data
if selected_topic != "All":
    df = df[df["topic"] == selected_topic]

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("📰 Total Articles", len(df))
col2.metric("📌 Unique Topics", df["topic"].nunique())
col3.metric("📊 Avg Topic ID", round(df["topic"].mean(), 2))

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Topics", 
    "📈 Visualization", 
    "📊 Distribution",
    "📰 Data"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Topic Overview")
    st.dataframe(topic_model.get_topic_info())

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Topic Visualization")
    try:
        st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)
    except:
        st.plotly_chart(topic_model.visualize_barchart(), use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Topic Distribution")
    st.bar_chart(df["topic"].value_counts())

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Filtered Data Preview")
    st.dataframe(df.head(20))

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💡 Built with NLP + BERTopic | Financial News Analysis")
