import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import gdown  # Make sure to add 'gdown' to your requirements.txt

# ========== SETTINGS ==========
CSV_DIR = "."
FILENAME_PATTERN = r"fetched_reddit_comments_(\w+)\.csv"

# ========== FILE DOWNLOAD ==========
# Replace these with actual Google Drive file IDs and names
file_ids = {
    "adidas": {"id": "16bo8ujiyiG61o3xzvGPR-pDrRVAIrSwj", "filename": "fetched_reddit_comments_adidas.csv"},
    "aliexpress": {"id": "19q8sJEBVskkqyg1VhOn_CGk6fEIvjWeK", "filename": "fetched_reddit_comments_aliexpress.csv"},
    "amazon": {"id": "1OQuHy-GYdYFSN9x2q5TGW4pMw3zkIVoe", "filename": "fetched_reddit_comments_amazon.csv"},
    "bmw": {"id": "1qVfjjlzFbjQBGheRaJjg4F-4c_iKvZcf", "filename": "fetched_reddit_comments_bmw.csv"},
    "iphone": {"id": "1Ln7iB7xUKq2bpYwkcsbZt61y0zh3m0zb", "filename": "fetched_reddit_comments_iphone.csv"},
    "nike": {"id": "1fr5KDslVOiX0z3i31HrLng2CqZyb3jRr", "filename": "fetched_reddit_comments_nike.csv"},
    "samsung": {"id": "1Pa1o470J38L8R3vvhJnto6xxQ3k6_kPQ", "filename": "fetched_reddit_comments_samsung.csv"},
    "tesla": {"id": "1sWSP2PPlvkK-XwQelUbGTK2D6-5nXJrE", "filename": "fetched_reddit_comments_tesla.csv"},
    "toyota": {"id": "14jXI55mgWgFvPqP0QMVTFD7f7gFDXL9_", "filename": "fetched_reddit_comments_toyota.csv"},
    # Add all 9 here
}

def download_csvs_if_missing():
    for brand, info in file_ids.items():
        file_path = os.path.join(CSV_DIR, info["filename"])
        if not os.path.exists(file_path):
            try:
                gdown.download(f"https://drive.google.com/uc?id={info['id']}", file_path, quiet=False)
                st.info(f"Downloaded {info['filename']}")
            except Exception as e:
                st.warning(f"Could not download {info['filename']}: {e}")

download_csvs_if_missing()

# ========== FUNCTIONS ==========

def get_available_brands():
    files = os.listdir(CSV_DIR)
    brands = []
    for file in files:
        match = re.match(FILENAME_PATTERN, file)
        if match:
            brands.append(match.group(1).lower())
    return sorted(brands)

def load_data_for_brands(brands, limit):
    dfs = []
    for brand in brands:
        filename = f"fetched_reddit_comments_{brand}.csv"
        filepath = os.path.join(CSV_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df["brand"] = brand
            dfs.append(df.head(limit))
        else:
            st.warning(f"⚠️ File not found: {filename}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def detect_comment_column(df):
    for col in df.columns:
        if col.strip().lower() in ["comment", "body", "text"]:
            return col
    return None

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def generate_wordcloud(text_series, title):
    wordcloud = WordCloud(width=400, height=300, background_color="white").generate(" ".join(text_series))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    return fig

# ========== STREAMLIT UI ==========

st.title("🧠 Reddit Brand Sentiment Analyzer")

available_brands = get_available_brands()
selected_brands = st.multiselect("Select brand(s) to analyze:", available_brands)

comment_limit = st.slider("Maximum number of comments per brand:", min_value=10, max_value=1000, value=100, step=10)

if st.button("Analyze"):
    if not selected_brands:
        st.warning("Please select at least one brand.")
    else:
        df = load_data_for_brands(selected_brands, comment_limit)

        if df.empty:
            st.error("No data found for the selected brand(s).")
        else:
            st.write(f"✅ Loaded {len(df)} comments from: {', '.join(selected_brands)}")

            comment_col = detect_comment_column(df)
            if comment_col is None:
                st.error("❌ Could not find a comment column (expected: 'comment', 'body', or 'text').")
                st.write("Available columns:", df.columns.tolist())
            else:
                df["cleaned_text"] = df[comment_col].astype(str).apply(clean_text)

                st.info("Running sentiment analysis...")

                pipe = get_sentiment_pipeline()
                try:
                    results = pipe(df["cleaned_text"].tolist(), truncation=True, batch_size=16)
                    df["label"] = [res["label"] for res in results]
                except Exception as e:
                    st.error(f"Sentiment analysis failed: {e}")
                    st.stop()

                pos = (df["label"] == "POSITIVE").sum()
                neg = (df["label"] == "NEGATIVE").sum()

                st.subheader("🔍 Sentiment Summary")
                st.write(f"**Positive**: {pos} | **Negative**: {neg}")

                st.subheader("📊 Sentiment Distribution")
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                ax.pie([pos, neg], labels=["Positive", "Negative"], colors=["#2D9F9F", "#FA7D4D"], autopct="%1.1f%%")
                st.pyplot(fig)

                st.subheader("☁ Word Clouds")
                pos_comments = df[df["label"] == "POSITIVE"]["cleaned_text"]
                neg_comments = df[df["label"] == "NEGATIVE"]["cleaned_text"]

                col1, col2 = st.columns(2)
                with col1:
                    if not pos_comments.empty:
                        st.pyplot(generate_wordcloud(pos_comments, "Positive Comments"))
                    else:
                        st.info("No positive comments found.")
                with col2:
                    if not neg_comments.empty:
                        st.pyplot(generate_wordcloud(neg_comments, "Negative Comments"))
                    else:
                        st.info("No negative comments found.")
