import streamlit as st
import torch
import torch.nn as nn
import pickle
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
from bs4 import BeautifulSoup

# NLTK requirements
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

st.set_page_config(page_title="BiLSTM Fake News Detector", layout="wide")

st.title("🧠 BiLSTM + Attention Fake News Detector")

def fetch_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text.strip()
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

@st.cache_resource
def load_bilstm_resources():
    try:
        with open("tokenizer.pkl", "rb") as f:
            word2idx = pickle.load(f)
        
        class BiLSTM_Attention(nn.Module):
            def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
                self.attention = nn.Linear(hidden_dim * 2, 1)
                self.dropout = nn.Dropout(0.3)
                self.fc = nn.Linear(hidden_dim * 2, 1)

            def forward(self, x):
                x = self.embedding(x)
                lstm_out, _ = self.lstm(x)
                attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context = torch.sum(attn_weights * lstm_out, dim=1)
                out = self.dropout(context)
                out = torch.sigmoid(self.fc(out))
                return out, attn_weights

        model = BiLSTM_Attention(vocab_size=10000)
        model.load_state_dict(torch.load("models/bilstm_attention.pth", map_location="cpu"))
        model.eval()
        return word2idx, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

word2idx, bilstm_model = load_bilstm_resources()

if not word2idx or not bilstm_model:
    st.stop()

input_mode = st.radio("Choose Input Mode:", ["Manual Text Entry", "Article URL Scraping"])

text_to_analyze = ""
if input_mode == "Manual Text Entry":
    text_to_analyze = st.text_area("Paste news article content here:", height=150)
else:
    url_input = st.text_input("Enter News Article URL:")
    if url_input:
        with st.spinner("Extracting text from URL..."):
            text_to_analyze = fetch_article_text(url_input)
            if text_to_analyze:
                st.success("Successfully extracted text!")

if st.button("🔍 Analyze Article", type="primary"):
    if not text_to_analyze.strip():
        st.warning("Please provide some text to analyze.")
    else:
        st.divider()
        with st.spinner("Analyzing..."):
            text_bilstm = text_to_analyze.lower()
            text_bilstm = re.sub(r"<.*?>", "", text_bilstm)
            text_bilstm = re.sub(r"[^a-zA-Z\s]", "", text_bilstm)
            
            try:
                b_tokens = word_tokenize(text_bilstm)
            except:
                b_tokens = text_bilstm.split()
                
            seq = [word2idx.get(w, 0) for w in b_tokens]
            MAX_LEN = 200
            if len(seq) < MAX_LEN:
                seq = seq + [0] * (MAX_LEN - len(seq))
            else:
                seq = seq[:MAX_LEN]
                b_tokens = b_tokens[:MAX_LEN] 
                
            X_bilstm = torch.tensor([seq]).long()
            
            with torch.no_grad():
                pred_bilstm, attn = bilstm_model(X_bilstm)
            score_bilstm = pred_bilstm.item()
            attn_weights = attn.squeeze().numpy()

            b_label = "FAKE" if score_bilstm > 0.5 else "REAL"
            st.markdown(f"## Prediction: **{b_label}**")
            st.progress(score_bilstm, text=f"Confidence Score (closer to 1 = Fake): {score_bilstm:.2f}")
            
            st.markdown("### Attention Heatmap")
            actual_len = min(len(b_tokens), MAX_LEN)
            vis_weights = attn_weights[:actual_len]
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.heatmap([vis_weights], xticklabels=b_tokens, cmap="Reds", ax=ax, cbar=False)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks([])
            st.pyplot(fig)
