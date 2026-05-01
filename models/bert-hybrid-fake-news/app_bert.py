import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import nltk
from transformers import BertTokenizer, BertModel
import requests
from bs4 import BeautifulSoup

# NLTK requirements
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt_tab')

st.set_page_config(page_title="BERT Hybrid Fake News Detector", layout="wide")

st.title("🔠 BERT Hybrid Fake News Detector")

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
def load_bert_resources():
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        class HybridModel(nn.Module):
            def __init__(self, feature_dim=5001):  
                super().__init__()
                self.bert = BertModel.from_pretrained("bert-base-uncased")
                self.fc = nn.Linear(768 + feature_dim, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, input_ids, attention_mask, features):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls = outputs.pooler_output
                combined = torch.cat((cls, features), dim=1)
                return self.sigmoid(self.fc(combined))

        model = HybridModel()
        model.load_state_dict(torch.load("models/hybrid_model.pth", map_location="cpu"))
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

bert_tokenizer, bert_model = load_bert_resources()

if not bert_tokenizer or not bert_model:
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
            nlp_tokens = nltk.word_tokenize(text_to_analyze)
            tags = nltk.pos_tag(nlp_tokens)
            nouns = sum(1 for _, t in tags if t.startswith("NN"))
            adjs = sum(1 for _, t in tags if t.startswith("JJ"))
            pos_ratio_val = adjs / (nouns + 1)
            
            features = np.zeros((1, 5001))
            features[0, 5000] = pos_ratio_val
            features_tensor = torch.tensor(features).float()
            
            encoded = bert_tokenizer(
                [text_to_analyze], 
                padding=True, 
                truncation=True, 
                max_length=200, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                out_bert = bert_model(encoded["input_ids"], encoded["attention_mask"], features_tensor)
            score_bert = out_bert.item()

            bert_label = "FAKE" if score_bert > 0.5 else "REAL"
            st.markdown(f"## Prediction: **{bert_label}**")
            st.progress(score_bert, text=f"Confidence Score (closer to 1 = Fake): {score_bert:.2f}")
            
            st.markdown("### NLP Feature Extraction Context")
            st.info(
                f"**Classical Signals:**\n"
                f"- Adjective/Noun Ratio: {pos_ratio_val:.4f}\n"
                f"- Noun Count: {nouns}\n"
                f"- Adjective Count: {adjs}\n\n"
                f"**BERT Context:**\n"
                f"- Tokens analyzed: {encoded['input_ids'].shape[1]}"
            )
