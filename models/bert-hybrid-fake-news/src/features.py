import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

vectorizer = TfidfVectorizer(max_features=5000)

def extract_tfidf(texts):
    return vectorizer.fit_transform(texts).toarray()

def pos_ratio(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)

    nouns = sum(1 for _, t in tags if t.startswith("NN"))
    adjs = sum(1 for _, t in tags if t.startswith("JJ"))

    return adjs / (nouns + 1)

def build_features(texts):
    tfidf = extract_tfidf(texts)
    pos = np.array([pos_ratio(t) for t in texts]).reshape(-1, 1)

    return np.hstack([tfidf, pos])