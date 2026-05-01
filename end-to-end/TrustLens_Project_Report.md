# TRUSTLENS: FORENSIC MISINFORMATION DETECTION
## Comprehensive Technical Research Report & System Documentation

**Team Members:**  
- **Arpita Mahapatra**  
- **Namradha Mani**

---

## 1. Executive Summary
TrustLens is a production-grade news verification suite developed to combat digital misinformation through multi-modal neural inference. The project consolidates two high-impact research modules:
1.  **Deep Learning (DL) Engine**: A 2-layer BiLSTM with Self-Attention for contextual sequence mapping.
2.  **NLP Hybrid Engine**: A BERT-base model augmented with classical stylometric features (TF-IDF, POS distribution, Sentiment).

These engines are integrated into a unified **TrustLens Dashboard**, providing real-time article auditing, attention visualizations, and forensic performance benchmarks.

---

## 2. Research Module I: Deep Learning (CSR311)
### **Objective: Sequential Context Mapping via BiLSTM + Attention**

#### **2.1 Dataset & Preprocessing**
The model was trained on the **LIAR** and **FakeNewsNet** datasets.
- **Normalisation**: Lowercasing, removal of HTML tags/special characters via Regex.
- **Tokenization**: NLTK-based word tokenization.
- **Vectorization**: Custom tokenizer mapping tokens to a 5000-word vocabulary.
- **Padding**: Constant sequence length of **200 tokens** to ensure uniform tensor shapes.

#### **2.2 Model Architecture**
The architecture consists of a 2-layer Bi-directional LSTM to capture dependencies from both ends of the sentence.
- **Embedding Layer**: 128-dimensional dense vectors.
- **BiLSTM Layer**: 256 hidden units with 0.3 dropout.
- **Self-Attention Layer**: A scaled dot-product attention mechanism that calculates importance weights for each token.
- **Output Layer**: Sigmoid activation for binary classification (Real vs. Fake).

#### **2.3 Code Implementation (Core Logic)**
```python
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Attention Mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return torch.sigmoid(self.fc(context)), attn_weights
```

#### **2.4 Evaluation & Visualization**
- **Accuracy**: 0.84 | **AUC-ROC**: 0.86.
- **Interpretability**: The Attention weights are rendered as a **Heatmap** in the TrustLens UI, allowing users to see exactly which words (e.g., "urgent", "secret", "exposed") triggered the "Fake" verdict.

---

## 3. Research Module II: NLP (CSR322)
### **Objective: Stylometric Fingerprinting via BERT Hybrid Model**

#### **3.1 Hybrid Feature Engineering**
The core innovation is the combination of deep contextual embeddings with explicit linguistic signals:
- **BERT [CLS]**: Captures the high-level semantic meaning of the article.
- **TF-IDF Vector**: Captures the frequency-based importance of rare keywords.
- **POS Distribution**: Calculates the **Adjective-to-Noun Ratio**, as fake news often employs excessive emotional modifiers.
- **Sentiment Polarity**: Analyzes emotional intensity using TextBlob/VADER.

#### **3.2 Model Architecture**
The model fine-tunes `bert-base-uncased` and concatenates its pooler output with a 5000-dimensional TF-IDF/Stylometric vector.
- **Concatenation Layer**: Merges 768 (BERT) + 5001 (Classical) features.
- **Final Classifier**: A dense linear layer reducing the 5769-dim vector to a single probability.

#### **3.3 Code Implementation (Core Logic)**
```python
class HybridBERT(nn.Module):
    def __init__(self, feature_dim=5001):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768 + feature_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, classical_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.pooler_output # [batch, 768]
        combined = torch.cat((cls_embedding, classical_features), dim=1)
        return self.sigmoid(self.fc(combined))
```

#### **3.4 Forensic Error Analysis**
We analyzed 20 misclassified examples to identify failure modes:
1.  **Sarcasm (40%)**: BERT struggles with high-contextual irony.
2.  **Domain Shift (30%)**: Technical reports misclassified due to "unusual" vocabulary.
3.  **Keyword Bias (30%)**: Genuine news containing words like "Trump" or "Iran" occasionally flagged as suspicious.

---

## 4. End-to-End System Integration
### **The TrustLens Unified Dashboard**

#### **4.1 System Backend (FastAPI)**
The unified `api.py` acts as an ensemble layer. It executes both models in parallel and applies a **Weighted Consensus**:
- **Consensus Score** = (0.4 * DL_Score) + (0.6 * NLP_Score).
- If the score exceeds 0.5, the article is flagged as **SUSPICIOUS**.

#### **4.2 AuraTruth UI/UX Design**
- **Home**: A high-impact landing page highlighting the "DNA of Truth".
- **Modules**: Interactive 3D flip-cards explaining technical specs.
- **Analyzer**: Real-time article auditing with "Step-by-Step" terminal logs.
- **Forensics**: Live benchmarks comparing the base vs. augmented versions of the models.

---

## 5. Deployment & GitHub
### **Source Code Repository**
The project is structured for seamless reproducibility:
- `/fake-news-bilstm-attention/`: Deep Learning module source code.
- `/bert-hybrid-fake-news/`: NLP Hybrid module source code.
- `/static/`: Frontend assets (Lumina UI).
- `api.py`: The unified ensemble server.

**GitHub Repository Structure:**
```text
TrustLens/
├── README.md (Project Overview)
├── requirements.txt (Dependency Manifest)
├── api.py (Ensemble Server)
├── static/ (Lumina UI Assets)
└── research_modules/
    ├── DL_BiLSTM_Attention/
    └── NLP_Hybrid_BERT/
```

---

## 6. Conclusion & Future Work
TrustLens successfully demonstrates that combining sequential context (BiLSTM) with stylistic fingerprints (BERT Hybrid) creates a robust defense against misinformation. 
**Future Enhancements:**
- Integration of a "Source Credibility API" (e.g., NewsGuard).
- Multi-lingual support for regional fake news detection.
- Graph-based propagation analysis to track how fake news spreads.

---
**Prepared by:**  
*Arpita Mahapatra & Namradha Mani*  
*Project Submission 2026*
