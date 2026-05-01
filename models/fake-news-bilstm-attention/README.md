# Fake News Detection using BiLSTM and Attention

This project implements and compares two deep learning models for classifying news articles as fake or real:
1. A standard **Bidirectional LSTM (BiLSTM)**
2. A **BiLSTM with an Attention Mechanism**

The models are built using PyTorch and evaluate the effectiveness of adding an attention layer for text classification tasks.

## Project Structure

```
fake-news-bilstm-attention/
│
├── data/                   # Dataset directory (contains raw/processed data)
├── models/                 # Saved PyTorch model weights
├── outputs/                # Evaluation metrics and generated plots
│   ├── attention_plot.png  # Visualization of attention weights
│   └── metrics.txt         # Accuracy and AUC scores
│
├── src/                    # Source code
│   ├── preprocess.py       # Data cleaning and tokenization
│   ├── split_test.py       # Train/test split utility
│   ├── model.py            # PyTorch model definitions (BiLSTM & BiLSTM_Attention)
│   ├── attention.py        # Attention mechanism implementation
│   ├── train.py            # Training loop and saving models
│   └── evaluate.py         # Model evaluation and metrics calculation
│
├── tokenizer.pkl           # Saved tokenizer for inference
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Models Overview

### 1. BiLSTM (Baseline)
A standard Bidirectional LSTM that processes the text sequence in both forward and backward directions. It takes the mean of the LSTM outputs across all timesteps to form the final context vector, which is then passed to a linear layer for classification.

### 2. BiLSTM with Attention
This model extends the baseline by adding a custom attention mechanism. Instead of taking a simple mean, it calculates attention weights for each timestep, allowing the model to focus on the most important words in a sentence before making a classification decision.

## Results

Both models perform exceptionally well, but the Attention mechanism provides a slight improvement in both Accuracy and AUC.

**Evaluation Metrics:**
- **BiLSTM Accuracy:** 0.9993
- **BiLSTM AUC:** 0.999979
- **Attention Accuracy:** 0.9996
- **Attention AUC:** 0.999982

## Requirements

To run this project, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `nltk`
- `matplotlib`
- `seaborn`

## Usage

1. **Preprocessing:** Run `src/preprocess.py` to clean the text data and prepare the vocabulary.
2. **Training:** Execute `src/train.py` to train both models. The trained weights will be saved in the `models/` directory.
3. **Evaluation:** Run `src/evaluate.py` to test the models on the hold-out set. The results will be saved in `outputs/metrics.txt`, and attention visualizations will be generated as `outputs/attention_plot.png`.

## Attention Visualization
The attention mechanism allows us to visualize which words the model focused on when making a prediction. Check `outputs/attention_plot.png` for examples of this interpretability feature.
