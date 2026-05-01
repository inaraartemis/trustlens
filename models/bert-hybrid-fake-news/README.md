# BERT-Hybrid Fake News Detector

This project implements a hybrid machine learning model to detect fake news. It combines the contextual embeddings from a pre-trained BERT model with traditional linguistic features (TF-IDF and Part-of-Speech ratios) to improve classification accuracy.

## Project Structure

- `data/`: Contains the datasets (`Fake.csv`, `True.csv`) and the processed dataset (`combined.csv`).
- `models/`: Directory where the trained models are saved (`hybrid_model.pth`).
- `outputs/`: Stores output files such as misclassified examples (`errors.txt`).
- `src/`: Contains the source code for the project.
  - `preprocess.py`: Merges the raw datasets (`Fake.csv` and `True.csv`) into a single processed dataset.
  - `features.py`: Extracts TF-IDF and POS (Part-of-Speech) features from the text.
  - `bert_model.py`: Defines a standalone BERT classification model.
  - `hybrid_model.py`: Defines the core Hybrid Model combining BERT outputs with the extracted features.
  - `train.py`: Script to build and save the model architecture.
  - `evaluate.py`: Evaluates the saved model on the test dataset and calculates the F1 score.
  - `error_analysis.py`: Runs the model and logs misclassified examples to `outputs/errors.txt` for further review.
- `requirements.txt`: Python dependencies required to run the project.

## Installation

1. Clone or download this repository.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**1. Prepare Data**
Place your `Fake.csv` and `True.csv` datasets into the `data/` directory. Run the preprocessing script to combine and shuffle them:
```bash
python src/preprocess.py
```

**2. Train the Model**
Run the training script to initialize the model and save its state to the `models/` directory:
```bash
python src/train.py
```

**3. Evaluate the Model**
Evaluate the model's F1 score on the test set:
```bash
python src/evaluate.py
```

**4. Error Analysis**
Generate an analysis of misclassified samples. The outputs will be saved to `outputs/errors.txt`:
```bash
python src/error_analysis.py
```

## Model Architecture

The `HybridModel` takes advantage of both deep contextual representations and traditional NLP features:
1. **BERT Embeddings**: Uses `bert-base-uncased` to process the text and extract contextual representations (Pooler Output).
2. **Feature Extraction**: Calculates a TF-IDF vector (up to 5000 features) and a Part-of-Speech ratio (Adjectives to Nouns) for each text sample.
3. **Classification**: Concatenates the 768-dimensional BERT output and the extracted custom features, passing them through a Linear layer with a Sigmoid activation to produce the final probability score (1 = Fake, 0 = True).
