# NLP Customer Feedback Classifier - BERT

A complete NLP project for classifying customer feedback into four categories: bug reports, feature requests, praise, and cancellation risk. This project implements both a TF-IDF baseline and a fine-tuned DistilBERT model for comparison.

## Project Structure

```
nlp-customer-feedback-classifier-bert/
├── data/
│   ├── raw/                    # Raw dataset
│   │   └── sample_raw_feedback.csv
│   └── processed/              # Preprocessed data
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis
├── src/
│   ├── config.py              # Configuration and paths
│   ├── preprocess.py          # Data preprocessing
│   ├── train_baseline.py      # TF-IDF + Logistic Regression
│   ├── train_bert.py          # DistilBERT fine-tuning
│   ├── evaluate.py            # Model evaluation and comparison
│   └── explain.py             # LIME explainability
├── models/
│   ├── baseline/              # Baseline model artifacts
│   └── bert/                  # BERT model artifacts
├── outputs/
│   ├── metrics/               # Performance metrics
│   ├── plots/                 # Visualizations
│   └── explainability/        # LIME explanations
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Dataset

The project uses a synthetic SaaS customer feedback dataset with 60 samples across 4 balanced categories:

- **bug_report**: Technical issues and errors
- **feature_request**: Requests for new functionality
- **praise**: Positive feedback and appreciation
- **cancellation_risk**: Dissatisfaction and churn signals

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone or navigate to the project directory:

```bash
cd nlp-customer-feedback-classifier-bert
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: If you have a GPU and want to use CUDA for faster training:

```bash
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Exploratory Data Analysis

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

This notebook provides:
- Label distribution analysis
- Text length statistics
- Common word analysis per category
- Sample feedback examples

### 2. Data Preprocessing

Preprocess the raw data:

```bash
python src/preprocess.py
```

This will:
- Clean and normalize text
- Split data into train/validation/test sets (70/10/20)
- Encode labels
- Save processed data to `data/processed/`

### 3. Train Baseline Model

Train the TF-IDF + Logistic Regression baseline:

```bash
python src/train_baseline.py
```

**Expected output:**
- Validation and test accuracy/F1 scores
- Classification report
- Model artifacts saved to `models/baseline/`

**Typical performance:**
- Test Accuracy: ~0.85-0.90
- Test F1 Score: ~0.85-0.90

### 4. Train BERT Model

Fine-tune DistilBERT on the feedback data:

```bash
python src/train_bert.py
```

**Training details:**
- Model: `distilbert-base-uncased`
- Max sequence length: 128 tokens
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 3

**Expected output:**
- Training loss per epoch
- Validation accuracy/F1 scores
- Best model saved to `models/bert/`

**Typical performance:**
- Test Accuracy: ~0.90-0.95
- Test F1 Score: ~0.90-0.95

**Note**: Training will automatically use GPU if available, otherwise CPU. CPU training may take 10-20 minutes.

### 5. Evaluate Models

Compare both models and generate visualizations:

```bash
python src/evaluate.py
```

**Outputs:**
- Confusion matrices for both models
- Side-by-side performance comparison chart
- Detailed classification reports
- F1 score improvement percentage

**Generated files:**
- `outputs/plots/baseline_confusion_matrix.png`
- `outputs/plots/bert_confusion_matrix.png`
- `outputs/plots/model_comparison.png`

### 6. Model Explainability

Generate LIME explanations for the baseline model:

```bash
python src/explain.py
```

**Outputs:**
- LIME explanations for sample predictions
- Feature importance analysis
- Visual explanations saved to `outputs/explainability/`

**Note**: BERT explainability is provided as a stub with recommended tools (SHAP, Captum, BertViz).

## Quick Start (Full Pipeline)

Run all steps in sequence:

```bash
python src/preprocess.py && \
python src/train_baseline.py && \
python src/train_bert.py && \
python src/evaluate.py && \
python src/explain.py
```

## Model Performance

### Baseline (TF-IDF + Logistic Regression)

**Strengths:**
- Fast training and inference
- Interpretable feature weights
- Low computational requirements
- Good performance on this task

**Typical metrics:**
- F1 Score: 0.85-0.90
- Training time: < 1 minute

### DistilBERT

**Strengths:**
- Captures contextual semantics
- Better handling of complex language
- Transfer learning from pre-trained weights
- Superior performance on edge cases

**Typical metrics:**
- F1 Score: 0.90-0.95
- Training time: 5-20 minutes (GPU/CPU)

## Configuration

Key parameters can be adjusted in `src/config.py`:

```python
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

TF_IDF_MAX_FEATURES = 5000
TF_IDF_NGRAM_RANGE = (1, 2)
```

## Inference Example

Use the trained models for prediction:

```python
import joblib
import pickle
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load baseline model
vectorizer = joblib.load('models/baseline/tfidf_vectorizer.pkl')
baseline_model = joblib.load('models/baseline/logistic_regression.pkl')

# Predict with baseline
text = "The app crashes every time I try to export data"
X = vectorizer.transform([text])
prediction = baseline_model.predict(X)[0]
print(f"Baseline prediction: {prediction}")

# Load BERT model
tokenizer = DistilBertTokenizer.from_pretrained('models/bert')
bert_model = DistilBertForSequenceClassification.from_pretrained('models/bert')

# Predict with BERT
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
with torch.no_grad():
    outputs = bert_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
print(f"BERT prediction: {prediction}")
```

## Troubleshooting

### Out of Memory (GPU)

If you encounter OOM errors during BERT training:

```python
# In src/config.py, reduce batch size
BATCH_SIZE = 8  # or even 4
```

### Slow CPU Training

BERT training on CPU can be slow. Options:

1. Use a smaller model: `bert-tiny` or `distilbert-base-uncased` (already used)
2. Reduce epochs: `NUM_EPOCHS = 2`
3. Use GPU via Google Colab or cloud services

### Import Errors

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

## Extensions

Potential improvements:

1. **Data augmentation**: Back-translation, synonym replacement
2. **Hyperparameter tuning**: Grid search for optimal parameters
3. **Ensemble methods**: Combine baseline + BERT predictions
4. **Multi-label classification**: Handle feedback with multiple categories
5. **Production deployment**: FastAPI endpoint with model serving
6. **Real-time monitoring**: Track model performance over time

## License

MIT License - Feel free to use this project for learning and development.

## Acknowledgments

- HuggingFace Transformers for pre-trained models
- scikit-learn for baseline implementation
- LIME for model interpretability
