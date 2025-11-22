import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

RAW_DATA_PATH = DATA_RAW_DIR / "sample_raw_feedback.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "processed_feedback.pkl"

BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
BERT_MODEL_DIR = MODELS_DIR / "bert"

METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
EXPLAINABILITY_DIR = OUTPUTS_DIR / "explainability"

RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

LABELS = ["bug_report", "feature_request", "praise", "cancellation_risk"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

TF_IDF_MAX_FEATURES = 5000
TF_IDF_NGRAM_RANGE = (1, 2)

for dir_path in [DATA_PROCESSED_DIR, BASELINE_MODEL_DIR, BERT_MODEL_DIR,
                 METRICS_DIR, PLOTS_DIR, EXPLAINABILITY_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
