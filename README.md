🧠 NLP Customer Feedback Classifier (BERT + ML Baseline)

This project builds a complete end-to-end NLP pipeline for classifying SaaS customer feedback into four actionable categories:

bug_report
feature_request
praise
cancellation_risk
It contains both:

✔ TF-IDF + Logistic Regression (baseline)
✔ Fine-tuned DistilBERT transformer model

allowing direct comparison between classical ML and modern transformer-based NLP.

The project is structured exactly like a real-world workflow used in Product Analytics, Customer Experience (CX), and NLP/ML teams.

🚀 Key Features

Synthetic SaaS feedback dataset (balanced across 4 categories)

Complete preprocessing pipeline:
text cleaning
normalization
label encoding
train/val/test split
Baseline model: TF-IDF + Logistic Regression
Transformer model: DistilBERT fine-tuning

Evaluation tools:
Accuracy, Precision, Recall, F1-score
Confusion matrices
Baseline vs BERT comparison plots

Explainability:
LIME (baseline)
SHAP/Captum recommendations (BERT)
Streamlit-based interactive demo UI
Fully reproducible, production-style project structure

Works on CPU or GPU

🧱 Project Structure

nlp-customer-feedback-classifier-bert/
├── data/
│   ├── raw/
│   │   └── sample_raw_feedback.csv
│   └── processed/
│       └── processed_feedback.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_bert.py
│   ├── evaluate.py
│   ├── explain.py
│   └── generate_synthetic_feedback.py
│
├── models/
│   ├── baseline/
│   └── bert/
│
├── outputs/
│   ├── metrics/
│   ├── plots/
│   │   └── model_comparison.png
│   └── explainability/
│
├── app.py
├── requirements.txt
└── README.md

📊 Dataset

The project uses a synthetic SaaS feedback dataset containing:
feedback_id
feedback_text
label
Labels include:
bug_report
feature_request
praise
cancellation_risk

You can replace this dataset with your own data from:
Zendesk
Intercom
Freshdesk
CRM exports
App reviews
NPS comments
Support tickets

🧪 Synthetic Dataset Generator

The original toy dataset had only 60 rows.
To create a more realistic training environment, the repo includes a synthetic dataset generator that expands it to:

→ 600 samples (150 per class)

Run:
python src/generate_synthetic_feedback.py

This will:
Backup original CSV →
data/raw/sample_raw_feedback_original_backup.csv

Generate a new balanced dataset →
data/raw/sample_raw_feedback.csv

Classes generated:

bug_report
feature_request
praise
cancellation_risk

🚀 Full Pipeline (Preprocess → Train → Evaluate → Demo)
1) Preprocess dataset
python src/preprocess.py

2) Train baseline model
python src/train_baseline.py

3) Train DistilBERT model
python src/train_bert.py

4) Evaluate both models
python src/evaluate.py


Outputs include:
Confusion matrices
Performance metrics
Class-level reports
Baseline vs BERT comparison plot

5) Run explainability
python src/explain.py

👩‍💻 Streamlit Demo App
Launch the interactive classification UI:
streamlit run app.py

Then open:
👉 http://localhost:8501

You can test feedback like:

Example Feedback	Expected Label
“The app crashes when I export data.”	bug_report
“Can you add a dark mode option?”	feature_request
“We might cancel if downtime continues.”	cancellation_risk
“Great UI and excellent performance!”	praise

🧪 Model Overview

1️⃣ Baseline — TF-IDF + Logistic Regression
Fast to train
Highly interpretable
Strong performance
Saved under models/baseline/

2️⃣ DistilBERT — Fine-Tuned Transformer
Context-aware
Handles complex expressions
Typically highest accuracy
Saved under models/bert/

📊 Model Comparison (Baseline vs BERT)
<p align="center"> <img src="https://raw.githubusercontent.com/abcanli/nlp-customer-feedback-classifier-bert/main/nlp-customer-feedback-classifier-bert/outputs/plots/model_comparison.png" width="500"> </p>

⚙️ Installation
git clone https://github.com/abcanli/nlp-customer-feedback-classifier-bert.git
cd nlp-customer-feedback-classifier-bert
python -m venv venv
venv\Scripts\activate  # on Windows
pip install -r requirements.txt

📈 Typical Performance
Model	F1 Score	Notes
TF-IDF + Logistic Regression	0.85–0.90	Strong baseline
DistilBERT	0.90–0.95	Best-performing

🧩 Future Work (Extend This Project)
Deploy BERT via FastAPI REST endpoint
Add SHAP explainability
Add human-in-the-loop feedback loop
Serve as a cloud function / Lambda
Build a full Product Analytics dashboard

👤 Author
Ali Berk Canlı
NLP/ML Analyst • Data Product Analyst
🔗 LinkedIn: https://www.linkedin.com/in/aliberkcanlı
🔗 GitHub: https://github.com/abcanli
