NLP Customer Feedback Classifier (BERT + ML Baseline)

This project builds an end-to-end NLP pipeline for classifying **SaaS customer feedback** into four actionable categories:

- `bug_report`
- `feature_request`
- `praise`
- `cancellation_risk`

It includes both a **TF-IDF + Logistic Regression** baseline and a **fine-tuned DistilBERT model**, enabling performance comparison between classical ML and modern transformer-based approaches.

The project is designed as a realistic, production-style ML workflow for Product Analytics, CX teams, and NLP/ML roles.

---

🚀 Key Features

- Synthetic SaaS feedback dataset (balanced across 4 categories)
- Preprocessing pipeline: cleaning, normalization, label encoding, dataset splits
- Baseline model: **TF-IDF + Logistic Regression**
- Transformer model: **DistilBERT fine-tuning** (HuggingFace)
- Evaluation tools:
  - Accuracy, precision, recall, F1-score
  - Confusion matrices & model comparison plots
- Explainability (LIME stub for baseline + SHAP recommendations for BERT)
- Fully structured & reproducible ML project layout
- Works on CPU or GPU

---

🧱 Project Structure

nlp-customer-feedback-classifier-bert/
├── data/

│ ├── raw/

│ │ └── sample_raw_feedback.csv

│ └── processed/

├── notebooks/

│ └── 01_eda.ipynb

├── src/

│ ├── config.py

│ ├── preprocess.py

│├── train_baseline.py

│ ├── train_bert.py

│ ├── evaluate.py

│ └── explain.py

├── models/

│ ├── baseline/

│ └── bert/

├── outputs/

│ ├── metrics/

│ ├── plots/

│ └── explainability/

├── requirements.txt

└── README.md

📊 Dataset

`data/raw/sample_raw_feedback.csv` contains synthetic SaaS feedback examples with:

- `feedback_id`
- `text`
- `label` (`bug_report`, `feature_request`, `praise`, `cancellation_risk`)

This dataset can be easily replaced with real feedback from:
- Zendesk  
- Intercom  
- Freshdesk  
- CRM exports  
- User reports  
- App reviews  

🧪 Synthetic Dataset

The original project ships with a small toy CSV (60 rows).  
To make the models more realistic and stable, this repo includes a simple **synthetic data generator** that expands the dataset to a larger, balanced corpus.

To generate a synthetic SaaS feedback dataset (600 rows, 4 balanced classes):

python src/generate_synthetic_feedback.py
This will:

Backup the original file to
data/raw/sample_raw_feedback_original_backup.csv

Create a new CSV at
data/raw/sample_raw_feedback.csv
with 150 examples per class:
bug_report
feature_request
praise
cancellation_risk

🚀 Run the Full Pipeline & Streamlit Demo

After generating (or updating) the dataset, you can run the full training pipeline:

1) Preprocess data
python src/preprocess.py

2) Train baseline (TF-IDF + Logistic Regression)
python src/train_baseline.py

3) Train DistilBERT classifier
python src/train_bert.py
Once the models are trained, launch the interactive demo app:

streamlit run app.py
Then open the URL shown in the terminal (usually http://localhost:8501) and:

Choose Baseline (TF-IDF + Logistic Regression) or DistilBERT

Type a piece of SaaS customer feedback such as:

“The app crashes when I try to export data.” → bug_report

“It would be great if you added a dark mode.” → feature_request

“We are considering cancelling because of frequent downtime.” → cancellation_risk

“I love how smooth the interface is, great job!” → praise

Click Classify to see the predicted label.

🧪 Models

1️⃣ Baseline — TF-IDF + Logistic Regression**

- Fast to train  
- Interpretable  
- Strong baseline performance (F1 ≈ 0.85–0.90)  
- Artifacts saved under `models/baseline/`

2️⃣ DistilBERT — Fine-Tuned Transformer**

- Captures contextual meaning  
- Handles complex phrasing  
- Typically higher accuracy (F1 ≈ 0.90–0.95)  
- Trained via HuggingFace Transformers  
- Saved under `models/bert/`

📊 Model Comparison

Below is the performance comparison between the **TF-IDF + Logistic Regression baseline**  
and the **fine-tuned DistilBERT model** used in this project:

<p align="center">
  <img src="https://raw.githubusercontent.com/abcanli/nlp-customer-feedback-classifier-bert/main/nlp-customer-feedback-classifier-bert/outputs/plots/model_comparison.png" width="550">
</p>
[[https://raw.githubusercontent.com/abcanli/nlp-customer-feedback-classifier-bert/main/outputs/plots/model_comparison.png](https://github.com/abcanli/nlp-customer-feedback-classifier-bert/blob/main/nlp-customer-feedback-classifier-bert/outputs/plots/model_comparison.png)](https://raw.githubusercontent.com/abcanli/nlp-customer-feedback-classifier-bert/main/nlp-customer-feedback-classifier-bert/outputs/plots/model_comparison.png
)

---

⚙️ Setup & Installation

git clone https://github.com/abcanli/nlp-customer-feedback-classifier-bert.git
cd nlp-customer-feedback-classifier-bert
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

▶️ How to Run the Pipeline

1. Preprocess Dataset
python src/preprocess.py

2. Train Baseline Model
python src/train_baseline.py

3. Train BERT Model
python src/train_bert.py

4. Evaluate Models
python src/evaluate.py
Outputs include:
Confusion matrices
Comparison plots
Classification reports

5. Explainability
python src/explain.py

📈 Example Performance (Typical)
Model	F1 Score	Notes
TF-IDF + Logistic Regression	0.85–0.90	Fast & simple
DistilBERT	0.90–0.95	Best performance

🧩 Extend This Project
Add FastAPI inference endpoint
Add SHAP explainability for BERT
Deploy model as a microservice
Build a Streamlit dashboard for predictions
Use a real SaaS feedback dataset

👤 Author
Ali Berk Canlı
NLP/ML Analyst • Data Product Analyst
LinkedIn: https://www.linkedin.com/in/aliberkcanlı
GitHub: https://github.com/abcanli
