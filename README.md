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

---

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

---

⚙️ Setup & Installation

```bash
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

### 🔍 Visual Comparison

![Model comparison](outputs/plots/model_comparison.png)
