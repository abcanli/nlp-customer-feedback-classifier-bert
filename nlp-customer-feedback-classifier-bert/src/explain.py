import pickle
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
import config

def explain_baseline_predictions():
    print("Loading baseline model and data...")

    with open(config.PROCESSED_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    vectorizer = joblib.load(config.BASELINE_MODEL_DIR / "tfidf_vectorizer.pkl")
    model = joblib.load(config.BASELINE_MODEL_DIR / "logistic_regression.pkl")

    label_names = data['label_encoder'].classes_
    test_df = data['test'].reset_index(drop=True)

    def predict_proba_wrapper(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)

    explainer = LimeTextExplainer(class_names=label_names)

    print("\nGenerating LIME explanations for baseline model...")
    print("Analyzing sample predictions...\n")

    sample_indices = [0, 5, 10, 15]
    explanations_text = []

    for idx in sample_indices:
        text = test_df.iloc[idx]['cleaned_text']
        true_label = label_names[test_df.iloc[idx]['label_encoded']]

        pred = model.predict(vectorizer.transform([text]))[0]
        pred_label = label_names[pred]
        pred_proba = model.predict_proba(vectorizer.transform([text]))[0]

        print(f"Sample {idx + 1}:")
        print(f"Text: {text[:100]}...")
        print(f"True Label: {true_label}")
        print(f"Predicted: {pred_label} (confidence: {pred_proba[pred]:.3f})")

        explanation = explainer.explain_instance(
            text,
            predict_proba_wrapper,
            num_features=10,
            top_labels=1
        )

        print(f"Top contributing words:")
        for word, weight in explanation.as_list(label=pred):
            print(f"  {word}: {weight:+.3f}")
        print()

        fig = explanation.as_pyplot_figure(label=pred)
        plt.title(f"LIME Explanation - Sample {idx + 1}\nTrue: {true_label} | Pred: {pred_label}")
        plt.tight_layout()
        save_path = config.EXPLAINABILITY_DIR / f"lime_baseline_sample_{idx + 1}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        explanations_text.append({
            'sample_idx': idx,
            'text': text,
            'true_label': true_label,
            'pred_label': pred_label,
            'confidence': float(pred_proba[pred]),
            'explanation': explanation.as_list(label=pred)
        })

    explanations_path = config.EXPLAINABILITY_DIR / "baseline_explanations.pkl"
    with open(explanations_path, 'wb') as f:
        pickle.dump(explanations_text, f)

    print(f"Explanations saved to {config.EXPLAINABILITY_DIR}")

    print("\nAnalyzing feature importance...")
    feature_names = vectorizer.get_feature_names_out()

    for label_idx, label_name in enumerate(label_names):
        coefficients = model.coef_[label_idx]
        top_indices = np.argsort(coefficients)[-10:][::-1]

        print(f"\nTop 10 features for '{label_name}':")
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank}. {feature_names[idx]}: {coefficients[idx]:.4f}")

def explain_bert_predictions_stub():
    print("\n" + "="*60)
    print("BERT Model Explainability (Stub)")
    print("="*60)
    print("\nFor BERT model explanations, consider using:")
    print("  - Integrated Gradients")
    print("  - Attention visualization")
    print("  - SHAP with transformers")
    print("  - BertViz for attention patterns")
    print("\nThis is a placeholder for BERT explainability implementation.")
    print("LIME can be computationally expensive for transformer models.")
    print("\nRecommended libraries:")
    print("  - transformers-interpret")
    print("  - captum (PyTorch)")
    print("  - shap")

    stub_info = {
        'note': 'BERT explainability stub',
        'recommended_tools': [
            'transformers-interpret',
            'captum',
            'shap',
            'bertviz'
        ]
    }

    stub_path = config.EXPLAINABILITY_DIR / "bert_explainability_stub.pkl"
    with open(stub_path, 'wb') as f:
        pickle.dump(stub_info, f)

if __name__ == "__main__":
    print("="*60)
    print("MODEL EXPLAINABILITY ANALYSIS")
    print("="*60)

    explain_baseline_predictions()
    explain_bert_predictions_stub()

    print("\n" + "="*60)
    print("Explainability analysis completed!")
    print(f"Results saved to: {config.EXPLAINABILITY_DIR}")
    print("="*60)
