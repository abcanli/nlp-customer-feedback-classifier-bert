import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)
import config

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_metrics_comparison(baseline_metrics, bert_metrics, save_path):
    metrics_names = ['Validation Accuracy', 'Validation F1', 'Test Accuracy', 'Test F1']
    baseline_values = [
        baseline_metrics['val_accuracy'],
        baseline_metrics['val_f1'],
        baseline_metrics['test_accuracy'],
        baseline_metrics['test_f1']
    ]
    bert_values = [
        bert_metrics['val_accuracy'],
        bert_metrics['val_f1'],
        bert_metrics['test_accuracy'],
        bert_metrics['test_f1']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_values, width, label='TF-IDF + LogReg', alpha=0.8)
    bars2 = ax.bar(x + width/2, bert_values, width, label='DistilBERT', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison saved to {save_path}")

def evaluate_models():
    print("Loading processed data...")
    with open(config.PROCESSED_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    label_names = data['label_encoder'].classes_
    test_df = data['test']
    y_true = test_df['label_encoded'].values

    baseline_metrics_path = config.METRICS_DIR / "baseline_metrics.pkl"
    bert_metrics_path = config.METRICS_DIR / "bert_metrics.pkl"

    print("\n" + "="*60)
    print("BASELINE MODEL (TF-IDF + Logistic Regression)")
    print("="*60)

    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, 'rb') as f:
            baseline_metrics = pickle.load(f)

        print(f"Validation Accuracy: {baseline_metrics['val_accuracy']:.4f}")
        print(f"Validation F1 Score: {baseline_metrics['val_f1']:.4f}")
        print(f"Test Accuracy: {baseline_metrics['test_accuracy']:.4f}")
        print(f"Test F1 Score: {baseline_metrics['test_f1']:.4f}")

        import joblib
        vectorizer = joblib.load(config.BASELINE_MODEL_DIR / "tfidf_vectorizer.pkl")
        model = joblib.load(config.BASELINE_MODEL_DIR / "logistic_regression.pkl")

        X_test = vectorizer.transform(test_df['cleaned_text'])
        baseline_pred = model.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_true, baseline_pred, target_names=label_names))

        cm_path = config.PLOTS_DIR / "baseline_confusion_matrix.png"
        plot_confusion_matrix(
            y_true,
            baseline_pred,
            label_names,
            "Baseline Model Confusion Matrix",
            cm_path
        )
    else:
        print("Baseline metrics not found. Run train_baseline.py first.")
        baseline_metrics = None

    print("\n" + "="*60)
    print("BERT MODEL (DistilBERT)")
    print("="*60)

    if bert_metrics_path.exists():
        with open(bert_metrics_path, 'rb') as f:
            bert_metrics = pickle.load(f)

        print(f"Validation Accuracy: {bert_metrics['val_accuracy']:.4f}")
        print(f"Validation F1 Score: {bert_metrics['val_f1']:.4f}")
        print(f"Test Accuracy: {bert_metrics['test_accuracy']:.4f}")
        print(f"Test F1 Score: {bert_metrics['test_f1']:.4f}")

        bert_pred = bert_metrics['test_predictions']

        print("\nClassification Report:")
        print(classification_report(y_true, bert_pred, target_names=label_names))

        cm_path = config.PLOTS_DIR / "bert_confusion_matrix.png"
        plot_confusion_matrix(
            y_true,
            bert_pred,
            label_names,
            "BERT Model Confusion Matrix",
            cm_path
        )
    else:
        print("BERT metrics not found. Run train_bert.py first.")
        bert_metrics = None

    if baseline_metrics and bert_metrics:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        comparison_path = config.PLOTS_DIR / "model_comparison.png"
        plot_metrics_comparison(baseline_metrics, bert_metrics, comparison_path)

        improvement = (bert_metrics['test_f1'] - baseline_metrics['test_f1']) / baseline_metrics['test_f1'] * 100
        print(f"\nF1 Score Improvement: {improvement:+.2f}%")

        if bert_metrics['test_f1'] > baseline_metrics['test_f1']:
            print(f"✓ BERT outperforms baseline by {improvement:.2f}%")
        else:
            print(f"✗ Baseline outperforms BERT by {abs(improvement):.2f}%")

if __name__ == "__main__":
    evaluate_models()
    print("\nEvaluation completed successfully!")
