import pickle
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import config

def train_baseline():
    print("Loading processed data...")
    with open(config.PROCESSED_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    train_df = data['train']
    val_df = data['val']
    test_df = data['test']

    print("\nTraining TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=config.TF_IDF_MAX_FEATURES,
        ngram_range=config.TF_IDF_NGRAM_RANGE,
        min_df=2,
        max_df=0.8
    )

    X_train = vectorizer.fit_transform(train_df['cleaned_text'])
    X_val = vectorizer.transform(val_df['cleaned_text'])
    X_test = vectorizer.transform(test_df['cleaned_text'])

    y_train = train_df['label_encoded'].values
    y_val = val_df['label_encoded'].values
    y_test = test_df['label_encoded'].values

    print(f"TF-IDF feature shape: {X_train.shape}")

    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_SEED,
        class_weight='balanced',
        C=1.0
    )

    model.fit(X_train, y_train)

    print("\nEvaluating on validation set...")
    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1 Score (weighted): {val_f1:.4f}")

    print("\nEvaluating on test set...")
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score (weighted): {test_f1:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(
        y_test,
        test_pred,
        target_names=data['label_encoder'].classes_
    ))

    vectorizer_path = config.BASELINE_MODEL_DIR / "tfidf_vectorizer.pkl"
    model_path = config.BASELINE_MODEL_DIR / "logistic_regression.pkl"

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)

    print(f"\nBaseline model saved to {config.BASELINE_MODEL_DIR}")

    metrics = {
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    }

    metrics_path = config.METRICS_DIR / "baseline_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

    return model, vectorizer, metrics

if __name__ == "__main__":
    model, vectorizer, metrics = train_baseline()
    print("\nBaseline training completed successfully!")
