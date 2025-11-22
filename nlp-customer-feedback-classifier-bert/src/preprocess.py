import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import config

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess():
    print(f"Loading data from {config.RAW_DATA_PATH}")
    df = pd.read_csv(config.RAW_DATA_PATH)

    print(f"Original dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    df['cleaned_text'] = df['feedback_text'].apply(clean_text)

    df = df.dropna(subset=['cleaned_text', 'label'])
    df = df[df['cleaned_text'].str.len() > 10]

    print(f"After cleaning shape: {df.shape}")

    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    train_val, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df['label_encoded']
    )

    train, val = train_test_split(
        train_val,
        test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_SEED,
        stratify=train_val['label_encoded']
    )

    print(f"\nTrain set: {len(train)} samples")
    print(f"Validation set: {len(val)} samples")
    print(f"Test set: {len(test)} samples")

    data_dict = {
        'train': train,
        'val': val,
        'test': test,
        'label_encoder': label_encoder
    }

    with open(config.PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"\nProcessed data saved to {config.PROCESSED_DATA_PATH}")

    return data_dict

if __name__ == "__main__":
    data = load_and_preprocess()
    print("\nPreprocessing completed successfully!")
    print(f"\nLabel mapping: {dict(enumerate(data['label_encoder'].classes_))}")
