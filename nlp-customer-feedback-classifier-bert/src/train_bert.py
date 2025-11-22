import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import config

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')

    return accuracy, f1, predictions, true_labels

def train_bert():
    print("Loading processed data...")
    with open(config.PROCESSED_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    train_df = data['train']
    val_df = data['val']
    test_df = data['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print("\nLoading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    train_dataset = FeedbackDataset(
        train_df['cleaned_text'].values,
        train_df['label_encoded'].values,
        tokenizer,
        config.MAX_LENGTH
    )

    val_dataset = FeedbackDataset(
        val_df['cleaned_text'].values,
        val_df['label_encoded'].values,
        tokenizer,
        config.MAX_LENGTH
    )

    test_dataset = FeedbackDataset(
        test_df['cleaned_text'].values,
        test_df['label_encoded'].values,
        tokenizer,
        config.MAX_LENGTH
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    print("\nInitializing DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME,
        num_labels=len(config.LABELS)
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    print(f"\nTraining for {config.NUM_EPOCHS} epochs...")
    best_val_f1 = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")

        val_accuracy, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(config.BERT_MODEL_DIR)
            tokenizer.save_pretrained(config.BERT_MODEL_DIR)
            print(f"Model saved (best val F1: {best_val_f1:.4f})")

    print("\nLoading best model for test evaluation...")
    model = DistilBertForSequenceClassification.from_pretrained(config.BERT_MODEL_DIR)
    model.to(device)

    test_accuracy, test_f1, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    metrics = {
        'val_accuracy': val_accuracy,
        'val_f1': best_val_f1,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_predictions': test_preds,
        'test_labels': test_labels
    }

    metrics_path = config.METRICS_DIR / "bert_metrics.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

    print(f"\nBERT model saved to {config.BERT_MODEL_DIR}")

    return model, tokenizer, metrics

if __name__ == "__main__":
    model, tokenizer, metrics = train_bert()
    print("\nBERT training completed successfully!")
