import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

start_time = time.time()
print("\U0001F680 Starting Fake Review Detection Model Training using ELECTRA...")

# Detect device
device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"\U0001F4BB Using device: {device}")

# Load dataset
print("\U0001F4C5 Loading dataset...")
df = pd.read_csv("cleaned_data.csv", encoding='utf-8')
df = df[['Text', 'Label']].dropna()
df['Text'] = df['Text'].astype(str).str.strip()
df = df[df['Text'] != ""]

# Sample for speed
df = df.sample(n=40000, random_state=42).reset_index(drop=True)
print(f"✅ Loaded {len(df)} cleaned rows.")

# Stratified split
print("🔀 Performing stratified train-test split...")
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(df['Text'], df['Label']):
    train_texts = df['Text'].iloc[train_idx].tolist()
    train_labels = df['Label'].iloc[train_idx].tolist()
    val_texts = df['Text'].iloc[val_idx].tolist()
    val_labels = df['Label'].iloc[val_idx].tolist()

# ELECTRA model
MODEL_NAME = "google/electra-small-discriminator"
print(f"\U0001F9E0 Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("✍️ Tokenizing data...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)

print(f"\U0001F4DA Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

print("⚙️ Setting up training configuration...")
training_args = TrainingArguments(
    output_dir='./results_electra',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    logging_steps=50,
    fp16=torch.cuda.is_available()
)

print("\U0001F6E0️ Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\U0001F3CB️ Training started...")
trainer.train()
print("✅ Training complete.")

print("\U0001F4BE Saving model and tokenizer...")
model.save_pretrained("./fake_review_model")
tokenizer.save_pretrained("./fake_review_model")
print("📁 Model saved to ./fake_review_model")

end_time = time.time()
print(f"⏱️ Total runtime: {round(end_time - start_time, 2)} seconds")