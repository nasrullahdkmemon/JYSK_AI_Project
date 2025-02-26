from transformers import DistilBertTokenizer, DistilBertForTokenClassification, Trainer, TrainingArguments
import torch
from datasets import load_dataset

# ✅ Load Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# ✅ Load Dataset (Replace with Your Data)
dataset = load_dataset("json", data_files={"train": "train.json", "test": "test.json"})

# ✅ Tokenize Function
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["text"], padding="max_length", truncation=True, return_tensors="pt")
    tokenized_inputs["labels"] = example["labels"]
    return tokenized_inputs

# ✅ Apply Tokenization
dataset = dataset.map(tokenize_and_align_labels, batched=True)

# ✅ Load Model with Correct Label Count
num_labels = 5
model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# ✅ Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Try increasing this if needed
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

# ✅ Start Training
trainer.train()

# ✅ Save Fine-Tuned Model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
