from transformers import DistilBertTokenizer, DistilBertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Define label mapping
LABELS = ["O", "B-DebtInstrumentInterestRateStatedPercentage", 
          "B-DebtInstrumentBasisSpreadOnVariableRate1", 
          "B-LineOfCreditFacilityMaximumBorrowingCapacity", 
          "B-AmortizationOfIntangibleAssets"]
label_map = {label: idx for idx, label in enumerate(LABELS)}

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load dataset (Make sure you have the correct path)
dataset = load_dataset("json", data_files={"train": "train.json", "test": "test.json"})

# Function to preprocess and tokenize data
def tokenize_and_align_labels(example):
    # Tokenize input text and handle padding/truncation
    tokenized_inputs = tokenizer(example["text"], padding='max_length', truncation=True, max_length=128, return_tensors="pt")

    # Initialize labels as "O" by default (length of tokens)
    labels = [label_map["O"]] * len(tokenized_inputs["input_ids"][0])

    # Align the labels with the tokenized text
    for i, word in enumerate(example["labels"]):
        if word in label_map:
            labels[i] = label_map[word]  # Assign proper label index based on the word
    
    # Padding the labels to match the length of the tokenized text (in case they are shorter)
    labels = labels + [label_map["O"]] * (len(tokenized_inputs["input_ids"][0]) - len(labels))

    # Ensure the labels are flattened (not a list of lists)
    labels = labels[:len(tokenized_inputs["input_ids"][0])]  # Truncate if necessary

    # Convert the labels list to tensor (ensure correct tensor shape)
    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

    return tokenized_inputs

# Apply tokenization and alignment to the dataset
dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Load the pre-trained DistilBERT model for token classification
model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(LABELS))

# Define training arguments with padding and truncation
training_args = TrainingArguments(
    output_dir="./results",  # Save results here
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir='./logs',  # Where to store logs
)

# Initialize Trainer with model, training arguments, and dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Start training
trainer.train()
