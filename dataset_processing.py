from transformers import DistilBertTokenizer
import torch

# Define the label mapping (indexing from 0)
LABELS = ["O", "B-DebtInstrumentInterestRateStatedPercentage", "B-DebtInstrumentBasisSpreadOnVariableRate1", 
          "B-LineOfCreditFacilityMaximumBorrowingCapacity", "B-AmortizationOfIntangibleAssets"]
label_map = {label: idx for idx, label in enumerate(LABELS)}

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_and_align_labels(example):
    # Tokenize input text
    tokenized_inputs = tokenizer(example["text"], padding="max_length", truncation=True, return_tensors="pt")
    
    # Map the labels to integers and align them with tokenized inputs
    labels = [label_map[label] for label in example["labels"]]
    
    # Ensure that the labels have the correct shape (matching token length)
    labels = labels + [label_map["O"]] * (tokenized_inputs.input_ids.size(1) - len(labels))
    
    # Add labels to tokenized inputs
    tokenized_inputs["labels"] = torch.tensor(labels)
    
    return tokenized_inputs

# Apply tokenization to your dataset
dataset = dataset.map(tokenize_and_align_labels, batched=True)
