from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForTokenClassification

# ✅ Correct Financial Entity Labels (Ensure These Match Your Training Labels)
LABELS = [
    "O", 
    "B-DebtInstrumentInterestRateStatedPercentage", 
    "B-DebtInstrumentBasisSpreadOnVariableRate1", 
    "B-LineOfCreditFacilityMaximumBorrowingCapacity", 
    "B-AmortizationOfIntangibleAssets"
]

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Enable CORS (For Frontend or External Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["http://localhost:3000"] if using frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load Tokenizer (Ensure it Matches Your Training Tokenizer)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")  # Change if different

# ✅ Load Your Fine-Tuned Model
model_path = "D:/jysk_ai_project/models/final_model.pt"
num_labels = len(LABELS)  # Ensure this matches your trained model

try:
    model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading PyTorch model: {str(e)}")

# ✅ Define Input Format for API
class InputData(BaseModel):
    text: str

# ✅ Preprocessing: Tokenize Input Text (Ensure It's the Same as Training)
def preprocess_text(text):
    try:
        encoded = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoded["input_ids"].to(torch.device("cpu")), encoded["attention_mask"].to(torch.device("cpu")), tokenizer.tokenize(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization Error: {str(e)}")

# ✅ Post-processing: Convert Model Output to Financial Labels
def postprocess_output(logits, tokens):
    try:
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()  # Convert logits to label indices
        predicted_labels = [LABELS[idx] for idx in predictions]

        # ✅ Merge Subword Tokens into Full Financial Entities
        entities = []
        current_entity = ""
        current_label = None

        for token, label in zip(tokens, predicted_labels):
            if label.startswith("B-"):  # New entity starts
                if current_entity:
                    entities.append({"entity": current_entity, "label": current_label})
                current_entity = token
                current_label = label[2:]  # Remove "B-" prefix
            elif label == "O":  # Outside entity
                if current_entity:
                    entities.append({"entity": current_entity, "label": current_label})
                    current_entity = ""
                    current_label = None
            else:  # Inside entity
                if current_entity:
                    current_entity += " " + token

        # ✅ Add last entity if not empty
        if current_entity:
            entities.append({"entity": current_entity, "label": current_label})

        return {"financial_entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Postprocessing Error: {str(e)}")

# ✅ Home Endpoint
@app.get("/")
def home():
    return {"message": "Financial NER API is running!"}

# ✅ Prediction Endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        input_ids, attention_mask, tokens = preprocess_text(data.text)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        return postprocess_output(outputs.logits, tokens)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# ✅ Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
