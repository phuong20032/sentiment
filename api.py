from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = BertTokenizer.from_pretrained("sentiment_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()

@app.post("/predict/")
async def predict(request: TextRequest):
    # Clean and tokenize text
    cleaned_text = clean_text(request.text)
    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process output
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, prediction = torch.max(probs, dim=-1)
    
    return {
        "sentiment": "Positive" if prediction.item() == 1 else "Negative",
        "confidence": confidence.item()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)