from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import ElectraTokenizer
import numpy as np
import onnxruntime as ort
import torch
import requests

app = FastAPI()

# Enable CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Electra tokenizer
tokenizer = ElectraTokenizer.from_pretrained("./fake_review_model")

# Load ONNX model (Electra version)
onnx_model_path = "onnx-electra/electra_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Input model for FastAPI
class ReviewInput(BaseModel):
    text: str

# Preprocess input text
def preprocess(text):
    encoded = tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=128)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }

# Run ONNX prediction
def predict(text):
    inputs = preprocess(text)
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    logits = ort_session.run(["logits"], ort_inputs)[0]
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[0]
    label = int(np.argmax(probs))
    confidence = float(probs[label]) * 100
    return label, confidence

# Get explanation using OpenRouter LLM
def get_llm_reason(text, label):
    try:
        explanation_prompt = (
            f"Explain why this review might be classified as {'fake' if label == 1 else 'real'}:\n\n\"{text}\""
        )
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",  # Replace with your key
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {"role": "user", "content": explanation_prompt}
                ]
            }
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return "⚠️ LLM explanation unavailable."
    except Exception:
        return "⚠️ LLM explanation unavailable."

# API endpoint
@app.post("/predict")
async def classify_review(input: ReviewInput):
    try:
        label, confidence = predict(input.text)
        reason = get_llm_reason(input.text, label)
        return {
            "prediction": "FAKE" if label == 1 else "REAL",
            "confidence": round(confidence, 2),
            "reason": reason
        }
    except Exception as e:
        return {"error": f"❌ Internal server error: {str(e)}"}
