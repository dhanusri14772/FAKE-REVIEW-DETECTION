
# 🛡️ Fake Review Detector with Explainable AI

This project detects **fake product reviews** in real-time using a fine-tuned **Electra Transformer model**, optimized with **ONNX Runtime** for blazing-fast inference. It integrates a **FastAPI backend** and an **explainable AI layer** powered by **OpenRouter LLMs** (like Mistral-7B) to justify predictions.  
Additionally, a **custom Chrome Extension** connects directly to the API and flags suspicious reviews directly on e-commerce websites.

---

## 🚀 Features

- 🔍 Real-time fake review classification  
- ⚡ Fast inference using ONNX  
- 🧠 Explanation support via LLM (OpenRouter)  
- 🌐 FastAPI backend for serving predictions  
- 🧩 Chrome Extension for browser-based detection  
- 🧪 Trained on custom balanced Amazon review dataset  
- ✅ High accuracy with fine-tuned Electra model  

---

## 🧰 Tech Stack

- **Model**: [Electra-base](https://huggingface.co/google/electra-base-discriminator)  
- **Serving**: FastAPI + ONNX Runtime  
- **LLM Explanation**: OpenRouter (e.g. Mistral-7B)  
- **Frontend**: Chrome Extension  
- **Dataset**: Amazon Reviews (rule-based label filtering)  

---

## ⚙️ Setup Instructions

### 1. Clone this repository


git clone https://github.com/dhanusri14772/fake-review-detector.git
cd fake-review-detector

### 2. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

## 3. Install dependencies
pip install -r requirements.txt

## 4. Start the FastAPI Backend
uvicorn app:app --reload


🌐 Chrome Extension Setup
1. Open Chrome and go to chrome://extensions/
2. Enable Developer Mode
3. Click Load unpacked
4. Select the extension/ folder from this repo
5. The extension will now detect and highlight fake reviews on product pages


🧪 API Endpoint
POST /predict

Request Body:

{
  "text": "Your review text here"
}
Response:

{
  "prediction": "FAKE",
  "confidence": 86.42,
  "reason": "This review seems overly generic and exaggerated..."
}

📄 License
This project is licensed under the MIT License, allowing anyone to use, modify, distribute, or contribute freely while disclaiming liability. See the LICENSE file for full terms.

