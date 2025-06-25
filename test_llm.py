import requests

API_KEY = "OPENAI KEY"  # Replace this with your real key

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "mistralai/mistral-7b-instruct",
    "messages": [
        {"role": "system", "content": "You are an AI that explains why a review is fake or real."},
        {"role": "user", "content": "Why is the review 'Excellent product, will buy again!' classified as fake?"}
    ]
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

print("🔍 Status Code:", response.status_code)
print("🧠 LLM Response:", response.text)

