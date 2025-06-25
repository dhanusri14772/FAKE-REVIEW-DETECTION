import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Paths
model_path = "./fake_review_model"
onnx_output_path = Path("onnx-electra")
onnx_output_path.mkdir(exist_ok=True)
onnx_file_path = onnx_output_path / "electra_model.onnx"

print("🚀 Loading ELECTRA model and tokenizer from:", model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("✅ Model and tokenizer loaded.")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dummy_text = "This product is amazing and exceeded expectations!"
inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

# Prepare inputs for ONNX
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

print("🔄 Exporting to ONNX format...")
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    str(onnx_file_path),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=13,
    do_constant_folding=True
)

print(f"✅ Model successfully exported to {onnx_file_path}")
