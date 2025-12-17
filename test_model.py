import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# -----------------------------
# Step 1: Set Hugging Face cache to D: drive
os.environ["HF_HOME"] = r"D:\HuggingFaceCache"

# -----------------------------
# Step 2: Load your model and tokenizer
MODEL_NAME = "humasfurquan/hatexplain-bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# -----------------------------
# Step 3: Label mapping
labels = ["Hate", "Offensive", "Normal"]

# -----------------------------
# Step 4: Function to predict label
def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    pred_label = labels[pred_idx]
    pred_confidence = probs[0, pred_idx].item()
    return pred_label, pred_confidence

# -----------------------------
# Step 5: Test
sample_text = "I really hate this!"
label, confidence = predict(sample_text)
print(f"Text: {sample_text}")
print(f"Prediction: {label} (Confidence: {confidence:.2f})")

# You can test with more examples
examples = [
    "You are amazing!",
    "I want to hurt you!",
    "This is okay."
]

for text in examples:
    label, confidence = predict(text)
    print(f"Text: {text} -> Prediction: {label} (Confidence: {confidence:.2f})")
