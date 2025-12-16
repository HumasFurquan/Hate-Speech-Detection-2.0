import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import pickle

# Load suggestions
with open("all_texts.pkl", "rb") as f:
    all_texts = pickle.load(f)

suggestions = all_texts  # use this list for autocomplete

# -----------------------------
# 1️⃣ Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2️⃣ Load tokenizer & model
NUM_LABELS = 3
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_LABELS
)
model.load_state_dict(torch.load("best_hatexplain_bert.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 3️⃣ Label mapping
label_map = {0: "normal", 1: "offensive", 2: "hatespeech"}
ui_label_map = {
    "normal": "neutral",
    "offensive": "abusive",
    "hatespeech": "toxic"
}

# -----------------------------
# 4️⃣ Prediction function
def predict_text(text, model, tokenizer, device, max_len=128):
    if not text.strip():
        return None, None
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        label_id = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        label = label_map[label_id]
        ui_label = ui_label_map[label]
    return ui_label, probs

# -----------------------------
# 5️⃣ Load your 23,810 suggestions
suggestions = all_texts  # replace with your list of suggestions

# -----------------------------
# 6️⃣ Streamlit UI
st.title("HateXplain Real-Time Classifier with Autocomplete")
st.write("Type a sentence below and see real-time predictions with confidence.")

input_text = st.text_input("Enter your text:")

# Filter suggestions while typing
filtered_suggestions = [s for s in suggestions if s.lower().startswith(input_text.lower()) and input_text.strip()]
filtered_suggestions = filtered_suggestions[:10]

if filtered_suggestions:
    selected = st.selectbox("Suggestions:", ["-- Select --"] + filtered_suggestions)
    if selected != "-- Select --":
        input_text = selected
        st.session_state.input = input_text

# Prediction and confidence chart
ui_label, probs = predict_text(input_text, model, tokenizer, device)
if probs is not None:
    st.subheader(f"Predicted Label: {ui_label}")
    
    labels = ["neutral", "abusive", "toxic"]
    fig, ax = plt.subplots()
    ax.bar(labels, probs, color=["green","orange","red"])
    ax.set_ylim([0,1])
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

    percentages = [f"{int(p*100)}% {l}" for p, l in zip(probs, labels)]
    st.write(" | ".join(percentages))
