import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import pickle

# -----------------------------
# Load suggestions
with open("all_texts.pkl", "rb") as f:
    all_texts = pickle.load(f)

suggestions = all_texts  # use this list for autocomplete

# -----------------------------
# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load tokenizer & model
NUM_LABELS = 3
MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)
model.load_state_dict(torch.load("best_hatexplain_bert.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Label mapping
label_map = {0: "normal", 1: "offensive", 2: "hatespeech"}
ui_label_map = {
    "normal": "neutral",
    "offensive": "abusive",
    "hatespeech": "toxic"
}

# -----------------------------
# Prediction function
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
# Streamlit UI
st.title("HateXplain Real-Time Classifier with Inline Autocomplete")
st.write("Type a sentence below and see live predictions!")

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "last_selected" not in st.session_state:
    st.session_state.last_selected = ""  # store last clicked suggestion

# Callback when suggestion is clicked
def set_suggestion(suggestion):
    st.session_state.input_text = suggestion
    st.session_state.last_selected = suggestion  # store the clicked suggestion

# Text input
input_text = st.text_input(
    "Enter your text:",
    value=st.session_state.input_text,
    key="input_text"
)

# -----------------------------
# Filter suggestions (top 5, shortest to longest, hide last selected)
if input_text.strip():
    filtered_suggestions = [
        s for s in suggestions
        if s.lower().startswith(input_text.lower())
        and s != st.session_state.last_selected  # hide last clicked suggestion
    ]
    # Sort by length (ascending)
    filtered_suggestions.sort(key=len)
    # Take top 5
    filtered_suggestions = filtered_suggestions[:5]

    # Display suggestions as buttons
    for s in filtered_suggestions:
        st.button(f"→ {s}", key=f"sugg_{s}", on_click=set_suggestion, args=(s,))

# -----------------------------
# Live prediction while typing
if input_text and len(input_text.strip()) > 2:
    ui_label, probs = predict_text(input_text, model, tokenizer, device)
    
    if probs is not None:
        st.subheader(f"Predicted Label: {ui_label}")
        labels = ["neutral", "abusive", "toxic"]
        fig, ax = plt.subplots()
        ax.bar(labels, probs, color=["green", "orange", "red"])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Confidence")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

        percentages = [f"{int(p*100)}% {l}" for p, l in zip(probs, labels)]
        st.write(" | ".join(percentages))
