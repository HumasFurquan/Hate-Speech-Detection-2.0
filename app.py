import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import pickle
import os

# -----------------------------
# Page config
st.set_page_config(page_title="HateXplain Classifier", layout="centered")

# -----------------------------
# Device (FORCE CPU – Streamlit Cloud)
device = torch.device("cpu")

# -----------------------------
# Load suggestions safely
suggestions = []
if os.path.exists("all_texts.pkl"):
    with open("all_texts.pkl", "rb") as f:
        suggestions = pickle.load(f)

# -----------------------------
# Load tokenizer & model from Hugging Face (cached)
@st.cache_resource
def load_model():
    MODEL_NAME = "humasfurquan/hatexplain-bert"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

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
def predict_text(text):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        label_id = torch.argmax(logits, dim=1).item()

    return ui_label_map[label_map[label_id]], probs

# -----------------------------
# UI
st.title("🛡️ HateXplain Real-Time Classifier")
st.write("Type a sentence and get live predictions")

# Session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "last_selected" not in st.session_state:
    st.session_state.last_selected = ""

def set_suggestion(text):
    st.session_state.input_text = text
    st.session_state.last_selected = text

# Text input
input_text = st.text_input(
    "Enter text:",
    value=st.session_state.input_text,
    key="input_text"
)

# -----------------------------
# Autocomplete suggestions
if input_text.strip() and suggestions:
    filtered = [
        s for s in suggestions
        if s.lower().startswith(input_text.lower())
        and s != st.session_state.last_selected
    ]
    filtered = sorted(filtered, key=len)[:5]

    for s in filtered:
        st.button(f"→ {s}", on_click=set_suggestion, args=(s,))

# -----------------------------
# Prediction output
if input_text.strip() and len(input_text.strip()) > 2:
    label, probs = predict_text(input_text)

    st.subheader(f"Predicted Label: **{label.upper()}**")

    labels = ["neutral", "abusive", "toxic"]
    fig, ax = plt.subplots()
    ax.bar(labels, probs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

    st.write(
        " | ".join(
            [f"{int(p * 100)}% {l}" for p, l in zip(probs, labels)]
        )
    )
