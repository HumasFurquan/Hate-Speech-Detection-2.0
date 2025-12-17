import torch
from transformers import BertForSequenceClassification, BertTokenizer

# -----------------------------
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 3
PTH_PATH = "best_hatexplain_bert.pth"
SAVE_DIR = "hf_hatexplain_bert"

# -----------------------------
print("Loading base BERT...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

print("Loading trained weights (.pth)...")
state_dict = torch.load(PTH_PATH, map_location="cpu")
model.load_state_dict(state_dict)

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

print("Saving Hugging Face model...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("✅ Conversion complete!")
