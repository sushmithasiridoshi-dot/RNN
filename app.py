# SMS spam Detection using RNN (Many to One) model - PyTorch version
# Dataset: spam.csv

import os
import re
import pickle
from collections import Counter

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# configurations
MODEL = "spam_model.pt"
VOCAB = "vocab.pkl"

MAX_WORDS = 10000
MAX_LEN = 50
PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"
EPOCHS = 20
PATIENCE = 4
BATCH_SIZE = 64

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# CLEANING TEXT
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# BUILD VOCABULARY (replaces keras Tokenizer)
def build_vocab(texts, max_words=MAX_WORDS):
    counter = Counter()
    for t in texts:
        counter.update(t.split())

    most_common = counter.most_common(max_words - 2)  # reserve slots for PAD/OOV
    vocab = {PAD_TOKEN: 0, OOV_TOKEN: 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab


def texts_to_sequences(texts, vocab):
    sequences = []
    for t in texts:
        seq = [vocab.get(word, vocab[OOV_TOKEN]) for word in t.split()]
        sequences.append(seq)
    return sequences


def pad_sequences(sequences, maxlen=MAX_LEN):
    padded = torch.zeros((len(sequences), maxlen), dtype=torch.long)
    for i, seq in enumerate(sequences):
        seq = seq[:maxlen]  # truncate
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded


# DATASET
class SMSDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# MODEL
class SpamRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.dense = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(64, 1)
        # NOTE: no sigmoid here anymore - we output raw logits.
        # BCEWithLogitsLoss (used during training) applies sigmoid internally,
        # and predict_sms() applies torch.sigmoid() manually at inference time.

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.relu(self.dense(hidden))
        out = self.dropout(out)
        out = self.output(out)
        return out.squeeze(1)


# TRAIN MODEL
def train_model():
    print("Training Dataset...")
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    print(df.head())
    print(df["label"].value_counts())

    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["message"] = df["message"].apply(clean_text)

    vocab = build_vocab(df["message"])
    sequences = texts_to_sequences(df["message"], vocab)
    x = pad_sequences(sequences, maxlen=MAX_LEN)
    y = df["label"]

    print("Shape of x:", x.shape)
    print("Shape of y:", y.shape)

    with open(VOCAB, "wb") as f:
        pickle.dump(vocab, f)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.5, random_state=42, stratify=y_holdout
    )

    train_ds = SMSDataset(X_train, y_train)
    val_ds = SMSDataset(X_val, y_val)
    test_ds = SMSDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = SpamRNN(vocab_size=len(vocab)).to(device)

    # handle class imbalance: spam is the minority class (~13% of data),
    # so weight positive (spam) examples higher in the loss
    n_ham = (y_train == 0).sum()
    n_spam = (y_train == 1).sum()
    pos_weight = torch.tensor([n_ham / n_spam], dtype=torch.float32).to(device)
    print(f"Class balance -> ham: {n_ham}, spam: {n_spam}, pos_weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(model)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs, val_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().tolist()
            val_probs.extend(probs)
            val_labels.extend(yb.int().tolist())

    best_threshold = 0.5
    best_f1 = -1.0
    for i in range(10, 91):
        threshold = i / 100
        preds = [1 if p >= threshold else 0 for p in val_probs]
        score = f1_score(val_labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print(f"Best decision threshold from validation: {best_threshold:.2f} (F1={best_f1:.4f})")

    # save model + vocab size for reconstruction
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": len(vocab),
            "threshold": best_threshold,
        },
        MODEL,
    )

    # evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb).cpu()
            preds = torch.sigmoid(logits)
            all_preds.extend((preds > best_threshold).int().tolist())
            all_labels.extend(yb.int().tolist())

    print(classification_report(all_labels, all_preds))
    print(confusion_matrix(all_labels, all_preds))


# PREDICT FUNCTION
@st.cache_resource
def get_inference_objects():
    if not os.path.exists(MODEL) or not os.path.exists(VOCAB):
        train_model()

    with open(VOCAB, "rb") as f:
        vocab = pickle.load(f)

    checkpoint = torch.load(MODEL, map_location=device)
    model = SpamRNN(vocab_size=checkpoint["vocab_size"]).to(device)

    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        print("Checkpoint incompatible with current model architecture. Retraining model.")
        train_model()
        with open(VOCAB, "rb") as rf:
            vocab = pickle.load(rf)
        checkpoint = torch.load(MODEL, map_location=device)
        model = SpamRNN(vocab_size=checkpoint["vocab_size"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5))
    return model, vocab, threshold


def assess_message_quality(raw_message, cleaned_message, vocab):
    tokens = cleaned_message.split()
    alpha_num_count = sum(ch.isalnum() for ch in str(raw_message))

    if alpha_num_count < 3 or len(tokens) == 0:
        return False, "Input has mostly symbols/no useful words. Please enter a normal SMS sentence."

    if len(tokens) < 2:
        return False, "Message is too short. Enter at least 2-3 words for reliable prediction."

    known_tokens = sum(1 for tok in tokens if tok in vocab)
    coverage = known_tokens / len(tokens)
    if len(tokens) >= 4 and coverage < 0.15:
        return (
            False,
            "Too many unknown words compared to training data. Please rephrase the message.",
        )

    return True, ""


def predict_sms(message):
    model, vocab, threshold = get_inference_objects()

    cleaned_message = clean_text(message)
    is_valid, reason = assess_message_quality(message, cleaned_message, vocab)
    if not is_valid:
        return None, None, None, threshold, reason

    sequence = texts_to_sequences([cleaned_message], vocab)
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN).to(device)

    with torch.no_grad():
        logit = model(padded_sequence)
        probability = torch.sigmoid(logit).item()

    label = "Spam" if probability >= threshold else "Ham"
    confidence = probability if label == "Spam" else (1 - probability)
    return label, probability, confidence, threshold, ""


# train once if no saved model/vocab exists
if not os.path.exists(MODEL) or not os.path.exists(VOCAB):
    train_model()


# STREAMLIT UI
st.title("SMS Spam Detection using RNN (Many to One) model")
st.write(
    "This is a simple web application that uses a Recurrent Neural Network (RNN) "
    "model to detect whether an SMS message is spam or not."
)
message = st.text_area("Enter your SMS message here:")

if st.button("Retrain Model"):
    with st.spinner("Retraining model. This may take around 1-2 minutes..."):
        train_model()
        st.cache_resource.clear()
    st.success("Model retrained with updated settings.")

if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction, _, confidence, _, quality_msg = predict_sms(message)
        if prediction is None:
            st.warning(quality_msg)
            st.caption("No prediction shown because the input quality is too low.")
        else:
            result = "Spam" if prediction == "Spam" else "Not Spam"
            st.success(f"Result: {result}")
            st.write("Confidence:", f"{round(confidence * 100, 2)}%")