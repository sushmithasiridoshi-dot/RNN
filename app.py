# SMS spam detection web app
# Dataset: spam.csv

import os
import re

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# configurations
MODEL = "spam_model.joblib"
EPOCHS = 20
PATIENCE = 4


# CLEANING TEXT
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.5, random_state=42, stratify=y_holdout
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=20000,
                    min_df=2,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_f1 = -1.0
    for i in range(10, 91):
        threshold = i / 100
        preds = [1 if p >= threshold else 0 for p in val_probs]
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print(f"Best decision threshold from validation: {best_threshold:.2f} (F1={best_f1:.4f})")

    joblib.dump({"model": model, "threshold": best_threshold}, MODEL)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = [1 if p >= best_threshold else 0 for p in test_probs]
    print(classification_report(y_test, test_preds))
    print(confusion_matrix(y_test, test_preds))


# PREDICT FUNCTION
@st.cache_resource
def get_inference_objects():
    if not os.path.exists(MODEL):
        train_model()

    checkpoint = joblib.load(MODEL)
    model = checkpoint["model"]
    threshold = float(checkpoint.get("threshold", 0.5))
    return model, threshold


def assess_message_quality(raw_message, cleaned_message):
    tokens = cleaned_message.split()
    alpha_num_count = sum(ch.isalnum() for ch in str(raw_message))

    if alpha_num_count < 3 or len(tokens) == 0:
        return False, "Input has mostly symbols/no useful words. Please enter a normal SMS sentence."

    if len(tokens) < 2:
        return False, "Message is too short. Enter at least 2-3 words for reliable prediction."

    return True, ""


def predict_sms(message):
    model, threshold = get_inference_objects()

    cleaned_message = clean_text(message)
    is_valid, reason = assess_message_quality(message, cleaned_message)
    if not is_valid:
        return None, None, None, threshold, reason

    probability = float(model.predict_proba([cleaned_message])[:, 1][0])

    label = "Spam" if probability >= threshold else "Ham"
    confidence = probability if label == "Spam" else (1 - probability)
    return label, probability, confidence, threshold, ""


# train once if no saved model exists
if not os.path.exists(MODEL):
    train_model()


# STREAMLIT UI
st.title("SMS Spam Detection using ML model")
st.write(
    "This is a simple web application that uses a text classification model "
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
