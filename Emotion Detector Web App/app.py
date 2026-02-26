# app.py
import streamlit as st
import pickle
import torch
from transformers import pipeline

# Load pickled model, tokenizer, label_map
with open("emotion_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
tokenizer = data["tokenizer"]
label_map = data["label_map"]

# Setup pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None
)

# Streamlit UI
st.title("Multi-Label Emotion Classifier ")

text_input = st.text_area("Enter text here:")

threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

if st.button("Predict Emotions"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        outputs = classifier(text_input)[0]
        preds = [(label_map[o['label']], o['score']) for o in outputs if o['score'] > threshold]

        if preds:
            st.success("Predicted Emotions:")
            for label, score in preds:
                st.write(f"{label}: {score:.2f}")
        else:
            st.info("No emotions passed the threshold.")