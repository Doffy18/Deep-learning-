# **Emotion Detector Web App**

A fun, interactive multi-label emotion detection project using Hugging Face Transformers and Streamlit!

> ⚡ *Just a small, personal project to play around with NLP and emotion detection — not a huge production-grade system, but a nice demonstration of transformer-based text classification.*

---

## **Project Overview**

This project demonstrates **multi-label emotion detection** on text using the [GoEmotions](https://huggingface.co/datasets/goemotions) dataset.
It’s powered by **DistilBERT** for efficient text encoding and includes a **Streamlit web app** to interactively test text inputs with a selectable confidence threshold.

* Trains a **multi-label classifier** using Hugging Face `Trainer`.
* Computes metrics like **micro, macro, weighted F1**, precision, and recall.
* Allows **custom thresholds at inference** to control which emotions are considered active.
* Provides a **user-friendly web interface** with Streamlit.

---

## **Model Performance**

Trained for 3 epochs on GoEmotions (simplified) dataset:

| Epoch | Training Loss | Validation Loss | Micro F1 | Macro F1 | Weighted F1 | Precision | Recall |
| ----- | ------------- | --------------- | -------- | -------- | ----------- | --------- | ------ |
| 1     | 0.095474      | 0.089055        | 0.534    | 0.311    | 0.472       | 0.721     | 0.424  |
| 2     | 0.081299      | 0.083841        | 0.563    | 0.389    | 0.527       | 0.729     | 0.459  |
| 3     | 0.071882      | 0.083861        | 0.577    | 0.409    | 0.543       | 0.709     | 0.486  |

> ✅ Metrics show **gradual improvement**, with Micro F1 reaching ~0.58 and weighted F1 ~0.54 after 3 epochs.

---

## **Demo Screenshot**

<img width="1886" height="870" alt="image" src="https://github.com/user-attachments/assets/b681442a-726b-4fee-8665-dbe6b491692d" />


## **Features**

* Multi-label emotion classification (e.g., joy, sadness, anger, surprise…)
* Adjustable confidence threshold via a slider
* Quick text input to predict emotions
* Easy-to-understand metrics and training workflow
* Lightweight and fast thanks to DistilBERT

---

## **Project Structure**

```text
.
├── app.py              # Streamlit web app
├── project.ipynb       # Jupyter notebook with training and evaluation
├── README.md           # Project overview (this file)
└── models/             # Pickle or Hugging Face saved models (not included due to size)
```

> **Note:** Model files are **not included** due to large size, but the notebook and `app.py` show how to train or load your own models.

---

## **Usage**

### **Run Streamlit App**

```bash
streamlit run app.py
```

* Type your text in the input box
* Adjust the **confidence threshold slider**
* See predicted emotions instantly

### **Train Model (optional)**

Open `project.ipynb` and run all cells to:

* Load GoEmotions dataset
* Preprocess & tokenize text
* Train multi-label classifier
* Evaluate metrics

---

## **Dependencies**

* `transformers` (Hugging Face)
* `datasets` (Hugging Face)
* `torch`
* `scikit-learn`
* `numpy`
* `streamlit`

---

## **Acknowledgements**

* [GoEmotions Dataset]([https://huggingface.co/datasets/goemotions](https://huggingface.co/datasets/google-research-datasets/go_emotions))
* Hugging Face Transformers & Trainer API
* Streamlit for interactive web apps

---

## **Notes / Fun Context**

This is a **personal, experimental project** — made just for learning and demonstrating some NLP skills.
It’s intentionally **small and approachable** compared to  other advanced projects , just to show a lightweight fun app for emotion detection.

---
