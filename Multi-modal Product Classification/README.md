# 🛍️ Multimodal Product Classification Pipeline

## 📘 Project Description
This project builds a **multimodal deep learning pipeline** that classifies products into predefined categories by processing both **images** and **text descriptions** (e.g., titles, specs, or reviews).  

It uses two parallel neural networks—one for images (**ResNet** or **Vision Transformer**) and one for text (**BERT** or **DistilBERT**)—and fuses their embeddings for classification.

**Key Highlights:**
- Handle heterogeneous input (**text + image**)
- Use transformer-based embeddings
- Fuse multimodal signals effectively
- Apply deep learning for classification tasks

---

## 📂 Dataset & License
- **Dataset:** [Fashion Product Text + Images](https://www.kaggle.com/datasets/nirmalsankalana/fashion-product-text-images-dataset) by *Nirmal Sankalana*
- **License:** MIT License — Free to use, modify, and distribute with attribution.

---

## ⚙️ What We Did (Updated Approach)

### ❌ Original Plan
The initial training loop processed **all three steps in each batch**:
1. Load & preprocess raw images → **ResNet50** → image embeddings
2. Tokenize descriptions → **DistilBERT** → text embeddings
3. Concatenate embeddings → train classifier with staged unfreezing (fusion head → text encoder → full model)

**Problem:**  
This was **CPU-intensive and slow** — repeated image loading, transformations, and tokenization every epoch created a bottleneck despite the GPU being underutilized.

---

### ✅ Updated Plan: Precompute, Then Train
We switched to a **two-phase pipeline**:

**1️⃣ Precomputation Phase (One-Time)**
- Load raw images & text once
- Generate:
  - **Image embeddings** via ResNet50 (2048-dim)
  - **Text embeddings** via DistilBERT (768-dim)
- Save embeddings & label mappings as `.pt` files for train/val/test

**2️⃣ Training Phase**
- Load precomputed embeddings (no repeated I/O or tokenization)
- Concatenate image + text embeddings → train only the **classifier head**
- Encoders are frozen — no fine-tuning in this mode

---

## 📊 Trade-Offs

**Advantages**
- 🚀 **Much faster training** — GPU-bound, ~800+ samples/sec
- 🖥️ Low CPU usage — avoids repeated preprocessing
- 🔁 Consistent data — same embeddings across runs

**Disadvantages**
- ❌ No encoder fine-tuning — ResNet & DistilBERT remain frozen
- 📉 Possible accuracy drop — fine-tuning could add +2–4%
- 🛑 Limited improvement — classifier only learns from fixed embeddings

---

## 🏆 Results

| Metric              | Value   | Notes |
|---------------------|---------|-------|
| **eval_loss**       | 0.5535  | Solid predictions; room for improvement |
| **eval_accuracy**   | 83.36%  | Strong for multi-class classification |
| **eval_f1**         | 0.8100  | Balanced precision & recall |
| **eval_runtime**    | 2.81 s  | Very fast evaluation |
| **eval_samples/sec**| 1775    | Extremely high throughput |
| **eval_steps/sec**  | 221     | Fast batch processing |


---

## 🌐 Streamlit Web App
An **interactive Streamlit application** was developed to make the model accessible through a user-friendly interface:  
- Users can **upload product images** and **enter product descriptions** to get real-time category predictions.  
- The app processes inputs, retrieves **precomputed embeddings**, fuses them, and displays predictions instantly.  


---

## 🚀 Future Improvements
1. **Enable Encoder Fine-Tuning** — gradually unfreeze ResNet & DistilBERT for better adaptation.
2. **Hybrid Mode** — allow fast precomputed runs & slower fine-tuning runs.
3. **Data Augmentation** —  
   - Images: crops, flips, color jitter  
   - Text: synonym replacement, paraphrasing
4. **Advanced Fusion Methods** — attention-based fusion or cross-modal transformers.
5. **Classifier Head Upgrades** — deeper MLP, dropout, batch norm.
6. **Error Analysis** — inspect per-class misclassifications & rebalance.
7. **Hyperparameter Tuning** — optimize LR, batch size, optimizer type.
8. **Model Ensembling** — combine multiple heads/architectures for robustness.

---

## 📌 Summary
This pipeline demonstrates efficient **multimodal learning** by combining precomputed image and text embeddings.  
With **83% accuracy** and **blazing-fast evaluation speed**, it provides a strong foundation for future **fine-tuned, end-to-end multimodal models**.
