# 📊 Earnings Call Risk Modeling using Transformer Regression

## 🚀 Overview

This project builds a transformer-based regression pipeline to estimate **linguistic risk density** from S&P 500 earnings call transcripts.
Using structured transcript data and a custom labeling mechanism, a DistilBERT-based model was trained to predict a continuous uncertainty score derived from language patterns.
Final validation performance:

* **R²:** 0.9682
* **MSE:** 0.0002
* **MAE:** 0.0108

---

# 📂 Dataset

**Source:** `kurry/sp500_earnings_transcripts` (HuggingFace Datasets: https://huggingface.co/datasets/kurry/sp500_earnings_transcripts)

The dataset contains structured earnings call transcripts with speaker segmentation and metadata.

### Preprocessing Steps

* Removed operator speech
* Removed forward-looking disclaimers
* Removed bracketed annotations
* Normalized whitespace
* Filtered transcripts shorter than 1000 characters
* Kept management discussion only

This ensures modeling focuses on meaningful executive commentary.

---

# 🏷 Label Construction

Because no external financial outcome variables were used, a linguistic heuristic was constructed.

### Uncertainty Word Set

```
may, might, could, uncertain, risk,
challenging, volatility, pressure,
decline, weakness, headwind,
concern, slowdown, impact
```

### Risk Formula (Per Chunk)

risk = min((uncertainty_word_count / total_words) × 20, 1.0)
This score represents **uncertainty density**, not actual financial risk.

---

# 🔎 Key Engineering Insight: Chunk-Level Supervision

Earnings transcripts exceed transformer token limits (512 tokens), requiring chunking.
Initially, transcript-level risk scores were assigned to all chunks.
This caused supervision misalignment and very low R² (~0.07).
After computing risk **per chunk**, the model performance increased to:

**R² = 0.9682**

This demonstrates a critical NLP principle:
> Label alignment matters more than model complexity.

---

# 🧠 Model Architecture

Base model: `distilbert-base-uncased`

Architecture:

* CLS token embedding
* Dropout (0.3)
* Linear regression head
* SmoothL1Loss
* AdamW optimizer
* Linear warmup scheduler
* Mixed precision training (AMP)

DistilBERT was selected for:

* Computational efficiency
* Sufficient capacity for linguistic density modeling
* Faster experimentation
---

# ⚙️ Training Setup

* 50,000 chunk samples
* 90% train / 10% validation split
* Batch size: 16
* Encoder LR: 1e-5
* Regressor LR: 3e-5
* Gradient clipping
* Mixed precision enabled

---

# 📈 Results

Epoch 2:
* R²: 0.9682
* MSE: 0.0002
* MAE: 0.0108

The model successfully learned to approximate the uncertainty scoring function.
---

# 🔬 What This Model Represents

This system does **not** predict stock movement or market risk.

It learns:
Text → Linguistic Uncertainty Density
The high R² reflects strong function approximation of the constructed heuristic target.

This project demonstrates:

* Long-document transformer handling
* Proper chunk supervision
* Regression head fine-tuning
* Training stabilization techniques
* Debugging label noise

---

# 🏦 How Real-World Financial NLP Differs

In industry and research settings, financial risk modeling typically involves:

1. Aligning transcripts with stock tickers and event dates
2. Merging with historical price data
3. Computing post-earnings metrics such as:

   * Abnormal returns
   * Volatility shifts
   * Earnings surprise reactions
4. Training models to predict future market behavior

That transforms the problem into:

Text → Future Financial Outcome

This project intentionally focuses only on the linguistic modeling layer, without incorporating external market data.

---

# 🧩 Key Lessons

* Label design dominates model performance.
* Chunk-level supervision is critical for long documents.
* High R² must be interpreted relative to label construction.
* Synthetic targets are useful for validating NLP pipelines.
* Debugging supervision alignment is often the hardest step.

---

# 🏁 Conclusion

This project builds and validates a transformer-based regression system for long-form financial text.

While the target variable is linguistically constructed rather than market-derived, the pipeline demonstrates:

* Proper preprocessing
* Supervision correction
* Efficient fine-tuning
* Strong regression performance

It serves as a solid demonstration of applied financial NLP engineering.
