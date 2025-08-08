

# ⚡ Harry Potter RAG Chatbot

A **Retrieval-Augmented Generation (RAG)**-powered chatbot that can answer questions from
**Harry Potter and the Sorcerer's Stone** and **Harry Potter and the Chamber of Secrets**.
It uses LangChain, FAISS vector storage, HuggingFace embeddings, and a quantized Mistral-7B model to bring the Hogwarts library to life.


---

## ✨ Features

* 📚 Answers only from the two specified Harry Potter books — no made-up trivia!
* 🧠 Retrieval-Augmented Generation pipeline for factual responses
* ⚡ GPU-accelerated Mistral-7B model with 4-bit quantization for efficiency
* 🌐 Streamlit web interface for an interactive experience
* 📖 Shows relevant book excerpts as sources for every answer

---

## 📂 Project Structure

```
PDF CHATBOT WITH RAG/
│
├── app.py                 # Streamlit frontend
├── RAG_pipeline.py        # Model loading, embeddings, FAISS index creation
├── faiss_index/           # Saved FAISS vector store
├── book/                  # Contains the two Harry Potter books in .pdf/.txt
├── hog.jpg                # Header image for the app
└── README.md              # Project documentation
```

---

## 🖥 Requirements

* Python 3.10+
* CUDA-enabled GPU (tested on NVIDIA RTX series)
* [PyTorch](https://pytorch.org/) with matching CUDA version
* Streamlit
* LangChain
* FAISS
* HuggingFace Transformers & Sentence-Transformers
* PyPDF2 (or pypdf)
* BitsAndBytes (for 4-bit quantization)
* Accelerate

*(Full list with versions can be generated as `requirements.txt`)*

---

## 🔮 Behind the Magic

This chatbot is more than just an LLM — it’s a mini Hogwarts library:

* **Document Processing** → Reads and chunks the books into overlapping segments.
* **Knowledge Storage** → Creates vector embeddings and stores them in FAISS.
* **Spell Casting (RAG)** → Finds the most relevant chunks for your question and feeds them to the Mistral-7B-Instruct model.
* **Truth Serum** → Uses a custom prompt to ensure the answer is grounded in the books and not hallucinated.

---

## 🧙 Fun Ideas to Try

* Play **“Stump the Sorting Hat”** — ask obscure book-only questions.
* Test **character timelines** — e.g., “Where was Hermione during the troll attack?”
* Create a **trivia quiz mode** by generating questions from the books.
* Ask the bot to **summarize a chapter** in a rhyming poem.

---

## 📜 License

This project is intended for **educational purposes only**.
The included books must be obtained legally by the user.
Harry Potter characters and books are © J.K. Rowling and Warner Bros.

---

