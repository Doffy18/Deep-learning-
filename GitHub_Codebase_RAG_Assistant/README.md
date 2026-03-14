# GitHub Codebase RAG Assistant

## Project Overview
Modern software repositories often contain thousands of lines of code distributed across many files. Understanding a new codebase can be difficult and time-consuming for developers.
This project builds an AI-powered assistant that allows users to **ask questions about a GitHub repository and receive explanations based on the repository’s code**.
The system uses **Retrieval-Augmented Generation (RAG)**. Relevant code snippets are retrieved from the repository and provided to a language model to generate accurate explanations.
The application is implemented in **Python** with a web interface built using **Streamlit** and a retrieval pipeline powered by **LangChain**.

---

# Objectives

The primary goals of this project are:

* Clone a GitHub repository automatically
* Extract Python source files from the repository
* Split code into meaningful chunks using AST parsing
* Convert code chunks into vector embeddings
* Store embeddings in a vector database
* Retrieve relevant code for user queries
* Generate explanations using a language model

---

# Technologies Used

## Programming Language

* Python

## Libraries and Tools

* Streamlit – Web interface for interacting with the system
* LangChain – Framework used to build the RAG pipeline
* FAISS – Vector database used for similarity search
* Hugging Face – Provides embedding models and hosted LLM access
* Git – Used to clone repositories
* Python **AST module** – Used to parse Python code and extract functions/classes
* Docker – Used to containerize the application

---

# System Architecture

The system follows a **Retrieval-Augmented Generation pipeline**.

```
User Query
     │
     ▼
Streamlit UI
     │
     ▼
Repository Cloning
     │
     ▼
Code Loading
     │
     ▼
AST Code Chunking
     │
     ▼
Embedding Generation
     │
     ▼
FAISS Vector Store
     │
     ▼
Retriever
     │
     ▼
Large Language Model
     │
     ▼
Generated Explanation
```

---

# Methodology

## 1. Repository Cloning

The system accepts a GitHub repository URL from the user.
The repository is cloned locally using Python's subprocess module and Git.

Example command executed internally:

```bash
git clone https://github.com/user/repo
```

This allows the system to analyze the repository locally.

---

# 2. Loading Python Files
The program recursively scans the repository directory and loads all `.py` files.
Each file is stored as a document object containing:

* File content
* File path metadata

---

# 3. Code Chunking using AST
Instead of randomly splitting text, the project uses the **Abstract Syntax Tree (AST)** to extract:
* Functions
* Classes

Each function or class becomes an independent chunk. This improves retrieval accuracy because most developer queries relate to specific functions or classes.

Example metadata:
```
file: fastapi/applications.py
name: FastAPI
type: ClassDef
```

---

# 4. Embedding Generation
Each code chunk is converted into a vector representation using the embedding model:
```
sentence-transformers/all-MiniLM-L6-v2
```
Embeddings capture the **semantic meaning** of code.

---

# 5. Vector Storage
Embeddings are stored using **FAISS**, a high-performance similarity search library.
The vector database is saved locally so embeddings do not need to be recomputed every time.
Example structure:

```
vector_store/
   repo_name/
       index.faiss
       index.pkl
```

---

# 6. Retrieval
When the user asks a question:
1. The query is converted into an embedding
2. The retriever searches the vector store
3. The top **k = 3** most relevant code chunks are retrieved

---

# 7. Response Generation
The retrieved code chunks are passed to a language model hosted through **Hugging Face**.
Model used:

```
deepseek-ai/DeepSeek-V3-0324
```

The model generates an explanation based **only on the retrieved context**, reducing hallucination.

---

# User Interface
The application interface is built using **Streamlit**.
Features include:
* Input field for GitHub repository URL
* Button to load repository
* Question input box
* Generated explanation display
* Source code references used in the answer

Example interaction:

```
Enter GitHub Repo URL:
https://github.com/tiangolo/typer

Question:
How does the Typer class work?

Answer:
Explanation of Typer class...

Sources:
typer/main.py — Typer (ClassDef)
```



---

# Docker Containerization
The project is containerized using **Docker** to ensure the application runs consistently across different systems.
The Docker container includes:
* Python environment
* Project dependencies
* Application code
* Streamlit server

This allows anyone to run the project without manually installing dependencies.
---

# Dockerfile Explanation
The Dockerfile defines how the container image is built.
```
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```
Explanation:

* `FROM python:3.12-slim` → Uses a lightweight Python base image
* `WORKDIR /app` → Sets the working directory inside the container
* `COPY requirements.txt .` → Copies dependency file
* `RUN pip install` → Installs project dependencies
* `COPY . .` → Copies project files into the container
* `EXPOSE 8501` → Opens the Streamlit port
* `CMD` → Starts the Streamlit application

---
# Running the Project with Docker
## 1. Clone the Repository
```bash
git clone <your-repository-url>
cd GitHub_Codebase_RAG_Assistant
```
---
## 2. Build the Docker Image
```bash
docker build -t codebase-rag .
```
 command builds the container image using the Dockerfile, name is your choice, but be sure to have it when you run the container.
---
## 3. Run the Docker Container
```bash
docker run -p 8501:8501 codebase-rag
```
This maps the container port to your local machine.
---
## 4. Open the Application
Open a browser and go to:
```
http://localhost:8501
```
The Streamlit application will start.
---
# Advantages of the System
* Helps developers understand unfamiliar codebases quickly
* Provides explanations grounded in real code
* Faster than manually browsing large repositories
* AST chunking improves retrieval quality
* RAG approach reduces hallucination
---

# Limitations
* Currently supports only Python repositories
* Large repositories may take longer to process initially
* Retrieval relies primarily on semantic similarity
* Limited multi-file reasoning capability
---

# Future Improvements
Possible improvements include:
* Hybrid retrieval (BM25 + embeddings)
* Multi-language repository support
* Code dependency graph analysis
* Conversational memory support
* Graph-based retrieval methods
---
# Conclusion
The **GitHub Codebase RAG Assistant** demonstrates how **Retrieval-Augmented Generation** can be applied to software engineering tasks.
By combining:
* code retrieval
* vector search
* large language models
* 
the system enables developers to interact with repositories using natural language.
This approach can significantly improve productivity when working with large or unfamiliar codebases.
---

