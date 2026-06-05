# 🎥 YouTube Video RAG Workspace

A full-stack Retrieval-Augmented Generation (RAG) application that lets users chat with the contents of any YouTube video.

Paste a YouTube URL, extract the transcript, generate embeddings, build a vector database, and ask context-aware questions about the video through a conversational interface.

---

## Features

- Extract transcripts from YouTube videos
- Retrieve video title and thumbnail automatically
- Semantic search using vector embeddings
- Context-aware conversational chat
- Follow-up question support with chat memory
- FastAPI backend + React frontend
- Dockerized deployment

---

## Tech Stack

### Frontend
- React 18
- TypeScript
- Vite

### Backend
- FastAPI
- Uvicorn
- Pydantic
- LangChain
- FAISS
- Hugging Face Embeddings
- DeepSeek-V3

### Infrastructure
- Docker
- Docker Compose
- WSL / Linux

---

## Architecture

```text
                  ┌──────────────────────────────┐
                  │      User Browser UI         │
                  │   (React / TypeScript)       │
                  └──────────────┬───────────────┘
                                 │
                    POST /       │      POST /
                  transcript     │       query
                                 ▼
                  ┌──────────────────────────────┐
                  │    REST API Routing Layer    │
                  │          (FastAPI)           │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
 [ Video Ingestion Pipeline ]                   [ Conversational RAG Chain ]
         │                                               │
         ├─► Parse URL & Fetch Metadata                  ├─► Contextualize Query
         │   (YouTube oEmbed API)                        │   (Chat History Check)
         │                                               │
         ├─► Extract Captions Data                       ├─► Vector Similarity Search
         │   (youtube-transcript-api)                    │   (FAISS Index Match k=3)
         │                                               │
         ├─► Chunk Document Matrices                     ├─► Construct Chat Template
         │   (RecursiveCharacterTextSplitter)            │   (System Prompts + Context)
         │                                               │
         └─► Vectorize & Index Memory                    └─► Run Remote Inference
             (HuggingFace Embeddings Engine)                 (DeepSeek-V3 Endpoint)
```

---

## How It Works

1. User submits a YouTube URL.
2. Backend validates and extracts the video ID.
3. Video metadata (title + thumbnail) is fetched.
4. Transcript is extracted.
5. Transcript is split into chunks.
6. Chunks are converted into embeddings.
7. FAISS creates a searchable vector index.
8. User questions are matched against relevant transcript chunks.
9. DeepSeek generates answers grounded in retrieved context.

---

## Key Components

### Transcript Processing
- `RecursiveCharacterTextSplitter`
- Chunk Size: `500`
- Chunk Overlap: `20`

### Vector Search
- Hugging Face `all-mpnet-base-v2`
- FAISS Retriever (`k=3`)

### Conversational Memory
- LangChain contextualization chain
- Follow-up question handling
- Session-based message history

---

## Running the Project with Docker

### 1. Prepare Environment Credentials

Inside docker-compose.yml, include your Hugging Face API credential:

```text
HF_TOKEN=hf_your_actual_token_string_goes_here
```

### 2. Fire Up the Infrastructure

Open your terminal (VS Code Terminal, Linux Terminal, or WSL Bash shell) in the project root directory and execute:

```bash
docker compose up --build
```

*Note: On your initial compilation pass, Docker will assemble the required base system modules. Subsequent boots will trigger almost instantly by using pre-cached layer structures.*

### 3. Open the Dashboard

Once the startup logs stabilize, open your web browser and navigate to:

```text
http://localhost:5173
```

Your application is fully live, operational, and ready to ingest information!

---

## Docker Configuration

### Backend (`backend/Dockerfile`)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend (`frontend/Dockerfile`)

```dockerfile
FROM node:20-slim

WORKDIR /app

COPY package*.json .
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
```

### Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
    ports:
      - "8000:8000"
    environment:
      - HF_TOKEN=${HF_TOKEN}

  frontend:
    build:
      context: ./frontend
    ports:
      - "5173:5173"
    depends_on:
      - backend
```
<img width="1594" height="1027" alt="Screenshot 2026-06-05 185534" src="https://github.com/user-attachments/assets/faee7229-0562-44ae-be13-cf519d5ca021" />

---

## Future Improvements

- Persistent vector storage (ChromaDB / PGVector)
- Hybrid search (BM25 + embeddings)
- Chat export and transcript summaries
- Source citations with timestamps
- Multi-video knowledge workspace

---

## Learning Outcomes

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases (FAISS)
- LangChain Workflows
- FastAPI API Development
- React + TypeScript Frontend Development
- Dockerized Full-Stack Deployment
- Conversational AI with Memory
