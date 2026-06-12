# 📧 Basic Email Agent (V1)

A basic, lightweight human-in-the-loop AI orchestration workspace that turns informal thoughts into professional corporate communications.

Instead of directly dispatching outputs, the system forces a state interruption—routing draft generation through a structured approval loop where users can review, provide text feedback, or instantly authorize a background SMTP relay using a shared system assistant account.

---

## 💡 The Problem It Solves

When people write professional emails, they waste significant time structuring formatting, refining tone corporate mechanics, or jumping back and forth between draft workspaces and actual email clients.

Conversely, completely hands-off AI email scripts are prone to hallucinations, incorrect formatting, or inappropriate context delivery without a baseline human safeguard.

**Basic Email Agent** fixes this asymmetry by balancing autonomous acceleration with strict security safeguards:

* **The Messy Input Problem:** Users can dump fragmented typing constraints (e.g., *"tell sarah im sick, check slack later"*). The agent dynamically isolates variables, drafts prose, and manages contextual requirements automatically.
* **The Ghost In The Machine Problem:** The graph code physically locks down execution parameters. An automated draft **cannot** reach the public internet until it successfully triggers a state check constraint and collects explicit validation text feedback or authorization patterns from the user layout surface.

---

## 🛠️ Tech Stack

### Frontend

* **React 18** (Functional State Hooks Layout)
* **Vite** (Build Layer Optimization)
* **Tailwind CSS** (Unified Scannable Component Structures)

### Backend

* **FastAPI** (Asynchronous Micro-routing Engine)
* **LangGraph** (Stateful Multi-Node Cyclical Flow Orchestration)
* **LangChain & LiteLLM** (Provider Agnostic Infrastructure Gateways)
* **Smtplib Engine** (Secure TLS Email Construction & Delivery)

### Infrastructure

* **Docker & Docker Compose** (Container Core)

---

## 📐 Architecture & System Flow

```text
                 ┌──────────────────────────────┐
                  │      User Browser UI         │
                  │       (Plain React)          │
                  └──────────────┬───────────────┘
                                 │
                    POST /       │       POST /
                 agent/start     │    agent/resume
                                 ▼
                  ┌──────────────────────────────┐
                  │    FastAPI Gateway Router    │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                     [ Stateful LangGraph Engine ]
                                 │
                      ┌──────────┴──────────┐
                      ▼                     ▼
               ( llm_tool Node ) ──► ( router_for_tools )
                      ▲                     │
                      │ ◄── [ Feedback ]    ├─► [ human_clarification ]
                      │      (Resume)       │   (State Paused via interrupt)
                      │                     │
                      └─────────────────────┼─► [ send_email_tool ]
                                            │   (SMTP Outbound Delivery)
                                            ▼
                                         ( END )
```

### Dynamic Execution Flow

1. **Initiate Generation:** The frontend shoots user constraints to `/agent/start`.
2. **LLM Gateway Abstraction & Routing:** The agent leverages `ChatLiteLLM` to trigger structural models. It targets **Groq (Llama 3.3 70B)** with an implicit network fallback to **Gemini 2.5 Flash** to maintain runtime availability.
3. **The State Interruption Hook:** The LLM *must* step through the `human_clarification` tool path first. This triggers a core LangGraph `interrupt()`, snapshots memory using `MemorySaver()`, stops engine processing, and pipes the string draft directly back to the screen interface.
4. **Human Evaluation Verification:** The UI locks until the operator interacts.
* If the user types modifications, the app passes feedback to `/agent/resume`, returning execution to the `llm_tool` node to rewrite the draft.
* If the user authorizes approval (*"ok", "looks good", "send"*), the agent bypasses the review node, shifts context execution tracking, and triggers the secure `send_email_tool` utility.



---

## 📦 Running the Project with Docker

Follow these step-by-step instructions to boot the complete environment up via containers.

### 1. Configure the Orchestration Credentials

Open up your `docker-compose.yml` file located inside the project root workspace directory. Inject your specific model tokens, system service address, and Gmail **App Password** directly into the `backend.environment` keys:

### 2. Boot the Containers

Launch your system terminal window in the repository root directory and process compilation layers cleanly:

```bash
docker compose up --build

```

### 3. Access Your Local Dashboard

Once your execution log outputs establish connection streams, initialize the workspace loop inside your browser structure:

* **Frontend UI Application:** `http://localhost:5173`
* **Interactive OpenAPI Backend Documentation:** `http://localhost:8000/docs`

---

## 🎓 Core Learning Outcomes

* **Decoupled LLM Gateway Architectures:** Implementing structural fallback bindings using `ChatLiteLLM` to maintain high application uptime across different API provider networks.
* **Stateful Interrupt Loops:** Pausing complex agentic sequences midpoint using stateful state engine checkpointers (`MemorySaver`).
* **Tool-Bound Automation:** Safeguarding low-level system communication processes (SMTP text framing) behind conditional agent router nodes.
