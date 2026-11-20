# 🩺 MediRAG: Medical Retrieval-Augmented Generation (RAG) System

[![CI/CD Pipeline](https://github.com/Maanit5117/Medical-Chatbot/actions/workflows/cicd.yaml/badge.svg)](.github/workflows/cicd.yaml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-cyan.svg)](Dockerfile)
[![Vector Database](https://img.shields.io/badge/Pinecone-Serverless%20(AWS%20us--east--1)-green.svg)](https://www.pinecone.io/)
[![LLM](https://img.shields.io/badge/LLM-Google%20Gemini%20Flash-orange.svg)](https://ai.google.dev/)

![MediRAG UI](docs/screenshot.png)

A production-style Medical Retrieval-Augmented Generation (RAG) system with automated CI/CD on Amazon Web Services (AWS). Grounded in *The Gale Encyclopedia of Medicine*, it couples a containerized Python backend with Pinecone Serverless vector search, local host-based Ollama embeddings, and Google Gemini Flash generation.

---

## 🏛 System Architecture

The architecture separates offline corpus indexing from the real-time query serving plane:

```
[ Ingestion ]
Medical Book (PDF) ──> PyPDFLoader ──> Recursive Splitter (1000/200)
    ──> Ollama (qwen3-embedding:0.6b, CPU) ──> Pinecone Serverless (AWS us-east-1)

[ Online Inference ]
User Query ──> Backend (/api/chat)
    ├──> Host Ollama (localhost:11434) ──[ Query Vector ]
    ├──> Pinecone Index (us-east-1 via Internet) ──[ Top-4 Chunks ]
    └──> Google Gemini Flash + Prompt Guardrails ──> JSON Response (Answer + Excerpts)
```

### Ingestion & Vector Indexing (`store_index.py`)
- **Corpus**: *The Gale Encyclopedia of Medicine* (`data/Medical_book.pdf`).
- **Chunking**: `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`) preserves semantic clinical context across split boundaries.
- **Embeddings**: `OllamaEmbeddings(model="qwen3-embedding:0.6b")` running locally in CPU-only mode; vector dimensions are resolved dynamically at runtime.
- **Vector Store**: Pinecone Serverless index (`medical-chatbot`) in `us-east-1` with `cosine` metric, populated via batched upserts (batch size: 20) with exponential backoff and resume support (`--start <N>`).

### Online Retrieval & LLM Generation (`src/helper.py`, `src/prompt.py`)
- **Query Embedding**: Each query is embedded on-the-fly via Ollama on the EC2 host.
- **Retrieval**: Pinecone returns the **Top-4 (`k=4`)** most semantically relevant text chunks over the public internet (cross-region: `ap-southeast-2` to `us-east-1`).
- **Prompt Guardrails**: A strict closed-domain system prompt mandates answering *only* from the retrieved context, instructs the model to admit when information is missing, and forbids personal diagnosis or prescription.
- **Inference**: Invoked via LangChain Expression Language (LCEL) using `gemini-3.6-flash` (`thinking_level="low"`, `max_output_tokens=4096`).

---

## ⚡ Backend Architecture (`app.py`)

The backend is built for resilience, providing an adaptive multi-tier server pattern:
1. **Flask (Primary)**: Production REST API handling `/api/chat`, `/api/health`, and web UI routing.
2. **FastAPI / Uvicorn (Alternative)**: Asynchronous non-blocking runtime with Pydantic schemas.
3. **Built-in `ThreadingHTTPServer` (Fallback)**: Zero-dependency standard library fallback ensuring the service can boot and answer queries in minimal Python environments.

- **Lazy Initialization**: Pinecone and Gemini clients load on first request, preventing startup crashes during boot.
- **Lightweight Health Check**: `/api/health` audits Pinecone index availability without burning LLM tokens.
- **Quota Resilience**: Intercepts Google Gemini HTTP 429 / `RESOURCE_EXHAUSTED` errors and maps them to clean user-facing responses.

### Key API Endpoints
- `POST /api/chat`: Accepts `{"message": "What is asthma?"}` and returns:
  ```json
  {
    "answer": "Asthma is a chronic inflammatory disorder of the airways...",
    "sources": [
      { "source": "data/Medical_book.pdf", "content": "Asthma is characterized by..." }
    ]
  }
  ```
  *(Note: `sources` contains text excerpts truncated to the first 400 characters).*
- `GET /api/health`: Returns vector store connectivity status (`200 OK` or `503 Unavailable`).
- `GET /get?msg=<query>`: Legacy GET/POST endpoint for backward compatibility.

---

## ☁️ AWS Cloud Infrastructure & CI/CD

```
[ Git Push main ] ──> GitHub Actions CI (ubuntu-latest)
                          └──> Build Docker Image ──> Push to Amazon ECR
                                                             │
[ Self-Hosted Runner on EC2 (ap-southeast-2) ] <─────────────┘
  ├── 1. docker pull <ecr-repo>:latest
  ├── 2. docker rm -f medicalbot (Automated Replacement)
  └── 3. docker run --network host (Accesses Host Ollama at localhost:11434)
```

- **Amazon EC2**: Self-hosted runner and container runtime in `ap-southeast-2`. Recommended instance: **`m7i-flex.large`** (2 vCPU, 8 GB RAM) to support host-level Ollama CPU embeddings and the app container simultaneously.
- **Host Ollama Integration**: Ollama runs directly on the EC2 host (CPU-only). The container runs with **`--network host`** specifically so it can connect to Ollama at `localhost:11434`.
- **Amazon ECR**: Private container registry for image versions.
- **Automated Container Replacement**: The runner stops the old container (`docker rm -f medicalbot`) and launches the new image. This automated deployment involves a few seconds of downtime during initialization.
- **Credentials**: `app.py` does not use AWS credentials; `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` are only needed by the GitHub Actions runner for ECR authentication.

### EC2 Host Setup
```bash
# Docker setup
sudo apt-get update -y && sudo apt-get install -y docker.io curl
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu

# Install Ollama & pull model (CPU mode)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-embedding:0.6b

# GitHub Self-Hosted Runner setup (ensure runner user is in docker group)
sudo usermod -aG docker <runner-user>
sudo ./svc.sh install && sudo ./svc.sh start
```

### GitHub Repository Secrets & IAM
Configure secrets under **Settings -> Secrets and variables -> Actions**:
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (`ap-southeast-2`), `ECR_REPO` (`medical-chatbot`), `PINECONE_API_KEY`, `GEMINI_API_KEY`.

Required IAM permissions for ECR:
`ecr:GetAuthorizationToken` (resource `*`), `ecr:BatchCheckLayerAvailability`, `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage`, `ecr:PutImage`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, and `ecr:CompleteLayerUpload`.

---

## 💻 Local Quickstart

### 1. Setup Environment
```bash
git clone https://github.com/Maanit5117/Medical-Chatbot.git
cd Medical-Chatbot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3-embedding:0.6b
```

### 2. Configure `.env` (gitignored)
```env
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="medical-chatbot"
GEMINI_API_KEY="your-gemini-api-key"
PORT=8080
```

### 3. Ingest Data & Run
```bash
python store_index.py    # Ingests PDF chunks into Pinecone (run once)
python app.py            # Starts server at http://localhost:8080
```

---

## 📂 Project Structure

```
├── .github/workflows/cicd.yaml # AWS CI/CD pipeline definition
├── data/Medical_book.pdf       # Clinical reference book
├── src/
│   ├── helper.py               # Ingestion, embeddings, retrieval & LCEL chain
│   └── prompt.py               # Closed-domain clinical system prompt
├── static/                     # Web UI assets (CSS, JS with excerpt accordion & TTS)
├── templates/index.html        # Web chat template
├── .env                        # Environment variables (gitignored, not committed)
├── Dockerfile                  # Container image definition (python:3.10-slim-buster)
├── requirements.txt            # Python dependencies
├── store_index.py              # Pinecone ingestion script with retry & resume
└── app.py                      # Multi-tier backend server
```

---

## ⚠️ Known Limitations

- **Host-Coupled CPU Embeddings**: Query embeddings run on CPU via Ollama (`qwen3-embedding:0.6b`), sharing memory and compute with the host EC2 instance.
- **Single-Instance Deployment**: Runs on a single EC2 instance over plain HTTP without a load balancer or TLS/SSL certificate.
- **No Authentication or Rate Limiting**: Endpoints are currently unauthenticated with no client-side rate limits.
- **Cross-Region Latency**: EC2 compute (`ap-southeast-2`) queries Pinecone Serverless (`us-east-1`) over the public internet, introducing cross-region round-trip latency.
- **No Automated RAG Evaluation**: Retrieval precision/recall and answer faithfulness are not evaluated through automated benchmarking pipelines.

---

## ⚠️ Medical Disclaimer

> **NOTICE**: MediRAG is strictly for informational, educational, and research purposes. It is not a certified diagnostic system and must not replace professional medical judgment or personalized consultation.
