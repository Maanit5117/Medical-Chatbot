# 🩺 MediRAG: Medical Retrieval-Augmented Generation (RAG) System

[![CI/CD Pipeline](https://github.com/Maanit5117/Medical-Chatbot/actions/workflows/cicd.yaml/badge.svg)](.github/workflows/cicd.yaml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-cyan.svg)](Dockerfile)
[![Vector Database](https://img.shields.io/badge/Pinecone-Serverless%20(AWS%20us--east--1)-green.svg)](https://www.pinecone.io/)
[![LLM](https://img.shields.io/badge/LLM-Google%20Gemini%20Flash-orange.svg)](https://ai.google.dev/)

![MediRAG UI](docs/screenshot.png)

A production-style Medical Retrieval-Augmented Generation (RAG) system with automated CI/CD on Amazon Web Services (AWS). Grounded in authoritative reference literature (*The Gale Encyclopedia of Medicine*), the application couples a containerized Python backend with Pinecone Serverless vector retrieval, local Ollama embeddings, and Google Gemini LLM inference.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
  - [1. Data Ingestion & Vector Indexing Pipeline](#1-data-ingestion--vector-indexing-pipeline)
  - [2. Online Inference & Query-Retrieval Pipeline](#2-online-inference--query-retrieval-pipeline)
- [RAG Engine & Modeling Deep-Dive](#-rag-engine--modeling-deep-dive)
  - [Clinical Knowledge Base](#clinical-knowledge-base)
  - [Chunking Strategy & Semantic Boundaries](#chunking-strategy--semantic-boundaries)
  - [Vector Embedding & Host Ollama Service](#vector-embedding--host-ollama-service)
  - [Pinecone Serverless Architecture](#pinecone-serverless-architecture)
  - [Strict Clinical Prompt Engineering & Guardrails](#strict-clinical-prompt-engineering--guardrails)
  - [Fault-Tolerant Ingestion Pipeline](#fault-tolerant-ingestion-pipeline)
- [Backend Architecture & Resiliency](#-backend-architecture--resiliency)
  - [Multi-Tier Engine Adapter Pattern](#multi-tier-engine-adapter-pattern)
  - [Lazy Pipeline Initialization & Health Auditing](#lazy-pipeline-initialization--health-auditing)
  - [API Reference](#api-reference)
- [AWS Cloud Infrastructure & Deployment](#-aws-cloud-infrastructure--deployment)
  - [Architecture Topology](#architecture-topology)
  - [Containerization Strategy (Docker)](#containerization-strategy-docker)
  - [Automated CI/CD Pipeline (GitHub Actions to EC2)](#automated-cicd-pipeline-github-actions-to-ec2)
  - [Host Network Mode & Ollama Communication](#host-network-mode--ollama-communication)
  - [Automated Container Replacement](#automated-container-replacement)
  - [Step-by-Step AWS Deployment Guide](#step-by-step-aws-deployment-guide)
- [Client Interface (Frontend Overview)](#-client-interface-frontend-overview)
- [Local Development Setup](#-local-development-setup)
- [Project Directory Structure](#-project-directory-structure)
- [Known Limitations](#-known-limitations)
- [Security & Clinical Disclaimers](#-security--clinical-disclaimers)

---

## 🏛 System Architecture

The MediRAG architecture operates across two decoupled operational planes: an **Offline Ingestion & Embedding Pipeline** and an **Online Retrieval-Augmented Generation (RAG) Serving Plane**.

```
                           OFFLINE DATA INGESTION PLANE
┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│  The Gale Encyclopedia  │ ──> │ PyPDF / DirectoryLoader│ ──> │ Recursive Character    │
│  of Medicine (PDF)      │     │ Document Parser        │     │ Text Splitter          │
└─────────────────────────┘     └────────────────────────┘     └───────────┬────────────┘
                                                                           │ (1000 chars / 200 overlap)
                                                                           ▼
┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│ Pinecone Serverless DB  │ <── │ Dynamic Dimension      │ <── │ Ollama Embeddings      │
│ (AWS us-east-1, Cosine) │     │ Batch Upsert Worker    │     │ (qwen3-embedding:0.6b) │
│ [Over Public Internet]  │     │                        │     │ [Host CPU Mode]        │
└─────────────────────────┘     └────────────────────────┘     └────────────────────────┘

                           ONLINE INFERENCE SERVING PLANE
                                                               ┌────────────────────────┐
                                                               │ User Query / Client    │
                                                               └───────────┬────────────┘
                                                                           │ HTTP POST /api/chat
                                                                           ▼
┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│ Host Ollama Daemon      │ <── │ Embed User Query       │ <── │ Backend Gateway        │
│ (qwen3-embedding:0.6b)  │ ──> │ Return Query Vector    │ ──> │ (Flask / FastAPI App)  │
│ [localhost:11434, CPU]  │     │                        │     └───────────┬────────────┘
└─────────────────────────┘     └────────────────────────┘                 │
                                                                           │ Vector Search (Public Internet)
                                                                           ▼
┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│ Pinecone Vector Index   │ ──> │ Top-4 Context Chunks   │ ──> │ Cross-Region Transfer  │
│ (AWS us-east-1, Cosine) │     │ (Truncated in backend) │     │ (ap-southeast-2 ◄- US) │
└─────────────────────────┘     └────────────────────────┘     └───────────┬────────────┘
                                                                           │
                                                                           ▼
┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│ Verified JSON Response  │ ◄── │ LangChain OutputParser │ ◄── │ Google Gemini Flash    │
│ (Answer + Excerpts)     │     │ Clean Text Synthesis   │     │ (Thinking: low, 4096T) │
└─────────────────────────┘     └────────────────────────┘     └────────────────────────┘
```

### 1. Data Ingestion & Vector Indexing Pipeline
- The reference PDF is parsed page-by-page using LangChain's `DirectoryLoader` with `PyPDFLoader`.
- Metadata is filtered to a minimal footprint (`source` reference and text) to reduce memory overhead.
- Chunks are generated using `RecursiveCharacterTextSplitter` preserving semantic paragraph boundaries.
- Chunks are vectorized using `OllamaEmbeddings(model="qwen3-embedding:0.6b")` and batch-upserted over the public internet into a Pinecone Serverless index in `us-east-1`.

### 2. Online Inference & Query-Retrieval Pipeline
- Incoming queries hit the backend REST API (`/api/chat`).
- The backend embeds the query on-the-fly via the local Ollama daemon on the host (`localhost:11434`).
- Vector similarity search queries Pinecone (`us-east-1`) over HTTPS to retrieve the **Top-4 (`k=4`)** context chunks.
- A LangChain Expression Language (LCEL) chain passes the formatted chunks and query into a constrained prompt.
- **Google Gemini Flash** synthesizes an evidence-grounded answer accompanied by retrieved context excerpts.

---

## 🔬 RAG Engine & Modeling Deep-Dive

### Clinical Knowledge Base
The system is grounded in ***The Gale Encyclopedia of Medicine*** (`data/Medical_book.pdf`), a comprehensive medical reference containing detailed entries on disorders, diseases, symptoms, diagnostic tests, surgical operations, and pharmaceutical therapies.

### Chunking Strategy & Semantic Boundaries
Medical concepts rely heavily on contiguous contextual clauses (e.g., contraindications, dosage qualifications, differential diagnoses). Fragmented chunking risks severing critical qualifications.

```python
CHUNK_SIZE = 1000        # Character window size
CHUNK_OVERLAP = 200      # Sliding character overlap window
```

- **Chunk Size (1,000 characters)**: Captures complete clinical descriptions, etiologies, and treatment summaries within a single retrieval block.
- **Chunk Overlap (200 characters)**: Ensures that medical definitions or symptom listings crossing chunk boundaries maintain semantic continuity.

### Vector Embedding & Host Ollama Service
- **Embedding Model**: `OllamaEmbeddings(model="qwen3-embedding:0.6b")`.
- **Runtime Environment**: Ollama runs directly on the EC2 host system in **CPU-only mode**, listening on port `11434`.
- **Dimensionality**: Automatically determined at runtime via `len(embeddings.embed_query("test"))`, ensuring vector index dimensions match model embeddings dynamically.
- **Online Embedding**: Every incoming user query is embedded on-the-fly through Ollama before querying Pinecone.

### Pinecone Serverless Architecture
- **Cloud Provider**: AWS (`cloud="aws"`).
- **Index Region**: `us-east-1` (N. Virginia).
- **Index Name**: `medical-chatbot`.
- **Distance Metric**: `cosine`.
- **Top-K Retrieval**: Retrieves the Top-4 nearest neighbor chunks (`TOP_K = 4`).

### Strict Clinical Prompt Engineering & Guardrails
To prevent hallucination in high-stakes healthcare scenarios, the prompt contract enforces strict closed-world domain boundary constraints (`src/prompt.py`):

```python
SYSTEM_PROMPT = (
    "You are a medical information assistant. Answer ONLY from the context below. "
    "If the context does not contain the answer, say you don't know. "
    "Do not diagnose or give personal medical advice; this is general information. "
    "Keep answers concise.\n\n"
    "Context:\n{context}"
)
```

1. **Context Boundary Constraint**: The model is mandated to answer strictly from the retrieved text. If context is absent, it responds that it does not know.
2. **Clinical Safety Boundary**: Instructed not to diagnose or provide personal medical advice.
3. **LCEL Invocation**: Executed as a functional pipe:
   ```python
   chain = prompt | llm | StrOutputParser()
   answer = chain.invoke({"context": format_docs(docs), "input": question})
   ```

### Fault-Tolerant Ingestion Pipeline
The indexing script (`store_index.py`) includes operational safeguards:
- **Batch Processing**: Ingests in batches of 20 documents to avoid request timeouts.
- **Exponential Retry Logic**: Up to 3 attempts per batch with exponential backoff.
- **Resumption Support**: Can resume from any chunk offset via `--start <N>` (e.g., `python store_index.py --start 740`).
- **Duplicate Prevention**: Checks `describe_index_stats().total_vector_count` before execution unless overridden by `--force`.

---

## ⚡ Backend Architecture & Resiliency

The backend server (`app.py`) is designed with adaptive runtime support and graceful degradation.

### Multi-Tier Engine Adapter Pattern
`app.py` checks installed dependencies at startup and launches the most appropriate web framework:

```
                      app.py Startup Strategy
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [Flask Installed?]              [FastAPI / Uvicorn?]
          │ Yes      │ No                  │ Yes      │ No
          ▼          └────────────────────►▼          ▼
    Run Flask Engine                 Run FastAPI    Run Built-in HTTP
    (Production Default)             (Async API)    (Zero-Dependency Fallback)
```

1. **Flask Tier (Primary)**: Default production server providing REST endpoints and template serving.
2. **FastAPI / Uvicorn Tier (Alternative)**: Asynchronous non-blocking concurrency with automatic Pydantic validation.
3. **Built-in `http.server.ThreadingHTTPServer` Tier (Fallback)**: Zero-dependency standard library fallback that can serve requests even in bare-bones container environments without third-party frameworks.

### Lazy Pipeline Initialization & Health Auditing
- **Lazy Initialization**: Pinecone and Google Gemini clients are initialized on the first request (`get_rag_pipeline()`), avoiding startup crashes if external services are briefly unreachable during boot.
- **Lightweight Health Check**: The `/api/health` endpoint verifies Pinecone connectivity via `pc.list_indexes()` without invoking the embedding model or burning LLM tokens.
- **Quota Error Handling**: Intercepts Google Gemini HTTP 429 and `RESOURCE_EXHAUSTED` responses and returns clear JSON error messages.

### API Reference

#### 1. Execute RAG Query
- **Endpoint**: `POST /api/chat`
- **Headers**: `Content-Type: application/json`
- **Request Payload**:
  ```json
  {
    "message": "What are the symptoms and treatments for asthma?"
  }
  ```
- **Response Payload (200 OK)**:
  ```json
  {
    "answer": "Asthma symptoms include wheezing, shortness of breath, chest tightness, and coughing. Treatments include inhaled corticosteroids, bronchodilators, and avoiding known triggers.",
    "sources": [
      {
        "source": "data/Medical_book.pdf",
        "content": "Asthma is a chronic inflammatory disease of the airways characterized by recurrent episodes of wheezing, breathlessness, chest tightness, and coughing..."
      }
    ]
  }
  ```
  *(Note: `sources` contains text excerpts truncated to the first 400 characters of each retrieved chunk).*

#### 2. Health & Vector Service Probe
- **Endpoint**: `GET /api/health`
- **Response Payload (200 OK)**:
  ```json
  {
    "status": "healthy",
    "pinecone": "online"
  }
  ```

#### 3. Legacy Web Endpoint
- **Endpoint**: `GET /get?msg=<query>` or `POST /get`
- **Purpose**: Backward-compatible interface for legacy chatbot frontends.

---

## ☁️ AWS Cloud Infrastructure & Deployment

The system is deployed on an Amazon Web Services (AWS) EC2 instance using a containerized backend connected to host services and managed via GitHub Actions CI/CD.

```
                          AWS CI/CD & RUNTIME TOPOLOGY
  
  [ Developer Push to main ]
             │
             ▼
  ┌────────────────────────────────────────────────────────┐
  │ GitHub Actions Runner (ubuntu-latest)                  │
  │   1. Checkout code                                     │
  │   2. Configure AWS Credentials via IAM                 │
  │   3. Docker build -t <ecr-repo>:latest                 │
  │   4. Docker push to Amazon ECR                         │
  └──────────────────────────┬─────────────────────────────┘
                             │ Container Image Push
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Amazon Elastic Container Registry (ECR)                │
  │ Private Image Repository (Tagged: latest)              │
  └──────────────────────────┬─────────────────────────────┘
                             │ Pull Image Trigger
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Amazon EC2 Instance (Self-Hosted Runner, ap-southeast-2│
  │                                                        │
  │   Host Processes:                                      │
  │   • Ollama Service (localhost:11434, CPU-only)         │
  │   • GitHub Actions Runner Daemon (in docker group)     │
  │                                                        │
  │   Docker Container (medicalbot, --network host):       │
  │   • Python 3.10 Slim | app.py | Port 8080              │
  │   • Communicates with host Ollama at localhost:11434   │
  └──────────────────────────┬─────────────────────────────┘
                             │ Public Internet (Cross-Region)
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Pinecone Serverless (AWS us-east-1)                    │
  │ Accessed over public internet via HTTPS API            │
  └────────────────────────────────────────────────────────┘
```

> [!NOTE]
> **Cross-Region Connectivity**: The EC2 compute instance resides in `ap-southeast-2` (Sydney) while the Pinecone index is hosted in `us-east-1` (N. Virginia). Retrieval queries travel over the public internet between regions.

### Containerization Strategy (Docker)
The backend is packaged using a lightweight Debian-based Python image (`Dockerfile`):

```dockerfile
FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]
```

### Automated CI/CD Pipeline (GitHub Actions to EC2)
The workflow in `.github/workflows/cicd.yaml` automates the release:

1. **Continuous Integration (CI) on `ubuntu-latest`**:
   - Authenticates with AWS credentials via `aws-actions/configure-aws-credentials@v1`.
   - Logs into Amazon ECR via `aws-actions/amazon-ecr-login@v1`.
   - Builds the Docker image and pushes it to Amazon ECR.
2. **Continuous Deployment (CD) on `self-hosted` EC2 Runner**:
   - Pulls the newly built image from Amazon ECR.
   - Replaces the running container via automated container replacement.

### Host Network Mode & Ollama Communication
The container is launched with `--network host`:
```bash
docker run -d --name medicalbot --restart unless-stopped --network host ...
```
**Why `--network host` is required**:
The container needs to communicate with the Ollama service running on the EC2 host at `localhost:11434` for query embeddings. By sharing the host network namespace, the container accesses Ollama directly via `localhost` without needing extra bridge network configurations.

### Automated Container Replacement
The deployment script performs a container teardown and replacement:
```bash
docker pull "$IMAGE"
docker rm -f medicalbot || true
docker run -d --name medicalbot --restart unless-stopped --network host \
  -e PINECONE_API_KEY="${{ secrets.PINECONE_API_KEY }}" \
  -e GEMINI_API_KEY="${{ secrets.GEMINI_API_KEY }}" \
  "$IMAGE"
```

> [!WARNING]
> Because `docker rm -f` removes the existing container before `docker run` starts the replacement, this deployment pattern involves a few seconds of downtime while the new container initializes.

> [!NOTE]
> **AWS Credentials in Container**: `app.py` does not use AWS credentials. AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) are required only by the GitHub Actions runner on the host to authenticate with Amazon ECR for image pulls and pushes.

---

### Step-by-Step AWS Deployment Guide

#### 1. Provision AWS Resources
1. **Amazon ECR Repository**:
   - Create a private repository named `medical-chatbot`.
2. **Amazon EC2 Instance**:
   - **Instance Type**: `m7i-flex.large` (2 vCPU, 8 GB RAM recommended, as Ollama and the backend share host CPU/memory).
   - **OS**: Ubuntu Server 22.04 LTS (x86_64).
   - **Region**: `ap-southeast-2` (or your preferred region).
   - **Storage**: 30 GB gp3 root volume.
   - **Security Groups**:
     - `SSH (22)` restricted to your IP.
     - `Custom TCP (8080)` open for web traffic (or routed via a reverse proxy).

#### 2. Configure EC2 Host Environment
SSH into your EC2 instance and set up Docker and Ollama:

```bash
# Update system packages and install Docker
sudo apt-get update -y
sudo apt-get install -y docker.io curl
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
newgrp docker

# Install Ollama on the host and pull embedding model (CPU mode)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-embedding:0.6b
```

#### 3. Install GitHub Actions Self-Hosted Runner on EC2
1. In your GitHub repository, go to **Settings -> Actions -> Runners -> New self-hosted runner**.
2. Follow the setup instructions on your EC2 instance.
3. Ensure the runner's user account belongs to the `docker` group so it can manage Docker containers:
   ```bash
   sudo usermod -aG docker <runner-user>
   ```
4. Install and start the runner as a background systemd service:
   ```bash
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

#### 4. Configure GitHub Repository Secrets
Go to **Settings -> Secrets and variables -> Actions** and configure the following secrets:

| Secret Name | Description | Example / Target |
|:---|:---|:---|
| `AWS_ACCESS_KEY_ID` | IAM User Access Key with ECR permissions | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | IAM User Secret Access Key | `wJalr...` |
| `AWS_DEFAULT_REGION` | AWS Region of your ECR repository | `ap-southeast-2` |
| `ECR_REPO` | Amazon ECR repository name | `medical-chatbot` |
| `PINECONE_API_KEY` | Pinecone vector database API key | `pcsk_...` |
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |

#### 5. IAM Permissions
Ensure the IAM user or role attached to the CI/CD credentials has the required ECR permissions:

```json
{
  "Version": "2012-10-17",
  "Effect": "Allow",
  "Action": [
    "ecr:GetAuthorizationToken",
    "ecr:BatchCheckLayerAvailability",
    "ecr:GetDownloadUrlForLayer",
    "ecr:BatchGetImage",
    "ecr:PutImage",
    "ecr:InitiateLayerUpload",
    "ecr:UploadLayerPart",
    "ecr:CompleteLayerUpload"
  ],
  "Resource": "*"
}
```

---

## 🖥 Client Interface (Frontend Overview)

The project includes a lightweight web interface in `static/` and `templates/`:

- **Retrieved Context Excerpt Accordion**: Displays an expandable list of the source document chunks (truncated to the first 400 characters) retrieved for each query.
- **Accessibility & TTS**: Uses the browser's built-in `SpeechSynthesis` API to read answers aloud.
- **In-Browser Markdown Parsing**: Converts bullet points, bold text, and numbered lists into readable message cards.
- **System Health Indicator**: Periodically checks `/api/health` to reflect vector index connectivity in the UI sidebar.

---

## 💻 Local Development Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) with `qwen3-embedding:0.6b` pulled locally (`ollama pull qwen3-embedding:0.6b`)
- A Pinecone account and API key
- A Google Gemini API key

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/Maanit5117/Medical-Chatbot.git
cd Medical-Chatbot

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME="medical-chatbot"
GEMINI_API_KEY="your-gemini-api-key"
PORT=8080
```

### 3. Run Knowledge Base Ingestion
Split, vectorize, and index the medical encyclopedia into Pinecone:
```bash
python store_index.py
```
*Resume from a specific chunk index if needed: `python store_index.py --start <chunk_number>`.*

### 4. Start the Application Server
```bash
python app.py
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser.

---

## 📂 Project Directory Structure

```
.
├── .github/
│   └── workflows/
│       └── cicd.yaml           # GitHub Actions CI/CD pipeline definition for AWS
├── data/
│   └── Medical_book.pdf        # Gale Encyclopedia of Medicine clinical reference
├── src/
│   ├── __init__.py
│   ├── helper.py               # Ingestion, document chunking, embeddings & chain logic
│   └── prompt.py               # Closed-domain clinical system prompt & guardrails
├── static/
│   ├── css/
│   │   └── style.css           # Clinical responsive stylesheet
│   ├── js/
│   │   └── app.js              # Client state, excerpt accordion & TTS integration
│   └── index.html              # Standalone client application
├── templates/
│   └── index.html              # Server-rendered web template
├── .env                        # Local environment variables (gitignored, not committed)
├── Dockerfile                  # Container image configuration
├── requirements.txt            # Core framework dependencies
├── store_index.py              # Pinecone vector ingestion script
├── app.py                      # Multi-tier backend server (Flask / FastAPI / Builtin)
└── README.md                   # Project documentation
```

---

## ⚠️ Known Limitations

- **Host-Coupled CPU Embeddings**: Query and document embeddings run on CPU via Ollama (`qwen3-embedding:0.6b`), which is tightly coupled to the host EC2 instance and shares CPU and memory resources with the application container.
- **Single-Instance Deployment**: Deployed on a single EC2 instance over plain HTTP with no Application Load Balancer (ALB), TLS/SSL termination, or auto-scaling.
- **No Authentication or Rate Limiting**: The API endpoints (`/api/chat`, `/api/health`) are unauthenticated and lack per-client rate limiting or throttling.
- **Cross-Region Network Latency**: The EC2 compute instance resides in `ap-southeast-2` (Sydney) while the Pinecone Serverless vector index is hosted in `us-east-1` (N. Virginia), introducing cross-region internet round-trip latency on every vector retrieval call.
- **No Automated Retrieval Evaluation**: Retrieval precision/recall and answer faithfulness are not evaluated through automated RAG benchmark pipelines (such as Ragas or TruLens).

---

## ⚠️ Security & Clinical Disclaimers

> **IMPORTANT CLINICAL NOTICE**
> 
> MediRAG is built strictly for **informational, educational, and research purposes**. It is not a licensed medical diagnostic tool and must not be used as a substitute for professional clinical judgment, patient-specific diagnosis, or medical prescription. Always consult a board-certified healthcare provider for individual medical concerns.

- **Data Privacy**: Ensure that no Protected Health Information (PHI) or personally identifiable information (PII) is transmitted via unencrypted channels or logged into application stdout in production.
- **IAM Principle of Least Privilege**: When configuring AWS IAM roles for CI/CD, grant permissions solely for the required ECR actions listed in the IAM Permissions section.
