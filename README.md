# 🩺 MediRAG: Enterprise Medical Retrieval-Augmented Generation (RAG) System

[![CI/CD Pipeline](https://github.com/actions/workflows/cicd.yaml/badge.svg)](.github/workflows/cicd.yaml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-cyan.svg)](Dockerfile)
[![Vector Database](https://img.shields.io/badge/Pinecone-Serverless%20(AWS%20us--east--1)-green.svg)](https://www.pinecone.io/)
[![LLM](https://img.shields.io/badge/LLM-Google%20Gemini%20Flash-orange.svg)](https://ai.google.dev/)

An enterprise-grade, domain-constrained **Medical Retrieval-Augmented Generation (RAG)** system designed to synthesize clinical knowledge from authoritative reference literature (*The Gale Encyclopedia of Medicine*). Built with a decoupled microservice backend, Pinecone Serverless vector storage on AWS, Google Gemini LLM inference, and an automated continuous integration / continuous deployment (CI/CD) pipeline deploying containerized instances onto **Amazon Web Services (AWS EC2 & ECR)**.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
  - [1. Data Ingestion & Vector Indexing Pipeline](#1-data-ingestion--vector-indexing-pipeline)
  - [2. Online Inference & Query-Retrieval Pipeline](#2-online-inference--query-retrieval-pipeline)
- [RAG Engine & Modeling Deep-Dive](#-rag-engine--modeling-deep-dive)
  - [Clinical Knowledge Base](#clinical-knowledge-base)
  - [Chunking Strategy & Semantic Boundaries](#chunking-strategy--semantic-boundaries)
  - [Vector Embedding & Dimensionality](#vector-embedding--dimensionality)
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
  - [Step-by-Step Production AWS Deployment Guide](#step-by-step-production-aws-deployment-guide)
- [Client Interface (Frontend Overview)](#-client-interface-frontend-overview)
- [Local Development Setup](#-local-development-setup)
- [Project Directory Structure](#-project-directory-structure)
- [Security & Clinical Disclaimers](#-security--clinical-disclaimers)

---

## 🏛 System Architecture

The MediRAG architecture operates across two distinct decoupled operational planes: an **Offline Ingestion & Embedding Pipeline** and an **Online Retrieval-Augmented Generation (RAG) Serving Plane**.

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
└─────────────────────────┘     └────────────────────────┘     └────────────────────────┘

                           ONLINE INFERENCE SERVING PLANE
                                                               ┌────────────────────────┐
                                                               │ User Query / Client    │
                                                               └───────────┬────────────┘
                                                                           │ HTTP POST /api/chat
                                                                           ▼
┌─────────────────────────┐                                    ┌────────────────────────┐
│ Pinecone Vector Index   │ ◄───────────────────────────────── │ Backend Gateway        │
│ (Top-4 Semantic Chunks) │ ──[ Top-K Context Vectors ]──────> │ (Flask / FastAPI App)  │
└─────────────────────────┘                                    └───────────┬────────────┘
                                                                           │ Formatted Prompt
                                                                           ▼
┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│ Verified JSON Response  │ ◄── │ LangChain OutputParser │ ◄── │ Google Gemini Flash    │
│ (Answer + Citations)    │     │ Clean Text Synthesis   │     │ (Thinking: low, 4096T) │
└─────────────────────────┘     └────────────────────────┘     └────────────────────────┘
```

### 1. Data Ingestion & Vector Indexing Pipeline
- Source corpus is loaded via `PyPDFLoader` through LangChain's `DirectoryLoader`.
- Metadata is stripped to minimal footprint (`source` reference + raw content) to maximize memory efficiency.
- Content is split using `RecursiveCharacterTextSplitter` preserving semantic paragraph boundaries.
- Chunks are vectorized using high-fidelity embedding representations and batch-upserted with retry-backoff into a Pinecone Serverless index hosted on AWS.

### 2. Online Inference & Query-Retrieval Pipeline
- Incoming queries hit the backend REST API (`/api/chat`).
- Vector similarity search runs against Pinecone using Cosine similarity metric to extract the **Top-4 (`k=4`)** contextually relevant passages.
- LangChain Expression Language (LCEL) chain injects retrieved passages into a strictly bounded system prompt.
- **Google Gemini Flash** synthesizes an evidence-grounded answer accompanied by verbatim source citations.

---

## 🔬 RAG Engine & Modeling Deep-Dive

### Clinical Knowledge Base
The primary corpus utilized is ***The Gale Encyclopedia of Medicine*** (`data/Medical_book.pdf`), a 4,000+ page comprehensive clinical reference encompassing symptoms, etiologies, diagnostic procedures, pharmacological treatments, and surgical interventions across thousands of diseases.

### Chunking Strategy & Semantic Boundaries
Medical concepts rely heavily on contiguous contextual clauses (e.g., contraindications, dosage qualifications, differential diagnoses). Fragmented chunking causes hallucinated advice or severed context.

```python
CHUNK_SIZE = 1000        # Character window size
CHUNK_OVERLAP = 200      # Sliding character overlap window
```

- **Chunk Size (1,000 chars)**: Chosen to capture full clinical symptom-disease-treatment descriptions within a single continuous vector.
- **Chunk Overlap (200 chars)**: Prevents semantic boundary truncation across complex medical definitions, ensuring that queries matching sentences spanning split boundaries retain retrieval fidelity.

### Vector Embedding & Dimensionality
The system leverages `OllamaEmbeddings(model="qwen3-embedding:0.6b")`:
- Dynamically derives vector dimensions via `len(embeddings.embed_query("test"))`, eliminating hardcoded vector mismatch errors.
- Cosine distance metric measures proximity in dense semantic space, ensuring invariance to passage length differences.

### Pinecone Serverless Architecture
- **Host Infrastructure**: AWS Cloud (`cloud="aws"`, `region="us-east-1"`).
- **Index Type**: Serverless Vector Index (`medical-chatbot`).
- **Distance Metric**: `cosine`.
- **Top-K Retrieval**: Returns top 4 nearest neighbors (`TOP_K = 4`). Multi-chunk aggregation balances context density against LLM context window cost and latency.

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

1. **Anti-Hallucination Isolation**: Explicit mandate to answer *only* from the provided context chunks. If information is absent, the model explicitly yields a fallback disclaimer.
2. **Clinical Safety Boundary**: Strictly prohibits prescribing regimens or diagnosing symptoms.
3. **LCEL Invocation**: Executed as a functional pipe:
   ```python
   chain = prompt | llm | StrOutputParser()
   answer = chain.invoke({"context": format_docs(docs), "input": question})
   ```

### Fault-Tolerant Ingestion Pipeline
The indexing script (`store_index.py`) is engineered for production-scale vector indexing:
- **Batch Processing**: Ingests in batches of 20 documents to prevent payload timeouts.
- **Exponential Retry Logic**: 3 automatic retries per batch with transient error sleeping.
- **Crash Recovery**: Supports resumption via `--start <index>` (e.g. `python store_index.py --start 740`).
- **Duplicate Prevention**: Audits `describe_index_stats().total_vector_count` prior to executing writes.

---

## ⚡ Backend Architecture & Resiliency

The backend server (`app.py`) is engineered with high resilience, graceful degradation, and production reliability.

### Multi-Tier Engine Adapter Pattern
`app.py` features an adaptive runtime discovery mechanism that runs across three operational tiers depending on the runtime environment:

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

1. **Flask Tier (Primary)**: Production-grade REST backend with modular route handlers and template rendering.
2. **FastAPI / Uvicorn Tier (Alternative)**: Asynchronous non-blocking concurrency with automatic Pydantic request validation and CORS middleware.
3. **Built-in `http.server.ThreadingHTTPServer` Tier (Fallback)**: Native Python standard library implementation ensuring that even in bare-metal container environments lacking pip packages, the system boots and serves chat queries.

### Lazy Pipeline Initialization & Health Auditing
- **Zero Cold-Start Lock**: Pinecone and Google Gemini clients are loaded lazily on the first request (`get_rag_pipeline()`), avoiding application crashes during initial container boot if network interfaces are still mounting.
- **Token-Free Health Checks**: `/api/health` queries Pinecone metadata directly (`pc.list_indexes()`) to determine vector database availability without incurring LLM inference billing or rate-limit consumption.
- **Upstream Rate Limit Resilience**: Explicitly intercepts Google Gemini HTTP 429 and `RESOURCE_EXHAUSTED` codes, mapping them to structured, human-interpretable API status responses.

### API Reference

#### 1. Execute RAG Query
- **Endpoint**: `POST /api/chat`
- **Headers**: `Content-Type: application/json`
- **Request Payload**:
  ```json
  {
    "message": "What are the common symptoms and treatment options for allergic rhinitis?"
  }
  ```
- **Response Payload (200 OK)**:
  ```json
  {
    "answer": "Allergic rhinitis, commonly known as hay fever, is characterized by symptoms such as sneezing, runny nose, nasal congestion, itching, and watery eyes. Treatments include antihistamines, nasal corticosteroids, and allergen avoidance.",
    "sources": [
      {
        "source": "data/Medical_book.pdf",
        "content": "Allergic rhinitis is an inflammation of the nasal passages caused by allergens..."
      }
    ]
  }
  ```

#### 2. Health & Vector Service Probe
- **Endpoint**: `GET /api/health`
- **Response Payload (200 OK)**:
  ```json
  {
    "status": "healthy",
    "pinecone": "online"
  }
  ```
- **Error Response (503 Service Unavailable)**:
  ```json
  {
    "status": "unhealthy",
    "pinecone": "offline",
    "detail": "Index 'medical-chatbot' not found"
  }
  ```

#### 3. Legacy Web Endpoint
- **Endpoint**: `GET /get?msg=<query>` or `POST /get`
- **Purpose**: Backward-compatible interface for legacy chatbot frontends.

---

## ☁️ AWS Cloud Infrastructure & Deployment

The system is deployed on Amazon Web Services using a containerized architecture managed by an automated GitHub Actions CI/CD pipeline.

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
  │ Amazon EC2 Instance (Self-Hosted GitHub Runner)        │
  │   1. Authenticate to Amazon ECR                        │
  │   2. Pull latest container image                       │
  │   3. Graceful termination: docker rm -f medicalbot     │
  │   4. docker run -d --restart unless-stopped            │
  │      --network host with production environment secrets│
  │                                                        │
  │   ┌────────────────────────────────────────────────┐   │
  │   │ Running Docker Container (medicalbot)          │   │
  │   │ Python 3.10 Slim | Flask | Port 8080           │   │
  │   └──────────────────────┬─────────────────────────┘   │
  └──────────────────────────┼─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Pinecone Serverless (AWS us-east-1)                    │
  │ Low-Latency Vector Queries via VPC Gateway             │
  └────────────────────────────────────────────────────────┘
```

### Containerization Strategy (Docker)
The service is encapsulated using a lean Debian-based Python image (`Dockerfile`):

```dockerfile
FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]
```
- **Base OS**: `python:3.10-slim-buster` minimizes image attack surface and reduces deployment footprint to under 400MB.
- **Process Model**: Single-container execution serving on port `8080`.

### Automated CI/CD Pipeline (GitHub Actions to EC2)
The automation workflow defined in `.github/workflows/cicd.yaml` executes across two stages:

#### Stage 1: Continuous Integration (CI) on `ubuntu-latest`
1. **Repository Checkout**: Clones the commit SHA.
2. **AWS Authentication**: Authenticates with AWS IAM credentials via `aws-actions/configure-aws-credentials@v1`.
3. **ECR Login**: Retrieves temporary ECR Docker registry authentication tokens via `aws-actions/amazon-ecr-login@v1`.
4. **Build & Push**: Compiles the Docker image, tags it with `:latest`, and pushes it to the Amazon ECR repository.

#### Stage 2: Continuous Deployment (CD) on AWS EC2 (Self-Hosted Runner)
1. **Self-Hosted Runner Execution**: Dispatches deployment jobs directly onto the target Amazon EC2 instance hosting the runner daemon.
2. **ECR Pull**: Pulls the newly updated Docker image from Amazon ECR.
3. **Container Recycling**:
   ```bash
   docker rm -f medicalbot || true
   ```
4. **Zero-Downtime Container Launch**:
   ```bash
   docker run -d --name medicalbot --restart unless-stopped --network host \
     -e AWS_ACCESS_KEY_ID="${{ secrets.AWS_ACCESS_KEY_ID }}" \
     -e AWS_SECRET_ACCESS_KEY="${{ secrets.AWS_SECRET_ACCESS_KEY }}" \
     -e AWS_DEFAULT_REGION="${{ secrets.AWS_DEFAULT_REGION }}" \
     -e PINECONE_API_KEY="${{ secrets.PINECONE_API_KEY }}" \
     -e GEMINI_API_KEY="${{ secrets.GEMINI_API_KEY }}" \
     "$IMAGE"
   ```
   - `--restart unless-stopped`: Ensures high availability across EC2 system reboots.
   - `--network host`: Minimizes network translation overhead and bridges port 8080 directly to host security groups.

---

### Step-by-Step Production AWS Deployment Guide

#### 1. Provision AWS Resources
1. **Create an Amazon ECR Repository**:
   - Go to AWS Management Console -> **ECR** -> **Create repository**.
   - Name: `medical-chatbot`.
2. **Launch an Amazon EC2 Instance**:
   - **OS**: Ubuntu Server 22.04 LTS (x86_64).
   - **Instance Type**: `t3.medium` or higher recommended.
   - **Storage**: Minimum 25 GB gp3.
   - **Security Group Inbound Rules**:
     - `SSH (22)` from your management IP.
     - `Custom TCP (8080)` from `0.0.0.0/0` (or behind an Application Load Balancer).

#### 2. Configure EC2 Host Environment
SSH into your EC2 instance and install Docker:
```bash
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
newgrp docker
```

#### 3. Install GitHub Actions Self-Hosted Runner on EC2
1. In your GitHub repository, navigate to **Settings -> Actions -> Runners -> New self-hosted runner**.
2. Select **Linux** architecture (x64) and execute the provided download, extraction, and configuration commands on your EC2 instance.
3. Install the runner as a background systemd service:
   ```bash
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

#### 4. Configure GitHub Repository Secrets
Navigate to **Settings -> Secrets and variables -> Actions** and create the following secrets:

| Secret Name | Description |
|:---|:---|
| `AWS_ACCESS_KEY_ID` | IAM User Access Key with ECR Read/Write permissions |
| `AWS_SECRET_ACCESS_KEY` | IAM User Secret Access Key |
| `AWS_DEFAULT_REGION` | AWS Region (e.g., `us-east-1`) |
| `ECR_REPO` | ECR repository name (`medical-chatbot`) |
| `PINECONE_API_KEY` | Production API key from Pinecone Console |
| `GEMINI_API_KEY` | Google Gemini API key |

#### 5. Deploy
Push any commit to the `main` branch. GitHub Actions will trigger, build the Docker container in the cloud, push to Amazon ECR, and your self-hosted EC2 runner will pull and run the containerized backend automatically.

---

## 🖥 Client Interface (Frontend Overview)

While the core focus of MediRAG is its retrieval engine and cloud infrastructure, a lightweight clinical client interface is provided in `static/` and `templates/`:

- **Grounded Source Attribution Accordion**: Dynamically renders verbatim excerpt cards and file references returned in the RAG payload so users can audit citations.
- **Accessibility & TTS**: Integrated client-side `SpeechSynthesis` audio player for vocalizing medical answers.
- **Zero-Dependency Markdown Renderer**: In-browser parser for structured lists, headers, bold emphasis, and warning banners.
- **Live Diagnostics**: Automatically pings `/api/health` to inform the user if the vector index or credentials require attention.

---

## 💻 Local Development Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) with `qwen3-embedding:0.6b` installed (`ollama pull qwen3-embedding:0.6b`)
- A Pinecone account and API key
- A Google Gemini API key

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/medical-chatbot.git
cd "medical-chatbot"

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
Extract, split, vectorize, and index the medical encyclopedia into Pinecone:
```bash
python store_index.py
```
*Note: If an upload is interrupted, resume seamlessly via `python store_index.py --start <chunk_number>`.*

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
│   │   └── app.js              # Client state, citation rendering & TTS integration
│   └── index.html              # Standalone client application
├── templates/
│   └── index.html              # Server-rendered web template
├── .env                        # Local environment variables (API credentials)
├── Dockerfile                  # Production container image configuration
├── requirements.txt            # Core framework dependencies
├── store_index.py              # Robust Pinecone vector ingestion script
├── app.py                      # Multi-tier backend server (Flask / FastAPI / Builtin)
└── README.md                   # Enterprise system documentation
```

---

## ⚠️ Security & Clinical Disclaimers

> **IMPORTANT CLINICAL NOTICE**
> 
> MediRAG is built strictly for **informational, educational, and research purposes**. It is not a licensed medical diagnostic tool and must not be used as a substitute for professional clinical judgment, patient-specific diagnosis, or medical prescription. Always consult a board-certified healthcare provider for individual medical concerns.

- **Data Privacy**: Ensure that no Protected Health Information (PHI) or personally identifiable information (PII) is transmitted via unencrypted channels or logged into application stdout in production.
- **IAM Principle of Least Privilege**: When configuring AWS IAM roles for CI/CD, grant permissions solely for `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage`, `ecr:PutImage`, `ecr:InitiateLayerUpload`, and `ecr:UploadLayerPart`.
