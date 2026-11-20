import os
from pathlib import Path
from typing import List

# Load .env file (with graceful fallback if python-dotenv is not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings

from .prompt import prompt

# --- Config -----------------------------------------------------------------
INDEX_NAME = "medical-chatbot"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "gemini-3.6-flash"
DATA_DIR = "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.50"))
GEMINI_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "30.0"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "1024"))


# --- Ingestion --------------------------------------------------------------
def load_pdf_files(data: str = DATA_DIR) -> List[Document]:
    """Extract text from every PDF in the given directory."""
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only page_content and the 'source' metadata of each document."""
    return [
        Document(page_content=doc.page_content, metadata={"source": doc.metadata.get("source")})
        for doc in docs
    ]


def text_split(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


# --- Models and retrieval ---------------------------------------------------
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_llm():
    # Imported here so the ingestion script doesn't need the Gemini package.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        thinking_level="minimal",
        max_retries=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout=GEMINI_TIMEOUT,
    )


def get_retriever(index_name: str = INDEX_NAME, k: int = TOP_K):
    """Connect to an existing Pinecone index (does not re-embed anything)."""
    from langchain_pinecone import PineconeVectorStore
    docsearch = PineconeVectorStore(index_name=index_name, embedding=get_embeddings())
    return docsearch.as_retriever(search_kwargs={"k": k})


# --- Question answering -----------------------------------------------------
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def is_greeting(question: str) -> bool:
    """Check if the user input is a greeting or brief conversational prompt."""
    clean = question.strip().lower().rstrip("?!.,")
    greetings = {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "how are you", "who are you", "what are you", "what is this", "help"
    }
    return clean in greetings or any(clean.startswith(g + " ") for g in ["hi", "hello", "hey"])


def ask(question: str, retriever, llm):
    """Return (answer_text, source_documents) with per-stage timing and similarity-score gating."""
    import logging
    import time

    logger = logging.getLogger("medirag")
    t_start = time.perf_counter()

    # Stage 1: Embed query
    t_embed_0 = time.perf_counter()
    vectorstore = getattr(retriever, "vectorstore", None)
    embeddings = getattr(vectorstore, "embedding", None) if vectorstore else None
    query_vector = None

    if embeddings is not None and hasattr(embeddings, "embed_query"):
        try:
            query_vector = embeddings.embed_query(question)
        except Exception as embed_err:
            logger.warning(f"Direct embedding failed: {embed_err}")
    embed_ms = (time.perf_counter() - t_embed_0) * 1000

    # Stage 2: Retrieve with similarity scores
    t_ret_0 = time.perf_counter()
    docs_with_scores = []
    if vectorstore is not None and query_vector is not None and hasattr(vectorstore, "similarity_search_by_vector_with_score"):
        try:
            docs_with_scores = vectorstore.similarity_search_by_vector_with_score(query_vector, k=TOP_K)
        except Exception as ret_err:
            logger.warning(f"Vector search with score failed: {ret_err}")

    if not docs_with_scores and vectorstore is not None and hasattr(vectorstore, "similarity_search_with_score"):
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(question, k=TOP_K)
        except Exception as ret_err:
            logger.warning(f"Similarity search with score failed: {ret_err}")

    if not docs_with_scores:
        raw_docs = retriever.invoke(question)
        docs_with_scores = [(d, 1.0) for d in raw_docs]

    retrieve_ms = (time.perf_counter() - t_ret_0) * 1000

    docs = [doc for doc, _ in docs_with_scores]
    scores = [score for _, score in docs_with_scores]
    top_score = max(scores) if scores else 0.0

    # Similarity-score threshold check: short-circuit greetings and off-topic queries
    if top_score < SIMILARITY_THRESHOLD or not docs:
        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[TIMING] embed: {embed_ms:.1f}ms | retrieve: {retrieve_ms:.1f}ms | "
            f"generate: 0.0ms (short-circuited: top_score={top_score:.3f} < {SIMILARITY_THRESHOLD}) | "
            f"total: {total_ms:.1f}ms"
        )
        if is_greeting(question):
            answer = (
                "Hello! I am MediRAG, your medical information assistant grounded in *The Gale Encyclopedia of Medicine*. "
                "How can I assist you with clinical concepts, symptoms, conditions, or treatments today?"
            )
        else:
            answer = (
                "I am a specialized medical assistant grounded strictly in *The Gale Encyclopedia of Medicine*. "
                "Your inquiry does not appear to match any indexed medical topics. "
                "Please ask a medical or health-related question."
            )
        return answer, []

    # Stage 3: Generate answer via Gemini
    t_gen_0 = time.perf_counter()
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(docs), "input": question}
    )
    generate_ms = (time.perf_counter() - t_gen_0) * 1000
    total_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"[TIMING] embed: {embed_ms:.1f}ms | retrieve: {retrieve_ms:.1f}ms | "
        f"generate: {generate_ms:.1f}ms | total: {total_ms:.1f}ms | top_score: {top_score:.3f}"
    )

    return answer, docs