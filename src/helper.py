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
        thinking_level="low",
        max_retries=0,
        max_output_tokens=4096,
    )


def get_retriever(index_name: str = INDEX_NAME, k: int = TOP_K):
    """Connect to an existing Pinecone index (does not re-embed anything)."""
    from langchain_pinecone import PineconeVectorStore
    docsearch = PineconeVectorStore(index_name=index_name, embedding=get_embeddings())
    return docsearch.as_retriever(search_kwargs={"k": k})


# --- Question answering -----------------------------------------------------
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def ask(question: str, retriever, llm):
    """Return (answer_text, source_documents) for a question."""
    docs = retriever.invoke(question)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(docs), "input": question}
    )
    return answer, docs