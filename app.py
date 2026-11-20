import json
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("medirag")

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

# Global RAG pipeline components (lazily initialized)
_retriever = None
_llm = None


def get_rag_pipeline():
    """Lazily initialize Pinecone retriever and Gemini LLM."""
    global _retriever, _llm
    if _retriever is None or _llm is None:
        print("[MediRAG] Initializing Pinecone Retriever and Gemini LLM...")
        try:
            from src.helper import get_llm, get_retriever
            _retriever = get_retriever()
            _llm = get_llm()
            print("[MediRAG] Pipeline successfully loaded and ready.")
        except ModuleNotFoundError as mne:
            msg = f"Missing required dependency: {mne.name}. Please install dependencies via: pip install -r requirements.txt"
            print(f"[MediRAG] Dependency error: {msg}")
            raise RuntimeError(msg) from mne
        except Exception as e:
            print(f"[MediRAG] Initialization warning: {e}")
            raise
    return _retriever, _llm


def check_pinecone_health():
    """Verify Pinecone connectivity without running a full LLM query."""
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        return False, "PINECONE_API_KEY not set"
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        indexes = [idx.name for idx in pc.list_indexes()]
        target_index = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
        if target_index not in indexes:
            return False, f"Index '{target_index}' not found"
        return True, "Online"
    except Exception as e:
        logger.warning(f"Pinecone health check failed: {e}")
        return False, str(e)


def process_query(question: str):
    """Query the RAG pipeline and format answer + sources."""
    if not question or not question.strip():
        return {"error": "Query cannot be empty."}, 400

    try:
        logger.info(f"Processing query: {question.strip()[:80]}")
        retriever, llm = get_rag_pipeline()
        from src.helper import ask

        answer, docs = ask(question.strip(), retriever, llm)
        sources = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source", "data/Medical_book.pdf"),
                "content": doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else "")
            })

        return {
            "answer": answer,
            "sources": sources
        }, 200
    except Exception as err:
        logger.exception("Error processing query:")
        err_str = str(err)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            return {"error": "RESOURCE_EXHAUSTED", "detail": err_str}, 429
        return {"error": "INTERNAL_ERROR", "detail": err_str}, 500


# ---------------------------------------------------------------------------
# Flask Server Implementation (Primary)
# ---------------------------------------------------------------------------
def run_flask(port: int = 8080):
    from flask import Flask, jsonify, render_template, request, send_from_directory

    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        online, msg = check_pinecone_health()
        if online:
            return jsonify({"status": "healthy", "pinecone": "online"}), 200
        return jsonify({"status": "unhealthy", "pinecone": "offline", "detail": msg}), 503

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(silent=True) or {}
        query = data.get("message") or data.get("question") or data.get("prompt")
        if not query:
            return jsonify({"error": "No query provided in request body."}), 400

        try:
            result, status = process_query(query)
            return jsonify(result), status
        except Exception as err:
            logger.exception("Unexpected error in /api/chat:")
            err_str = str(err)
            status = 429 if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) else 500
            err_type = "RESOURCE_EXHAUSTED" if status == 429 else "INTERNAL_ERROR"
            return jsonify({"error": err_type, "detail": err_str}), status

    # Support traditional /get endpoint
    @app.route("/get", methods=["GET", "POST"])
    def get_bot_response():
        query = request.args.get("msg") or request.form.get("msg")
        if not query and request.is_json:
            query = (request.get_json(silent=True) or {}).get("msg")

        if not query:
            return "Please provide a query.", 400

        try:
            result, status = process_query(query)
            return jsonify(result), status
        except Exception as err:
            logger.exception("Unexpected error in /get:")
            err_str = str(err)
            status = 429 if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) else 500
            err_type = "RESOURCE_EXHAUSTED" if status == 429 else "INTERNAL_ERROR"
            return jsonify({"error": err_type, "detail": err_str}), status

    print(f"\n========================================================")
    print(f"  MediRAG Chatbot is running at http://localhost:{port}")
    print(f"========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)


# ---------------------------------------------------------------------------
# FastAPI / Uvicorn Server Implementation (Alternative)
# ---------------------------------------------------------------------------
def run_fastapi(port: int = 8080):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn

    fast_app = FastAPI(title="MediRAG Medical Chatbot")

    fast_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fast_app.mount("/static", StaticFiles(directory="static"), name="static")

    class ChatRequest(BaseModel):
        message: str

    @fast_app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = Path("templates/index.html")
        if not index_file.exists():
            index_file = Path("static/index.html")
        return FileResponse(index_file)

    @fast_app.get("/api/health")
    async def get_health():
        online, msg = check_pinecone_health()
        if online:
            return {"status": "healthy", "pinecone": "online"}
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "pinecone": "offline", "detail": msg})

    @fast_app.post("/api/chat")
    async def post_chat(req: ChatRequest):
        try:
            res, status = process_query(req.message)
            if status != 200:
                raise HTTPException(status_code=status, detail=res.get("error", "Error"))
            return res
        except HTTPException:
            raise
        except Exception as err:
            logger.exception("Unexpected error in FastAPI /api/chat:")
            err_str = str(err)
            status = 429 if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) else 500
            raise HTTPException(status_code=status, detail="RESOURCE_EXHAUSTED" if status == 429 else "INTERNAL_ERROR")

    @fast_app.get("/get")
    async def get_route(msg: str):
        res, status = process_query(msg)
        return res

    print(f"\n========================================================")
    print(f"  MediRAG Chatbot (FastAPI) running at http://localhost:{port}")
    print(f"========================================================\n")
    uvicorn.run(fast_app, host="0.0.0.0", port=port)


# ---------------------------------------------------------------------------
# Built-in HTTP Server Fallback (No extra libraries required)
# ---------------------------------------------------------------------------
def run_builtin(port: int = 8080):
    import http.server
    import urllib.parse

    class MedicalHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                template_path = Path("templates/index.html")
                content = template_path.read_bytes() if template_path.exists() else Path("static/index.html").read_bytes()
                self.wfile.write(content)
                return

            if parsed.path.startswith("/get"):
                qs = urllib.parse.parse_qs(parsed.query)
                query = qs.get("msg", [""])[0]
                try:
                    result, status = process_query(query)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode("utf-8"))
                except Exception as err:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(err)}).encode("utf-8"))
                return

            return super().do_GET()

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in ("/api/chat", "/get"):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                try:
                    data = json.loads(body)
                    query = data.get("message") or data.get("question") or data.get("msg", "")
                    result, status = process_query(query)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode("utf-8"))
                except Exception as err:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(err)}).encode("utf-8"))
                return

            self.send_response(404)
            self.end_headers()

    print(f"\n========================================================")
    print(f"  MediRAG Chatbot (Standard HTTP) running at http://localhost:{port}")
    print(f"========================================================\n")
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), MedicalHandler)
    server.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    # Detect available framework
    try:
        import flask
        run_flask(port)
    except ImportError:
        try:
            import fastapi
            import uvicorn
            run_fastapi(port)
        except ImportError:
            run_builtin(port)
