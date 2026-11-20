# 🩺 MediRAG - Medical Knowledge Chatbot Frontend

A responsive web frontend (HTML5, CSS3, Vanilla JavaScript) designed specifically for the Medical RAG chatbot. Grounded in *The Gale Encyclopedia of Medicine*, Pinecone Serverless vector retrieval, and Google Gemini.

---

## 📁 Frontend File Structure

```
├── templates/
│   └── index.html          # Main HTML template for Flask / Web server
├── static/
│   ├── index.html          # Standalone static HTML (works with live-server or file://)
│   ├── css/
│   │   └── style.css       # Medical UI styling, animations & responsive layout
│   └── js/
│       └── app.js          # Interactive chat engine, markdown, RAG citations & TTS
├── app.py                  # Multi-engine backend server (Flask / FastAPI / Built-in HTTP)
└── requirements.txt        # Python package dependencies
```

---

## ✨ Features

- **Clinical Healthcare Aesthetic**: Soothing teal/slate color scheme, crisp typography, and high-contrast readable message cards.
- **RAG Source Citation Accordion**:
  - Automatically receives retrieved chunks from Pinecone.
  - Displays an expandable dropdown showing source documents and verbatim text excerpts.
- **Interactive Markdown Support**:
  - Renders bold, italics, numbered lists, bullet points, headers, and code blocks safely without third-party bloat.
- **Accessibility & Utilities**:
  - **Text-to-Speech (TTS)**: Built-in `SpeechSynthesis` audio player to listen to medical explanations aloud.
  - **One-Click Copy**: Copy responses directly to clipboard with animated toast notification.
  - **Quick Suggestion Cards**: Preset inquiries to test the knowledge base immediately.
  - **Auto-Expanding Input**: Textarea resizes smoothly as you type with character counter (`0 / 2000`).
  - **Keyboard Friendly**: `Enter` to send, `Shift + Enter` for new lines.
- **Responsive & Mobile Ready**: Collapsible drawer sidebar with system stats and toggle navigation on mobile.
- **Error Resilient**: Displays friendly diagnostics if Pinecone, Ollama, or Gemini API keys need configuration.

---

## 🚀 How to Run

### 1. Configure Environment Variables
Ensure your `.env` file contains your Pinecone and Gemini credentials:
```env
PINECONE_API_KEY="your-pinecone-key"
GEMINI_API_KEY="your-gemini-key"
```

### 2. Start the Application
Run the backend web server with Python:

```bash
python app.py
```

*Note: `app.py` automatically detects your environment (Flask, FastAPI, or built-in HTTP server) and serves the frontend at:*

👉 **[http://localhost:8080](http://localhost:8080)**

---

## 🔌 API Endpoints

- `GET /` — Serves the MediRAG chat interface.
- `POST /api/chat` — Accepts `{"message": "What is acne?"}` and returns:
  ```json
  {
    "answer": "Acne is a skin disease marked by pimples...",
    "sources": [
      {
        "source": "data/Medical_book.pdf",
        "content": "Acne is a skin disease marked by the occurrence of..."
      }
    ]
  }
  ```
- `GET /get?msg=<query>` — Traditional Flask chatbot query endpoint for backward compatibility.
- `GET /api/health` — System health check.
