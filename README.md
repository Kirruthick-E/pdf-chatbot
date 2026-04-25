# PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF and ask questions about its contents. Text is extracted from the PDF, chunked, embedded with a local sentence-transformers model, stored in ChromaDB, and retrieved at query time to ground answers from an LLM (Google Gemini or a local Ollama model).

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- **One of the following LLM backends:**
  - **Google Gemini** — a `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey)
  - **Ollama** — [Ollama](https://ollama.com) running locally with your chosen model pulled (e.g. `ollama pull gemma4:e4b`)

## Installation

1. Clone the repo and enter the project directory:

   ```bash
   cd pdf_chatbot
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

   Or with pip:

   ```bash
   pip install -e .
   ```

3. Copy the example env file and fill in your credentials:

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:

   | Variable | Description |
   |---|---|
   | `LLM_PROVIDER` | `gemini` or `ollama` |
   | `LLM_MODEL` | Model name — leave blank to use the default |
   | `GEMINI_API_KEY` | Required when `LLM_PROVIDER=gemini` |
   | `OLLAMA_BASE_URL` | Ollama endpoint (default: `http://localhost:11434`) |
   | `EMBEDDING_MODEL` | Sentence-transformers model key (see options below) |
   | `HF_TOKEN` | Optional — raises HuggingFace rate limits |

   **Embedding model options** (downloaded automatically on first use):

   | Key | Size | Quality |
   |---|---|---|
   | `all-MiniLM-L6-v2` | ~90 MB | Fast (default) |
   | `all-MiniLM-L12-v2` | ~120 MB | Balanced |
   | `all-mpnet-base-v2` | ~420 MB | High quality |

## Running

Start the Streamlit app:

```bash
uv run streamlit run app.py
```

Or with pip/python:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a PDF from the sidebar, choose an embedding model, and start chatting.
