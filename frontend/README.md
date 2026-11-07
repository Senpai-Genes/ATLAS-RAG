# Atlas — Document Q&A frontend

React + Vite frontend for your existing FastAPI RAPTOR/ColBERT/Ollama backend.

## Start backend (WSL Ubuntu)

```bash
cd /mnt/d/RAGGG
source colbert-venv/bin/activate
CUDA_VISIBLE_DEVICES="" python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Start frontend (Windows PowerShell)

Extract this folder to `D:\RAGGG\frontend`, then:

```powershell
cd D:\RAGGG\frontend
npm install
npm run dev
```

Open http://localhost:5173

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000/*`. No CORS change is needed during development. If WSL networking does not expose port 8000 to Windows localhost, run the frontend in WSL or adjust `vite.config.js` to your backend address.

## Features

- Upload PDFs (25 MB limit, as enforced by the backend)
- Automatically create unique collection names
- List indexed documents and switch between them
- Ask questions, show responses and page-level source references
- Per-collection chat display retained in memory until page refresh
- Upload progress indicator (indexing is synchronous on the backend)

## Limitations

- PDF only; OCR and other formats require backend work
- Conversation history is frontend-only, not sent to the RAG model; follow-up questions are not context-aware
- No authentication, user isolation, persistent chat history, or background job tracking
- For local development only, not a public deployment
