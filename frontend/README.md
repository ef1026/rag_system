# RAG System Frontend

Next.js frontend for the FastAPI RAG backend. The browser never receives LLM keys, embedding settings, vector database details, or document parsing logic.

## Local Development

```powershell
npm install
npm run dev
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Run the backend from the repository root:

```powershell
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

## Vercel

Set `NEXT_PUBLIC_API_BASE_URL` to the public URL of the FastAPI backend. Vercel only deploys this frontend; the Python RAG runtime must be hosted separately.
