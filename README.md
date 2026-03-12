# LangChain + FastAPI Starter

A minimal API service that exposes a LangChain-powered LLM endpoint via FastAPI.

## 1) Install dependencies with uv

```bash
uv sync
```

## 2) Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` and set `OLLAMA_*` values for your environment.

- Local Ollama default: `OLLAMA_BASE_URL=http://localhost:11434`
- Hosted Ollama: set your hosted URL in `OLLAMA_BASE_URL` and set `OLLAMA_API_KEY`

## 3) Run the API

```bash
uv run uvicorn app.main:app --reload
```

## Endpoints

- `GET /health` - health check
- `POST /ask` - asks the configured model
- `POST /ask/stream` - streams model response as plain text chunks

## API Docs

- Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is FastAPI in one sentence?"}'
```

Streaming request:

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain FastAPI in 3 short bullets."}'
```
