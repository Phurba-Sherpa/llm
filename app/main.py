from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chain import ask_llm, stream_llm
from app.settings import settings

app = FastAPI(
    title=settings.app_name,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    try:
        answer = await ask_llm(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return AskResponse(answer=answer)


@app.post("/ask/stream")
async def ask_stream(payload: AskRequest) -> StreamingResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    return StreamingResponse(
        stream_llm(payload.question), media_type="text/plain; charset=utf-8"
    )
