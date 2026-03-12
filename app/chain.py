import os
from typing import AsyncIterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.settings import settings


def _build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise and helpful assistant."),
            ("human", "{question}"),
        ]
    )
    if settings.ollama_api_key:
        os.environ["OLLAMA_API_KEY"] = settings.ollama_api_key

    llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
    return prompt | llm | StrOutputParser()


async def ask_llm(question: str) -> str:
    chain = _build_chain()
    return await chain.ainvoke({"question": question})


async def stream_llm(question: str) -> AsyncIterator[str]:
    chain = _build_chain()
    async for chunk in chain.astream({"question": question}):
        yield chunk
