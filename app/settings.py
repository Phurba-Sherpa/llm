from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LangChain FastAPI Service"
    ollama_api_key: Optional[str] = None
    ollama_model: str = "gpt-oss:120b-cloud"
    ollama_base_url: str = "https://ollama.com"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
