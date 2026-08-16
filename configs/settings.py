import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App & Logging
    APP_NAME: str = "ArXiv-Deep-Research-Agent"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # Cache & Storage
    BASE_DIR: Path = BASE_DIR
    CACHE_DIR: Path = BASE_DIR / "data" / "cache"
    PDF_CACHE_DIR: Path = BASE_DIR / "data" / "cache" / "pdfs"
    PARSED_CACHE_DIR: Path = BASE_DIR / "data" / "cache" / "parsed"

    # LLM Settings
    DEFAULT_LLM_PROVIDER: Literal["gemini", "openai_compatible"] = "gemini"

    # Google Gemini Settings (Default: gemini-3.7-flash)
    GEMINI_API_KEY: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    GEMINI_MODEL: str = "gemini-3.7-flash"
    GEMINI_TEMPERATURE: float = 0.2

    # OpenAI-Compatible / VseLLM / OpenRouter Settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    OPENAI_MODEL: str = "deepseek-chat"
    OPENAI_TEMPERATURE: float = 0.2

    # Retrieval Settings
    ARXIV_MAX_RESULTS: int = 10
    ARXIV_DEFAULT_CATEGORIES: list[str] = [
        "cs.AI",
        "cs.LG",
        "cs.CL",
        "cs.CV",
        "stat.ML",
    ]
    REQUEST_TIMEOUT: int = 30
    USER_AGENT: str = "ArXiv-Deep-Research-Agent/1.0"


# Instantiate global settings
settings = Settings()

# Ensure directories exist
settings.PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.PARSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
