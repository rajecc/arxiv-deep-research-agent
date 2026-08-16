from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from configs.settings import settings
from src.utils.logger import logger


class LLMFactory:
    """Factory to create LLM chat model instances (Gemini or OpenAI-compatible / VseLLM)."""

    @staticmethod
    def get_llm(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Instantiate and return the configured BaseChatModel.

        Args:
            provider: 'gemini' or 'openai_compatible' (defaults to settings.DEFAULT_LLM_PROVIDER)
            model_name: override model name (e.g. 'gemini-3.7-flash', 'deepseek-chat', 'gpt-4o')
            temperature: sampling temperature
        """
        active_provider = provider or settings.DEFAULT_LLM_PROVIDER

        if active_provider == "gemini":
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                logger.warning(
                    "GEMINI_API_KEY is not set in environment or .env file."
                )

            model = model_name or settings.GEMINI_MODEL
            temp = temperature if temperature is not None else settings.GEMINI_TEMPERATURE

            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key or "",
                temperature=temp,
                **kwargs,
            )

        elif active_provider == "openai_compatible":
            api_key = settings.OPENAI_API_KEY or "dummy-key"
            base_url = settings.OPENAI_BASE_URL
            model = model_name or settings.OPENAI_MODEL
            temp = temperature if temperature is not None else settings.OPENAI_TEMPERATURE

            if not settings.OPENAI_API_KEY:
                logger.warning(
                    "OPENAI_API_KEY is not set. Using dummy key (required if using local non-auth endpoint)."
                )

            return ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temp,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider: '{active_provider}'. Supported: 'gemini', 'openai_compatible'"
            )


# Global helper instance
llm_factory = LLMFactory()
