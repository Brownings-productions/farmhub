"""The LLM module (SPEC §8.2): a thin OpenAI-compatible client. Exposes no tools."""

from farmhub.modules.llm.client import OpenAICompatBackend
from farmhub.modules.llm.module import LlmModule, LLMService

__all__ = ["LLMService", "LlmModule", "OpenAICompatBackend"]
