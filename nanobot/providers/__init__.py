"""LLM provider abstraction module."""

from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.litellm_provider import LiteLLMProvider
# 🌟 註釋掉下面呢行，防止載入不存在的依賴
# from nanobot.providers.openai_codex_provider import OpenAICodexProvider

# 🌟 同樣喺 __all__ 入面拎走 OpenAICodexProvider
__all__ = ["LLMProvider", "LLMResponse", "LiteLLMProvider"]