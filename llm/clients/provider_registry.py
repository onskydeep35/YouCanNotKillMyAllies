# llm/clients/provider_registry.py
import os
from openai import OpenAI
import google.genai as genai


class ProviderClientRegistry:
    _clients: dict[str, object] = {}

    @classmethod
    def get_openai_client(cls) -> OpenAI:
        if "openai" not in cls._clients:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            cls._clients["openai"] = OpenAI(api_key=api_key)
        return cls._clients["openai"]

    @classmethod
    def get_gemini_client(cls) -> genai.Client:
        if "gemini" not in cls._clients:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY not set")
            cls._clients["gemini"] = genai.Client(api_key=api_key)
        return cls._clients["gemini"]
