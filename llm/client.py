import os
import asyncio
import google.genai as genai
from google.genai import types


def create_gemini_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    return genai.Client(api_key=api_key)
