import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://127.0.0.1:11434/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "ollama")
VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "ministral-3:latest")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "ministral-3:latest")

validation_llm = ChatOpenAI(
    openai_api_base=LITELLM_API_BASE,
    api_key=LITELLM_API_KEY,
    model=VALIDATION_MODEL,
    temperature=0.0,
    max_tokens=2048,
)

answer_llm = ChatOpenAI(
    openai_api_base=LITELLM_API_BASE,
    api_key=LITELLM_API_KEY,
    model=ANSWER_MODEL,
    temperature=0.3,
    max_tokens=2048,
)