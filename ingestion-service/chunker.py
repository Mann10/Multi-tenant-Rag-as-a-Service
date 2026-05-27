import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from typing import List
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def chunk_text(text: str, strategy: str = "recursive") -> List[str]:
    if not text or not text.strip():
        return []

    if strategy == "semantic":
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url="http://localhost:11434"
        )
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    return splitter.split_text(text)