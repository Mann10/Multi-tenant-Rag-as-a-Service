import os
import voyageai
from typing import List
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
vo = voyageai.Client(api_key=VOYAGE_API_KEY)


def get_embeddings(texts: List[str],input_type: str = "document", batch_size: int = 128) -> List[List[float]]:
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = vo.embed(
            texts=batch,
            model="voyage-3.5",
            input_type=input_type,
            truncation=True
        )
        all_embeddings.extend(result.embeddings)

    return all_embeddings