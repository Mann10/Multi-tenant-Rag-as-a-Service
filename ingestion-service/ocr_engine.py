import base64
import httpx

OLLAMA_HOST = "http://127.0.0.1:11434"

def ocr_image(image_path: str):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": "glm-ocr:latest",
        "messages": [
            {
                "role": "user",
                "content": "Text Recognition:",
                "images": [image_b64]
            }
        ],
        "stream": False
    }

    response = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()
    return result["message"]["content"]
