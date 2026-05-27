
from langchain_ollama import OllamaEmbeddings
embed = OllamaEmbeddings(model="nomic-embed-text:latest",base_url="http://localhost:11434")
input_text = "King cross the road"
vector = embed.embed_query(input_text)
print(vector[:3])
      