# main.py

import torch
import clip
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest

COLLECTION_NAME = "video-frames"

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

def get_query_embedding(text):
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_token).squeeze().cpu().numpy()
    return text_embedding

def search_qdrant(text_embedding):
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=text_embedding,
        limit=5
    )
    return results

if __name__ == "__main__":
    user_query = input("Search your video (e.g., 'a dog running'): ")
    embedding = get_query_embedding(user_query)
    results = search_qdrant(embedding)

    print("\nTop matching timestamps:")
    for r in results:
        print(f"Timestamp: {r.payload['timestamp']}s, Score: {r.score:.4f}")
