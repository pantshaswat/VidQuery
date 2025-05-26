# search_captions.py

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "caption-embeddings"

# Load BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

def get_query_embedding(query):
    return model.encode(query).tolist()

def search_query(embedding):
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=5
    )
    return results

if __name__ == "__main__":
    query = input("Enter caption-style query: ")
    embedding = get_query_embedding(query)
    results = search_query(embedding)

    print("\nTop matches:")
    for res in results:
        print(f"Timestamp: {res.payload['timestamp']}s, Caption: {res.payload['caption']}, Score: {res.score:.4f}")
