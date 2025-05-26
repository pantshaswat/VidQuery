# store_caption_embeddings.py

import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "caption-embeddings"
CAPTION_FILE = "captions.txt"

# Load BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

# Create Qdrant collection if it doesn't exist
def setup_qdrant():
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

# Parse captions file
def load_captions():
    captions = []
    with open(CAPTION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                timestamp_part, text = line.split("]", 1)
                timestamp = float(timestamp_part.strip("[").split("-")[0].replace("s", "").strip())
                captions.append((timestamp, text.strip()))
            except Exception as e:
                print(f"Skipping line due to error: {e}")
    return captions

# Generate and upload embeddings
def upload_caption_embeddings():
    captions = load_captions()
    texts = [text for _, text in captions]
    embeddings = model.encode(texts, convert_to_numpy=True)

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=embedding.tolist(), payload={"timestamp": timestamp, "caption": text})
        for (timestamp, text), embedding in zip(captions, embeddings)
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Uploaded {len(points)} caption embeddings to Qdrant.")

if __name__ == "__main__":
    setup_qdrant()
    upload_caption_embeddings()
