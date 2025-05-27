import torch
import clip
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Constants
VIDEO_COLLECTION = "video-frames"
CAPTION_COLLECTION = "caption-embeddings"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

# Scene-based (CLIP) embedding
def get_scene_embedding(query):
    text_token = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_token).squeeze().cpu().numpy()
    return text_embedding

# Caption-based (BERT) embedding
def get_caption_embedding(query):
    return bert_model.encode(query).tolist()

# Generic search
def search_qdrant(collection_name, embedding, top_k=5):
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=top_k
    )
    return results

def main():
    print("🎬 Welcome to Smart Video Search")
    print("Choose search type:")
    print("1. Scene-based search (visual description)")
    print("2. Dialogue-based search (spoken words)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        query = input("Describe the scene you're looking for (e.g., 'a man making coffee'): ")
        embedding = get_scene_embedding(query)
        results = search_qdrant(VIDEO_COLLECTION, embedding)

        print("\n🔍 Top matching scene timestamps:")
        for r in results:
            print(f"⏱️ Timestamp: {r.payload['timestamp']}s | Score: {r.score:.4f}")

    elif choice == "2":
        query = input("Type part of a sentence or dialogue (e.g., 'strawberries are added'): ")
        embedding = get_caption_embedding(query)
        results = search_qdrant(CAPTION_COLLECTION, embedding)

        print("\n🔍 Top matching captions:")
        for r in results:
            print(f"⏱️ Timestamp: {r.payload['timestamp']}s | 💬 Caption: {r.payload['caption']} | Score: {r.score:.4f}")

    else:
        print("❌ Invalid option. Please run the script again and choose 1 or 2.")

if __name__ == "__main__":
    main()
