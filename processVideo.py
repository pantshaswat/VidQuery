# process_video.py

import cv2
import torch
import clip
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

COLLECTION_NAME = "video-frames"

# Setup CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Connect to Qdrant
qdrant = QdrantClient("localhost", port=6333)

# Create collection if not exists
def setup_qdrant():
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

def extract_embeddings(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_id = 0
    points = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_id % frame_rate == 0:  # Every 1 second
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_input = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model.encode_image(image_input).squeeze().cpu().numpy()

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # seconds
            points.append(
                PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"timestamp": timestamp})
            )

        frame_id += 1

    cap.release()
    return points

def upload_embeddings(points):
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Uploaded {len(points)} frame embeddings to Qdrant.")

if __name__ == "__main__":
    setup_qdrant()
    points = extract_embeddings("Video/Coffee.mp4")
    upload_embeddings(points)
