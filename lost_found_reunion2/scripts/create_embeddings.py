import pandas as pd
from PIL import Image
import torch
import chromadb
import open_clip

print("Loading OpenCLIP model...")

# Load OpenCLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

device = "cpu"
model = model.to(device)

# Connect vector database
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_or_create_collection(name="lost_items")

# Load dataset
df = pd.read_csv("dataset/lost_found_dataset_cleaned.csv")

success = 0
failed = 0

for _, row in df.iterrows():

    try:

        # Load and preprocess image
        image = preprocess(
            Image.open(row["image_path"]).convert("RGB")
        ).unsqueeze(0).to(device)

        # Generate image embedding
        with torch.no_grad():
            image_features = model.encode_image(image)

        # Normalize embedding (important for similarity search)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embedding = image_features[0].cpu().numpy().tolist()

        # Store embedding in vector DB
        collection.add(
            embeddings=[embedding],
            ids=[str(row["id"])],
            metadatas=[{
                "product_name": row["product_name"],
                "category": row["category"],
                "image_path": row["image_path"],
                "description": row["description"]
            }]
        )

        success += 1
        print("Embedded:", row["image_path"])

    except Exception as e:

        failed += 1
        print("Failed:", row["image_path"])
        print("Reason:", e)

print("\nEmbedding complete")
print("Success:", success)
print("Failed:", failed)