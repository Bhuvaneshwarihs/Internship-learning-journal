import chromadb
import open_clip
import torch

print("Loading model...")

model, _, _ = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)

tokenizer = open_clip.get_tokenizer('ViT-B-32')

device = "cpu"
model = model.to(device)

client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection(name="lost_items")

query = input("Enter lost item description: ")

tokens = tokenizer([query]).to(device)

with torch.no_grad():
    text_features = model.encode_text(tokens)

query_embedding = text_features[0].cpu().numpy().tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=4
)

print("\nTop Matches:\n")

for i in range(len(results["ids"][0])):

    meta = results["metadatas"][0][i]

    print("Product:", meta["product_name"])
    print("Category:", meta["category"])
    print("Image:", meta["image_path"])
    print("Description:", meta["description"])
    print("---------------------")