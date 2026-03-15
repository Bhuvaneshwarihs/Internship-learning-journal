import chromadb

client = chromadb.PersistentClient(path="vector_db")

collection = client.get_or_create_collection(name="lost_items")

count = collection.count()

print("Total items in DB:", count)