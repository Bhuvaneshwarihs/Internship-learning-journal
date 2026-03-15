import os
import pandas as pd
from duckduckgo_search import DDGS
import requests
import time

categories = {
    "laptops": 70,
    "watches": 70,
    "headphones": 70,
    "earbuds": 70,
    "shoes": 70,
    "speakers": 50,
    "phones": 50,
    "bottles": 30,
    "chairs": 20
}

os.makedirs("dataset/images", exist_ok=True)

data = []
image_id = 1

with DDGS() as ddgs:

    for category, count in categories.items():

        print(f"\nDownloading {category} images...")

        results = ddgs.images(
            query=category + " product",
            max_results=count * 2
        )

        downloaded = 0

        for r in results:

            if downloaded >= count:
                break

            try:
                url = r["image"]

                response = requests.get(url, timeout=5)

                if response.status_code == 200:

                    filename = f"{image_id}.jpg"
                    path = f"dataset/images/{filename}"

                    with open(path, "wb") as f:
                        f.write(response.content)

                    product_name = f"{category}_{image_id}"
                    description = f"Lost {category} item found on campus"

                    data.append([
                        image_id,
                        product_name,
                        description,
                        category,
                        path
                    ])

                    print("Saved", filename)

                    image_id += 1
                    downloaded += 1

                    time.sleep(1)

            except Exception as e:
                continue


df = pd.DataFrame(data, columns=[
    "id",
    "product_name",
    "description",
    "category",
    "image_path"
])

df.to_csv("dataset/lost_found_dataset_cleaned.csv", index=False)

print("\n✅ Dataset created successfully")
print("Total images:", len(df))