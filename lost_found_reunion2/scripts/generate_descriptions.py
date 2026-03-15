import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Loading caption model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

df = pd.read_csv("dataset/lost_found_dataset_cleaned.csv")

new_descriptions = []

for i, row in df.iterrows():

    try:

        image = Image.open(row["image_path"]).convert("RGB")

        inputs = processor(image, return_tensors="pt")

        out = model.generate(**inputs)

        caption = processor.decode(out[0], skip_special_tokens=True)

        new_descriptions.append(caption)

        print(i + 1, caption)

    except:

        new_descriptions.append("Lost item found on campus")

df["description"] = new_descriptions

df.to_csv("dataset/lost_found_dataset_cleaned.csv", index=False)

print("Descriptions updated successfully")