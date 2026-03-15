from PIL import Image
import os

folder="dataset/images"

for file in os.listdir(folder):

    path=os.path.join(folder,file)

    try:
        img=Image.open(path).convert("RGB")
        img=img.resize((256,256))
        img.save(path)

    except:
        print("Skipped",file)

print("Images resized successfully")